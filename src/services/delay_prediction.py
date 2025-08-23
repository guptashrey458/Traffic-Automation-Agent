"""Delay risk prediction models using LightGBM for flight scheduling optimization."""

import os
import pickle
import warnings
from datetime import datetime, date, time, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from pathlib import Path

# Suppress LightGBM warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')

try:
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        mean_absolute_error, mean_squared_error, roc_auc_score,
        classification_report, confusion_matrix
    )
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn and/or lightgbm not available. Delay prediction will use fallback methods.")

from ..models.flight import Flight
from .database import FlightDatabaseService, QueryResult


class DelayRiskLevel(Enum):
    """Delay risk classification levels."""
    LOW = "low"           # < 5% probability of >15min delay
    MODERATE = "moderate" # 5-20% probability
    HIGH = "high"         # 20-50% probability
    CRITICAL = "critical" # > 50% probability


class PredictionConfidence(Enum):
    """Confidence levels for predictions."""
    LOW = "low"       # < 70% model confidence
    MEDIUM = "medium" # 70-85% confidence
    HIGH = "high"     # > 85% confidence


@dataclass
class DelayPrediction:
    """Delay prediction result with confidence metrics."""
    flight_id: str
    prediction_type: str  # "departure" or "arrival"
    
    # Binary classification results
    delay_probability: float  # Probability of >15min delay
    risk_level: DelayRiskLevel
    is_likely_delayed: bool
    
    # Regression results
    expected_delay_minutes: float
    delay_range_min: float  # Lower bound of prediction interval
    delay_range_max: float  # Upper bound of prediction interval
    
    # Model confidence and features
    confidence: PredictionConfidence
    confidence_score: float
    key_risk_factors: List[str] = field(default_factory=list)
    
    # Metadata
    prediction_timestamp: datetime = field(default_factory=datetime.now)
    model_version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert prediction to dictionary for API responses."""
        return {
            "flight_id": self.flight_id,
            "prediction_type": self.prediction_type,
            "delay_probability": round(self.delay_probability, 3),
            "risk_level": self.risk_level.value,
            "is_likely_delayed": self.is_likely_delayed,
            "expected_delay_minutes": round(self.expected_delay_minutes, 1),
            "delay_range": {
                "min": round(self.delay_range_min, 1),
                "max": round(self.delay_range_max, 1)
            },
            "confidence": self.confidence.value,
            "confidence_score": round(self.confidence_score, 3),
            "key_risk_factors": self.key_risk_factors,
            "prediction_timestamp": self.prediction_timestamp.isoformat(),
            "model_version": self.model_version
        }


@dataclass
class TurnaroundEstimate:
    """Turnaround time estimation for same-tail operations."""
    aircraft_registration: str
    airport_code: str
    
    # Turnaround time estimates
    p50_turnaround_minutes: float  # Median turnaround time
    p90_turnaround_minutes: float  # 90th percentile (planning buffer)
    p95_turnaround_minutes: float  # 95th percentile (conservative buffer)
    
    # Historical context
    sample_size: int
    min_observed: float
    max_observed: float
    
    # Operational factors
    aircraft_type: str
    typical_route_type: str  # "domestic", "international", "mixed"
    
    def is_feasible_departure(self, arrival_time: datetime, 
                            departure_time: datetime) -> bool:
        """Check if departure time is feasible given arrival time."""
        turnaround_minutes = (departure_time - arrival_time).total_seconds() / 60
        return turnaround_minutes >= self.p90_turnaround_minutes


@dataclass
class TaxiTimeEstimate:
    """Taxi time estimation for EXOT/EXIN calculations."""
    airport_code: str
    runway: str
    operation_type: str  # "departure" or "arrival"
    
    # Taxi time estimates
    expected_taxi_minutes: float
    p90_taxi_minutes: float
    
    # Factors affecting taxi time
    terminal_distance: str = "unknown"  # "near", "medium", "far"
    congestion_factor: float = 1.0
    weather_impact: float = 1.0


@dataclass
class DelayRiskResult:
    """Result of delay risk analysis for a flight."""
    flight_id: str
    departure_risk: 'DelayPrediction'
    arrival_risk: 'DelayPrediction'
    risk_factors: List[str]
    recommendations: List[str]


@dataclass
class OperationalContext:
    """Operational context for delay prediction."""
    airport_code: str
    analysis_datetime: datetime
    
    # Traffic conditions
    current_demand: int = 0
    runway_capacity: int = 30
    utilization_rate: float = 0.0
    
    # Weather conditions
    weather_regime: str = "calm"  # "calm", "medium", "strong", "severe"
    visibility_km: Optional[float] = None
    wind_speed_kts: Optional[float] = None
    
    # Time-based factors
    is_peak_hour: bool = False
    is_weekend: bool = False
    is_holiday: bool = False
    
    # Historical delay context
    recent_avg_delay: float = 0.0
    cascade_risk_score: float = 0.0


class DelayRiskPredictor:
    """Machine learning-based delay risk prediction system."""
    
    def __init__(self, db_service: Optional[FlightDatabaseService] = None,
                 model_dir: str = "models"):
        """
        Initialize the delay risk predictor.
        
        Args:
            db_service: Database service for historical data
            model_dir: Directory to store trained models
        """
        self.db_service = db_service or FlightDatabaseService()
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Model storage
        self.departure_classifier = None
        self.departure_regressor = None
        self.arrival_classifier = None
        self.arrival_regressor = None
        
        # Feature encoders
        self.label_encoders = {}
        
        # Model metadata
        self.model_trained = False
        self.training_date = None
        self.feature_importance = {}
        
        # Fallback statistics for when ML models aren't available
        self.fallback_stats = {
            'airline_delay_rates': {},
            'airport_delay_rates': {},
            'time_delay_patterns': {},
            'aircraft_delay_rates': {}
        }
        
        # Load existing models if available
        self._load_models()
    
    def predict_departure_delay(self, flight: Flight, 
                              context: OperationalContext) -> DelayPrediction:
        """
        Predict departure delay risk for a flight.
        
        Args:
            flight: Flight object with route and timing information
            context: Operational context including traffic and weather
            
        Returns:
            DelayPrediction with probability and expected delay
        """
        if not SKLEARN_AVAILABLE or not self.model_trained:
            return self._fallback_departure_prediction(flight, context)
        
        # Extract features for prediction
        features = self._extract_features(flight, context, "departure")
        feature_df = pd.DataFrame([features])
        
        # Make predictions
        try:
            # Binary classification (probability of >15min delay)
            delay_prob = self.departure_classifier.predict_proba(feature_df)[0][1]
            
            # Regression (expected delay minutes)
            expected_delay = max(0, self.departure_regressor.predict(feature_df)[0])
            
            # Calculate prediction intervals (simple approach using model uncertainty)
            delay_std = expected_delay * 0.3  # Assume 30% standard deviation
            delay_range_min = max(0, expected_delay - 1.96 * delay_std)
            delay_range_max = expected_delay + 1.96 * delay_std
            
        except Exception as e:
            print(f"Error in ML prediction: {e}")
            return self._fallback_departure_prediction(flight, context)
        
        # Classify risk level
        risk_level = self._classify_risk_level(delay_prob)
        is_likely_delayed = delay_prob > 0.5
        
        # Determine confidence
        confidence, confidence_score = self._calculate_confidence(features, delay_prob)
        
        # Identify key risk factors
        key_factors = self._identify_risk_factors(features, "departure")
        
        return DelayPrediction(
            flight_id=flight.flight_id,
            prediction_type="departure",
            delay_probability=delay_prob,
            risk_level=risk_level,
            is_likely_delayed=is_likely_delayed,
            expected_delay_minutes=expected_delay,
            delay_range_min=delay_range_min,
            delay_range_max=delay_range_max,
            confidence=confidence,
            confidence_score=confidence_score,
            key_risk_factors=key_factors
        )
    
    def predict_arrival_delay(self, flight: Flight, 
                            context: OperationalContext) -> DelayPrediction:
        """
        Predict arrival delay risk for a flight.
        
        Args:
            flight: Flight object with route and timing information
            context: Operational context including traffic and weather
            
        Returns:
            DelayPrediction with probability and expected delay
        """
        if not SKLEARN_AVAILABLE or not self.model_trained:
            return self._fallback_arrival_prediction(flight, context)
        
        # Extract features for prediction
        features = self._extract_features(flight, context, "arrival")
        feature_df = pd.DataFrame([features])
        
        # Make predictions
        try:
            # Binary classification
            delay_prob = self.arrival_classifier.predict_proba(feature_df)[0][1]
            
            # Regression
            expected_delay = max(0, self.arrival_regressor.predict(feature_df)[0])
            
            # Prediction intervals
            delay_std = expected_delay * 0.3
            delay_range_min = max(0, expected_delay - 1.96 * delay_std)
            delay_range_max = expected_delay + 1.96 * delay_std
            
        except Exception as e:
            print(f"Error in ML prediction: {e}")
            return self._fallback_arrival_prediction(flight, context)
        
        # Classify and format results
        risk_level = self._classify_risk_level(delay_prob)
        is_likely_delayed = delay_prob > 0.5
        confidence, confidence_score = self._calculate_confidence(features, delay_prob)
        key_factors = self._identify_risk_factors(features, "arrival")
        
        return DelayPrediction(
            flight_id=flight.flight_id,
            prediction_type="arrival",
            delay_probability=delay_prob,
            risk_level=risk_level,
            is_likely_delayed=is_likely_delayed,
            expected_delay_minutes=expected_delay,
            delay_range_min=delay_range_min,
            delay_range_max=delay_range_max,
            confidence=confidence,
            confidence_score=confidence_score,
            key_risk_factors=key_factors
        )
    
    def estimate_turnaround_time(self, aircraft_registration: str, 
                               airport_code: str) -> TurnaroundEstimate:
        """
        Estimate turnaround time for same-tail operations.
        
        Args:
            aircraft_registration: Aircraft tail number
            airport_code: Airport where turnaround occurs
            
        Returns:
            TurnaroundEstimate with P90 and other percentiles
        """
        try:
            # Query historical turnaround data
            turnaround_data = self._get_turnaround_data(aircraft_registration, airport_code)
            
            if not turnaround_data:
                # Use default estimates based on aircraft type and airport
                return self._default_turnaround_estimate(aircraft_registration, airport_code)
            
            # Calculate percentiles
            turnaround_times = [t['turnaround_minutes'] for t in turnaround_data]
            p50 = np.percentile(turnaround_times, 50)
            p90 = np.percentile(turnaround_times, 90)
            p95 = np.percentile(turnaround_times, 95)
            
            # Determine aircraft type and route type
            aircraft_type = turnaround_data[0].get('aircraft_type', 'UNKNOWN')
            route_types = [t.get('route_type', 'domestic') for t in turnaround_data]
            typical_route_type = max(set(route_types), key=route_types.count)
            
            return TurnaroundEstimate(
                aircraft_registration=aircraft_registration,
                airport_code=airport_code,
                p50_turnaround_minutes=p50,
                p90_turnaround_minutes=p90,
                p95_turnaround_minutes=p95,
                sample_size=len(turnaround_times),
                min_observed=min(turnaround_times),
                max_observed=max(turnaround_times),
                aircraft_type=aircraft_type,
                typical_route_type=typical_route_type
            )
            
        except Exception as e:
            print(f"Error estimating turnaround time: {e}")
            return self._default_turnaround_estimate(aircraft_registration, airport_code)
    
    def predict_taxi_time(self, flight: Flight, runway: str) -> TaxiTimeEstimate:
        """
        Predict taxi time for EXOT/EXIN calculations.
        
        Args:
            flight: Flight object
            runway: Runway identifier
            
        Returns:
            TaxiTimeEstimate for the operation
        """
        # Determine operation type
        operation_type = "departure" if flight.origin else "arrival"
        airport_code = flight.origin.code if flight.origin else flight.destination.code
        
        try:
            # Query historical taxi time data
            taxi_data = self._get_taxi_time_data(airport_code, runway, operation_type)
            
            if taxi_data:
                taxi_times = [t['taxi_minutes'] for t in taxi_data]
                expected_taxi = np.mean(taxi_times)
                p90_taxi = np.percentile(taxi_times, 90)
            else:
                # Use default estimates
                expected_taxi, p90_taxi = self._default_taxi_estimates(airport_code, operation_type)
            
            return TaxiTimeEstimate(
                airport_code=airport_code,
                runway=runway,
                operation_type=operation_type,
                expected_taxi_minutes=expected_taxi,
                p90_taxi_minutes=p90_taxi
            )
            
        except Exception as e:
            print(f"Error predicting taxi time: {e}")
            expected_taxi, p90_taxi = self._default_taxi_estimates(airport_code, operation_type)
            
            return TaxiTimeEstimate(
                airport_code=airport_code,
                runway=runway,
                operation_type=operation_type,
                expected_taxi_minutes=expected_taxi,
                p90_taxi_minutes=p90_taxi
            )
    
    def train_models(self, start_date: date, end_date: date, 
                    airports: List[str] = None) -> Dict[str, Any]:
        """
        Train delay prediction models using historical data.
        
        Args:
            start_date: Start date for training data
            end_date: End date for training data
            airports: List of airports to include (None for all)
            
        Returns:
            Training results and model performance metrics
        """
        if not SKLEARN_AVAILABLE:
            return {"error": "scikit-learn and lightgbm are required for model training"}
        
        print(f"Training delay prediction models from {start_date} to {end_date}")
        
        # Load training data
        training_data = self._load_training_data(start_date, end_date, airports)
        
        if len(training_data) < 100:
            return {"error": f"Insufficient training data: {len(training_data)} samples"}
        
        print(f"Loaded {len(training_data)} training samples")
        
        # Prepare features and targets
        features_df = pd.DataFrame([self._extract_features_from_historical(row) 
                                  for row in training_data])
        
        # Train departure models
        dep_results = self._train_departure_models(features_df, training_data)
        
        # Train arrival models
        arr_results = self._train_arrival_models(features_df, training_data)
        
        # Save models
        self._save_models()
        
        # Update training metadata
        self.model_trained = True
        self.training_date = datetime.now()
        
        return {
            "training_samples": len(training_data),
            "training_date": self.training_date.isoformat(),
            "departure_models": dep_results,
            "arrival_models": arr_results,
            "feature_importance": self.feature_importance
        } 
   
    def _extract_features(self, flight: Flight, context: OperationalContext, 
                         prediction_type: str) -> Dict[str, Any]:
        """Extract features for ML prediction."""
        features = {}
        
        # Time-based features
        if flight.departure.scheduled:
            features['hour_of_day'] = flight.departure.scheduled.hour
            features['minute_of_hour'] = flight.departure.scheduled.minute
            features['day_of_week'] = flight.flight_date.weekday() if flight.flight_date else 0
            features['is_weekend'] = context.is_weekend
            features['is_peak_hour'] = context.is_peak_hour
        else:
            features['hour_of_day'] = 12  # Default to noon
            features['minute_of_hour'] = 0
            features['day_of_week'] = 0
            features['is_weekend'] = False
            features['is_peak_hour'] = False
        
        # Airline features
        features['airline_code'] = flight.airline_code or 'UNKNOWN'
        
        # Route features
        features['origin_code'] = flight.origin.code if flight.origin else 'UNKNOWN'
        features['destination_code'] = flight.destination.code if flight.destination else 'UNKNOWN'
        features['route_key'] = flight.get_route_key()
        
        # Aircraft features
        features['aircraft_type'] = flight.aircraft_type or 'UNKNOWN'
        
        # Operational context features
        features['current_demand'] = context.current_demand
        features['runway_capacity'] = context.runway_capacity
        features['utilization_rate'] = context.utilization_rate
        features['weather_regime'] = context.weather_regime
        features['recent_avg_delay'] = context.recent_avg_delay
        features['cascade_risk_score'] = context.cascade_risk_score
        
        # Weather features
        features['visibility_km'] = context.visibility_km or 10.0
        features['wind_speed_kts'] = context.wind_speed_kts or 5.0
        
        return features
    
    def _extract_features_from_historical(self, flight_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from historical flight data for training."""
        features = {}
        
        # Parse datetime fields
        std_utc = flight_data.get('std_utc')
        if isinstance(std_utc, str):
            std_utc = datetime.fromisoformat(std_utc.replace('Z', '+00:00'))
        
        if std_utc:
            features['hour_of_day'] = std_utc.hour
            features['minute_of_hour'] = std_utc.minute
            features['day_of_week'] = std_utc.weekday()
            features['is_weekend'] = std_utc.weekday() >= 5
            features['is_peak_hour'] = 7 <= std_utc.hour <= 9 or 17 <= std_utc.hour <= 19
        else:
            features['hour_of_day'] = 12
            features['minute_of_hour'] = 0
            features['day_of_week'] = 0
            features['is_weekend'] = False
            features['is_peak_hour'] = False
        
        # Flight features
        features['airline_code'] = flight_data.get('airline_code', 'UNKNOWN')
        features['origin_code'] = flight_data.get('origin_code', 'UNKNOWN')
        features['destination_code'] = flight_data.get('destination_code', 'UNKNOWN')
        features['aircraft_type'] = flight_data.get('aircraft_type', 'UNKNOWN')
        features['route_key'] = f"{features['origin_code']}-{features['destination_code']}"
        
        # Default operational features (would be enhanced with real-time data)
        features['current_demand'] = 15  # Average demand
        features['runway_capacity'] = 30
        features['utilization_rate'] = 0.5
        features['weather_regime'] = 'calm'
        features['recent_avg_delay'] = flight_data.get('dep_delay_min', 0) or 0
        features['cascade_risk_score'] = 0.0
        features['visibility_km'] = 10.0
        features['wind_speed_kts'] = 5.0
        
        return features
    
    def _train_departure_models(self, features_df: pd.DataFrame, 
                              training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train departure delay prediction models."""
        # Prepare targets
        dep_delays = [max(0, row.get('dep_delay_min', 0) or 0) for row in training_data]
        dep_binary = [1 if delay > 15 else 0 for delay in dep_delays]
        
        # Encode categorical features
        features_encoded = self._encode_features(features_df.copy())
        
        # Split data
        X_train, X_test, y_class_train, y_class_test = train_test_split(
            features_encoded, dep_binary, test_size=0.2, random_state=42
        )
        _, _, y_reg_train, y_reg_test = train_test_split(
            features_encoded, dep_delays, test_size=0.2, random_state=42
        )
        
        # Train binary classifier
        self.departure_classifier = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            verbose=-1
        )
        self.departure_classifier.fit(X_train, y_class_train)
        
        # Train regressor
        self.departure_regressor = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            verbose=-1
        )
        self.departure_regressor.fit(X_train, y_reg_train)
        
        # Evaluate models
        class_pred = self.departure_classifier.predict(X_test)
        class_prob = self.departure_classifier.predict_proba(X_test)[:, 1]
        reg_pred = self.departure_regressor.predict(X_test)
        
        # Store feature importance
        self.feature_importance['departure_classifier'] = dict(
            zip(features_encoded.columns, self.departure_classifier.feature_importances_)
        )
        self.feature_importance['departure_regressor'] = dict(
            zip(features_encoded.columns, self.departure_regressor.feature_importances_)
        )
        
        return {
            "classifier": {
                "accuracy": accuracy_score(y_class_test, class_pred),
                "precision": precision_score(y_class_test, class_pred),
                "recall": recall_score(y_class_test, class_pred),
                "f1": f1_score(y_class_test, class_pred),
                "auc": roc_auc_score(y_class_test, class_prob)
            },
            "regressor": {
                "mae": mean_absolute_error(y_reg_test, reg_pred),
                "rmse": np.sqrt(mean_squared_error(y_reg_test, reg_pred))
            }
        }
    
    def _train_arrival_models(self, features_df: pd.DataFrame, 
                            training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train arrival delay prediction models."""
        # Prepare targets
        arr_delays = [max(0, row.get('arr_delay_min', 0) or 0) for row in training_data]
        arr_binary = [1 if delay > 15 else 0 for delay in arr_delays]
        
        # Use same encoded features
        features_encoded = self._encode_features(features_df.copy())
        
        # Split data
        X_train, X_test, y_class_train, y_class_test = train_test_split(
            features_encoded, arr_binary, test_size=0.2, random_state=42
        )
        _, _, y_reg_train, y_reg_test = train_test_split(
            features_encoded, arr_delays, test_size=0.2, random_state=42
        )
        
        # Train models
        self.arrival_classifier = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            verbose=-1
        )
        self.arrival_classifier.fit(X_train, y_class_train)
        
        self.arrival_regressor = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            verbose=-1
        )
        self.arrival_regressor.fit(X_train, y_reg_train)
        
        # Evaluate models
        class_pred = self.arrival_classifier.predict(X_test)
        class_prob = self.arrival_classifier.predict_proba(X_test)[:, 1]
        reg_pred = self.arrival_regressor.predict(X_test)
        
        # Store feature importance
        self.feature_importance['arrival_classifier'] = dict(
            zip(features_encoded.columns, self.arrival_classifier.feature_importances_)
        )
        self.feature_importance['arrival_regressor'] = dict(
            zip(features_encoded.columns, self.arrival_regressor.feature_importances_)
        )
        
        return {
            "classifier": {
                "accuracy": accuracy_score(y_class_test, class_pred),
                "precision": precision_score(y_class_test, class_pred),
                "recall": recall_score(y_class_test, class_pred),
                "f1": f1_score(y_class_test, class_pred),
                "auc": roc_auc_score(y_class_test, class_prob)
            },
            "regressor": {
                "mae": mean_absolute_error(y_reg_test, reg_pred),
                "rmse": np.sqrt(mean_squared_error(y_reg_test, reg_pred))
            }
        }
    
    def _encode_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features for ML models."""
        categorical_columns = [
            'airline_code', 'origin_code', 'destination_code', 
            'aircraft_type', 'route_key', 'weather_regime'
        ]
        
        for col in categorical_columns:
            if col in features_df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    features_df[col] = self.label_encoders[col].fit_transform(
                        features_df[col].astype(str)
                    )
                else:
                    # Handle unseen categories
                    unique_values = set(self.label_encoders[col].classes_)
                    features_df[col] = features_df[col].astype(str).apply(
                        lambda x: x if x in unique_values else 'UNKNOWN'
                    )
                    features_df[col] = self.label_encoders[col].transform(features_df[col])
        
        return features_df
    
    def _classify_risk_level(self, delay_probability: float) -> DelayRiskLevel:
        """Classify delay risk level based on probability."""
        if delay_probability < 0.05:
            return DelayRiskLevel.LOW
        elif delay_probability < 0.20:
            return DelayRiskLevel.MODERATE
        elif delay_probability < 0.50:
            return DelayRiskLevel.HIGH
        else:
            return DelayRiskLevel.CRITICAL
    
    def _calculate_confidence(self, features: Dict[str, Any], 
                            delay_prob: float) -> Tuple[PredictionConfidence, float]:
        """Calculate prediction confidence based on feature quality and model certainty."""
        confidence_score = 0.8  # Base confidence
        
        # Reduce confidence for missing or default values
        if features.get('airline_code') == 'UNKNOWN':
            confidence_score -= 0.1
        if features.get('aircraft_type') == 'UNKNOWN':
            confidence_score -= 0.1
        if features.get('recent_avg_delay', 0) == 0:
            confidence_score -= 0.05
        
        # Adjust based on prediction certainty
        if 0.4 <= delay_prob <= 0.6:  # Uncertain predictions
            confidence_score -= 0.15
        elif delay_prob < 0.1 or delay_prob > 0.9:  # Very certain predictions
            confidence_score += 0.1
        
        # Classify confidence level
        if confidence_score >= 0.85:
            confidence = PredictionConfidence.HIGH
        elif confidence_score >= 0.70:
            confidence = PredictionConfidence.MEDIUM
        else:
            confidence = PredictionConfidence.LOW
        
        return confidence, max(0.0, min(1.0, confidence_score))
    
    def _identify_risk_factors(self, features: Dict[str, Any], 
                             prediction_type: str) -> List[str]:
        """Identify key risk factors contributing to delay prediction."""
        risk_factors = []
        
        # High utilization
        if features.get('utilization_rate', 0) > 0.8:
            risk_factors.append("High runway utilization")
        
        # Peak hour operations
        if features.get('is_peak_hour', False):
            risk_factors.append("Peak hour operations")
        
        # Weather conditions
        if features.get('weather_regime') in ['strong', 'severe']:
            risk_factors.append("Adverse weather conditions")
        
        # High recent delays
        if features.get('recent_avg_delay', 0) > 20:
            risk_factors.append("Recent high delays at airport")
        
        # Cascade risk
        if features.get('cascade_risk_score', 0) > 0.5:
            risk_factors.append("High cascade delay risk")
        
        return risk_factors
    
    def _fallback_departure_prediction(self, flight: Flight, 
                                     context: OperationalContext) -> DelayPrediction:
        """Fallback prediction method when ML models are not available."""
        # Use simple heuristics based on operational context
        base_delay_prob = 0.15  # Base 15% chance of delay
        expected_delay = 5.0    # Base 5 minutes expected delay
        
        # Adjust based on context
        if context.utilization_rate > 0.8:
            base_delay_prob += 0.2
            expected_delay += 10
        
        if context.is_peak_hour:
            base_delay_prob += 0.15
            expected_delay += 8
        
        if context.weather_regime in ['strong', 'severe']:
            base_delay_prob += 0.25
            expected_delay += 15
        
        if context.recent_avg_delay > 20:
            base_delay_prob += 0.1
            expected_delay += context.recent_avg_delay * 0.3
        
        # Cap probability at 0.9
        base_delay_prob = min(0.9, base_delay_prob)
        
        # Classify risk level
        risk_level = self._classify_risk_level(base_delay_prob)
        
        return DelayPrediction(
            flight_id=flight.flight_id,
            prediction_type="departure",
            delay_probability=base_delay_prob,
            risk_level=risk_level,
            is_likely_delayed=base_delay_prob > 0.5,
            expected_delay_minutes=expected_delay,
            delay_range_min=max(0, expected_delay - 10),
            delay_range_max=expected_delay + 20,
            confidence=PredictionConfidence.LOW,
            confidence_score=0.6,
            key_risk_factors=["Using fallback prediction method"]
        )
    
    def _fallback_arrival_prediction(self, flight: Flight, 
                                   context: OperationalContext) -> DelayPrediction:
        """Fallback prediction method for arrival delays."""
        # Similar to departure but with different base rates
        base_delay_prob = 0.12  # Slightly lower base rate for arrivals
        expected_delay = 4.0
        
        # Adjust based on context (similar logic to departure)
        if context.utilization_rate > 0.8:
            base_delay_prob += 0.15
            expected_delay += 8
        
        if context.is_peak_hour:
            base_delay_prob += 0.1
            expected_delay += 6
        
        if context.weather_regime in ['strong', 'severe']:
            base_delay_prob += 0.2
            expected_delay += 12
        
        # Cap probability
        base_delay_prob = min(0.9, base_delay_prob)
        
        risk_level = self._classify_risk_level(base_delay_prob)
        
        return DelayPrediction(
            flight_id=flight.flight_id,
            prediction_type="arrival",
            delay_probability=base_delay_prob,
            risk_level=risk_level,
            is_likely_delayed=base_delay_prob > 0.5,
            expected_delay_minutes=expected_delay,
            delay_range_min=max(0, expected_delay - 8),
            delay_range_max=expected_delay + 15,
            confidence=PredictionConfidence.LOW,
            confidence_score=0.6,
            key_risk_factors=["Using fallback prediction method"]
        )
    
    def _load_training_data(self, start_date: date, end_date: date, 
                          airports: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Load training data from the database."""
        try:
            conn = self.db_service.connect()
            
            base_query = """
            SELECT 
                flight_id, flight_number, airline_code, aircraft_type,
                origin_code, destination_code, route,
                flight_date, std_utc, atd_utc, sta_utc, ata_utc,
                dep_delay_min, arr_delay_min
            FROM flights
            WHERE flight_date BETWEEN ? AND ?
                AND std_utc IS NOT NULL
            """
            
            params = [start_date, end_date]
            
            if airports:
                placeholders = ','.join(['?' for _ in airports])
                base_query += f" AND (origin_code IN ({placeholders}) OR destination_code IN ({placeholders}))"
                params.extend(airports)
                params.extend(airports)
            
            base_query += " ORDER BY flight_date, std_utc"
            
            result = conn.execute(base_query, params).fetchall()
            columns = [desc[0] for desc in conn.description]
            
            return [dict(zip(columns, row)) for row in result]
            
        except Exception as e:
            print(f"Error loading training data: {e}")
            return []
    
    def _get_turnaround_data(self, aircraft_registration: str, 
                           airport_code: str) -> List[Dict[str, Any]]:
        """Query historical turnaround data for an aircraft at an airport."""
        try:
            conn = self.db_service.connect()
            
            # Query for same-tail turnarounds at the airport
            query = """
            WITH turnarounds AS (
                SELECT 
                    arr.aircraft_registration,
                    arr.ata_utc as arrival_time,
                    dep.std_utc as departure_time,
                    dep.aircraft_type,
                    arr.destination_code as turnaround_airport,
                    dep.origin_code,
                    CASE 
                        WHEN arr.route LIKE '%-%' AND LENGTH(arr.route) <= 7 THEN 'domestic'
                        ELSE 'international'
                    END as arrival_route_type,
                    CASE 
                        WHEN dep.route LIKE '%-%' AND LENGTH(dep.route) <= 7 THEN 'domestic'
                        ELSE 'international'
                    END as departure_route_type,
                    (EXTRACT(EPOCH FROM dep.std_utc - arr.ata_utc) / 60) as turnaround_minutes
                FROM flights arr
                JOIN flights dep ON (
                    arr.aircraft_registration = dep.aircraft_registration
                    AND arr.destination_code = dep.origin_code
                    AND dep.std_utc > arr.ata_utc
                    AND dep.std_utc <= arr.ata_utc + INTERVAL '24 hours'
                )
                WHERE arr.aircraft_registration = ?
                    AND arr.destination_code = ?
                    AND arr.ata_utc IS NOT NULL
                    AND dep.std_utc IS NOT NULL
                    AND (EXTRACT(EPOCH FROM dep.std_utc - arr.ata_utc) / 60) BETWEEN 30 AND 720
            )
            SELECT * FROM turnarounds
            ORDER BY arrival_time DESC
            LIMIT 50
            """
            
            result = conn.execute(query, [aircraft_registration, airport_code]).fetchall()
            columns = [desc[0] for desc in conn.description]
            
            return [dict(zip(columns, row)) for row in result]
            
        except Exception as e:
            print(f"Error querying turnaround data: {e}")
            return []
    
    def _get_taxi_time_data(self, airport_code: str, runway: str, 
                          operation_type: str) -> List[Dict[str, Any]]:
        """Query historical taxi time data for an airport and runway."""
        try:
            conn = self.db_service.connect()
            
            # For now, return empty list as we don't have taxi time data in the schema
            # This would be enhanced with actual taxi time tracking
            return []
            
        except Exception as e:
            print(f"Error querying taxi time data: {e}")
            return []
    
    def _default_turnaround_estimate(self, aircraft_registration: str, 
                                   airport_code: str) -> TurnaroundEstimate:
        """Generate default turnaround estimate when no historical data is available."""
        # Default turnaround times by aircraft type (in minutes)
        default_times = {
            "A320": {"p50": 45, "p90": 75, "p95": 90},
            "A321": {"p50": 50, "p90": 80, "p95": 95},
            "B737": {"p50": 45, "p90": 75, "p95": 90},
            "B738": {"p50": 45, "p90": 75, "p95": 90},
            "B777": {"p50": 120, "p90": 180, "p95": 210},
            "B787": {"p50": 110, "p90": 170, "p95": 200},
            "default": {"p50": 45, "p90": 75, "p95": 90}
        }
        
        # Try to extract aircraft type from registration
        aircraft_type = self._extract_aircraft_type_from_registration(aircraft_registration)
        defaults = default_times.get(aircraft_type, default_times["default"])
        
        return TurnaroundEstimate(
            aircraft_registration=aircraft_registration,
            airport_code=airport_code,
            p50_turnaround_minutes=defaults["p50"],
            p90_turnaround_minutes=defaults["p90"],
            p95_turnaround_minutes=defaults["p95"],
            sample_size=0,
            min_observed=defaults["p50"],
            max_observed=defaults["p95"],
            aircraft_type=aircraft_type,
            typical_route_type="domestic"
        )
    
    def _default_taxi_estimates(self, airport_code: str, 
                              operation_type: str) -> Tuple[float, float]:
        """Get default taxi time estimates for an airport and operation type."""
        # Default taxi times by airport (in minutes)
        defaults = {
            "BOM": {"departure": (15, 25), "arrival": (12, 20)},  # (expected, p90)
            "DEL": {"departure": (18, 28), "arrival": (15, 25)},
            "BLR": {"departure": (12, 20), "arrival": (10, 18)},
            "default": {"departure": (15, 25), "arrival": (12, 20)}
        }
        
        airport_defaults = defaults.get(airport_code, defaults["default"])
        return airport_defaults.get(operation_type, (15, 25))
    
    def _extract_aircraft_type_from_registration(self, registration: str) -> str:
        """Extract aircraft type from registration if possible."""
        if not registration:
            return "UNKNOWN"
        
        # Common patterns in Indian aircraft registrations
        registration_upper = registration.upper()
        
        if "320" in registration_upper:
            return "A320"
        elif "321" in registration_upper:
            return "A321"
        elif "737" in registration_upper or "738" in registration_upper:
            return "B737"
        elif "777" in registration_upper:
            return "B777"
        elif "787" in registration_upper:
            return "B787"
        else:
            return "UNKNOWN"
        
        # Weather conditions
        weather = features.get('weather_regime', 'calm')
        if weather in ['strong', 'severe']:
            risk_factors.append(f"Adverse weather ({weather})")
        
        # High recent delays
        if features.get('recent_avg_delay', 0) > 20:
            risk_factors.append("Recent high delays at airport")
        
        # Weekend operations
        if features.get('is_weekend', False):
            risk_factors.append("Weekend operations")
        
        # High cascade risk
        if features.get('cascade_risk_score', 0) > 0.5:
            risk_factors.append("High cascade delay risk")
        
        # Low visibility
        if features.get('visibility_km', 10) < 5:
            risk_factors.append("Low visibility conditions")
        
        return risk_factors[:5]  # Return top 5 factors
    
    def _fallback_departure_prediction(self, flight: Flight, 
                                     context: OperationalContext) -> DelayPrediction:
        """Fallback prediction method when ML models aren't available."""
        # Use simple heuristics based on operational factors
        base_delay_prob = 0.15  # Base 15% chance of delay
        expected_delay = 5.0    # Base 5 minutes expected delay
        
        # Adjust based on context
        if context.utilization_rate > 0.9:
            base_delay_prob += 0.3
            expected_delay += 15
        elif context.utilization_rate > 0.7:
            base_delay_prob += 0.15
            expected_delay += 8
        
        # Weather impact
        weather_multipliers = {
            'calm': 1.0,
            'medium': 1.3,
            'strong': 1.8,
            'severe': 2.5
        }
        multiplier = weather_multipliers.get(context.weather_regime, 1.0)
        base_delay_prob = min(0.95, base_delay_prob * multiplier)
        expected_delay *= multiplier
        
        # Peak hour impact
        if context.is_peak_hour:
            base_delay_prob += 0.1
            expected_delay += 5
        
        # Recent delay impact
        if context.recent_avg_delay > 20:
            base_delay_prob += 0.2
            expected_delay += context.recent_avg_delay * 0.3
        
        # Airline-specific adjustments (simplified)
        airline_factors = {
            'AI': 1.2,  # Air India - historically higher delays
            '6E': 0.9,  # IndiGo - generally punctual
            'UK': 1.1,  # Vistara
            'SG': 1.0   # SpiceJet
        }
        airline_factor = airline_factors.get(flight.airline_code, 1.0)
        base_delay_prob = min(0.95, base_delay_prob * airline_factor)
        expected_delay *= airline_factor
        
        # Calculate prediction intervals
        delay_std = expected_delay * 0.4
        delay_range_min = max(0, expected_delay - 1.96 * delay_std)
        delay_range_max = expected_delay + 1.96 * delay_std
        
        # Determine risk level and confidence
        risk_level = self._classify_risk_level(base_delay_prob)
        confidence = PredictionConfidence.MEDIUM  # Moderate confidence for heuristics
        
        # Identify risk factors
        risk_factors = []
        if context.utilization_rate > 0.8:
            risk_factors.append("High airport utilization")
        if context.weather_regime in ['strong', 'severe']:
            risk_factors.append(f"Adverse weather ({context.weather_regime})")
        if context.is_peak_hour:
            risk_factors.append("Peak hour operations")
        if context.recent_avg_delay > 20:
            risk_factors.append("Recent delays at airport")
        
        return DelayPrediction(
            flight_id=flight.flight_id,
            prediction_type="departure",
            delay_probability=base_delay_prob,
            risk_level=risk_level,
            is_likely_delayed=base_delay_prob > 0.5,
            expected_delay_minutes=expected_delay,
            delay_range_min=delay_range_min,
            delay_range_max=delay_range_max,
            confidence=confidence,
            confidence_score=0.7,
            key_risk_factors=risk_factors,
            model_version="fallback_1.0"
        )
    
    def _fallback_arrival_prediction(self, flight: Flight, 
                                   context: OperationalContext) -> DelayPrediction:
        """Fallback arrival prediction method."""
        # Arrival delays are often correlated with departure delays
        dep_prediction = self._fallback_departure_prediction(flight, context)
        
        # Arrival delays tend to be slightly lower due to airborne recovery
        arrival_prob = dep_prediction.delay_probability * 0.8
        arrival_delay = dep_prediction.expected_delay_minutes * 0.7
        
        # Adjust for destination airport conditions
        if context.airport_code in ['BOM', 'DEL']:  # Busy airports
            arrival_prob += 0.1
            arrival_delay += 3
        
        delay_std = arrival_delay * 0.4
        delay_range_min = max(0, arrival_delay - 1.96 * delay_std)
        delay_range_max = arrival_delay + 1.96 * delay_std
        
        risk_level = self._classify_risk_level(arrival_prob)
        
        return DelayPrediction(
            flight_id=flight.flight_id,
            prediction_type="arrival",
            delay_probability=arrival_prob,
            risk_level=risk_level,
            is_likely_delayed=arrival_prob > 0.5,
            expected_delay_minutes=arrival_delay,
            delay_range_min=delay_range_min,
            delay_range_max=delay_range_max,
            confidence=PredictionConfidence.MEDIUM,
            confidence_score=0.7,
            key_risk_factors=dep_prediction.key_risk_factors,
            model_version="fallback_1.0"
        )
    
    def _load_training_data(self, start_date: date, end_date: date, 
                          airports: List[str] = None) -> List[Dict[str, Any]]:
        """Load historical flight data for model training."""
        try:
            result = self.db_service.query_flights_by_date_range(
                start_date=start_date,
                end_date=end_date,
                airport_code=airports[0] if airports else None
            )
            
            # Filter for flights with delay information
            training_data = []
            for flight in result.data:
                if (flight.get('dep_delay_min') is not None or 
                    flight.get('arr_delay_min') is not None):
                    training_data.append(flight)
            
            return training_data
            
        except Exception as e:
            print(f"Error loading training data: {e}")
            return []
    
    def _get_turnaround_data(self, aircraft_registration: str, 
                           airport_code: str) -> List[Dict[str, Any]]:
        """Get historical turnaround data for an aircraft at an airport."""
        # This would query the database for same-tail operations
        # For now, return empty list to trigger default estimates
        return []
    
    def _default_turnaround_estimate(self, aircraft_registration: str, 
                                   airport_code: str) -> TurnaroundEstimate:
        """Provide default turnaround estimates based on aircraft type and airport."""
        # Extract aircraft type from registration if possible
        aircraft_type = "UNKNOWN"
        if aircraft_registration:
            if aircraft_registration.startswith("VT-"):
                # Indian aircraft registration
                if "320" in aircraft_registration or "319" in aircraft_registration:
                    aircraft_type = "A320"
                elif "737" in aircraft_registration or "738" in aircraft_registration:
                    aircraft_type = "B737"
        
        # Default turnaround times by aircraft type (in minutes)
        default_turnarounds = {
            "A320": {"p50": 45, "p90": 75, "p95": 90},
            "A321": {"p50": 50, "p90": 80, "p95": 95},
            "B737": {"p50": 45, "p90": 75, "p95": 90},
            "B738": {"p50": 45, "p90": 75, "p95": 90},
            "B777": {"p50": 60, "p90": 100, "p95": 120},
            "B787": {"p50": 55, "p90": 90, "p95": 110},
            "UNKNOWN": {"p50": 50, "p90": 80, "p95": 100}
        }
        
        turnaround = default_turnarounds.get(aircraft_type, default_turnarounds["UNKNOWN"])
        
        # Adjust for airport size
        if airport_code in ["BOM", "DEL"]:  # Major hubs
            multiplier = 1.2
        elif airport_code in ["BLR", "MAA", "HYD", "CCU"]:  # Large airports
            multiplier = 1.1
        else:  # Smaller airports
            multiplier = 0.9
        
        return TurnaroundEstimate(
            aircraft_registration=aircraft_registration,
            airport_code=airport_code,
            p50_turnaround_minutes=turnaround["p50"] * multiplier,
            p90_turnaround_minutes=turnaround["p90"] * multiplier,
            p95_turnaround_minutes=turnaround["p95"] * multiplier,
            sample_size=0,  # No historical data
            min_observed=turnaround["p50"] * 0.7 * multiplier,
            max_observed=turnaround["p95"] * 1.3 * multiplier,
            aircraft_type=aircraft_type,
            typical_route_type="domestic"
        )
    
    def _get_taxi_time_data(self, airport_code: str, runway: str, 
                          operation_type: str) -> List[Dict[str, Any]]:
        """Get historical taxi time data."""
        # This would query the database for taxi time data
        # For now, return empty list to trigger default estimates
        return []
    
    def _default_taxi_estimates(self, airport_code: str, 
                              operation_type: str) -> Tuple[float, float]:
        """Provide default taxi time estimates."""
        # Default taxi times by airport and operation type (in minutes)
        taxi_defaults = {
            "BOM": {"departure": (12, 18), "arrival": (8, 12)},
            "DEL": {"departure": (15, 22), "arrival": (10, 15)},
            "BLR": {"departure": (10, 15), "arrival": (7, 10)},
            "MAA": {"departure": (8, 12), "arrival": (6, 9)},
            "HYD": {"departure": (9, 13), "arrival": (6, 10)},
            "CCU": {"departure": (8, 12), "arrival": (6, 9)}
        }
        
        if airport_code in taxi_defaults:
            expected, p90 = taxi_defaults[airport_code][operation_type]
        else:
            # Default for smaller airports
            if operation_type == "departure":
                expected, p90 = 6, 10
            else:
                expected, p90 = 4, 7
        
        return float(expected), float(p90)
    
    def _save_models(self) -> None:
        """Save trained models to disk."""
        if not SKLEARN_AVAILABLE:
            return
        
        model_files = {
            'departure_classifier.pkl': self.departure_classifier,
            'departure_regressor.pkl': self.departure_regressor,
            'arrival_classifier.pkl': self.arrival_classifier,
            'arrival_regressor.pkl': self.arrival_regressor,
            'label_encoders.pkl': self.label_encoders,
            'feature_importance.pkl': self.feature_importance
        }
        
        for filename, model in model_files.items():
            if model is not None:
                try:
                    with open(self.model_dir / filename, 'wb') as f:
                        pickle.dump(model, f)
                except Exception as e:
                    print(f"Error saving {filename}: {e}")
    
    def _load_models(self) -> None:
        """Load trained models from disk."""
        if not SKLEARN_AVAILABLE:
            return
        
        model_files = {
            'departure_classifier.pkl': 'departure_classifier',
            'departure_regressor.pkl': 'departure_regressor',
            'arrival_classifier.pkl': 'arrival_classifier',
            'arrival_regressor.pkl': 'arrival_regressor',
            'label_encoders.pkl': 'label_encoders',
            'feature_importance.pkl': 'feature_importance'
        }
        
        for filename, attr_name in model_files.items():
            filepath = self.model_dir / filename
            if filepath.exists():
                try:
                    with open(filepath, 'rb') as f:
                        setattr(self, attr_name, pickle.load(f))
                    
                    if attr_name in ['departure_classifier', 'departure_regressor', 
                                   'arrival_classifier', 'arrival_regressor']:
                        self.model_trained = True
                        
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current models."""
        return {
            "model_trained": self.model_trained,
            "training_date": self.training_date.isoformat() if self.training_date else None,
            "sklearn_available": SKLEARN_AVAILABLE,
            "models_loaded": {
                "departure_classifier": self.departure_classifier is not None,
                "departure_regressor": self.departure_regressor is not None,
                "arrival_classifier": self.arrival_classifier is not None,
                "arrival_regressor": self.arrival_regressor is not None
            },
            "feature_encoders": len(self.label_encoders),
            "model_dir": str(self.model_dir)
        }
    
    def predict_delay_risks(self, airport: str, date: date, 
                          flight_ids: Optional[List[str]] = None,
                          risk_threshold: float = 0.2) -> List['DelayRiskResult']:
        """
        Predict delay risks for flights at an airport on a specific date.
        
        Args:
            airport: Airport code (e.g., "BOM", "DEL")
            date: Analysis date
            flight_ids: Optional list of specific flight IDs to analyze
            risk_threshold: Minimum risk threshold to include in results
            
        Returns:
            List of DelayRiskResult objects with risk assessments
        """
        try:
            # Get flights for the specified airport and date
            query_result = self.db_service.query_flights_by_date_and_airport(date, airport)
            
            if not query_result.success or not query_result.data:
                return []
            
            flights_data = query_result.data
            if flight_ids:
                flights_data = [f for f in flights_data if f.get('flight_id') in flight_ids]
            
            results = []
            
            for flight_data in flights_data:
                try:
                    # Convert to Flight object
                    flight = self._convert_dict_to_flight(flight_data)
                    if not flight:
                        continue
                    
                    # Create operational context
                    context = OperationalContext(
                        airport_code=airport,
                        analysis_time=datetime.combine(date, time(6, 0)),  # Default to 6 AM
                        weather_conditions="normal",
                        runway_config="standard",
                        traffic_level="medium"
                    )
                    
                    # Get predictions
                    dep_prediction = self.predict_departure_delay(flight, context)
                    arr_prediction = self.predict_arrival_delay(flight, context)
                    
                    # Check if meets risk threshold
                    if (dep_prediction.delay_probability >= risk_threshold or 
                        arr_prediction.delay_probability >= risk_threshold):
                        
                        # Identify risk factors
                        risk_factors = self._identify_combined_risk_factors(flight, context, dep_prediction, arr_prediction)
                        
                        # Generate recommendations
                        recommendations = self._generate_risk_recommendations(flight, dep_prediction, arr_prediction)
                        
                        result = DelayRiskResult(
                            flight_id=flight.flight_id,
                            departure_risk=dep_prediction,
                            arrival_risk=arr_prediction,
                            risk_factors=risk_factors,
                            recommendations=recommendations
                        )
                        results.append(result)
                        
                except Exception as e:
                    logger.error(f"Error predicting risk for flight {flight_data.get('flight_id', 'unknown')}: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error in predict_delay_risks: {e}")
            return []
    
    def _convert_dict_to_flight(self, flight_data: Dict[str, Any]) -> Optional[Flight]:
        """Convert flight data dictionary to Flight object."""
        try:
            from ..models.flight import Flight, Airport
            
            # Parse airports
            origin = Airport.from_string(flight_data.get('origin', ''))
            destination = Airport.from_string(flight_data.get('destination', ''))
            
            # Parse timestamps
            std_utc = None
            atd_utc = None
            sta_utc = None
            ata_utc = None
            
            if flight_data.get('std_utc'):
                std_utc = datetime.fromisoformat(str(flight_data['std_utc']))
            if flight_data.get('atd_utc'):
                atd_utc = datetime.fromisoformat(str(flight_data['atd_utc']))
            if flight_data.get('sta_utc'):
                sta_utc = datetime.fromisoformat(str(flight_data['sta_utc']))
            if flight_data.get('ata_utc'):
                ata_utc = datetime.fromisoformat(str(flight_data['ata_utc']))
            
            return Flight(
                flight_id=flight_data.get('flight_id', ''),
                flight_number=flight_data.get('flight_no', ''),
                airline_code=flight_data.get('airline_code', ''),
                origin=origin,
                destination=destination,
                aircraft_type=flight_data.get('aircraft_type', ''),
                tail_number=flight_data.get('tail_number'),
                std_utc=std_utc,
                atd_utc=atd_utc,
                sta_utc=sta_utc,
                ata_utc=ata_utc,
                dep_delay_min=flight_data.get('dep_delay_min'),
                arr_delay_min=flight_data.get('arr_delay_min'),
                runway=flight_data.get('runway'),
                stand=flight_data.get('stand'),
                source_file=flight_data.get('source_file', '')
            )
            
        except Exception as e:
            logger.error(f"Error converting flight data: {e}")
            return None
    
    def _identify_combined_risk_factors(self, flight: Flight, context: OperationalContext,
                                      dep_prediction: DelayPrediction, arr_prediction: DelayPrediction) -> List[str]:
        """Identify combined risk factors from both departure and arrival predictions."""
        factors = set()
        
        # Add departure risk factors
        dep_factors = self._identify_risk_factors(
            self._extract_features(flight, context, "departure"), "departure"
        )
        factors.update(dep_factors)
        
        # Add arrival risk factors
        arr_factors = self._identify_risk_factors(
            self._extract_features(flight, context, "arrival"), "arrival"
        )
        factors.update(arr_factors)
        
        return list(factors)[:5]  # Return top 5 unique factors
    
    def _generate_risk_recommendations(self, flight: Flight, dep_prediction: DelayPrediction,
                                     arr_prediction: DelayPrediction) -> List[str]:
        """Generate recommendations based on risk predictions."""
        recommendations = []
        
        if dep_prediction.delay_probability > 0.3:
            recommendations.append("Consider earlier departure slot to reduce delay risk")
        
        if arr_prediction.delay_probability > 0.3:
            recommendations.append("Monitor arrival conditions and prepare for potential delays")
        
        if dep_prediction.risk_level == DelayRiskLevel.CRITICAL:
            recommendations.append("High departure delay risk - prioritize for optimization")
        
        if arr_prediction.risk_level == DelayRiskLevel.CRITICAL:
            recommendations.append("High arrival delay risk - coordinate with destination airport")
        
        # Add generic recommendations if none specific
        if not recommendations:
            recommendations.append("Monitor flight status closely")
            recommendations.append("Consider weather and traffic conditions")
        
        return recommendations[:3]  # Return top 3 recommendations

    def evaluate_model_performance(self, test_start_date: date, 
                                 test_end_date: date) -> Dict[str, Any]:
        """Evaluate model performance on test data."""
        if not SKLEARN_AVAILABLE or not self.model_trained:
            return {"error": "Models not available for evaluation"}
        
        # Load test data
        test_data = self._load_training_data(test_start_date, test_end_date)
        
        if len(test_data) < 10:
            return {"error": "Insufficient test data"}
        
        # Extract features and targets
        features_df = pd.DataFrame([self._extract_features_from_historical(row) 
                                  for row in test_data])
        features_encoded = self._encode_features(features_df)
        
        # Departure evaluation
        dep_delays = [max(0, row.get('dep_delay_min', 0) or 0) for row in test_data]
        dep_binary = [1 if delay > 15 else 0 for delay in dep_delays]
        
        dep_class_pred = self.departure_classifier.predict(features_encoded)
        dep_class_prob = self.departure_classifier.predict_proba(features_encoded)[:, 1]
        dep_reg_pred = self.departure_regressor.predict(features_encoded)
        
        # Arrival evaluation
        arr_delays = [max(0, row.get('arr_delay_min', 0) or 0) for row in test_data]
        arr_binary = [1 if delay > 15 else 0 for delay in arr_delays]
        
        arr_class_pred = self.arrival_classifier.predict(features_encoded)
        arr_class_prob = self.arrival_classifier.predict_proba(features_encoded)[:, 1]
        arr_reg_pred = self.arrival_regressor.predict(features_encoded)
        
        return {
            "test_samples": len(test_data),
            "test_period": f"{test_start_date} to {test_end_date}",
            "departure_performance": {
                "classification": {
                    "accuracy": accuracy_score(dep_binary, dep_class_pred),
                    "precision": precision_score(dep_binary, dep_class_pred),
                    "recall": recall_score(dep_binary, dep_class_pred),
                    "f1": f1_score(dep_binary, dep_class_pred),
                    "auc": roc_auc_score(dep_binary, dep_class_prob)
                },
                "regression": {
                    "mae": mean_absolute_error(dep_delays, dep_reg_pred),
                    "rmse": np.sqrt(mean_squared_error(dep_delays, dep_reg_pred))
                }
            },
            "arrival_performance": {
                "classification": {
                    "accuracy": accuracy_score(arr_binary, arr_class_pred),
                    "precision": precision_score(arr_binary, arr_class_pred),
                    "recall": recall_score(arr_binary, arr_class_pred),
                    "f1": f1_score(arr_binary, arr_class_pred),
                    "auc": roc_auc_score(arr_binary, arr_class_prob)
                },
                "regression": {
                    "mae": mean_absolute_error(arr_delays, arr_reg_pred),
                    "rmse": np.sqrt(mean_squared_error(arr_delays, arr_reg_pred))
                }
            }
        }