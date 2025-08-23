"""
Continuous Learning Pipeline for Flight Delay Prediction Models

This module implements a comprehensive continuous learning system that provides:
- Online/offline feature engineering parity
- Model performance monitoring and drift detection
- Automated retraining triggers
- Ensemble prediction methods
- Model versioning and deployment pipeline
- Incremental learning capabilities

Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6
"""

import os
import json
import pickle
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, accuracy_score, roc_auc_score
from sklearn.ensemble import VotingRegressor, VotingClassifier
import lightgbm as lgb
import joblib
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class ModelMetrics:
    """Model performance metrics for monitoring"""
    model_id: str
    timestamp: datetime
    mae: float
    accuracy: float
    auc: float
    prediction_count: int
    drift_score: float
    confidence_score: float

@dataclass
class FeatureVector:
    """Standardized feature vector for online/offline parity"""
    flight_id: str
    timestamp: datetime
    features: Dict[str, float]
    feature_hash: str
    source: str  # "online" or "offline"

@dataclass
class ModelVersion:
    """Model version metadata"""
    version_id: str
    model_type: str
    created_at: datetime
    performance_metrics: ModelMetrics
    feature_importance: Dict[str, float]
    training_data_hash: str
    is_active: bool
    rollback_available: bool

@dataclass
class DriftReport:
    """Model drift detection report"""
    model_id: str
    timestamp: datetime
    drift_detected: bool
    drift_score: float
    threshold: float
    affected_features: List[str]
    recommendation: str

@dataclass
class EnsemblePrediction:
    """Ensemble prediction with confidence and component breakdown"""
    prediction: float
    confidence: float
    component_predictions: Dict[str, float]
    component_weights: Dict[str, float]
    uncertainty: float

class FeatureEngineer:
    """Feature engineering with online/offline parity"""
    
    def __init__(self):
        self.feature_definitions = self._load_feature_definitions()
        self.scaler_params = {}
        self.categorical_encodings = {}
    
    def _load_feature_definitions(self) -> Dict[str, Any]:
        """Load feature engineering definitions"""
        return {
            'temporal_features': [
                'hour_of_day', 'day_of_week', 'month', 'is_weekend',
                'is_peak_hour', 'minutes_since_midnight'
            ],
            'operational_features': [
                'scheduled_turnaround_time', 'runway_demand_ratio',
                'weather_severity_score', 'airline_delay_history',
                'aircraft_type_risk_score', 'route_complexity'
            ],
            'contextual_features': [
                'preceding_delay', 'downstream_flights_count',
                'gate_change_indicator', 'maintenance_flag',
                'crew_change_indicator', 'fuel_stop_required'
            ]
        }
    
    def extract_features_online(self, flight_data: Dict[str, Any], 
                              context_data: Dict[str, Any]) -> FeatureVector:
        """Extract features for online/real-time prediction"""
        features = {}
        
        # Temporal features
        timestamp = pd.to_datetime(flight_data['std_utc'])
        features.update(self._extract_temporal_features(timestamp))
        
        # Operational features
        features.update(self._extract_operational_features(flight_data, context_data))
        
        # Contextual features
        features.update(self._extract_contextual_features(flight_data, context_data))
        
        # Apply scaling and encoding
        features = self._apply_transformations(features)
        
        # Create feature hash for consistency checking
        feature_hash = self._compute_feature_hash(features)
        
        return FeatureVector(
            flight_id=flight_data['flight_id'],
            timestamp=datetime.now(),
            features=features,
            feature_hash=feature_hash,
            source="online"
        )
    
    def extract_features_offline(self, flight_df: pd.DataFrame) -> List[FeatureVector]:
        """Extract features for offline/batch training"""
        feature_vectors = []
        
        for _, row in flight_df.iterrows():
            flight_data = row.to_dict()
            context_data = self._build_offline_context(flight_df, row.name)
            
            features = {}
            
            # Use same feature extraction logic as online
            timestamp = pd.to_datetime(flight_data['std_utc'])
            features.update(self._extract_temporal_features(timestamp))
            features.update(self._extract_operational_features(flight_data, context_data))
            features.update(self._extract_contextual_features(flight_data, context_data))
            
            # Apply same transformations
            features = self._apply_transformations(features)
            feature_hash = self._compute_feature_hash(features)
            
            feature_vectors.append(FeatureVector(
                flight_id=flight_data['flight_id'],
                timestamp=timestamp,
                features=features,
                feature_hash=feature_hash,
                source="offline"
            ))
        
        return feature_vectors
    
    def _extract_temporal_features(self, timestamp: pd.Timestamp) -> Dict[str, float]:
        """Extract time-based features"""
        return {
            'hour_of_day': float(timestamp.hour),
            'day_of_week': float(timestamp.dayofweek),
            'month': float(timestamp.month),
            'is_weekend': float(timestamp.dayofweek >= 5),
            'is_peak_hour': float(timestamp.hour in [7, 8, 9, 17, 18, 19]),
            'minutes_since_midnight': float(timestamp.hour * 60 + timestamp.minute)
        }
    
    def _extract_operational_features(self, flight_data: Dict[str, Any], 
                                    context_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract operational features"""
        return {
            'scheduled_turnaround_time': context_data.get('turnaround_time', 45.0),
            'runway_demand_ratio': context_data.get('runway_utilization', 0.7),
            'weather_severity_score': context_data.get('weather_score', 0.0),
            'airline_delay_history': context_data.get('airline_performance', 0.85),
            'aircraft_type_risk_score': context_data.get('aircraft_risk', 0.1),
            'route_complexity': context_data.get('route_score', 0.5)
        }
    
    def _extract_contextual_features(self, flight_data: Dict[str, Any], 
                                   context_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract contextual features"""
        return {
            'preceding_delay': context_data.get('inbound_delay', 0.0),
            'downstream_flights_count': context_data.get('connections', 0.0),
            'gate_change_indicator': float(context_data.get('gate_changed', False)),
            'maintenance_flag': float(context_data.get('maintenance_due', False)),
            'crew_change_indicator': float(context_data.get('crew_change', False)),
            'fuel_stop_required': float(context_data.get('fuel_stop', False))
        }
    
    def _apply_transformations(self, features: Dict[str, float]) -> Dict[str, float]:
        """Apply scaling and encoding transformations"""
        # Apply saved scaling parameters
        for feature_name, value in features.items():
            if feature_name in self.scaler_params:
                mean, std = self.scaler_params[feature_name]
                features[feature_name] = (value - mean) / (std + 1e-8)
        
        return features
    
    def _compute_feature_hash(self, features: Dict[str, float]) -> str:
        """Compute hash of feature vector for consistency checking"""
        feature_str = json.dumps(features, sort_keys=True)
        return hashlib.md5(feature_str.encode()).hexdigest()
    
    def _build_offline_context(self, df: pd.DataFrame, row_idx: int) -> Dict[str, Any]:
        """Build context data for offline feature extraction"""
        # This would normally query historical data
        # For now, return default values
        return {
            'turnaround_time': 45.0,
            'runway_utilization': 0.7,
            'weather_score': 0.0,
            'airline_performance': 0.85,
            'aircraft_risk': 0.1,
            'route_score': 0.5,
            'inbound_delay': 0.0,
            'connections': 0.0,
            'gate_changed': False,
            'maintenance_due': False,
            'crew_change': False,
            'fuel_stop': False
        }
    
    def fit_transformations(self, feature_vectors: List[FeatureVector]) -> None:
        """Fit scaling parameters from training data"""
        all_features = {}
        
        for fv in feature_vectors:
            for name, value in fv.features.items():
                if name not in all_features:
                    all_features[name] = []
                all_features[name].append(value)
        
        # Compute scaling parameters
        for name, values in all_features.items():
            values_array = np.array(values)
            self.scaler_params[name] = (np.mean(values_array), np.std(values_array))

class ModelPerformanceMonitor:
    """Monitor model performance and detect drift"""
    
    def __init__(self, drift_threshold: float = 0.1):
        self.drift_threshold = drift_threshold
        self.performance_history: List[ModelMetrics] = []
        self.baseline_metrics: Optional[ModelMetrics] = None
    
    def record_prediction(self, model_id: str, prediction: float, 
                         actual: Optional[float] = None) -> None:
        """Record a prediction for performance tracking"""
        # This would be called for each prediction
        # Store in database or metrics store
        pass
    
    def evaluate_model_performance(self, model_id: str, 
                                 predictions: List[float], 
                                 actuals: List[float]) -> ModelMetrics:
        """Evaluate model performance metrics"""
        mae = mean_absolute_error(actuals, predictions)
        
        # Convert to binary for accuracy/AUC (delay > 15 minutes)
        binary_actuals = [1 if a > 15 else 0 for a in actuals]
        binary_preds = [1 if p > 15 else 0 for p in predictions]
        
        accuracy = accuracy_score(binary_actuals, binary_preds)
        
        # For AUC, use probability scores (normalized predictions)
        prob_scores = np.array(predictions) / (np.max(predictions) + 1e-8)
        auc = roc_auc_score(binary_actuals, prob_scores) if len(set(binary_actuals)) > 1 else 0.5
        
        # Calculate drift score
        drift_score = self._calculate_drift_score(predictions, actuals)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(predictions, actuals)
        
        metrics = ModelMetrics(
            model_id=model_id,
            timestamp=datetime.now(),
            mae=mae,
            accuracy=accuracy,
            auc=auc,
            prediction_count=len(predictions),
            drift_score=drift_score,
            confidence_score=confidence_score
        )
        
        self.performance_history.append(metrics)
        
        if self.baseline_metrics is None:
            self.baseline_metrics = metrics
        
        return metrics
    
    def detect_drift(self, model_id: str, recent_metrics: ModelMetrics) -> DriftReport:
        """Detect model drift based on performance degradation and statistical tests"""
        drift_detected = False
        affected_features = []
        recommendation = "Continue monitoring"
        
        if self.baseline_metrics is not None:
            # Check for significant performance degradation
            mae_degradation = 0.0
            accuracy_degradation = 0.0
            
            if self.baseline_metrics.mae > 0:
                mae_degradation = (recent_metrics.mae - self.baseline_metrics.mae) / self.baseline_metrics.mae
            
            if self.baseline_metrics.accuracy > 0:
                accuracy_degradation = (self.baseline_metrics.accuracy - recent_metrics.accuracy) / self.baseline_metrics.accuracy
            
            # More sensitive drift detection
            if mae_degradation > self.drift_threshold or accuracy_degradation > self.drift_threshold:
                drift_detected = True
                recommendation = f"Retrain model - performance degradation detected (MAE: {mae_degradation:.3f}, Acc: {accuracy_degradation:.3f})"
        
        # Check drift score (now includes statistical tests)
        if recent_metrics.drift_score > self.drift_threshold:
            drift_detected = True
            if "degradation" not in recommendation:
                recommendation = f"Retrain model - statistical drift detected (score: {recent_metrics.drift_score:.3f})"
        
        return DriftReport(
            model_id=model_id,
            timestamp=datetime.now(),
            drift_detected=drift_detected,
            drift_score=recent_metrics.drift_score,
            threshold=self.drift_threshold,
            affected_features=affected_features,
            recommendation=recommendation
        )
    
    def _calculate_drift_score(self, predictions: List[float], actuals: List[float]) -> float:
        """Calculate drift score based on prediction distribution using statistical tests"""
        if len(self.performance_history) < 2:
            return 0.0
        
        # Get baseline predictions from history (if available)
        if hasattr(self, '_baseline_predictions') and len(self._baseline_predictions) > 0:
            # Use Kolmogorov-Smirnov test for distribution drift
            from scipy import stats
            try:
                ks_stat, p_value = stats.ks_2samp(self._baseline_predictions, predictions)
                # Convert KS statistic to drift score (0-1 range)
                drift_score = min(ks_stat * 2, 1.0)  # Scale KS stat to 0-1
                return drift_score
            except ImportError:
                # Fallback to variance-based drift if scipy not available
                pass
        
        # Fallback: Simple drift detection based on prediction variance
        current_var = np.var(predictions)
        if not hasattr(self, '_baseline_variance'):
            self._baseline_variance = current_var
            self._baseline_predictions = predictions.copy()
            return 0.0
        
        if self._baseline_variance > 0:
            variance_drift = abs(current_var - self._baseline_variance) / self._baseline_variance
            return min(variance_drift, 1.0)
        
        return 0.0
    
    def _calculate_confidence_score(self, predictions: List[float], actuals: List[float]) -> float:
        """Calculate confidence score for predictions"""
        if len(predictions) == 0:
            return 0.0
        
        # Simple confidence based on prediction consistency
        pred_std = np.std(predictions)
        pred_mean = np.mean(predictions)
        
        if pred_mean > 0:
            coefficient_of_variation = pred_std / pred_mean
            return max(0.0, 1.0 - coefficient_of_variation)
        
        return 0.5

class EnsemblePredictor:
    """Ensemble prediction methods combining multiple models"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.model_weights: Dict[str, float] = {}
        self.performance_tracker = ModelPerformanceMonitor()
    
    def add_model(self, model_id: str, model: Any, weight: float = 1.0) -> None:
        """Add a model to the ensemble"""
        self.models[model_id] = model
        self.model_weights[model_id] = weight
    
    def predict_ensemble(self, features: FeatureVector) -> EnsemblePrediction:
        """Make ensemble prediction combining all models"""
        component_predictions = {}
        
        # Get predictions from all models
        for model_id, model in self.models.items():
            try:
                feature_array = np.array(list(features.features.values())).reshape(1, -1)
                pred = model.predict(feature_array)[0]
                component_predictions[model_id] = float(pred)
            except Exception as e:
                logger.warning(f"Model {model_id} prediction failed: {e}")
                component_predictions[model_id] = 0.0
        
        if not component_predictions:
            return EnsemblePrediction(
                prediction=0.0,
                confidence=0.0,
                component_predictions={},
                component_weights={},
                uncertainty=1.0
            )
        
        # Calculate weighted average
        total_weight = sum(self.model_weights.get(mid, 1.0) for mid in component_predictions.keys())
        weighted_sum = sum(pred * self.model_weights.get(mid, 1.0) 
                          for mid, pred in component_predictions.items())
        
        ensemble_prediction = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # Calculate confidence and uncertainty
        pred_values = list(component_predictions.values())
        uncertainty = np.std(pred_values) if len(pred_values) > 1 else 0.0
        confidence = max(0.0, 1.0 - uncertainty / (abs(ensemble_prediction) + 1.0))
        
        return EnsemblePrediction(
            prediction=ensemble_prediction,
            confidence=confidence,
            component_predictions=component_predictions,
            component_weights=dict(self.model_weights),
            uncertainty=uncertainty
        )
    
    def update_weights(self, performance_metrics: Dict[str, ModelMetrics]) -> None:
        """Update model weights based on recent performance"""
        for model_id, metrics in performance_metrics.items():
            if model_id in self.model_weights:
                # Weight based on inverse of MAE (better models get higher weight)
                self.model_weights[model_id] = 1.0 / (metrics.mae + 0.1)
        
        # Normalize weights
        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            for model_id in self.model_weights:
                self.model_weights[model_id] /= total_weight

class ModelVersionManager:
    """Manage model versions and deployment pipeline"""
    
    def __init__(self, model_store_path: str = "models"):
        self.model_store_path = Path(model_store_path)
        self.model_store_path.mkdir(exist_ok=True)
        self.versions: Dict[str, ModelVersion] = {}
        self.active_models: Dict[str, str] = {}  # model_type -> version_id
    
    def save_model_version(self, model_type: str, model: Any, 
                          metrics: ModelMetrics, 
                          feature_importance: Dict[str, float],
                          training_data_hash: str) -> str:
        """Save a new model version"""
        import time
        # Use microseconds to ensure unique version IDs
        timestamp = datetime.now()
        version_id = f"{model_type}_{timestamp.strftime('%Y%m%d_%H%M%S')}_{timestamp.microsecond}"
        
        # Save model file
        model_path = self.model_store_path / f"{version_id}.pkl"
        joblib.dump(model, model_path)
        
        # Create version metadata
        version = ModelVersion(
            version_id=version_id,
            model_type=model_type,
            created_at=datetime.now(),
            performance_metrics=metrics,
            feature_importance=feature_importance,
            training_data_hash=training_data_hash,
            is_active=False,
            rollback_available=True
        )
        
        self.versions[version_id] = version
        
        # Save metadata
        metadata_path = self.model_store_path / f"{version_id}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(asdict(version), f, default=str, indent=2)
        
        logger.info(f"Saved model version {version_id}")
        return version_id
    
    def deploy_model(self, version_id: str) -> bool:
        """Deploy a model version to production"""
        if version_id not in self.versions:
            logger.error(f"Version {version_id} not found")
            return False
        
        version = self.versions[version_id]
        
        # Deactivate current active model
        current_active = self.active_models.get(version.model_type)
        if current_active and current_active in self.versions:
            self.versions[current_active].is_active = False
        
        # Activate new model
        version.is_active = True
        self.active_models[version.model_type] = version_id
        
        logger.info(f"Deployed model version {version_id}")
        return True
    
    def rollback_model(self, model_type: str) -> bool:
        """Rollback to previous model version"""
        # Find all versions of this model type
        model_versions = [v for v in self.versions.values() 
                         if v.model_type == model_type]
        
        if len(model_versions) < 2:
            logger.error(f"No rollback version available for {model_type}")
            return False
        
        # Sort by creation time and get second most recent (excluding current active)
        model_versions.sort(key=lambda x: x.created_at, reverse=True)
        
        # Find the most recent non-active version
        rollback_version = None
        for version in model_versions:
            if not version.is_active:
                rollback_version = version
                break
        
        if rollback_version is None:
            # If all versions are active (shouldn't happen), use second most recent
            rollback_version = model_versions[1] if len(model_versions) > 1 else None
        
        if rollback_version is None:
            logger.error(f"No suitable rollback version found for {model_type}")
            return False
        
        return self.deploy_model(rollback_version.version_id)
    
    def load_active_model(self, model_type: str) -> Optional[Any]:
        """Load the currently active model"""
        version_id = self.active_models.get(model_type)
        if not version_id:
            return None
        
        model_path = self.model_store_path / f"{version_id}.pkl"
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return None
        
        return joblib.load(model_path)
    
    def cleanup_old_versions(self, keep_versions: int = 5) -> None:
        """Clean up old model versions"""
        for model_type in set(v.model_type for v in self.versions.values()):
            type_versions = [v for v in self.versions.values() if v.model_type == model_type]
            type_versions.sort(key=lambda x: x.created_at, reverse=True)
            
            # Keep only the most recent versions
            for version in type_versions[keep_versions:]:
                model_path = self.model_store_path / f"{version.version_id}.pkl"
                metadata_path = self.model_store_path / f"{version.version_id}_metadata.json"
                
                if model_path.exists():
                    model_path.unlink()
                if metadata_path.exists():
                    metadata_path.unlink()
                
                del self.versions[version.version_id]
                logger.info(f"Cleaned up old version {version.version_id}")

class IncrementalLearner:
    """Incremental learning capabilities for real-time model updates"""
    
    def __init__(self, learning_rate: float = 0.01, batch_size: int = 50):
        self.learning_rate = learning_rate
        self.batch_size = batch_size  # Increased default batch size
        self.update_buffer: List[Tuple[FeatureVector, float]] = []
        self.model = None
        self.feature_names = None
    
    def add_training_sample(self, features: FeatureVector, target: float) -> None:
        """Add a new training sample to the update buffer"""
        self.update_buffer.append((features, target))
        
        # Trigger update if buffer is full
        if len(self.update_buffer) >= self.batch_size:
            self.update_model()
    
    def update_model(self) -> bool:
        """Update model with buffered samples"""
        if len(self.update_buffer) == 0:
            return False
        
        try:
            # Prepare training data
            X = []
            y = []
            
            for features, target in self.update_buffer:
                feature_values = list(features.features.values())
                # Ensure all features are numeric
                feature_values = [float(v) if v is not None else 0.0 for v in feature_values]
                X.append(feature_values)
                y.append(target)
            
            X = np.array(X)
            y = np.array(y)
            
            # Store feature names for consistency
            if self.feature_names is None and len(self.update_buffer) > 0:
                self.feature_names = list(self.update_buffer[0][0].features.keys())
            
            if self.model is None:
                # Initialize model with better parameters for small datasets
                self.model = lgb.LGBMRegressor(
                    n_estimators=50,  # Reduced for faster training
                    learning_rate=self.learning_rate,
                    min_data_in_leaf=1,  # Allow small leaves
                    min_data_in_bin=1,   # Allow small bins
                    min_child_samples=1, # Allow small child nodes
                    num_leaves=10,       # Reduced complexity
                    random_state=42,
                    verbose=-1           # Suppress warnings
                )
                self.model.fit(X, y)
            else:
                # For incremental updates, retrain with new data
                # In production, consider using online learning algorithms
                # or maintaining a sliding window of recent data
                self.model.fit(X, y)
            
            # Clear buffer
            self.update_buffer.clear()
            
            logger.info(f"Updated model with {len(X)} new samples")
            return True
            
        except Exception as e:
            logger.error(f"Model update failed: {e}")
            return False
    
    def predict(self, features: FeatureVector) -> float:
        """Make prediction with current model"""
        if self.model is None:
            return 0.0
        
        try:
            feature_array = np.array(list(features.features.values())).reshape(1, -1)
            return float(self.model.predict(feature_array)[0])
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return 0.0

class ContinuousLearningPipeline:
    """Main continuous learning pipeline orchestrator"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.feature_engineer = FeatureEngineer()
        self.performance_monitor = ModelPerformanceMonitor()
        self.ensemble_predictor = EnsemblePredictor()
        
        # Use persistent model store instead of temp directory
        model_store_path = self.config.get('model_store_path', 'models')
        self.version_manager = ModelVersionManager(model_store_path)
        
        # Configure incremental learner with better batch size
        batch_size = self.config.get('batch_size', 50)
        learning_rate = self.config.get('learning_rate', 0.01)
        self.incremental_learner = IncrementalLearner(learning_rate, batch_size)
        
        # Configuration
        self.retrain_threshold = self.config.get('retrain_threshold', 0.15)
        self.monitoring_interval = self.config.get('monitoring_interval', 3600)  # 1 hour
        self.auto_deploy = self.config.get('auto_deploy', False)
    
    def update_features_online(self, flight_data: Dict[str, Any], 
                             context_data: Dict[str, Any]) -> FeatureVector:
        """Update features for online prediction"""
        return self.feature_engineer.extract_features_online(flight_data, context_data)
    
    def retrain_models_incremental(self, new_data: pd.DataFrame) -> Dict[str, Any]:
        """Retrain models with new data incrementally"""
        result = {
            'success': False,
            'models_updated': [],
            'performance_improvement': {},
            'errors': []
        }
        
        try:
            # Extract features from new data
            feature_vectors = self.feature_engineer.extract_features_offline(new_data)
            
            # Update incremental learner
            for fv in feature_vectors:
                # Assume target is delay in minutes (would come from actual data)
                target = 0.0  # Placeholder
                self.incremental_learner.add_training_sample(fv, target)
            
            # Evaluate current performance
            # This would use actual predictions vs actuals
            predictions = [self.incremental_learner.predict(fv) for fv in feature_vectors]
            actuals = [0.0] * len(predictions)  # Placeholder
            
            metrics = self.performance_monitor.evaluate_model_performance(
                "incremental_model", predictions, actuals
            )
            
            # Check if retraining is needed
            drift_report = self.performance_monitor.detect_drift("incremental_model", metrics)
            
            if drift_report.drift_detected:
                # Trigger full retraining
                self._trigger_full_retrain(new_data)
                result['models_updated'].append('incremental_model')
            
            result['success'] = True
            
        except Exception as e:
            result['errors'].append(str(e))
            logger.error(f"Incremental retraining failed: {e}")
        
        return result
    
    def predict_with_ensemble(self, features: FeatureVector) -> EnsemblePrediction:
        """Make ensemble prediction"""
        return self.ensemble_predictor.predict_ensemble(features)
    
    def monitor_model_drift(self, predictions: List[float], 
                           actuals: List[float]) -> DriftReport:
        """Monitor model drift"""
        metrics = self.performance_monitor.evaluate_model_performance(
            "ensemble", predictions, actuals
        )
        return self.performance_monitor.detect_drift("ensemble", metrics)
    
    def trigger_retraining(self, drift_threshold: float) -> bool:
        """Check if retraining should be triggered"""
        if not self.performance_monitor.performance_history:
            return False
        
        latest_metrics = self.performance_monitor.performance_history[-1]
        return latest_metrics.drift_score > drift_threshold
    
    def _trigger_full_retrain(self, training_data: pd.DataFrame) -> None:
        """Trigger full model retraining"""
        try:
            # Extract features
            feature_vectors = self.feature_engineer.extract_features_offline(training_data)
            
            # Fit transformations
            self.feature_engineer.fit_transformations(feature_vectors)
            
            # Prepare training data
            X = np.array([list(fv.features.values()) for fv in feature_vectors])
            y = np.zeros(len(X))  # Placeholder targets
            
            # Train new model
            model = lgb.LGBMRegressor(n_estimators=200, random_state=42)
            model.fit(X, y)
            
            # Evaluate model
            predictions = model.predict(X)
            metrics = self.performance_monitor.evaluate_model_performance(
                "retrained_model", predictions.tolist(), y.tolist()
            )
            
            # Save model version
            feature_importance = dict(zip(
                [f"feature_{i}" for i in range(X.shape[1])],
                model.feature_importances_
            ))
            
            training_hash = hashlib.md5(str(training_data.values.tobytes()).encode()).hexdigest()
            
            version_id = self.version_manager.save_model_version(
                "delay_predictor", model, metrics, feature_importance, training_hash
            )
            
            # Auto-deploy if configured
            if self.auto_deploy:
                self.version_manager.deploy_model(version_id)
            
            logger.info(f"Full retraining completed, version: {version_id}")
            
        except Exception as e:
            logger.error(f"Full retraining failed: {e}")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return {
            'active_models': dict(self.version_manager.active_models),
            'total_versions': len(self.version_manager.versions),
            'performance_history_length': len(self.performance_monitor.performance_history),
            'ensemble_models': list(self.ensemble_predictor.models.keys()),
            'incremental_buffer_size': len(self.incremental_learner.update_buffer),
            'last_update': datetime.now().isoformat()
        }