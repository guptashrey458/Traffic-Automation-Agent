"""Tests for delay risk prediction models and turnaround time analysis."""

import pytest
from datetime import datetime, date, time, timedelta
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

from src.services.delay_prediction import (
    DelayRiskPredictor, DelayPrediction, TurnaroundEstimate, TaxiTimeEstimate,
    OperationalContext, DelayRiskLevel, PredictionConfidence
)
from src.models.flight import Flight, Airport, FlightTime
from src.services.database import FlightDatabaseService, QueryResult


class TestDelayPrediction:
    """Test DelayPrediction data class."""
    
    def test_delay_prediction_initialization(self):
        """Test DelayPrediction initialization and methods."""
        prediction = DelayPrediction(
            flight_id="AI2509_001",
            prediction_type="departure",
            delay_probability=0.35,
            risk_level=DelayRiskLevel.MODERATE,
            is_likely_delayed=False,
            expected_delay_minutes=12.5,
            delay_range_min=5.0,
            delay_range_max=20.0,
            confidence=PredictionConfidence.HIGH,
            confidence_score=0.85,
            key_risk_factors=["Peak hour operations", "High utilization"]
        )
        
        assert prediction.flight_id == "AI2509_001"
        assert prediction.delay_probability == 0.35
        assert prediction.risk_level == DelayRiskLevel.MODERATE
        assert not prediction.is_likely_delayed
        assert prediction.expected_delay_minutes == 12.5
        assert prediction.confidence == PredictionConfidence.HIGH
        assert len(prediction.key_risk_factors) == 2
    
    def test_delay_prediction_to_dict(self):
        """Test DelayPrediction dictionary conversion."""
        prediction = DelayPrediction(
            flight_id="6E123_001",
            prediction_type="arrival",
            delay_probability=0.65,
            risk_level=DelayRiskLevel.HIGH,
            is_likely_delayed=True,
            expected_delay_minutes=25.3,
            delay_range_min=15.0,
            delay_range_max=35.0,
            confidence=PredictionConfidence.MEDIUM,
            confidence_score=0.75,
            key_risk_factors=["Adverse weather", "Recent delays"]
        )
        
        result_dict = prediction.to_dict()
        
        assert result_dict["flight_id"] == "6E123_001"
        assert result_dict["prediction_type"] == "arrival"
        assert result_dict["delay_probability"] == 0.65
        assert result_dict["risk_level"] == "high"
        assert result_dict["is_likely_delayed"] is True
        assert result_dict["expected_delay_minutes"] == 25.3
        assert result_dict["delay_range"]["min"] == 15.0
        assert result_dict["delay_range"]["max"] == 35.0
        assert result_dict["confidence"] == "medium"
        assert result_dict["confidence_score"] == 0.75
        assert len(result_dict["key_risk_factors"]) == 2


class TestTurnaroundEstimate:
    """Test TurnaroundEstimate functionality."""
    
    def test_turnaround_estimate_initialization(self):
        """Test TurnaroundEstimate initialization."""
        estimate = TurnaroundEstimate(
            aircraft_registration="VT-ABC",
            airport_code="BOM",
            p50_turnaround_minutes=45.0,
            p90_turnaround_minutes=75.0,
            p95_turnaround_minutes=90.0,
            sample_size=150,
            min_observed=25.0,
            max_observed=120.0,
            aircraft_type="A320",
            typical_route_type="domestic"
        )
        
        assert estimate.aircraft_registration == "VT-ABC"
        assert estimate.airport_code == "BOM"
        assert estimate.p90_turnaround_minutes == 75.0
        assert estimate.sample_size == 150
        assert estimate.aircraft_type == "A320"
    
    def test_is_feasible_departure(self):
        """Test feasible departure time checking."""
        estimate = TurnaroundEstimate(
            aircraft_registration="VT-ABC",
            airport_code="BOM",
            p50_turnaround_minutes=45.0,
            p90_turnaround_minutes=75.0,
            p95_turnaround_minutes=90.0,
            sample_size=100,
            min_observed=30.0,
            max_observed=120.0,
            aircraft_type="A320",
            typical_route_type="domestic"
        )
        
        arrival_time = datetime(2024, 1, 15, 10, 0)
        
        # Feasible departure (90 minutes later)
        feasible_departure = arrival_time + timedelta(minutes=90)
        assert estimate.is_feasible_departure(arrival_time, feasible_departure)
        
        # Infeasible departure (only 60 minutes)
        infeasible_departure = arrival_time + timedelta(minutes=60)
        assert not estimate.is_feasible_departure(arrival_time, infeasible_departure)
        
        # Exactly at P90 threshold
        threshold_departure = arrival_time + timedelta(minutes=75)
        assert estimate.is_feasible_departure(arrival_time, threshold_departure)


class TestOperationalContext:
    """Test OperationalContext data class."""
    
    def test_operational_context_initialization(self):
        """Test OperationalContext initialization."""
        context = OperationalContext(
            airport_code="DEL",
            analysis_datetime=datetime(2024, 1, 15, 8, 30),
            current_demand=25,
            runway_capacity=35,
            utilization_rate=0.71,
            weather_regime="medium",
            is_peak_hour=True,
            is_weekend=False,
            recent_avg_delay=15.5,
            cascade_risk_score=0.3
        )
        
        assert context.airport_code == "DEL"
        assert context.current_demand == 25
        assert context.utilization_rate == 0.71
        assert context.weather_regime == "medium"
        assert context.is_peak_hour
        assert not context.is_weekend
        assert context.recent_avg_delay == 15.5


class TestDelayRiskPredictor:
    """Test DelayRiskPredictor functionality."""
    
    @pytest.fixture
    def mock_db_service(self):
        """Create a mock database service."""
        mock_db = Mock(spec=FlightDatabaseService)
        return mock_db
    
    @pytest.fixture
    def predictor(self, mock_db_service):
        """Create DelayRiskPredictor with mock database."""
        return DelayRiskPredictor(db_service=mock_db_service, model_dir="test_models")
    
    @pytest.fixture
    def sample_flight(self):
        """Create a sample flight for testing."""
        return Flight(
            flight_id="AI2509_001",
            flight_number="AI2509",
            airline_code="AI",
            origin=Airport.from_string("Mumbai (BOM)"),
            destination=Airport.from_string("Delhi (DEL)"),
            aircraft_type="A320",
            aircraft_registration="VT-ABC",
            flight_date=date(2024, 1, 15),
            departure=FlightTime(scheduled=time(8, 0)),
            arrival=FlightTime(scheduled=time(10, 30))
        )
    
    @pytest.fixture
    def sample_context(self):
        """Create sample operational context."""
        return OperationalContext(
            airport_code="BOM",
            analysis_datetime=datetime(2024, 1, 15, 8, 0),
            current_demand=20,
            runway_capacity=30,
            utilization_rate=0.67,
            weather_regime="calm",
            is_peak_hour=True,
            is_weekend=False,
            recent_avg_delay=10.0,
            cascade_risk_score=0.2
        )
    
    def test_predictor_initialization(self, predictor):
        """Test DelayRiskPredictor initialization."""
        assert predictor.db_service is not None
        assert predictor.model_dir.name == "test_models"
        assert not predictor.model_trained
        assert predictor.training_date is None
        assert len(predictor.label_encoders) == 0
    
    def test_extract_features(self, predictor, sample_flight, sample_context):
        """Test feature extraction for ML prediction."""
        features = predictor._extract_features(sample_flight, sample_context, "departure")
        
        # Check required features are present
        required_features = [
            'hour_of_day', 'minute_of_hour', 'day_of_week', 'is_weekend', 'is_peak_hour',
            'airline_code', 'origin_code', 'destination_code', 'route_key', 'aircraft_type',
            'current_demand', 'runway_capacity', 'utilization_rate', 'weather_regime',
            'recent_avg_delay', 'cascade_risk_score', 'visibility_km', 'wind_speed_kts'
        ]
        
        for feature in required_features:
            assert feature in features
        
        # Check specific values
        assert features['hour_of_day'] == 8
        assert features['minute_of_hour'] == 0
        assert features['airline_code'] == 'AI'
        assert features['origin_code'] == 'BOM'
        assert features['destination_code'] == 'DEL'
        assert features['route_key'] == 'BOM-DEL'
        assert features['aircraft_type'] == 'A320'
        assert features['current_demand'] == 20
        assert features['utilization_rate'] == 0.67
        assert features['is_peak_hour'] is True
    
    def test_extract_features_missing_data(self, predictor, sample_context):
        """Test feature extraction with missing flight data."""
        incomplete_flight = Flight(
            flight_id="TEST_001",
            flight_number="TEST123"
            # Missing most fields
        )
        
        features = predictor._extract_features(incomplete_flight, sample_context, "departure")
        
        # Should handle missing data gracefully
        assert features['hour_of_day'] == 12  # Default
        # The airline code is extracted from flight number, so "TEST123" -> "TE"
        assert features['airline_code'] == 'TE'  # Extracted from flight number
        assert features['origin_code'] == 'UNKNOWN'
        assert features['aircraft_type'] == 'UNKNOWN'
    
    def test_classify_risk_level(self, predictor):
        """Test delay risk level classification."""
        assert predictor._classify_risk_level(0.03) == DelayRiskLevel.LOW
        assert predictor._classify_risk_level(0.12) == DelayRiskLevel.MODERATE
        assert predictor._classify_risk_level(0.35) == DelayRiskLevel.HIGH
        assert predictor._classify_risk_level(0.75) == DelayRiskLevel.CRITICAL
        
        # Boundary cases
        assert predictor._classify_risk_level(0.05) == DelayRiskLevel.MODERATE
        assert predictor._classify_risk_level(0.20) == DelayRiskLevel.HIGH
        assert predictor._classify_risk_level(0.50) == DelayRiskLevel.CRITICAL
    
    def test_calculate_confidence(self, predictor):
        """Test prediction confidence calculation."""
        # High quality features
        good_features = {
            'airline_code': 'AI',
            'aircraft_type': 'A320',
            'recent_avg_delay': 10.0
        }
        
        confidence, score = predictor._calculate_confidence(good_features, 0.8)
        assert confidence in [PredictionConfidence.MEDIUM, PredictionConfidence.HIGH]
        assert 0.7 <= score <= 1.0
        
        # Poor quality features
        poor_features = {
            'airline_code': 'UNKNOWN',
            'aircraft_type': 'UNKNOWN',
            'recent_avg_delay': 0.0
        }
        
        confidence, score = predictor._calculate_confidence(poor_features, 0.5)
        assert confidence in [PredictionConfidence.LOW, PredictionConfidence.MEDIUM]
        assert score < 0.8
    
    def test_identify_risk_factors(self, predictor):
        """Test risk factor identification."""
        high_risk_features = {
            'utilization_rate': 0.9,
            'is_peak_hour': True,
            'weather_regime': 'strong',
            'recent_avg_delay': 25.0,
            'is_weekend': True,
            'cascade_risk_score': 0.7,
            'visibility_km': 3.0
        }
        
        risk_factors = predictor._identify_risk_factors(high_risk_features, "departure")
        
        assert len(risk_factors) <= 5  # Should return max 5 factors
        assert any("utilization" in factor.lower() for factor in risk_factors)
        assert any("peak hour" in factor.lower() for factor in risk_factors)
        assert any("weather" in factor.lower() for factor in risk_factors)
        assert any("delay" in factor.lower() for factor in risk_factors)
    
    def test_fallback_departure_prediction(self, predictor, sample_flight, sample_context):
        """Test fallback departure prediction when ML models aren't available."""
        prediction = predictor._fallback_departure_prediction(sample_flight, sample_context)
        
        assert prediction.flight_id == sample_flight.flight_id
        assert prediction.prediction_type == "departure"
        assert 0.0 <= prediction.delay_probability <= 1.0
        assert prediction.expected_delay_minutes >= 0
        assert prediction.delay_range_min >= 0
        assert prediction.delay_range_max > prediction.delay_range_min
        assert prediction.confidence == PredictionConfidence.MEDIUM
        assert prediction.model_version == "fallback_1.0"
        assert len(prediction.key_risk_factors) >= 0
    
    def test_fallback_arrival_prediction(self, predictor, sample_flight, sample_context):
        """Test fallback arrival prediction."""
        prediction = predictor._fallback_arrival_prediction(sample_flight, sample_context)
        
        assert prediction.flight_id == sample_flight.flight_id
        assert prediction.prediction_type == "arrival"
        assert 0.0 <= prediction.delay_probability <= 1.0
        assert prediction.expected_delay_minutes >= 0
        assert prediction.model_version == "fallback_1.0"
    
    def test_fallback_prediction_high_utilization(self, predictor, sample_flight):
        """Test fallback prediction with high utilization."""
        high_util_context = OperationalContext(
            airport_code="BOM",
            analysis_datetime=datetime(2024, 1, 15, 8, 0),
            current_demand=28,
            runway_capacity=30,
            utilization_rate=0.93,  # Very high utilization
            weather_regime="calm",
            is_peak_hour=True,
            recent_avg_delay=5.0
        )
        
        prediction = predictor._fallback_departure_prediction(sample_flight, high_util_context)
        
        # Should have higher delay probability due to high utilization
        assert prediction.delay_probability > 0.3
        assert prediction.expected_delay_minutes > 10
        assert any("utilization" in factor.lower() for factor in prediction.key_risk_factors)
    
    def test_fallback_prediction_severe_weather(self, predictor, sample_flight):
        """Test fallback prediction with severe weather."""
        severe_weather_context = OperationalContext(
            airport_code="BOM",
            analysis_datetime=datetime(2024, 1, 15, 8, 0),
            current_demand=15,
            runway_capacity=30,
            utilization_rate=0.5,
            weather_regime="severe",  # Severe weather
            is_peak_hour=False,
            recent_avg_delay=5.0
        )
        
        prediction = predictor._fallback_departure_prediction(sample_flight, severe_weather_context)
        
        # Should have higher delay probability due to severe weather
        assert prediction.delay_probability > 0.4
        assert prediction.expected_delay_minutes >= 15  # Allow for exactly 15 minutes
        assert any("weather" in factor.lower() for factor in prediction.key_risk_factors)
    
    def test_predict_departure_delay_fallback(self, predictor, sample_flight, sample_context):
        """Test departure delay prediction using fallback method."""
        # Ensure ML models are not available
        predictor.model_trained = False
        
        prediction = predictor.predict_departure_delay(sample_flight, sample_context)
        
        assert isinstance(prediction, DelayPrediction)
        assert prediction.prediction_type == "departure"
        assert prediction.flight_id == sample_flight.flight_id
        assert 0.0 <= prediction.delay_probability <= 1.0
        assert prediction.expected_delay_minutes >= 0
    
    def test_predict_arrival_delay_fallback(self, predictor, sample_flight, sample_context):
        """Test arrival delay prediction using fallback method."""
        # Ensure ML models are not available
        predictor.model_trained = False
        
        prediction = predictor.predict_arrival_delay(sample_flight, sample_context)
        
        assert isinstance(prediction, DelayPrediction)
        assert prediction.prediction_type == "arrival"
        assert prediction.flight_id == sample_flight.flight_id
        assert 0.0 <= prediction.delay_probability <= 1.0
        assert prediction.expected_delay_minutes >= 0
    
    def test_default_turnaround_estimate(self, predictor):
        """Test default turnaround time estimation."""
        # Test A320 at major hub
        estimate = predictor._default_turnaround_estimate("VT-ABC320", "BOM")
        
        assert estimate.aircraft_registration == "VT-ABC320"
        assert estimate.airport_code == "BOM"
        assert estimate.aircraft_type == "A320"
        assert estimate.p90_turnaround_minutes > estimate.p50_turnaround_minutes
        assert estimate.p95_turnaround_minutes > estimate.p90_turnaround_minutes
        assert estimate.sample_size == 0  # No historical data
        
        # Test unknown aircraft at smaller airport
        estimate_small = predictor._default_turnaround_estimate("VT-XYZ", "GOI")
        
        assert estimate_small.aircraft_type == "UNKNOWN"
        # Should have shorter turnaround times than major hub
        assert estimate_small.p90_turnaround_minutes < estimate.p90_turnaround_minutes
    
    def test_estimate_turnaround_time(self, predictor):
        """Test turnaround time estimation."""
        estimate = predictor.estimate_turnaround_time("VT-ABC", "BOM")
        
        assert isinstance(estimate, TurnaroundEstimate)
        assert estimate.aircraft_registration == "VT-ABC"
        assert estimate.airport_code == "BOM"
        assert estimate.p90_turnaround_minutes > 0
        assert estimate.p95_turnaround_minutes >= estimate.p90_turnaround_minutes
    
    def test_default_taxi_estimates(self, predictor):
        """Test default taxi time estimates."""
        # Test major airport departure
        expected, p90 = predictor._default_taxi_estimates("BOM", "departure")
        assert expected > 0
        assert p90 > expected
        
        # Test major airport arrival
        expected_arr, p90_arr = predictor._default_taxi_estimates("BOM", "arrival")
        assert expected_arr > 0
        assert p90_arr > expected_arr
        # Arrivals should generally have shorter taxi times
        assert expected_arr < expected
        
        # Test smaller airport
        expected_small, p90_small = predictor._default_taxi_estimates("GOI", "departure")
        assert expected_small < expected  # Smaller airports have shorter taxi times
    
    def test_predict_taxi_time(self, predictor, sample_flight):
        """Test taxi time prediction."""
        estimate = predictor.predict_taxi_time(sample_flight, "09L")
        
        assert isinstance(estimate, TaxiTimeEstimate)
        assert estimate.airport_code == "BOM"  # From sample_flight origin
        assert estimate.runway == "09L"
        assert estimate.operation_type == "departure"
        assert estimate.expected_taxi_minutes > 0
        assert estimate.p90_taxi_minutes >= estimate.expected_taxi_minutes
    
    def test_get_model_info(self, predictor):
        """Test model information retrieval."""
        info = predictor.get_model_info()
        
        assert "model_trained" in info
        assert "training_date" in info
        assert "sklearn_available" in info
        assert "models_loaded" in info
        assert "feature_encoders" in info
        assert "model_dir" in info
        
        assert info["model_trained"] is False  # No training in test
        assert info["training_date"] is None
        assert isinstance(info["models_loaded"], dict)
    
    @patch('src.services.delay_prediction.SKLEARN_AVAILABLE', False)
    def test_predictor_without_sklearn(self, mock_db_service):
        """Test predictor behavior when sklearn is not available."""
        predictor = DelayRiskPredictor(db_service=mock_db_service)
        
        sample_flight = Flight(flight_id="TEST_001", flight_number="TEST123")
        sample_context = OperationalContext(
            airport_code="BOM",
            analysis_datetime=datetime.now()
        )
        
        # Should use fallback methods
        dep_prediction = predictor.predict_departure_delay(sample_flight, sample_context)
        arr_prediction = predictor.predict_arrival_delay(sample_flight, sample_context)
        
        assert dep_prediction.model_version == "fallback_1.0"
        assert arr_prediction.model_version == "fallback_1.0"
    
    def test_load_training_data_error_handling(self, predictor):
        """Test error handling in training data loading."""
        # Mock database error
        predictor.db_service.query_flights_by_date_range.side_effect = Exception("DB Error")
        
        training_data = predictor._load_training_data(
            date(2024, 1, 1), 
            date(2024, 1, 31)
        )
        
        assert training_data == []  # Should return empty list on error
    
    def test_extract_features_from_historical(self, predictor):
        """Test feature extraction from historical flight data."""
        historical_data = {
            'flight_id': 'AI2509_001',
            'flight_number': 'AI2509',
            'airline_code': 'AI',
            'origin_code': 'BOM',
            'destination_code': 'DEL',
            'aircraft_type': 'A320',
            'std_utc': '2024-01-15T08:00:00Z',
            'dep_delay_min': 15,
            'arr_delay_min': 10
        }
        
        features = predictor._extract_features_from_historical(historical_data)
        
        assert features['hour_of_day'] == 8
        assert features['airline_code'] == 'AI'
        assert features['origin_code'] == 'BOM'
        assert features['destination_code'] == 'DEL'
        assert features['aircraft_type'] == 'A320'
        assert features['route_key'] == 'BOM-DEL'
        assert features['recent_avg_delay'] == 15  # Uses dep_delay_min
    
    def test_airline_specific_adjustments(self, predictor, sample_context):
        """Test airline-specific delay adjustments in fallback prediction."""
        # Test Air India (historically higher delays)
        ai_flight = Flight(
            flight_id="AI_001",
            flight_number="AI2509",
            airline_code="AI",
            origin=Airport.from_string("Mumbai (BOM)"),
            destination=Airport.from_string("Delhi (DEL)")
        )
        
        # Test IndiGo (generally punctual)
        indigo_flight = Flight(
            flight_id="6E_001",
            flight_number="6E123",
            airline_code="6E",
            origin=Airport.from_string("Mumbai (BOM)"),
            destination=Airport.from_string("Delhi (DEL)")
        )
        
        ai_prediction = predictor._fallback_departure_prediction(ai_flight, sample_context)
        indigo_prediction = predictor._fallback_departure_prediction(indigo_flight, sample_context)
        
        # Air India should have higher delay probability than IndiGo
        assert ai_prediction.delay_probability >= indigo_prediction.delay_probability
        assert ai_prediction.expected_delay_minutes >= indigo_prediction.expected_delay_minutes


class TestModelTraining:
    """Test model training functionality (when sklearn is available)."""
    
    @pytest.fixture
    def predictor_with_data(self):
        """Create predictor with mock training data."""
        mock_db = Mock(spec=FlightDatabaseService)
        
        # Mock training data
        training_data = []
        for i in range(200):  # Sufficient training samples
            training_data.append({
                'flight_id': f'TEST_{i:03d}',
                'flight_number': f'AI{2500+i}',
                'airline_code': 'AI',
                'origin_code': 'BOM',
                'destination_code': 'DEL',
                'aircraft_type': 'A320',
                'std_utc': f'2024-01-{(i%30)+1:02d}T{8+(i%12):02d}:00:00Z',
                'dep_delay_min': max(0, np.random.normal(10, 15)),
                'arr_delay_min': max(0, np.random.normal(8, 12))
            })
        
        mock_db.query_flights_by_date_range.return_value = QueryResult(
            data=training_data,
            row_count=len(training_data)
        )
        
        return DelayRiskPredictor(db_service=mock_db, model_dir="test_models")
    
    @patch('src.services.delay_prediction.SKLEARN_AVAILABLE', True)
    def test_train_models_insufficient_data(self, predictor_with_data):
        """Test model training with insufficient data."""
        # Mock insufficient data
        predictor_with_data.db_service.query_flights_by_date_range.return_value = QueryResult(
            data=[],  # No data
            row_count=0
        )
        
        result = predictor_with_data.train_models(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31)
        )
        
        assert "error" in result
        assert "Insufficient training data" in result["error"]
    
    @patch('src.services.delay_prediction.SKLEARN_AVAILABLE', False)
    def test_train_models_no_sklearn(self, predictor_with_data):
        """Test model training when sklearn is not available."""
        result = predictor_with_data.train_models(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31)
        )
        
        assert "error" in result
        assert "scikit-learn and lightgbm are required" in result["error"]


if __name__ == "__main__":
    pytest.main([__file__])