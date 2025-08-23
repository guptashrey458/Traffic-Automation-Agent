"""Integration tests for FastAPI backend endpoints."""

import pytest
from fastapi.testclient import TestClient
from datetime import date, datetime
import json
from unittest.mock import Mock, patch

from src.api.main import app
from src.services.analytics import PeakAnalysis, TimeBucket, OverloadWindow, WeatherRegime
from src.services.schedule_optimizer import OptimizationResult, DelayMetrics
from src.services.whatif_simulator import ImpactCard, BeforeAfterComparison
from src.services.delay_prediction import DelayRiskResult, DelayPrediction


# Create test client
client = TestClient(app)


class TestBasicEndpoints:
    """Test basic API endpoints."""
    
    def test_root_endpoint(self):
        """Test root endpoint returns correct information."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Agentic Flight Scheduler API"
        assert data["version"] == "0.1.0"
        assert "status" in data
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_supported_airports(self):
        """Test airports endpoint returns supported airports."""
        response = client.get("/airports")
        assert response.status_code == 200
        data = response.json()
        assert "airports" in data
        assert len(data["airports"]) > 0
        
        # Check BOM is in the list
        airport_codes = [airport["code"] for airport in data["airports"]]
        assert "BOM" in airport_codes
        assert "DEL" in airport_codes


class TestPeaksEndpoint:
    """Test /flights/peaks endpoint."""
    
    @patch('src.api.main.analytics_engine')
    def test_peaks_analysis_success(self, mock_analytics):
        """Test successful peaks analysis."""
        # Mock the analytics engine response
        mock_bucket = Mock()
        mock_bucket.to_dict.return_value = {
            "start_time": "2024-01-01T06:00:00",
            "end_time": "2024-01-01T06:10:00",
            "total_demand": 15,
            "capacity": 12,
            "overload": 3
        }
        
        mock_window = Mock()
        mock_window.to_dict.return_value = {
            "start_time": "2024-01-01T06:00:00",
            "end_time": "2024-01-01T06:30:00",
            "severity": "high",
            "overload_flights": 8
        }
        
        mock_analysis = Mock()
        mock_analysis.airport = "BOM"
        mock_analysis.bucket_minutes = 10
        mock_analysis.analysis_date = date(2024, 1, 1)
        mock_analysis.time_buckets = [mock_bucket]
        mock_analysis.overload_windows = [mock_window]
        mock_analysis.capacity_utilization = 0.85
        mock_analysis.recommendations = ["Consider redistributing flights from 06:00-06:30"]
        
        mock_analytics.analyze_peaks.return_value = mock_analysis
        
        # Make request
        response = client.get("/flights/peaks?airport=BOM&bucket_minutes=10")
        
        assert response.status_code == 200
        data = response.json()
        assert data["airport"] == "BOM"
        assert data["bucket_minutes"] == 10
        assert len(data["time_buckets"]) == 1
        assert len(data["overload_windows"]) == 1
        assert data["capacity_utilization"] == 0.85
    
    def test_peaks_invalid_airport(self):
        """Test peaks analysis with invalid airport."""
        response = client.get("/flights/peaks?airport=INVALID&bucket_minutes=10")
        assert response.status_code == 400
        assert "Unsupported airport code" in response.json()["detail"]
    
    def test_peaks_invalid_bucket_size(self):
        """Test peaks analysis with invalid bucket size."""
        response = client.get("/flights/peaks?airport=BOM&bucket_minutes=7")
        assert response.status_code == 400
        assert "bucket_minutes must be" in response.json()["detail"]


class TestOptimizeEndpoint:
    """Test /optimize endpoint."""
    
    @patch('src.api.main.schedule_optimizer')
    def test_optimization_success(self, mock_optimizer):
        """Test successful schedule optimization."""
        # Mock optimization result
        mock_result = Mock()
        mock_result.optimization_id = "opt_123"
        mock_result.status = "completed"
        mock_result.original_metrics = Mock()
        mock_result.original_metrics.to_dict.return_value = {"avg_delay": 12.5}
        mock_result.optimized_metrics = Mock()
        mock_result.optimized_metrics.to_dict.return_value = {"avg_delay": 8.2}
        mock_result.recommended_changes = []
        mock_result.cost_reduction = 0.35
        mock_result.execution_time_seconds = 2.1
        
        mock_optimizer.optimize_schedule.return_value = mock_result
        
        # Make request
        request_data = {
            "airport": "BOM",
            "date": "2024-01-01",
            "objectives": {"delay_weight": 1.0, "taxi_weight": 0.3}
        }
        
        response = client.post("/optimize", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["optimization_id"] == "opt_123"
        assert data["status"] == "completed"
        assert data["cost_reduction"] == 0.35
    
    def test_optimization_missing_required_fields(self):
        """Test optimization with missing required fields."""
        request_data = {"date": "2024-01-01"}  # Missing airport
        
        response = client.post("/optimize", json=request_data)
        assert response.status_code == 422  # Validation error


class TestWhatIfEndpoint:
    """Test /whatif endpoint."""
    
    @patch('src.api.main.whatif_simulator')
    def test_whatif_analysis_success(self, mock_simulator):
        """Test successful what-if analysis."""
        # Mock what-if result (ImpactCard)
        mock_result = Mock()
        mock_result.scenario_description = "Move flight AI2739 by +10 minutes"
        mock_result.delay_change_minutes = -2.3
        mock_result.co2_change_kg = -15.2
        mock_result.confidence_level = "high"
        mock_result.affected_flights_count = 2
        mock_result.peak_overload_change = -1
        
        mock_simulator.analyze_single_flight_change.return_value = mock_result
        
        # Make request
        request_data = {
            "flight_id": "AI2739",
            "change_type": "time_shift",
            "change_value": "+10m",
            "airport": "BOM",
            "date": "2024-01-01"
        }
        
        response = client.post("/whatif", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["flight_id"] == "AI2739"
        assert data["co2_impact_kg"] == -15.2
        assert data["impact_summary"]["affected_flights"] == 2


class TestDelayRisksEndpoint:
    """Test /flights/risks endpoint."""
    
    @patch('src.api.main.delay_predictor')
    def test_delay_risks_success(self, mock_predictor):
        """Test successful delay risk analysis."""
        # Mock delay risk result
        mock_risk = Mock()
        mock_risk.flight_id = "AI2739"
        mock_risk.departure_risk = Mock()
        mock_risk.departure_risk.to_dict.return_value = {
            "probability": 0.35,
            "expected_delay": 8.2,
            "risk_level": "moderate"
        }
        mock_risk.arrival_risk = Mock()
        mock_risk.arrival_risk.to_dict.return_value = {
            "probability": 0.28,
            "expected_delay": 6.1,
            "risk_level": "moderate"
        }
        mock_risk.risk_factors = ["Peak hour slot", "Weather conditions"]
        mock_risk.recommendations = ["Consider earlier slot", "Monitor weather"]
        
        mock_predictor.predict_delay_risks.return_value = [mock_risk]
        
        # Make request
        response = client.get("/flights/risks?airport=BOM&date=2024-01-01")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["flight_id"] == "AI2739"
        assert len(data[0]["risk_factors"]) == 2
        assert len(data[0]["recommendations"]) == 2


class TestConstraintsEndpoint:
    """Test /constraints endpoint."""
    
    @patch('src.api.main.schedule_optimizer')
    def test_constraints_success(self, mock_optimizer):
        """Test successful constraints retrieval."""
        # Mock constraints
        mock_constraints = Mock()
        mock_constraints.operational_rules = {
            "min_turnaround_time": 45,
            "wake_turbulence_separation": True
        }
        mock_constraints.capacity_limits = {
            "runway_09": 30,
            "runway_27": 25
        }
        mock_constraints.weather_adjustments = {
            "calm": 1.0,
            "medium": 0.8,
            "strong": 0.6
        }
        mock_constraints.curfew_hours = {
            "start": "23:00",
            "end": "06:00"
        }
        
        mock_optimizer.get_constraints.return_value = mock_constraints
        
        # Make request
        response = client.get("/constraints?airport=BOM")
        
        assert response.status_code == 200
        data = response.json()
        assert data["airport"] == "BOM"
        assert "operational_rules" in data
        assert "capacity_limits" in data
        assert "weather_adjustments" in data
        assert "curfew_hours" in data


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @patch('src.api.main.analytics_engine')
    def test_peaks_service_error(self, mock_analytics):
        """Test handling of service errors in peaks endpoint."""
        mock_analytics.analyze_peaks.side_effect = Exception("Database connection failed")
        
        response = client.get("/flights/peaks?airport=BOM&bucket_minutes=10")
        assert response.status_code == 500
        assert "Peak analysis failed" in response.json()["detail"]
    
    @patch('src.api.main.schedule_optimizer')
    def test_optimization_service_error(self, mock_optimizer):
        """Test handling of service errors in optimization endpoint."""
        mock_optimizer.optimize_schedule.side_effect = Exception("Optimization failed")
        
        request_data = {
            "airport": "BOM",
            "date": "2024-01-01"
        }
        
        response = client.post("/optimize", json=request_data)
        assert response.status_code == 500
        assert "Schedule optimization failed" in response.json()["detail"]


class TestResponseSchemas:
    """Test response schema validation."""
    
    @patch('src.api.main.analytics_engine')
    def test_peaks_response_schema(self, mock_analytics):
        """Test that peaks response matches expected schema."""
        # Mock minimal valid response
        mock_bucket = Mock()
        mock_bucket.to_dict.return_value = {
            "start_time": "2024-01-01T06:00:00",
            "end_time": "2024-01-01T06:10:00",
            "total_demand": 15,
            "capacity": 12,
            "overload": 3
        }
        
        mock_analysis = Mock()
        mock_analysis.airport = "BOM"
        mock_analysis.bucket_minutes = 10
        mock_analysis.analysis_date = date(2024, 1, 1)
        mock_analysis.time_buckets = [mock_bucket]
        mock_analysis.overload_windows = []
        mock_analysis.capacity_utilization = 0.85
        mock_analysis.recommendations = []
        
        mock_analytics.analyze_peaks.return_value = mock_analysis
        
        response = client.get("/flights/peaks?airport=BOM&bucket_minutes=10")
        assert response.status_code == 200
        
        # Validate required fields are present
        data = response.json()
        required_fields = [
            "airport", "bucket_minutes", "analysis_date", "time_buckets",
            "overload_windows", "capacity_utilization", "recommendations", "weather_regime"
        ]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])