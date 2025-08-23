"""Tests for turnaround time analysis service."""

import pytest
from datetime import datetime, date, time, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.services.turnaround_analysis import (
    TurnaroundAnalysisService, TurnaroundEstimate, TaxiTimeEstimate, 
    TurnaroundValidation, TurnaroundType, TaxiPhase
)
from src.models.flight import Flight, Airport, FlightTime
from src.services.database import FlightDatabaseService


class TestTurnaroundAnalysisService:
    """Test cases for TurnaroundAnalysisService."""
    
    @pytest.fixture
    def mock_db_service(self):
        """Mock database service."""
        return Mock(spec=FlightDatabaseService)
    
    @pytest.fixture
    def service(self, mock_db_service):
        """Create turnaround analysis service with mocked database."""
        return TurnaroundAnalysisService(db_service=mock_db_service)
    
    @pytest.fixture
    def sample_flight(self):
        """Create a sample flight for testing."""
        flight = Flight()
        flight.flight_id = "test-flight-123"
        flight.flight_number = "AI2509"
        flight.aircraft_registration = "VT-EXA"
        flight.aircraft_type = "A320"
        flight.origin = Airport(code="BOM", name="Mumbai", city="Mumbai")
        flight.destination = Airport(code="DEL", name="Delhi", city="Delhi")
        flight.departure = FlightTime()
        flight.departure.scheduled = time(10, 30)
        flight.flight_date = date(2024, 1, 15)
        return flight
    
    def test_estimate_turnaround_time_with_historical_data(self, service, mock_db_service):
        """Test turnaround time estimation with historical data."""
        # Mock historical data
        historical_data = [
            {
                'aircraft_registration': 'VT-EXA',
                'turnaround_minutes': 65.0,
                'aircraft_type': 'A320',
                'arrival_route_type': 'domestic',
                'departure_route_type': 'domestic'
            },
            {
                'aircraft_registration': 'VT-EXA',
                'turnaround_minutes': 75.0,
                'aircraft_type': 'A320',
                'arrival_route_type': 'domestic',
                'departure_route_type': 'domestic'
            },
            {
                'aircraft_registration': 'VT-EXA',
                'turnaround_minutes': 85.0,
                'aircraft_type': 'A320',
                'arrival_route_type': 'domestic',
                'departure_route_type': 'domestic'
            },
            {
                'aircraft_registration': 'VT-EXA',
                'turnaround_minutes': 55.0,
                'aircraft_type': 'A320',
                'arrival_route_type': 'domestic',
                'departure_route_type': 'domestic'
            },
            {
                'aircraft_registration': 'VT-EXA',
                'turnaround_minutes': 70.0,
                'aircraft_type': 'A320',
                'arrival_route_type': 'domestic',
                'departure_route_type': 'domestic'
            }
        ]
        
        service._get_turnaround_data = Mock(return_value=historical_data)
        
        # Test estimation
        estimate = service.estimate_turnaround_time("VT-EXA", "BOM")
        
        assert estimate.aircraft_registration == "VT-EXA"
        assert estimate.airport_code == "BOM"
        assert estimate.sample_size == 5
        assert estimate.aircraft_type == "A320"
        assert estimate.typical_route_type == "domestic"
        assert estimate.turnaround_type == TurnaroundType.DOMESTIC_DOMESTIC
        assert estimate.confidence_level == "low"  # < 10 samples
        
        # Check percentiles are reasonable
        assert 50 <= estimate.p50_turnaround_minutes <= 80
        assert 70 <= estimate.p90_turnaround_minutes <= 90
        assert 75 <= estimate.p95_turnaround_minutes <= 95
        assert estimate.p50_turnaround_minutes < estimate.p90_turnaround_minutes < estimate.p95_turnaround_minutes
    
    def test_estimate_turnaround_time_no_historical_data(self, service):
        """Test turnaround time estimation without historical data."""
        service._get_turnaround_data = Mock(return_value=[])
        
        estimate = service.estimate_turnaround_time("VT-EXA", "BOM")
        
        assert estimate.aircraft_registration == "VT-EXA"
        assert estimate.airport_code == "BOM"
        assert estimate.sample_size == 0
        assert estimate.confidence_level == "low"
        
        # Should use default values for A320 domestic
        assert estimate.p50_turnaround_minutes == 45
        assert estimate.p90_turnaround_minutes == 75
        assert estimate.p95_turnaround_minutes == 90
    
    def test_predict_taxi_time_departure(self, service, sample_flight):
        """Test taxi time prediction for departure."""
        service._get_taxi_time_data = Mock(return_value=[])
        
        taxi_estimate = service.predict_taxi_time(sample_flight, "09R")
        
        assert taxi_estimate.airport_code == "BOM"
        assert taxi_estimate.runway == "09R"
        assert taxi_estimate.operation_type == "departure"
        assert taxi_estimate.expected_taxi_minutes == 15  # BOM departure default
        assert taxi_estimate.p90_taxi_minutes == 25
        assert taxi_estimate.confidence_level == "low"
    
    def test_predict_taxi_time_arrival(self, service):
        """Test taxi time prediction for arrival."""
        # Create arrival flight
        flight = Flight()
        flight.flight_id = "test-arrival-123"
        flight.destination = Airport(code="DEL", name="Delhi", city="Delhi")
        
        service._get_taxi_time_data = Mock(return_value=[])
        
        taxi_estimate = service.predict_taxi_time(flight, "28")
        
        assert taxi_estimate.airport_code == "DEL"
        assert taxi_estimate.runway == "28"
        assert taxi_estimate.operation_type == "arrival"
        assert taxi_estimate.expected_taxi_minutes == 15  # DEL arrival default
        assert taxi_estimate.p90_taxi_minutes == 25
    
    def test_validate_departure_slot_feasible(self, service, sample_flight):
        """Test departure slot validation when turnaround is feasible."""
        # Mock previous arrival flight
        arrival_flight = Flight()
        arrival_flight.aircraft_registration = "VT-EXA"
        arrival_flight.arrival = FlightTime()
        arrival_flight.arrival.actual = datetime(2024, 1, 15, 8, 30)  # Arrived at 8:30
        
        # Mock turnaround estimate
        turnaround_estimate = TurnaroundEstimate(
            aircraft_registration="VT-EXA",
            airport_code="BOM",
            p50_turnaround_minutes=45,
            p90_turnaround_minutes=75,
            p95_turnaround_minutes=90,
            sample_size=10,
            min_observed=35,
            max_observed=120,
            aircraft_type="A320",
            typical_route_type="domestic",
            turnaround_type=TurnaroundType.DOMESTIC_DOMESTIC
        )
        
        service.estimate_turnaround_time = Mock(return_value=turnaround_estimate)
        
        # Proposed departure at 10:30 (120 minutes after arrival)
        proposed_departure = datetime(2024, 1, 15, 10, 30)
        
        validation = service.validate_departure_slot(sample_flight, proposed_departure, arrival_flight)
        
        assert validation.flight_id == "test-flight-123"
        assert validation.aircraft_registration == "VT-EXA"
        assert validation.required_turnaround_minutes == 120
        assert validation.is_feasible_p90 is True
        assert validation.is_feasible_p95 is True
        assert validation.risk_level == "low"
        assert validation.recommended_departure_time is None  # No recommendation needed
    
    def test_validate_departure_slot_not_feasible(self, service, sample_flight):
        """Test departure slot validation when turnaround is not feasible."""
        # Mock previous arrival flight
        arrival_flight = Flight()
        arrival_flight.aircraft_registration = "VT-EXA"
        arrival_flight.arrival = FlightTime()
        arrival_flight.arrival.actual = datetime(2024, 1, 15, 9, 45)  # Arrived at 9:45
        
        # Mock turnaround estimate
        turnaround_estimate = TurnaroundEstimate(
            aircraft_registration="VT-EXA",
            airport_code="BOM",
            p50_turnaround_minutes=45,
            p90_turnaround_minutes=75,
            p95_turnaround_minutes=90,
            sample_size=10,
            min_observed=35,
            max_observed=120,
            aircraft_type="A320",
            typical_route_type="domestic",
            turnaround_type=TurnaroundType.DOMESTIC_DOMESTIC
        )
        
        service.estimate_turnaround_time = Mock(return_value=turnaround_estimate)
        
        # Proposed departure at 10:15 (30 minutes after arrival - too tight)
        proposed_departure = datetime(2024, 1, 15, 10, 15)
        
        validation = service.validate_departure_slot(sample_flight, proposed_departure, arrival_flight)
        
        assert validation.required_turnaround_minutes == 30
        assert validation.is_feasible_p90 is False
        assert validation.is_feasible_p95 is False
        assert validation.risk_level == "critical"  # Below median
        assert "Below median turnaround time" in validation.risk_factors
        assert validation.recommended_departure_time is not None
        assert validation.buffer_minutes == 45  # 75 - 30
    
    def test_validate_departure_slot_no_aircraft_registration(self, service):
        """Test validation when aircraft registration is missing."""
        flight = Flight()
        flight.flight_id = "test-flight-no-reg"
        flight.aircraft_registration = ""
        flight.origin = Airport(code="BOM", name="Mumbai", city="Mumbai")
        
        proposed_departure = datetime(2024, 1, 15, 10, 30)
        
        validation = service.validate_departure_slot(flight, proposed_departure)
        
        assert validation.aircraft_registration == "UNKNOWN"
        assert validation.risk_level == "medium"
        assert "Unknown aircraft registration" in validation.risk_factors
    
    def test_validate_departure_slot_no_previous_arrival(self, service, sample_flight):
        """Test validation when no previous arrival is found."""
        service._find_previous_arrival = Mock(return_value=None)
        
        proposed_departure = datetime(2024, 1, 15, 10, 30)
        
        validation = service.validate_departure_slot(sample_flight, proposed_departure)
        
        assert validation.risk_level == "medium"
        assert "No previous arrival flight found" in validation.risk_factors
    
    def test_turnaround_estimate_feasibility_check(self):
        """Test the feasibility check method of TurnaroundEstimate."""
        estimate = TurnaroundEstimate(
            aircraft_registration="VT-EXA",
            airport_code="BOM",
            p50_turnaround_minutes=45,
            p90_turnaround_minutes=75,
            p95_turnaround_minutes=90,
            sample_size=10,
            min_observed=35,
            max_observed=120,
            aircraft_type="A320",
            typical_route_type="domestic",
            turnaround_type=TurnaroundType.DOMESTIC_DOMESTIC
        )
        
        arrival_time = datetime(2024, 1, 15, 8, 30)
        
        # Test feasible departure (120 minutes later)
        departure_time = datetime(2024, 1, 15, 10, 30)
        is_feasible, required = estimate.is_feasible_departure(arrival_time, departure_time, 90)
        assert is_feasible is True
        assert required == 120
        
        # Test infeasible departure (60 minutes later)
        departure_time = datetime(2024, 1, 15, 9, 30)
        is_feasible, required = estimate.is_feasible_departure(arrival_time, departure_time, 90)
        assert is_feasible is False
        assert required == 60
    
    def test_aircraft_type_extraction(self, service):
        """Test aircraft type extraction from registration."""
        assert service._extract_aircraft_type_from_registration("VT-EXA320") == "A320"
        assert service._extract_aircraft_type_from_registration("VT-EXB321") == "A321"
        assert service._extract_aircraft_type_from_registration("VT-EXC737") == "B737"
        assert service._extract_aircraft_type_from_registration("VT-EXD777") == "B777"
        assert service._extract_aircraft_type_from_registration("VT-EXE787") == "B787"
        assert service._extract_aircraft_type_from_registration("VT-EXF") == "UNKNOWN"
        assert service._extract_aircraft_type_from_registration("") == "UNKNOWN"
    
    def test_turnaround_estimate_to_dict(self):
        """Test TurnaroundEstimate serialization to dictionary."""
        estimate = TurnaroundEstimate(
            aircraft_registration="VT-EXA",
            airport_code="BOM",
            p50_turnaround_minutes=45.5,
            p90_turnaround_minutes=75.2,
            p95_turnaround_minutes=90.8,
            sample_size=15,
            min_observed=35.0,
            max_observed=120.0,
            aircraft_type="A320",
            typical_route_type="domestic",
            turnaround_type=TurnaroundType.DOMESTIC_DOMESTIC,
            confidence_level="medium"
        )
        
        result = estimate.to_dict()
        
        assert result["aircraft_registration"] == "VT-EXA"
        assert result["airport_code"] == "BOM"
        assert result["turnaround_estimates"]["p50_minutes"] == 45.5
        assert result["turnaround_estimates"]["p90_minutes"] == 75.2
        assert result["turnaround_estimates"]["p95_minutes"] == 90.8
        assert result["historical_context"]["sample_size"] == 15
        assert result["operational_factors"]["aircraft_type"] == "A320"
        assert result["operational_factors"]["turnaround_type"] == "domestic_domestic"
        assert result["metadata"]["confidence_level"] == "medium"
    
    def test_taxi_time_estimate_to_dict(self):
        """Test TaxiTimeEstimate serialization to dictionary."""
        estimate = TaxiTimeEstimate(
            airport_code="BOM",
            runway="09R",
            operation_type="departure",
            expected_taxi_minutes=15.5,
            p90_taxi_minutes=25.2,
            p95_taxi_minutes=30.8,
            terminal_distance="medium",
            congestion_factor=1.2,
            weather_impact=1.1,
            sample_size=50,
            confidence_level="high"
        )
        
        result = estimate.to_dict()
        
        assert result["airport_code"] == "BOM"
        assert result["runway"] == "09R"
        assert result["operation_type"] == "departure"
        assert result["taxi_estimates"]["expected_minutes"] == 15.5
        assert result["taxi_estimates"]["p90_minutes"] == 25.2
        assert result["taxi_estimates"]["p95_minutes"] == 30.8
        assert result["factors"]["terminal_distance"] == "medium"
        assert result["factors"]["congestion_factor"] == 1.2
        assert result["factors"]["weather_impact"] == 1.1
        assert result["metadata"]["sample_size"] == 50
        assert result["metadata"]["confidence_level"] == "high"
    
    def test_turnaround_validation_to_dict(self):
        """Test TurnaroundValidation serialization to dictionary."""
        turnaround_estimate = TurnaroundEstimate(
            aircraft_registration="VT-EXA",
            airport_code="BOM",
            p50_turnaround_minutes=45,
            p90_turnaround_minutes=75,
            p95_turnaround_minutes=90,
            sample_size=10,
            min_observed=35,
            max_observed=120,
            aircraft_type="A320",
            typical_route_type="domestic",
            turnaround_type=TurnaroundType.DOMESTIC_DOMESTIC
        )
        
        validation = TurnaroundValidation(
            flight_id="test-123",
            aircraft_registration="VT-EXA",
            airport_code="BOM",
            arrival_time=datetime(2024, 1, 15, 8, 30),
            proposed_departure_time=datetime(2024, 1, 15, 10, 30),
            required_turnaround_minutes=120.0,
            is_feasible_p90=True,
            is_feasible_p95=True,
            turnaround_estimate=turnaround_estimate,
            risk_level="low",
            risk_factors=[],
            recommended_departure_time=None,
            buffer_minutes=45.0
        )
        
        result = validation.to_dict()
        
        assert result["flight_id"] == "test-123"
        assert result["aircraft_registration"] == "VT-EXA"
        assert result["airport_code"] == "BOM"
        assert result["timing"]["required_turnaround_minutes"] == 120.0
        assert result["validation"]["is_feasible_p90"] is True
        assert result["validation"]["is_feasible_p95"] is True
        assert result["risk_assessment"]["risk_level"] == "low"
        assert result["recommendations"]["recommended_departure_time"] is None
        assert result["recommendations"]["buffer_minutes"] == 45.0
    
    @patch('src.services.turnaround_analysis.np.percentile')
    def test_calculate_historical_turnaround_with_mixed_routes(self, mock_percentile, service):
        """Test historical turnaround calculation with mixed route types."""
        mock_percentile.side_effect = lambda data, p: {50: 60, 90: 85, 95: 95}[p]
        
        historical_data = [
            {
                'aircraft_registration': 'VT-EXA',
                'turnaround_minutes': 65.0,
                'aircraft_type': 'A320',
                'arrival_route_type': 'domestic',
                'departure_route_type': 'international'
            },
            {
                'aircraft_registration': 'VT-EXA',
                'turnaround_minutes': 75.0,
                'aircraft_type': 'A320',
                'arrival_route_type': 'international',
                'departure_route_type': 'domestic'
            }
        ]
        
        result = service._calculate_historical_turnaround(historical_data, "VT-EXA", "BOM")
        
        assert result.typical_route_type == "mixed"
        assert result.turnaround_type == TurnaroundType.DOMESTIC_INTERNATIONAL
        assert result.confidence_level == "low"  # < 10 samples
    
    def test_default_turnaround_times_for_different_aircraft(self, service):
        """Test default turnaround times for different aircraft types."""
        # Test A320
        estimate_a320 = service._default_turnaround_estimate("VT-EXA320", "BOM")
        assert estimate_a320.aircraft_type == "A320"
        assert estimate_a320.p90_turnaround_minutes == 75
        
        # Test B777 (should use default since not in domestic defaults)
        estimate_b777 = service._default_turnaround_estimate("VT-EXB777", "BOM")
        assert estimate_b777.aircraft_type == "B777"
        assert estimate_b777.p90_turnaround_minutes == 75  # Uses default domestic
        
        # Test unknown aircraft
        estimate_unknown = service._default_turnaround_estimate("VT-UNKNOWN", "BOM")
        assert estimate_unknown.aircraft_type == "UNKNOWN"
        assert estimate_unknown.p90_turnaround_minutes == 75


class TestTurnaroundDataModels:
    """Test the data models used in turnaround analysis."""
    
    def test_turnaround_type_enum(self):
        """Test TurnaroundType enum values."""
        assert TurnaroundType.DOMESTIC_DOMESTIC.value == "domestic_domestic"
        assert TurnaroundType.DOMESTIC_INTERNATIONAL.value == "domestic_international"
        assert TurnaroundType.INTERNATIONAL_DOMESTIC.value == "international_domestic"
        assert TurnaroundType.INTERNATIONAL_INTERNATIONAL.value == "international_international"
    
    def test_taxi_phase_enum(self):
        """Test TaxiPhase enum values."""
        assert TaxiPhase.TAXI_OUT.value == "taxi_out"
        assert TaxiPhase.TAXI_IN.value == "taxi_in"


if __name__ == "__main__":
    pytest.main([__file__])