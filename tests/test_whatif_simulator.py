"""Tests for what-if simulation system."""

import pytest
from datetime import datetime, date, time, timedelta
from unittest.mock import Mock, patch
import pandas as pd

from src.services.whatif_simulator import (
    WhatIfSimulator, WhatIfScenario, ImpactCard, BeforeAfterComparison,
    CO2Factors, ChangeType
)
from src.services.schedule_optimizer import FlightChange, Schedule, DelayMetrics
from src.services.analytics import PeakAnalysis, WeatherRegime, OverloadWindow
from src.models.flight import Flight, Airport, FlightTime
from src.services.database import QueryResult


class TestCO2Factors:
    """Test CO2 emission factors."""
    
    def test_default_factors(self):
        """Test default CO2 factors."""
        factors = CO2Factors()
        
        assert factors.delay_factor_kg_per_min == 2.5
        assert factors.taxi_factor_kg_per_min == 8.0
        assert factors.routing_optimization_kg == 15.0
        assert "A320" in factors.aircraft_factors
        assert "DEFAULT" in factors.aircraft_factors
    
    def test_get_aircraft_factor(self):
        """Test aircraft-specific CO2 factors."""
        factors = CO2Factors()
        
        # Test known aircraft
        assert factors.get_aircraft_factor("A320") == 12.0
        assert factors.get_aircraft_factor("B777") == 35.0
        
        # Test unknown aircraft
        assert factors.get_aircraft_factor("UNKNOWN") == 15.0  # DEFAULT


class TestWhatIfSimulator:
    """Test what-if simulation system."""
    
    @pytest.fixture
    def mock_db_service(self):
        """Mock database service."""
        db_service = Mock()
        
        # Sample flight data
        flight_data = [
            {
                'flight_id': 'AI2509_001',
                'flight_number': 'AI2509',
                'airline_code': 'AI',
                'origin_code': 'BOM',
                'origin_name': 'Mumbai (BOM)',
                'destination_code': 'DEL',
                'destination_name': 'Delhi (DEL)',
                'std_utc': '2024-01-15T08:30:00Z',
                'atd_utc': '2024-01-15T08:45:00Z',
                'dep_delay_min': 15,
                'arr_delay_min': 10
            },
            {
                'flight_id': '6E123_001',
                'flight_number': '6E123',
                'airline_code': '6E',
                'origin_code': 'BOM',
                'origin_name': 'Mumbai (BOM)',
                'destination_code': 'BLR',
                'destination_name': 'Bangalore (BLR)',
                'std_utc': '2024-01-15T09:00:00Z',
                'atd_utc': '2024-01-15T09:05:00Z',
                'dep_delay_min': 5,
                'arr_delay_min': 0
            }
        ]
        
        db_service.query_flights_by_date_range.return_value = QueryResult(
            data=flight_data,
            row_count=len(flight_data),
            execution_time_ms=100
        )
        
        return db_service
    
    @pytest.fixture
    def mock_analytics_engine(self):
        """Mock analytics engine."""
        analytics = Mock()
        
        # Mock peak analysis
        peak_analysis = PeakAnalysis(
            airport="BOM",
            analysis_date=date(2024, 1, 15),
            bucket_minutes=10,
            overload_windows=[],
            peak_hour=9,
            peak_demand=25,
            total_capacity=300,
            avg_utilization=0.75
        )
        
        analytics.analyze_peaks.return_value = peak_analysis
        return analytics
    
    @pytest.fixture
    def mock_schedule_optimizer(self):
        """Mock schedule optimizer."""
        optimizer = Mock()
        
        # Mock what-if analysis result
        from src.services.schedule_optimizer import ImpactAnalysis, DelayMetrics
        
        impact_analysis = ImpactAnalysis(
            delay_delta=-5.0,
            peak_overload_change=0,
            co2_impact=-25.0,
            fairness_score=0.8,
            affected_flights=['AI2509_001'],
            before_metrics=DelayMetrics(100, 10, 30, 5, 75),
            after_metrics=DelayMetrics(95, 9.5, 28, 4, 80)
        )
        
        optimizer.what_if_analysis.return_value = impact_analysis
        return optimizer
    
    @pytest.fixture
    def simulator(self, mock_db_service, mock_analytics_engine, mock_schedule_optimizer):
        """Create what-if simulator with mocked dependencies."""
        return WhatIfSimulator(
            db_service=mock_db_service,
            analytics_engine=mock_analytics_engine,
            schedule_optimizer=mock_schedule_optimizer
        )
    
    def test_initialization(self):
        """Test simulator initialization."""
        simulator = WhatIfSimulator()
        
        assert simulator.db_service is not None
        assert simulator.analytics_engine is not None
        assert simulator.schedule_optimizer is not None
        assert isinstance(simulator.co2_factors, CO2Factors)
        assert simulator.max_simulation_time_seconds == 30
    
    def test_analyze_single_flight_change(self, simulator):
        """Test single flight change analysis."""
        impact_card = simulator.analyze_single_flight_change(
            flight_id="AI2509",
            time_change_minutes=-10,  # Move 10 minutes earlier
            airport="BOM",
            analysis_date=date(2024, 1, 15)
        )
        
        assert isinstance(impact_card, ImpactCard)
        assert impact_card.scenario_id == "move_AI2509_-10m"
        assert "Move flight AI2509 by -10 minutes" in impact_card.scenario_description
        assert impact_card.affected_flights_count == 1
        assert impact_card.delay_direction in ["improvement", "degradation", "neutral"]
        assert impact_card.confidence_level in ["high", "medium", "low"]
    
    def test_analyze_nonexistent_flight(self, simulator):
        """Test analysis of non-existent flight."""
        impact_card = simulator.analyze_single_flight_change(
            flight_id="NONEXISTENT",
            time_change_minutes=5,
            airport="BOM",
            analysis_date=date(2024, 1, 15)
        )
        
        assert isinstance(impact_card, ImpactCard)
        assert "not found" in impact_card.recommendation.lower()
        assert impact_card.confidence_level == "low"
        assert impact_card.affected_flights_count == 0
    
    def test_simulate_scenario(self, simulator):
        """Test complete scenario simulation."""
        # Create test scenario
        flight_change = FlightChange(
            flight_id="AI2509_001",
            original_time=datetime(2024, 1, 15, 8, 30),
            new_time=datetime(2024, 1, 15, 8, 20),  # 10 minutes earlier
            change_type="departure"
        )
        
        scenario = WhatIfScenario(
            scenario_id="test_scenario",
            description="Test scenario",
            changes=[flight_change],
            base_date=date(2024, 1, 15),
            airport="BOM"
        )
        
        impact_card = simulator.simulate_scenario(scenario)
        
        assert isinstance(impact_card, ImpactCard)
        assert impact_card.scenario_id == "test_scenario"
        assert impact_card.scenario_description == "Test scenario"
        assert impact_card.affected_flights_count == 1
    
    def test_compare_before_after(self, simulator):
        """Test before/after comparison."""
        flight_change = FlightChange(
            flight_id="AI2509_001",
            original_time=datetime(2024, 1, 15, 8, 30),
            new_time=datetime(2024, 1, 15, 8, 20),
            change_type="departure"
        )
        
        scenario = WhatIfScenario(
            scenario_id="comparison_test",
            description="Comparison test",
            changes=[flight_change],
            base_date=date(2024, 1, 15),
            airport="BOM"
        )
        
        comparison = simulator.compare_before_after(scenario)
        
        assert isinstance(comparison, BeforeAfterComparison)
        assert isinstance(comparison.before_metrics, DelayMetrics)
        assert isinstance(comparison.after_metrics, DelayMetrics)
        assert isinstance(comparison.before_peak_analysis, PeakAnalysis)
        assert isinstance(comparison.after_peak_analysis, PeakAnalysis)
        
        summary = comparison.get_summary()
        assert "delay_metrics" in summary
        assert "capacity_metrics" in summary
        assert "detailed_changes" in summary
    
    def test_calculate_delay_impact(self, simulator):
        """Test delay impact calculation."""
        # Create test schedules
        base_flights = [
            self._create_test_flight("AI2509", dep_delay=15),
            self._create_test_flight("6E123", dep_delay=5)
        ]
        
        modified_flights = [
            self._create_test_flight("AI2509", dep_delay=5),  # Improved
            self._create_test_flight("6E123", dep_delay=5)   # Same
        ]
        
        base_schedule = Schedule(flights=base_flights)
        modified_schedule = Schedule(flights=modified_flights)
        
        delay_impact = simulator._calculate_delay_impact(base_schedule, modified_schedule)
        
        assert "total_change" in delay_impact
        assert "direction" in delay_impact
        assert delay_impact["direction"] in ["improvement", "degradation", "neutral"]
        assert delay_impact["total_change"] < 0  # Should be improvement
    
    def test_calculate_co2_impact(self, simulator):
        """Test CO2 impact calculation."""
        # Create test changes
        changes = [
            FlightChange(
                flight_id="AI2509_001",
                original_time=datetime(2024, 1, 15, 8, 30),
                new_time=datetime(2024, 1, 15, 8, 20),  # 10 minutes earlier
                change_type="departure"
            )
        ]
        
        # Create test schedule
        flights = [self._create_test_flight("AI2509", aircraft_type="A320")]
        base_schedule = Schedule(flights=flights)
        
        co2_impact = simulator._calculate_co2_impact(changes, base_schedule)
        
        assert "total_kg" in co2_impact
        assert "fuel_liters" in co2_impact
        assert "direction" in co2_impact
        assert co2_impact["direction"] in ["improvement", "degradation", "neutral"]
        assert co2_impact["total_kg"] < 0  # Earlier departure should reduce CO2
    
    def test_calculate_fairness_score(self, simulator):
        """Test fairness score calculation."""
        # Test with evenly distributed changes
        even_changes = [
            FlightChange("flight1", datetime.now(), datetime.now() + timedelta(minutes=5), "departure"),
            FlightChange("flight2", datetime.now(), datetime.now() + timedelta(minutes=5), "departure"),
            FlightChange("flight3", datetime.now(), datetime.now() + timedelta(minutes=5), "departure")
        ]
        
        base_schedule = Schedule(flights=[])
        even_score = simulator._calculate_fairness_score(even_changes, base_schedule)
        
        # Test with uneven changes
        uneven_changes = [
            FlightChange("flight1", datetime.now(), datetime.now() + timedelta(minutes=1), "departure"),
            FlightChange("flight2", datetime.now(), datetime.now() + timedelta(minutes=30), "departure"),
            FlightChange("flight3", datetime.now(), datetime.now() + timedelta(minutes=2), "departure")
        ]
        
        uneven_score = simulator._calculate_fairness_score(uneven_changes, base_schedule)
        
        assert 0 <= even_score <= 1
        assert 0 <= uneven_score <= 1
        assert even_score >= uneven_score  # Even distribution should be more fair
    
    def test_generate_recommendation(self, simulator):
        """Test recommendation generation."""
        # Test positive impact
        delay_impact = {"direction": "improvement", "total_change": -10, "otp_change": 5, "base_total": 100}
        capacity_impact = {"direction": "improvement", "overload_change": -1, "new_windows": 0}
        co2_impact = {"direction": "improvement", "total_kg": -20}
        fairness_score = 0.8
        
        recommendation, confidence = simulator._generate_recommendation(
            delay_impact, capacity_impact, co2_impact, fairness_score
        )
        
        assert isinstance(recommendation, str)
        assert confidence in ["high", "medium", "low"]
        assert "recommended" in recommendation.lower()
        
        # Test negative impact
        delay_impact = {"direction": "degradation", "total_change": 20, "otp_change": -5, "base_total": 100}
        capacity_impact = {"direction": "degradation", "overload_change": 2, "new_windows": 2}
        co2_impact = {"direction": "degradation", "total_kg": 50}
        fairness_score = 0.3
        
        recommendation, confidence = simulator._generate_recommendation(
            delay_impact, capacity_impact, co2_impact, fairness_score
        )
        
        assert "not recommended" in recommendation.lower()
    
    def test_impact_card_to_dict(self):
        """Test impact card dictionary conversion."""
        impact_card = ImpactCard(
            scenario_id="test_scenario",
            scenario_description="Test scenario",
            delay_change_minutes=-5.5,
            delay_direction="improvement",
            affected_flights_count=2,
            peak_overload_change=-1,
            capacity_direction="improvement",
            new_overload_windows=0,
            co2_change_kg=-25.3,
            co2_direction="improvement",
            fuel_savings_liters=10.1,
            on_time_performance_change=3.2,
            fairness_score=0.85,
            constraint_violations=0,
            recommendation="Highly recommended",
            confidence_level="high"
        )
        
        result_dict = impact_card.to_dict()
        
        assert "scenario" in result_dict
        assert "delay_impact" in result_dict
        assert "capacity_impact" in result_dict
        assert "environmental_impact" in result_dict
        assert "operational_metrics" in result_dict
        assert "recommendation" in result_dict
        
        assert result_dict["scenario"]["id"] == "test_scenario"
        assert result_dict["delay_impact"]["change_minutes"] == -5.5
        assert result_dict["environmental_impact"]["co2_change_kg"] == -25.3
        assert result_dict["recommendation"]["confidence"] == "high"
    
    def test_edge_cases(self, simulator):
        """Test edge cases and error handling."""
        # Test with empty schedule
        impact_card = simulator.analyze_single_flight_change(
            flight_id="AI2509",
            time_change_minutes=10,
            airport="UNKNOWN",
            analysis_date=date(2024, 1, 15)
        )
        
        assert isinstance(impact_card, ImpactCard)
        assert impact_card.confidence_level in ["low", "medium"]  # Can be either based on data
        
        # Test with zero time change
        impact_card = simulator.analyze_single_flight_change(
            flight_id="AI2509",
            time_change_minutes=0,
            airport="BOM",
            analysis_date=date(2024, 1, 15)
        )
        
        assert isinstance(impact_card, ImpactCard)
        assert impact_card.delay_change_minutes == 0 or abs(impact_card.delay_change_minutes) < 1
    
    def test_weather_regime_impact(self, simulator):
        """Test impact of different weather regimes."""
        # Test with severe weather
        impact_card = simulator.analyze_single_flight_change(
            flight_id="AI2509",
            time_change_minutes=-15,
            airport="BOM",
            analysis_date=date(2024, 1, 15),
            weather_regime=WeatherRegime.SEVERE
        )
        
        assert isinstance(impact_card, ImpactCard)
        
        # Test with calm weather
        impact_card_calm = simulator.analyze_single_flight_change(
            flight_id="AI2509",
            time_change_minutes=-15,
            airport="BOM",
            analysis_date=date(2024, 1, 15),
            weather_regime=WeatherRegime.CALM
        )
        
        assert isinstance(impact_card_calm, ImpactCard)
    
    def _create_test_flight(self, flight_number: str, dep_delay: int = 0, 
                          aircraft_type: str = "A320") -> Flight:
        """Create a test flight object."""
        flight = Flight()
        flight.flight_id = f"{flight_number}_001"
        flight.flight_number = flight_number
        flight.airline_code = flight_number[:2]
        flight.aircraft_type = aircraft_type
        
        flight.origin = Airport(code="BOM", name="Mumbai (BOM)", city="Mumbai")
        flight.destination = Airport(code="DEL", name="Delhi (DEL)", city="Delhi")
        
        flight.flight_date = date(2024, 1, 15)
        flight.departure.scheduled = time(8, 30)
        flight.departure.actual = datetime(2024, 1, 15, 8, 30) + timedelta(minutes=dep_delay)
        flight.dep_delay_min = dep_delay
        
        return flight


class TestWhatIfScenario:
    """Test what-if scenario data structure."""
    
    def test_scenario_creation(self):
        """Test scenario creation."""
        changes = [
            FlightChange(
                flight_id="AI2509",
                original_time=datetime(2024, 1, 15, 8, 30),
                new_time=datetime(2024, 1, 15, 8, 20),
                change_type="departure"
            )
        ]
        
        scenario = WhatIfScenario(
            scenario_id="test_scenario",
            description="Test scenario description",
            changes=changes,
            base_date=date(2024, 1, 15),
            airport="BOM"
        )
        
        assert scenario.scenario_id == "test_scenario"
        assert scenario.description == "Test scenario description"
        assert len(scenario.changes) == 1
        assert scenario.base_date == date(2024, 1, 15)
        assert scenario.airport == "BOM"
        assert scenario.weather_regime == WeatherRegime.CALM
        assert isinstance(scenario.created_at, datetime)


class TestBeforeAfterComparison:
    """Test before/after comparison functionality."""
    
    def test_comparison_summary(self):
        """Test comparison summary generation."""
        before_metrics = DelayMetrics(100, 10, 30, 5, 75)
        after_metrics = DelayMetrics(80, 8, 25, 3, 85)
        
        before_analysis = PeakAnalysis(
            airport="BOM",
            analysis_date=date(2024, 1, 15),
            bucket_minutes=10,
            overload_windows=[OverloadWindow(
                start_time=datetime(2024, 1, 15, 9, 0),
                end_time=datetime(2024, 1, 15, 9, 30),
                duration_minutes=30,
                peak_overload=5,
                avg_overload=3.0,
                affected_flights=15
            )],
            avg_utilization=0.85
        )
        
        after_analysis = PeakAnalysis(
            airport="BOM",
            analysis_date=date(2024, 1, 15),
            bucket_minutes=10,
            overload_windows=[],
            avg_utilization=0.75
        )
        
        comparison = BeforeAfterComparison(
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            before_peak_analysis=before_analysis,
            after_peak_analysis=after_analysis,
            delay_improvements=[],
            delay_degradations=[],
            capacity_changes=[]
        )
        
        summary = comparison.get_summary()
        
        assert "delay_metrics" in summary
        assert "capacity_metrics" in summary
        assert "detailed_changes" in summary
        
        assert summary["delay_metrics"]["before"]["total_delay"] == 100
        assert summary["delay_metrics"]["after"]["total_delay"] == 80
        assert summary["capacity_metrics"]["before_overloads"] == 1
        assert summary["capacity_metrics"]["after_overloads"] == 0


if __name__ == "__main__":
    pytest.main([__file__])