"""Tests for schedule optimization engine."""

import pytest
from datetime import datetime, date, time, timedelta
from unittest.mock import Mock, patch
import numpy as np

from src.services.schedule_optimizer import (
    ScheduleOptimizer, Schedule, Constraints, ObjectiveWeights,
    TimeSlot, FlightChange, ConstraintViolation, DelayMetrics,
    OptimizationResult, ImpactAnalysis, ConstraintType,
    WeatherRegime, OptimizationObjective
)
from src.models.flight import Flight, Airport, FlightTime
from src.services.analytics import WeatherRegime as AnalyticsWeatherRegime


class TestScheduleOptimizer:
    """Test suite for ScheduleOptimizer class."""
    
    @pytest.fixture
    def sample_flights(self):
        """Create sample flights for testing."""
        flights = []
        
        # Create airports
        bom = Airport(code="BOM", name="Mumbai", city="Mumbai")
        del_airport = Airport(code="DEL", name="Delhi", city="Delhi")
        
        # Flight 1: Early morning departure
        flight1 = Flight(
            flight_id="AI101",
            flight_number="AI101",
            airline_code="AI",
            origin=bom,
            destination=del_airport,
            aircraft_type="A320",
            aircraft_registration="VT-ABC",
            flight_date=date.today(),
            departure=FlightTime(scheduled=time(6, 30)),
            arrival=FlightTime(scheduled=time(8, 45))
        )
        
        # Flight 2: Peak hour departure
        flight2 = Flight(
            flight_id="6E202",
            flight_number="6E202",
            airline_code="6E",
            origin=bom,
            destination=del_airport,
            aircraft_type="A320",
            aircraft_registration="VT-DEF",
            flight_date=date.today(),
            departure=FlightTime(scheduled=time(8, 15)),
            arrival=FlightTime(scheduled=time(10, 30))
        )
        
        # Flight 3: Another peak hour departure (creates congestion)
        flight3 = Flight(
            flight_id="AI303",
            flight_number="AI303",
            airline_code="AI",
            origin=bom,
            destination=del_airport,
            aircraft_type="B737",
            aircraft_registration="VT-GHI",
            flight_date=date.today(),
            departure=FlightTime(scheduled=time(8, 20)),  # 5 minutes after flight2
            arrival=FlightTime(scheduled=time(10, 35))
        )
        
        # Flight 4: Curfew hour departure (should be penalized)
        flight4 = Flight(
            flight_id="AI404",
            flight_number="AI404",
            airline_code="AI",
            origin=bom,
            destination=del_airport,
            aircraft_type="A321",
            aircraft_registration="VT-JKL",
            flight_date=date.today(),
            departure=FlightTime(scheduled=time(2, 30)),  # Curfew hour
            arrival=FlightTime(scheduled=time(4, 45))
        )
        
        flights.extend([flight1, flight2, flight3, flight4])
        return flights
    
    @pytest.fixture
    def sample_constraints(self):
        """Create sample constraints for testing."""
        return Constraints(
            min_turnaround_minutes={
                "A320": 45,
                "A321": 50,
                "B737": 45,
                "DEFAULT": 60
            },
            runway_capacity={"DEFAULT": 30},
            curfew_hours=[1, 2, 3, 4, 5],
            min_separation_minutes=2.0
        )
    
    @pytest.fixture
    def optimizer(self):
        """Create ScheduleOptimizer instance with mocked dependencies."""
        with patch('src.services.schedule_optimizer.FlightDatabaseService'), \
             patch('src.services.schedule_optimizer.AnalyticsEngine'), \
             patch('src.services.schedule_optimizer.DelayRiskPredictor'):
            return ScheduleOptimizer()
    
    def test_optimizer_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert optimizer is not None
        assert optimizer.max_optimization_time_seconds == 300
        assert optimizer.solution_time_limit_seconds == 60
    
    def test_schedule_creation(self, sample_flights):
        """Test Schedule creation and delay metrics calculation."""
        # Add some delays to flights
        sample_flights[0].dep_delay_min = 10
        sample_flights[1].dep_delay_min = 25  # Delayed flight
        sample_flights[2].dep_delay_min = 5
        sample_flights[3].dep_delay_min = 0
        
        schedule = Schedule(flights=sample_flights)
        metrics = schedule.get_delay_metrics()
        
        assert metrics.total_delay_minutes == 40
        assert metrics.avg_delay_minutes == 10
        assert metrics.delayed_flights_count == 1  # Only flight with >15min delay
        assert metrics.on_time_performance == 75.0  # 3 out of 4 flights on time
    
    def test_constraints_validation(self, optimizer, sample_flights, sample_constraints):
        """Test constraint validation."""
        schedule = Schedule(flights=sample_flights)
        violations = optimizer.validate_constraints(schedule, sample_constraints)
        
        # Should find curfew violation for flight4
        curfew_violations = [v for v in violations if v.constraint_type == ConstraintType.CURFEW_HOURS]
        assert len(curfew_violations) >= 1
        assert curfew_violations[0].flight_id == "AI404"
        assert curfew_violations[0].severity == "critical"
    
    def test_turnaround_constraint_validation(self, optimizer, sample_constraints):
        """Test turnaround time constraint validation."""
        # Create flights with same aircraft for turnaround
        bom = Airport(code="BOM", name="Mumbai", city="Mumbai")
        del_airport = Airport(code="DEL", name="Delhi", city="Delhi")
        
        # Arrival flight
        arrival_flight = Flight(
            flight_id="AI501",
            flight_number="AI501",
            airline_code="AI",
            origin=del_airport,
            destination=bom,
            aircraft_type="A320",
            aircraft_registration="VT-TURN",
            flight_date=date.today(),
            departure=FlightTime(scheduled=time(10, 0)),
            arrival=FlightTime(scheduled=time(12, 15))
        )
        
        # Departure flight with insufficient turnaround time
        departure_flight = Flight(
            flight_id="AI502",
            flight_number="AI502",
            airline_code="AI",
            origin=bom,
            destination=del_airport,
            aircraft_type="A320",
            aircraft_registration="VT-TURN",  # Same aircraft
            flight_date=date.today(),
            departure=FlightTime(scheduled=time(12, 30)),  # Only 15 minutes turnaround
            arrival=FlightTime(scheduled=time(14, 45))
        )
        
        schedule = Schedule(flights=[arrival_flight, departure_flight])
        violations = optimizer.validate_constraints(schedule, sample_constraints)
        
        # Should find turnaround violation
        turnaround_violations = [v for v in violations if v.constraint_type == ConstraintType.TURNAROUND_TIME]
        assert len(turnaround_violations) >= 1
        assert turnaround_violations[0].flight_id == "AI502"
    
    def test_wake_separation_validation(self, optimizer, sample_constraints):
        """Test wake turbulence separation validation."""
        bom = Airport(code="BOM", name="Mumbai", city="Mumbai")
        del_airport = Airport(code="DEL", name="Delhi", city="Delhi")
        
        # Heavy aircraft followed by light aircraft (needs more separation)
        heavy_flight = Flight(
            flight_id="AI777",
            flight_number="AI777",
            airline_code="AI",
            origin=bom,
            destination=del_airport,
            aircraft_type="B777",  # Heavy aircraft
            flight_date=date.today(),
            departure=FlightTime(scheduled=time(10, 0))
        )
        
        light_flight = Flight(
            flight_id="AI100",
            flight_number="AI100",
            airline_code="AI",
            origin=bom,
            destination=del_airport,
            aircraft_type="ATR72",  # Light aircraft
            flight_date=date.today(),
            departure=FlightTime(scheduled=time(10, 1))  # Only 1 minute separation
        )
        
        schedule = Schedule(flights=[heavy_flight, light_flight])
        violations = optimizer.validate_constraints(schedule, sample_constraints)
        
        # Should find wake separation violation
        wake_violations = [v for v in violations if v.constraint_type == ConstraintType.WAKE_TURBULENCE]
        assert len(wake_violations) >= 1
    
    def test_time_slot_creation(self, optimizer, sample_flights, sample_constraints):
        """Test time slot creation for optimization."""
        time_slots = optimizer._create_time_slots(sample_flights, WeatherRegime.CALM, sample_constraints)
        
        assert len(time_slots) > 0
        
        # Check that slots cover the flight time range
        flight_times = [datetime.combine(f.flight_date or date.today(), f.departure.scheduled or time(12, 0)) 
                       for f in sample_flights if f.departure.scheduled]
        min_flight_time = min(flight_times)
        max_flight_time = max(flight_times)
        
        slot_times = [slot.timestamp for slot in time_slots]
        min_slot_time = min(slot_times)
        max_slot_time = max(slot_times)
        
        assert min_slot_time <= min_flight_time
        assert max_slot_time >= max_flight_time
        
        # Check curfew slots have reduced capacity
        curfew_slots = [slot for slot in time_slots if slot.is_curfew]
        normal_slots = [slot for slot in time_slots if not slot.is_curfew]
        
        if curfew_slots and normal_slots:
            avg_curfew_capacity = sum(slot.capacity for slot in curfew_slots) / len(curfew_slots)
            avg_normal_capacity = sum(slot.capacity for slot in normal_slots) / len(normal_slots)
            assert avg_curfew_capacity < avg_normal_capacity
    
    def test_assignment_cost_calculation(self, optimizer, sample_flights, sample_constraints):
        """Test cost calculation for flight-slot assignments."""
        flight = sample_flights[0]
        weights = ObjectiveWeights()
        
        # Create time slots
        original_time = datetime.combine(flight.flight_date or date.today(), 
                                       flight.departure.scheduled or time(12, 0))
        
        # Slot at original time (should have low cost)
        original_slot = TimeSlot(
            runway="DEFAULT",
            timestamp=original_time,
            capacity=10,
            weather_regime=WeatherRegime.CALM
        )
        
        # Slot 30 minutes later (should have higher cost)
        delayed_slot = TimeSlot(
            runway="DEFAULT",
            timestamp=original_time + timedelta(minutes=30),
            capacity=10,
            weather_regime=WeatherRegime.CALM
        )
        
        # Curfew slot (should have very high cost)
        curfew_slot = TimeSlot(
            runway="DEFAULT",
            timestamp=original_time.replace(hour=2),
            capacity=5,
            weather_regime=WeatherRegime.CALM,
            is_curfew=True
        )
        
        original_cost = optimizer._calculate_assignment_cost(flight, original_slot, weights)
        delayed_cost = optimizer._calculate_assignment_cost(flight, delayed_slot, weights)
        curfew_cost = optimizer._calculate_assignment_cost(flight, curfew_slot, weights)
        
        assert original_cost < delayed_cost
        assert delayed_cost < curfew_cost
    
    def test_fallback_optimization(self, optimizer, sample_flights, sample_constraints):
        """Test fallback optimization when OR-Tools is not available."""
        weights = ObjectiveWeights()
        original_schedule = Schedule(flights=sample_flights)
        
        result = optimizer._fallback_optimization(original_schedule, sample_constraints, weights)
        
        assert isinstance(result, OptimizationResult)
        assert result.solver_status == "HEURISTIC"
        assert len(result.affected_flights) >= 0
        assert result.optimization_time_seconds > 0
    
    @patch('src.services.schedule_optimizer.ORTOOLS_AVAILABLE', False)
    def test_optimize_schedule_without_ortools(self, optimizer, sample_flights, sample_constraints):
        """Test schedule optimization when OR-Tools is not available."""
        result = optimizer.optimize_schedule(sample_flights, sample_constraints)
        
        assert isinstance(result, OptimizationResult)
        assert result.solver_status == "HEURISTIC"
        assert result.original_schedule is not None
        assert result.optimized_schedule is not None
    
    def test_what_if_analysis(self, optimizer, sample_flights):
        """Test what-if analysis functionality."""
        base_schedule = Schedule(flights=sample_flights)
        
        # Propose moving a flight 15 minutes later
        flight_to_change = sample_flights[1]
        original_time = datetime.combine(flight_to_change.flight_date or date.today(),
                                       flight_to_change.departure.scheduled or time(12, 0))
        new_time = original_time + timedelta(minutes=15)
        
        changes = [FlightChange(
            flight_id=flight_to_change.flight_id,
            original_time=original_time,
            new_time=new_time,
            change_type="departure"
        )]
        
        impact = optimizer.what_if_analysis(base_schedule, changes)
        
        assert isinstance(impact, ImpactAnalysis)
        assert len(impact.affected_flights) == 1
        assert impact.affected_flights[0] == flight_to_change.flight_id
        assert impact.before_metrics is not None
        assert impact.after_metrics is not None
        
        # Test impact card generation
        impact_card = impact.get_impact_card()
        assert "delay_impact" in impact_card
        assert "capacity_impact" in impact_card
        assert "environmental_impact" in impact_card
        assert "overall_recommendation" in impact_card
    
    def test_objective_weights_normalization(self):
        """Test objective weights normalization."""
        weights = ObjectiveWeights(
            delay_weight=2.0,
            taxi_weight=1.0,
            runway_change_weight=0.5,
            fairness_weight=1.5,
            curfew_weight=3.0
        )
        
        normalized = weights.normalize()
        
        total = (normalized.delay_weight + normalized.taxi_weight + 
                normalized.runway_change_weight + normalized.fairness_weight + 
                normalized.curfew_weight)
        
        assert abs(total - 1.0) < 0.001  # Should sum to 1.0
    
    def test_delay_metrics_improvement_calculation(self):
        """Test delay metrics improvement calculation."""
        baseline = DelayMetrics(
            total_delay_minutes=100,
            avg_delay_minutes=20,
            p95_delay_minutes=45,
            delayed_flights_count=3,
            on_time_performance=70.0
        )
        
        improved = DelayMetrics(
            total_delay_minutes=80,
            avg_delay_minutes=16,
            p95_delay_minutes=35,
            delayed_flights_count=2,
            on_time_performance=80.0
        )
        
        improvement = improved.improvement_over(baseline)
        
        assert improvement["total_delay_reduction"] == 20
        assert improvement["avg_delay_reduction"] == 4
        assert improvement["p95_delay_reduction"] == 10
        assert improvement["otp_improvement"] == 10.0
    
    def test_constraint_violation_creation(self):
        """Test constraint violation object creation."""
        violation = ConstraintViolation(
            constraint_type=ConstraintType.TURNAROUND_TIME,
            flight_id="AI123",
            description="Insufficient turnaround time",
            severity="major",
            suggested_fix="Delay departure by 15 minutes"
        )
        
        assert violation.constraint_type == ConstraintType.TURNAROUND_TIME
        assert violation.flight_id == "AI123"
        assert violation.severity == "major"
        assert violation.suggested_fix is not None
    
    def test_flight_change_time_delta(self):
        """Test FlightChange time delta calculation."""
        original_time = datetime(2024, 1, 1, 10, 0)
        new_time = datetime(2024, 1, 1, 10, 15)
        
        change = FlightChange(
            flight_id="AI123",
            original_time=original_time,
            new_time=new_time,
            change_type="departure"
        )
        
        assert change.time_delta_minutes == 15.0
        
        # Test negative delta (earlier time)
        earlier_change = FlightChange(
            flight_id="AI124",
            original_time=original_time,
            new_time=original_time - timedelta(minutes=10),
            change_type="departure"
        )
        
        assert earlier_change.time_delta_minutes == -10.0
    
    def test_time_slot_utilization(self):
        """Test TimeSlot utilization calculation."""
        slot = TimeSlot(
            runway="RW09",
            timestamp=datetime.now(),
            capacity=10,
            current_demand=7
        )
        
        assert slot.utilization == 0.7
        assert slot.available_capacity == 3
        
        # Test overloaded slot
        overloaded_slot = TimeSlot(
            runway="RW09",
            timestamp=datetime.now(),
            capacity=10,
            current_demand=12
        )
        
        assert overloaded_slot.utilization == 1.2
        assert overloaded_slot.available_capacity == 0  # Can't be negative
    
    def test_wake_category_classification(self, optimizer):
        """Test wake turbulence category classification."""
        assert optimizer._get_wake_category("B777") == "HEAVY"
        assert optimizer._get_wake_category("B787") == "HEAVY"
        assert optimizer._get_wake_category("A320") == "MEDIUM"
        assert optimizer._get_wake_category("B737") == "MEDIUM"
        assert optimizer._get_wake_category("ATR72") == "LIGHT"
        assert optimizer._get_wake_category("CRJ900") == "LIGHT"
        assert optimizer._get_wake_category("") == "DEFAULT"
        assert optimizer._get_wake_category(None) == "DEFAULT"
    
    def test_co2_impact_estimation(self, optimizer):
        """Test CO2 impact estimation for flight changes."""
        changes = [
            FlightChange(
                flight_id="AI123",
                original_time=datetime(2024, 1, 1, 10, 0),
                new_time=datetime(2024, 1, 1, 9, 45),  # 15 minutes earlier
                change_type="departure"
            ),
            FlightChange(
                flight_id="AI124",
                original_time=datetime(2024, 1, 1, 11, 0),
                new_time=datetime(2024, 1, 1, 11, 20),  # 20 minutes later
                change_type="departure"
            )
        ]
        
        co2_impact = optimizer._estimate_co2_impact(changes)
        
        # Earlier departure should reduce CO2, later should increase
        # Net impact depends on the weights (0.5 for earlier, 0.3 for later)
        expected_impact = (-15 * 0.5) + (20 * 0.3)  # -7.5 + 6.0 = -1.5
        assert abs(co2_impact - expected_impact) < 0.1
    
    def test_fairness_score_calculation(self, optimizer):
        """Test fairness score calculation."""
        # Evenly distributed changes (high fairness)
        even_changes = [
            FlightChange("AI1", datetime.now(), datetime.now() + timedelta(minutes=10), "departure"),
            FlightChange("AI2", datetime.now(), datetime.now() + timedelta(minutes=10), "departure"),
            FlightChange("AI3", datetime.now(), datetime.now() + timedelta(minutes=10), "departure")
        ]
        
        even_score = optimizer._calculate_fairness_score(even_changes)
        
        # Unevenly distributed changes (low fairness)
        uneven_changes = [
            FlightChange("AI1", datetime.now(), datetime.now() + timedelta(minutes=5), "departure"),
            FlightChange("AI2", datetime.now(), datetime.now() + timedelta(minutes=30), "departure"),
            FlightChange("AI3", datetime.now(), datetime.now() + timedelta(minutes=60), "departure")
        ]
        
        uneven_score = optimizer._calculate_fairness_score(uneven_changes)
        
        assert even_score > uneven_score
        assert 0 <= even_score <= 1
        assert 0 <= uneven_score <= 1
    
    def test_optimization_result_summary(self, sample_flights):
        """Test OptimizationResult summary generation."""
        original_schedule = Schedule(flights=sample_flights)
        optimized_schedule = Schedule(flights=sample_flights)
        
        # Add some sample delays to test metrics
        optimized_schedule.flights[0].dep_delay_min = 5  # Improved from 10
        optimized_schedule.flights[1].dep_delay_min = 20  # Improved from 25
        
        flight_changes = [
            FlightChange("AI101", datetime.now(), datetime.now() + timedelta(minutes=5), "departure"),
            FlightChange("6E202", datetime.now(), datetime.now() - timedelta(minutes=5), "departure")
        ]
        
        result = OptimizationResult(
            original_schedule=original_schedule,
            optimized_schedule=optimized_schedule,
            cost_reduction=15.0,
            delay_improvement=optimized_schedule.get_delay_metrics(),
            affected_flights=flight_changes,
            constraint_violations=[],
            optimization_time_seconds=45.2,
            solver_status="OPTIMAL",
            objective_value=15.0
        )
        
        summary = result.get_summary()
        
        assert summary["flights_affected"] == 2
        assert summary["optimization_time_seconds"] == 45.2
        assert summary["solver_status"] == "OPTIMAL"
        assert summary["cost_reduction"] == 15.0
        assert "total_delay_reduction_minutes" in summary
        assert "constraint_violations" in summary


class TestConstraintsClass:
    """Test suite for Constraints class."""
    
    def test_constraints_initialization(self):
        """Test Constraints initialization with defaults."""
        constraints = Constraints()
        
        # Check default values are set
        assert "A320" in constraints.min_turnaround_minutes
        assert "DEFAULT" in constraints.min_turnaround_minutes
        assert "DEFAULT" in constraints.runway_capacity
        assert WeatherRegime.CALM in constraints.weather_capacity_reduction
        assert len(constraints.curfew_hours) > 0
        assert constraints.min_separation_minutes > 0
    
    def test_constraints_custom_values(self):
        """Test Constraints with custom values."""
        custom_constraints = Constraints(
            min_turnaround_minutes={"A320": 30, "B737": 35},
            runway_capacity={"RW09": 25, "RW27": 30},
            curfew_hours=[2, 3, 4],
            min_separation_minutes=3.0
        )
        
        assert custom_constraints.min_turnaround_minutes["A320"] == 30
        assert custom_constraints.runway_capacity["RW09"] == 25
        assert custom_constraints.curfew_hours == [2, 3, 4]
        assert custom_constraints.min_separation_minutes == 3.0


if __name__ == "__main__":
    pytest.main([__file__])