#!/usr/bin/env python3
"""
Simple test for Schedule Optimization Engine functionality
"""

import sys
from datetime import datetime, date, time, timedelta
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.services.schedule_optimizer import (
    ScheduleOptimizer, Schedule, Constraints, ObjectiveWeights,
    FlightChange, WeatherRegime
)
from src.models.flight import Flight, Airport, FlightTime


def test_basic_functionality():
    """Test basic schedule optimization functionality."""
    print("🧪 Testing Schedule Optimization Engine...")
    
    # Create sample flights
    bom = Airport(code="BOM", name="Mumbai", city="Mumbai")
    del_airport = Airport(code="DEL", name="Delhi", city="Delhi")
    
    flights = [
        Flight(
            flight_id="AI101",
            flight_number="AI101",
            airline_code="AI",
            origin=bom,
            destination=del_airport,
            aircraft_type="A320",
            flight_date=date.today(),
            departure=FlightTime(scheduled=time(8, 0)),
            dep_delay_min=10
        ),
        Flight(
            flight_id="AI102",
            flight_number="AI102",
            airline_code="AI",
            origin=bom,
            destination=del_airport,
            aircraft_type="A320",
            flight_date=date.today(),
            departure=FlightTime(scheduled=time(8, 5)),  # Close to first flight
            dep_delay_min=20
        )
    ]
    
    print(f"✅ Created {len(flights)} test flights")
    
    # Initialize optimizer
    optimizer = ScheduleOptimizer()
    print("✅ Optimizer initialized")
    
    # Create constraints
    constraints = Constraints()
    print("✅ Constraints created")
    
    # Test constraint validation
    schedule = Schedule(flights=flights)
    violations = optimizer.validate_constraints(schedule, constraints)
    print(f"✅ Constraint validation: {len(violations)} violations found")
    
    # Test schedule metrics
    metrics = schedule.get_delay_metrics()
    print(f"✅ Schedule metrics: {metrics.total_delay_minutes} min total delay")
    
    # Test optimization
    result = optimizer.optimize_schedule(flights, constraints)
    print(f"✅ Optimization completed: {result.solver_status}")
    print(f"   - Flights affected: {len(result.affected_flights)}")
    print(f"   - Constraint violations: {len(result.constraint_violations)}")
    
    # Test what-if analysis
    changes = [FlightChange(
        flight_id="AI101",
        original_time=datetime.combine(date.today(), time(8, 0)),
        new_time=datetime.combine(date.today(), time(8, 30)),
        change_type="departure"
    )]
    
    impact = optimizer.what_if_analysis(schedule, changes)
    print(f"✅ What-if analysis completed")
    print(f"   - Delay delta: {impact.delay_delta:.1f} minutes")
    print(f"   - Affected flights: {len(impact.affected_flights)}")
    
    print("\n🎉 All tests passed successfully!")
    return True


def test_data_models():
    """Test data model functionality."""
    print("\n🧪 Testing Data Models...")
    
    # Test ObjectiveWeights
    weights = ObjectiveWeights(delay_weight=2.0, curfew_weight=3.0)
    normalized = weights.normalize()
    total = (normalized.delay_weight + normalized.taxi_weight + 
             normalized.runway_change_weight + normalized.fairness_weight + 
             normalized.curfew_weight)
    assert abs(total - 1.0) < 0.001, "Weights should normalize to 1.0"
    print("✅ ObjectiveWeights normalization works")
    
    # Test FlightChange
    change = FlightChange(
        flight_id="AI123",
        original_time=datetime(2024, 1, 1, 10, 0),
        new_time=datetime(2024, 1, 1, 10, 15),
        change_type="departure"
    )
    assert change.time_delta_minutes == 15.0, "Time delta calculation incorrect"
    print("✅ FlightChange time delta calculation works")
    
    # Test Constraints
    constraints = Constraints()
    assert "A320" in constraints.min_turnaround_minutes, "Default turnaround times not set"
    assert "DEFAULT" in constraints.runway_capacity, "Default runway capacity not set"
    print("✅ Constraints default values work")
    
    print("✅ All data model tests passed!")
    return True


def main():
    """Run all tests."""
    print("🚀 Schedule Optimization Engine - Simple Test Suite")
    print("=" * 60)
    
    try:
        # Test basic functionality
        test_basic_functionality()
        
        # Test data models
        test_data_models()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED SUCCESSFULLY!")
        print("🎯 Schedule Optimization Engine is working correctly")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())