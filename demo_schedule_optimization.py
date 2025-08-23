#!/usr/bin/env python3
"""
Demo script for Schedule Optimization Engine

This script demonstrates the schedule optimization capabilities including:
- Constraint-based optimization using min-cost flow and CP-SAT algorithms
- Multi-objective optimization with weighted objectives
- What-if analysis for schedule changes
- Constraint validation and violation detection
"""

import os
import sys
from datetime import datetime, date, time, timedelta
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.services.schedule_optimizer import (
    ScheduleOptimizer, Schedule, Constraints, ObjectiveWeights,
    FlightChange, WeatherRegime, OptimizationObjective
)
from src.models.flight import Flight, Airport, FlightTime
from src.services.database import FlightDatabaseService


def create_sample_flights() -> list[Flight]:
    """Create sample flights for optimization demo."""
    print("Creating sample flight schedule...")
    
    # Create airports
    bom = Airport(code="BOM", name="Mumbai (BOM)", city="Mumbai")
    del_airport = Airport(code="DEL", name="Delhi (DEL)", city="Delhi")
    blr = Airport(code="BLR", name="Bangalore (BLR)", city="Bangalore")
    
    flights = []
    today = date.today()
    
    # Morning rush - multiple flights clustered together (creates congestion)
    flights.extend([
        Flight(
            flight_id="AI101",
            flight_number="AI101",
            airline_code="AI",
            origin=bom,
            destination=del_airport,
            aircraft_type="A320",
            aircraft_registration="VT-ABC",
            flight_date=today,
            departure=FlightTime(scheduled=time(8, 0)),
            arrival=FlightTime(scheduled=time(10, 15)),
            dep_delay_min=5  # Minor delay
        ),
        Flight(
            flight_id="6E202",
            flight_number="6E202",
            airline_code="6E",
            origin=bom,
            destination=del_airport,
            aircraft_type="A320",
            aircraft_registration="VT-DEF",
            flight_date=today,
            departure=FlightTime(scheduled=time(8, 5)),  # 5 minutes after AI101
            arrival=FlightTime(scheduled=time(10, 20)),
            dep_delay_min=15  # Moderate delay
        ),
        Flight(
            flight_id="AI303",
            flight_number="AI303",
            airline_code="AI",
            origin=bom,
            destination=blr,
            aircraft_type="B737",
            aircraft_registration="VT-GHI",
            flight_date=today,
            departure=FlightTime(scheduled=time(8, 10)),  # Another clustered flight
            arrival=FlightTime(scheduled=time(9, 30)),
            dep_delay_min=25  # Significant delay
        ),
        Flight(
            flight_id="SG404",
            flight_number="SG404",
            airline_code="SG",
            origin=bom,
            destination=del_airport,
            aircraft_type="B737",
            aircraft_registration="VT-JKL",
            flight_date=today,
            departure=FlightTime(scheduled=time(8, 15)),  # Peak congestion
            arrival=FlightTime(scheduled=time(10, 30)),
            dep_delay_min=30  # Major delay
        )
    ])
    
    # Add some flights with turnaround constraints (same aircraft)
    flights.extend([
        # Arrival flight
        Flight(
            flight_id="AI505",
            flight_number="AI505",
            airline_code="AI",
            origin=del_airport,
            destination=bom,
            aircraft_type="A321",
            aircraft_registration="VT-TURN",
            flight_date=today,
            departure=FlightTime(scheduled=time(11, 0)),
            arrival=FlightTime(scheduled=time(13, 15))
        ),
        # Departure flight with tight turnaround
        Flight(
            flight_id="AI506",
            flight_number="AI506",
            airline_code="AI",
            origin=bom,
            destination=del_airport,
            aircraft_type="A321",
            aircraft_registration="VT-TURN",  # Same aircraft
            flight_date=today,
            departure=FlightTime(scheduled=time(13, 45)),  # Only 30 min turnaround
            arrival=FlightTime(scheduled=time(16, 0))
        )
    ])
    
    # Add a curfew violation flight
    flights.append(
        Flight(
            flight_id="AI999",
            flight_number="AI999",
            airline_code="AI",
            origin=bom,
            destination=del_airport,
            aircraft_type="A320",
            aircraft_registration="VT-NIGHT",
            flight_date=today,
            departure=FlightTime(scheduled=time(2, 30)),  # Curfew hour
            arrival=FlightTime(scheduled=time(4, 45))
        )
    )
    
    # Add some well-spaced flights (good baseline)
    flights.extend([
        Flight(
            flight_id="AI701",
            flight_number="AI701",
            airline_code="AI",
            origin=bom,
            destination=del_airport,
            aircraft_type="A320",
            aircraft_registration="VT-GOOD1",
            flight_date=today,
            departure=FlightTime(scheduled=time(14, 0)),
            arrival=FlightTime(scheduled=time(16, 15))
        ),
        Flight(
            flight_id="6E802",
            flight_number="6E802",
            airline_code="6E",
            origin=bom,
            destination=blr,
            aircraft_type="A320",
            aircraft_registration="VT-GOOD2",
            flight_date=today,
            departure=FlightTime(scheduled=time(16, 30)),
            arrival=FlightTime(scheduled=time(17, 50))
        )
    ])
    
    print(f"Created {len(flights)} sample flights")
    return flights


def create_optimization_constraints() -> Constraints:
    """Create realistic optimization constraints."""
    print("Setting up optimization constraints...")
    
    constraints = Constraints(
        min_turnaround_minutes={
            "A320": 45,
            "A321": 50,
            "B737": 45,
            "B777": 90,
            "DEFAULT": 60
        },
        runway_capacity={
            "DEFAULT": 20  # Reduced capacity to create optimization opportunities
        },
        curfew_hours=[1, 2, 3, 4, 5],  # 1 AM to 5 AM
        min_separation_minutes=3.0,  # 3 minutes minimum separation
        curfew_penalty=200.0  # High penalty for curfew violations
    )
    
    print("Constraints configured:")
    print(f"  - Turnaround times: {constraints.min_turnaround_minutes}")
    print(f"  - Runway capacity: {constraints.runway_capacity['DEFAULT']} ops/hour")
    print(f"  - Curfew hours: {constraints.curfew_hours}")
    print(f"  - Minimum separation: {constraints.min_separation_minutes} minutes")
    
    return constraints


def demonstrate_constraint_validation(optimizer: ScheduleOptimizer, 
                                    flights: list[Flight], 
                                    constraints: Constraints):
    """Demonstrate constraint validation functionality."""
    print("\n" + "="*60)
    print("CONSTRAINT VALIDATION DEMO")
    print("="*60)
    
    schedule = Schedule(flights=flights)
    violations = optimizer.validate_constraints(schedule, constraints)
    
    print(f"\nFound {len(violations)} constraint violations:")
    
    for i, violation in enumerate(violations, 1):
        print(f"\n{i}. {violation.constraint_type.value.upper()} VIOLATION")
        print(f"   Flight: {violation.flight_id}")
        print(f"   Severity: {violation.severity}")
        print(f"   Description: {violation.description}")
        if violation.suggested_fix:
            print(f"   Suggested fix: {violation.suggested_fix}")
    
    if not violations:
        print("✅ No constraint violations found!")
    
    return violations


def demonstrate_schedule_optimization(optimizer: ScheduleOptimizer,
                                    flights: list[Flight],
                                    constraints: Constraints):
    """Demonstrate schedule optimization."""
    print("\n" + "="*60)
    print("SCHEDULE OPTIMIZATION DEMO")
    print("="*60)
    
    # Configure optimization weights
    weights = ObjectiveWeights(
        delay_weight=1.0,      # Primary objective: minimize delays
        taxi_weight=0.3,       # Secondary: minimize taxi time
        runway_change_weight=0.2,  # Minimize runway changes
        fairness_weight=0.4,   # Ensure fair distribution of changes
        curfew_weight=2.0      # High penalty for curfew violations
    )
    
    print(f"\nOptimization objectives (weights):")
    print(f"  - Minimize delays: {weights.delay_weight}")
    print(f"  - Minimize taxi time: {weights.taxi_weight}")
    print(f"  - Minimize runway changes: {weights.runway_change_weight}")
    print(f"  - Maximize fairness: {weights.fairness_weight}")
    print(f"  - Avoid curfew violations: {weights.curfew_weight}")
    
    # Calculate original schedule metrics
    original_schedule = Schedule(flights=flights)
    original_metrics = original_schedule.get_delay_metrics()
    
    print(f"\nOriginal schedule metrics:")
    print(f"  - Total delay: {original_metrics.total_delay_minutes:.1f} minutes")
    print(f"  - Average delay: {original_metrics.avg_delay_minutes:.1f} minutes")
    print(f"  - P95 delay: {original_metrics.p95_delay_minutes:.1f} minutes")
    print(f"  - Delayed flights: {original_metrics.delayed_flights_count}")
    print(f"  - On-time performance: {original_metrics.on_time_performance:.1f}%")
    
    # Run optimization
    print(f"\n🚀 Running schedule optimization...")
    print("   (This may take a few seconds...)")
    
    start_time = datetime.now()
    result = optimizer.optimize_schedule(
        flights=flights,
        constraints=constraints,
        weights=weights,
        weather_regime=WeatherRegime.CALM
    )
    optimization_time = (datetime.now() - start_time).total_seconds()
    
    print(f"✅ Optimization completed in {optimization_time:.2f} seconds")
    print(f"   Solver status: {result.solver_status}")
    
    # Display results
    print(f"\nOptimization results:")
    summary = result.get_summary()
    print(f"  - Flights affected: {summary['flights_affected']}")
    print(f"  - Total delay reduction: {summary['total_delay_reduction_minutes']:.1f} minutes")
    print(f"  - Average delay reduction: {summary['avg_delay_reduction_minutes']:.1f} minutes")
    print(f"  - OTP improvement: {summary['otp_improvement_percent']:.1f}%")
    print(f"  - Constraint violations: {summary['constraint_violations']}")
    print(f"  - Cost reduction: {summary['cost_reduction']:.1f}")
    
    # Show flight changes
    if result.affected_flights:
        print(f"\nFlight changes made:")
        for i, change in enumerate(result.affected_flights, 1):
            direction = "earlier" if change.time_delta_minutes < 0 else "later"
            print(f"  {i}. {change.flight_id}: moved {abs(change.time_delta_minutes):.0f} min {direction}")
            print(f"     {change.original_time.strftime('%H:%M')} → {change.new_time.strftime('%H:%M')}")
    else:
        print("\n✅ No flight changes needed - schedule is already optimal!")
    
    # Show remaining violations
    if result.constraint_violations:
        print(f"\nRemaining constraint violations:")
        for violation in result.constraint_violations:
            print(f"  - {violation.flight_id}: {violation.description}")
    else:
        print("\n✅ All constraints satisfied!")
    
    return result


def demonstrate_what_if_analysis(optimizer: ScheduleOptimizer, flights: list[Flight]):
    """Demonstrate what-if analysis functionality."""
    print("\n" + "="*60)
    print("WHAT-IF ANALYSIS DEMO")
    print("="*60)
    
    base_schedule = Schedule(flights=flights)
    
    # Propose some changes
    print("\nProposing schedule changes:")
    
    # Move a peak-hour flight to off-peak
    peak_flight = next((f for f in flights if f.flight_number == "6E202"), None)
    if peak_flight:
        original_time = datetime.combine(peak_flight.flight_date or date.today(),
                                       peak_flight.departure.scheduled or time(12, 0))
        new_time = original_time + timedelta(hours=2)  # Move to off-peak
        
        change1 = FlightChange(
            flight_id=peak_flight.flight_id,
            original_time=original_time,
            new_time=new_time,
            change_type="departure"
        )
        
        print(f"  1. Move {peak_flight.flight_number} from {original_time.strftime('%H:%M')} to {new_time.strftime('%H:%M')}")
    
    # Move curfew flight to daytime
    curfew_flight = next((f for f in flights if f.flight_number == "AI999"), None)
    if curfew_flight:
        original_time = datetime.combine(curfew_flight.flight_date or date.today(),
                                       curfew_flight.departure.scheduled or time(12, 0))
        new_time = original_time.replace(hour=9, minute=0)  # Move to 9 AM
        
        change2 = FlightChange(
            flight_id=curfew_flight.flight_id,
            original_time=original_time,
            new_time=new_time,
            change_type="departure"
        )
        
        print(f"  2. Move {curfew_flight.flight_number} from {original_time.strftime('%H:%M')} to {new_time.strftime('%H:%M')}")
    
    changes = [change1, change2] if peak_flight and curfew_flight else []
    
    if not changes:
        print("  No suitable flights found for what-if analysis")
        return
    
    # Run what-if analysis
    print(f"\n🔍 Analyzing impact of proposed changes...")
    
    impact = optimizer.what_if_analysis(base_schedule, changes)
    
    print(f"\nImpact analysis results:")
    print(f"  - Delay impact: {impact.delay_delta:.1f} minutes")
    print(f"  - Peak overload change: {impact.peak_overload_change}")
    print(f"  - CO₂ impact: {impact.co2_impact:.1f} kg")
    print(f"  - Fairness score: {impact.fairness_score:.2f}")
    print(f"  - Affected flights: {len(impact.affected_flights)}")
    
    # Generate impact card
    impact_card = impact.get_impact_card()
    
    print(f"\n📊 Impact Card:")
    print(f"  Delay Impact: {impact_card['delay_impact']['change_minutes']:.1f} min ({impact_card['delay_impact']['direction']})")
    print(f"  Capacity Impact: {impact_card['capacity_impact']['overload_change']} ({impact_card['capacity_impact']['direction']})")
    print(f"  Environmental Impact: {impact_card['environmental_impact']['co2_change_kg']:.1f} kg CO₂ ({impact_card['environmental_impact']['direction']})")
    print(f"  Fairness Score: {impact_card['fairness_score']:.2f}")
    print(f"  Overall Recommendation: {impact_card['overall_recommendation']}")
    
    return impact


def demonstrate_weather_impact(optimizer: ScheduleOptimizer, 
                             flights: list[Flight], 
                             constraints: Constraints):
    """Demonstrate optimization under different weather conditions."""
    print("\n" + "="*60)
    print("WEATHER IMPACT DEMO")
    print("="*60)
    
    weather_conditions = [
        (WeatherRegime.CALM, "Clear weather"),
        (WeatherRegime.MEDIUM, "Moderate winds"),
        (WeatherRegime.STRONG, "Strong winds"),
        (WeatherRegime.SEVERE, "Severe weather")
    ]
    
    weights = ObjectiveWeights()
    
    print("Comparing optimization results under different weather conditions:\n")
    
    for weather, description in weather_conditions:
        print(f"🌤️  {description} ({weather.value}):")
        
        result = optimizer.optimize_schedule(
            flights=flights,
            constraints=constraints,
            weights=weights,
            weather_regime=weather
        )
        
        summary = result.get_summary()
        print(f"   - Flights affected: {summary['flights_affected']}")
        print(f"   - Delay reduction: {summary['total_delay_reduction_minutes']:.1f} min")
        print(f"   - Constraint violations: {summary['constraint_violations']}")
        print(f"   - Optimization time: {summary['optimization_time_seconds']:.2f}s")
        print()


def main():
    """Main demo function."""
    print("🛫 Schedule Optimization Engine Demo")
    print("=" * 60)
    
    try:
        # Initialize components
        print("Initializing schedule optimizer...")
        optimizer = ScheduleOptimizer()
        
        # Create sample data
        flights = create_sample_flights()
        constraints = create_optimization_constraints()
        
        # Run demonstrations
        demonstrate_constraint_validation(optimizer, flights, constraints)
        demonstrate_schedule_optimization(optimizer, flights, constraints)
        demonstrate_what_if_analysis(optimizer, flights)
        demonstrate_weather_impact(optimizer, flights, constraints)
        
        print("\n" + "="*60)
        print("✅ Schedule Optimization Demo Completed Successfully!")
        print("="*60)
        
        print("\nKey Features Demonstrated:")
        print("  ✓ Constraint validation and violation detection")
        print("  ✓ Multi-objective schedule optimization")
        print("  ✓ What-if analysis for proposed changes")
        print("  ✓ Weather impact on optimization")
        print("  ✓ Turnaround time constraint handling")
        print("  ✓ Curfew violation detection and resolution")
        print("  ✓ Fairness scoring for schedule changes")
        print("  ✓ CO₂ impact estimation")
        
        print("\nOptimization Algorithms Used:")
        print("  • CP-SAT (Constraint Programming)")
        print("  • Min-Cost Flow (Network Optimization)")
        print("  • Heuristic Fallback (When OR-Tools unavailable)")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())