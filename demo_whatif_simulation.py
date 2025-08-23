#!/usr/bin/env python3
"""
Demo script for What-If Simulation System

This script demonstrates the what-if simulation capabilities for analyzing
the impact of flight schedule changes on delays, capacity, and CO2 emissions.
"""

import os
import sys
from datetime import datetime, date, time, timedelta
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.services.whatif_simulator import (
    WhatIfSimulator, WhatIfScenario, CO2Factors, ChangeType
)
from src.services.schedule_optimizer import FlightChange
from src.services.analytics import WeatherRegime
from src.services.database import FlightDatabaseService
from src.models.flight import Flight, Airport, FlightTime


def print_header(title: str) -> None:
    """Print formatted header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_section(title: str) -> None:
    """Print formatted section header."""
    print(f"\n{'-'*40}")
    print(f" {title}")
    print(f"{'-'*40}")


def create_sample_flights() -> list:
    """Create sample flight data for demonstration."""
    flights = []
    
    # Flight 1: AI2509 BOM-DEL (currently delayed)
    flight1 = Flight()
    flight1.flight_id = "AI2509_001"
    flight1.flight_number = "AI2509"
    flight1.airline_code = "AI"
    flight1.aircraft_type = "A320"
    flight1.origin = Airport(code="BOM", name="Mumbai (BOM)", city="Mumbai")
    flight1.destination = Airport(code="DEL", name="Delhi (DEL)", city="Delhi")
    flight1.flight_date = date.today()
    flight1.departure.scheduled = time(8, 30)
    flight1.departure.actual = datetime.combine(date.today(), time(8, 45))  # 15 min delay
    flight1.dep_delay_min = 15
    flights.append(flight1)
    
    # Flight 2: 6E123 BOM-BLR (on time)
    flight2 = Flight()
    flight2.flight_id = "6E123_001"
    flight2.flight_number = "6E123"
    flight2.airline_code = "6E"
    flight2.aircraft_type = "A320"
    flight2.origin = Airport(code="BOM", name="Mumbai (BOM)", city="Mumbai")
    flight2.destination = Airport(code="BLR", name="Bangalore (BLR)", city="Bangalore")
    flight2.flight_date = date.today()
    flight2.departure.scheduled = time(9, 0)
    flight2.departure.actual = datetime.combine(date.today(), time(9, 5))  # 5 min delay
    flight2.dep_delay_min = 5
    flights.append(flight2)
    
    # Flight 3: UK955 BOM-CCU (significantly delayed)
    flight3 = Flight()
    flight3.flight_id = "UK955_001"
    flight3.flight_number = "UK955"
    flight3.airline_code = "UK"
    flight3.aircraft_type = "B737"
    flight3.origin = Airport(code="BOM", name="Mumbai (BOM)", city="Mumbai")
    flight3.destination = Airport(code="CCU", name="Kolkata (CCU)", city="Kolkata")
    flight3.flight_date = date.today()
    flight3.departure.scheduled = time(9, 15)
    flight3.departure.actual = datetime.combine(date.today(), time(9, 45))  # 30 min delay
    flight3.dep_delay_min = 30
    flights.append(flight3)
    
    return flights


def demo_co2_factors():
    """Demonstrate CO2 emission factors."""
    print_section("CO2 Emission Factors")
    
    factors = CO2Factors()
    
    print(f"Delay factor: {factors.delay_factor_kg_per_min} kg CO2 per minute of delay")
    print(f"Taxi factor: {factors.taxi_factor_kg_per_min} kg CO2 per minute of taxi")
    print(f"Routing optimization savings: {factors.routing_optimization_kg} kg CO2 per flight")
    
    print("\nAircraft-specific factors (kg CO2 per minute):")
    for aircraft, factor in factors.aircraft_factors.items():
        print(f"  {aircraft}: {factor} kg/min")
    
    # Test factor lookup
    print(f"\nA320 factor: {factors.get_aircraft_factor('A320')} kg/min")
    print(f"Unknown aircraft factor: {factors.get_aircraft_factor('UNKNOWN')} kg/min")


def demo_single_flight_analysis(simulator: WhatIfSimulator):
    """Demonstrate single flight change analysis."""
    print_section("Single Flight Change Analysis")
    
    # Scenario 1: Move AI2509 earlier by 10 minutes
    print("Scenario 1: Move AI2509 departure 10 minutes earlier")
    print("Current: 08:30 (delayed to 08:45)")
    print("Proposed: 08:20")
    
    impact_card = simulator.analyze_single_flight_change(
        flight_id="AI2509",
        time_change_minutes=-10,
        airport="BOM",
        analysis_date=date.today()
    )
    
    print(f"\nImpact Analysis:")
    print(f"  Delay Impact: {impact_card.delay_change_minutes:.1f} minutes ({impact_card.delay_direction})")
    print(f"  Capacity Impact: {impact_card.peak_overload_change} overload change ({impact_card.capacity_direction})")
    print(f"  CO2 Impact: {impact_card.co2_change_kg:.1f} kg ({impact_card.co2_direction})")
    print(f"  Fuel Savings: {impact_card.fuel_savings_liters:.1f} liters")
    print(f"  OTP Change: {impact_card.on_time_performance_change:.1f}%")
    print(f"  Fairness Score: {impact_card.fairness_score:.2f}")
    print(f"  Recommendation: {impact_card.recommendation}")
    print(f"  Confidence: {impact_card.confidence_level}")
    
    # Scenario 2: Move UK955 later by 20 minutes
    print("\n" + "="*50)
    print("Scenario 2: Move UK955 departure 20 minutes later")
    print("Current: 09:15 (delayed to 09:45)")
    print("Proposed: 09:35")
    
    impact_card2 = simulator.analyze_single_flight_change(
        flight_id="UK955",
        time_change_minutes=20,
        airport="BOM",
        analysis_date=date.today()
    )
    
    print(f"\nImpact Analysis:")
    print(f"  Delay Impact: {impact_card2.delay_change_minutes:.1f} minutes ({impact_card2.delay_direction})")
    print(f"  Capacity Impact: {impact_card2.peak_overload_change} overload change ({impact_card2.capacity_direction})")
    print(f"  CO2 Impact: {impact_card2.co2_change_kg:.1f} kg ({impact_card2.co2_direction})")
    print(f"  Recommendation: {impact_card2.recommendation}")
    print(f"  Confidence: {impact_card2.confidence_level}")


def demo_scenario_simulation(simulator: WhatIfSimulator):
    """Demonstrate complete scenario simulation."""
    print_section("Multi-Flight Scenario Simulation")
    
    # Create a scenario with multiple changes
    changes = [
        FlightChange(
            flight_id="AI2509_001",
            original_time=datetime.combine(date.today(), time(8, 30)),
            new_time=datetime.combine(date.today(), time(8, 20)),  # 10 min earlier
            change_type="departure"
        ),
        FlightChange(
            flight_id="UK955_001",
            original_time=datetime.combine(date.today(), time(9, 15)),
            new_time=datetime.combine(date.today(), time(9, 35)),  # 20 min later
            change_type="departure"
        )
    ]
    
    scenario = WhatIfScenario(
        scenario_id="peak_optimization",
        description="Optimize peak hour congestion by spreading flights",
        changes=changes,
        base_date=date.today(),
        airport="BOM",
        weather_regime=WeatherRegime.CALM
    )
    
    print(f"Scenario: {scenario.description}")
    print(f"Changes: {len(scenario.changes)} flights affected")
    print(f"Weather: {scenario.weather_regime.value}")
    
    # Simulate the scenario
    impact_card = simulator.simulate_scenario(scenario)
    
    print(f"\nOverall Impact:")
    print(f"  Total Delay Change: {impact_card.delay_change_minutes:.1f} minutes")
    print(f"  Affected Flights: {impact_card.affected_flights_count}")
    print(f"  Peak Overload Change: {impact_card.peak_overload_change}")
    print(f"  New Overload Windows: {impact_card.new_overload_windows}")
    print(f"  CO2 Impact: {impact_card.co2_change_kg:.1f} kg")
    print(f"  Fuel Impact: {impact_card.fuel_savings_liters:.1f} liters")
    print(f"  OTP Change: {impact_card.on_time_performance_change:.1f}%")
    print(f"  Fairness Score: {impact_card.fairness_score:.2f}")
    print(f"  Constraint Violations: {impact_card.constraint_violations}")
    
    print(f"\nRecommendation: {impact_card.recommendation}")
    print(f"Confidence Level: {impact_card.confidence_level}")


def demo_before_after_comparison(simulator: WhatIfSimulator):
    """Demonstrate detailed before/after comparison."""
    print_section("Before/After Detailed Comparison")
    
    # Create scenario
    changes = [
        FlightChange(
            flight_id="AI2509_001",
            original_time=datetime.combine(date.today(), time(8, 30)),
            new_time=datetime.combine(date.today(), time(8, 15)),  # 15 min earlier
            change_type="departure"
        )
    ]
    
    scenario = WhatIfScenario(
        scenario_id="early_departure_test",
        description="Test impact of earlier departure",
        changes=changes,
        base_date=date.today(),
        airport="BOM"
    )
    
    # Get detailed comparison
    comparison = simulator.compare_before_after(scenario)
    summary = comparison.get_summary()
    
    print("Before vs After Metrics:")
    print(f"\nDelay Metrics:")
    print(f"  Before - Total: {summary['delay_metrics']['before']['total_delay']:.1f} min, "
          f"Avg: {summary['delay_metrics']['before']['avg_delay']:.1f} min, "
          f"OTP: {summary['delay_metrics']['before']['otp']:.1f}%")
    print(f"  After  - Total: {summary['delay_metrics']['after']['total_delay']:.1f} min, "
          f"Avg: {summary['delay_metrics']['after']['avg_delay']:.1f} min, "
          f"OTP: {summary['delay_metrics']['after']['otp']:.1f}%")
    
    improvements = summary['delay_metrics']['improvements']
    print(f"\nImprovements:")
    print(f"  Total delay reduction: {improvements['total_delay_reduction']:.1f} minutes")
    print(f"  Average delay reduction: {improvements['avg_delay_reduction']:.1f} minutes")
    print(f"  OTP improvement: {improvements['otp_improvement']:.1f} percentage points")
    
    print(f"\nCapacity Metrics:")
    print(f"  Before overload windows: {summary['capacity_metrics']['before_overloads']}")
    print(f"  After overload windows: {summary['capacity_metrics']['after_overloads']}")
    print(f"  Peak utilization change: {summary['capacity_metrics']['peak_utilization_change']:.3f}")
    
    print(f"\nDetailed Changes:")
    print(f"  Improvements: {summary['detailed_changes']['improvements']}")
    print(f"  Degradations: {summary['detailed_changes']['degradations']}")
    print(f"  Capacity changes: {summary['detailed_changes']['capacity_changes']}")


def demo_weather_impact(simulator: WhatIfSimulator):
    """Demonstrate weather regime impact on analysis."""
    print_section("Weather Regime Impact Analysis")
    
    flight_id = "AI2509"
    time_change = -15  # 15 minutes earlier
    
    weather_regimes = [
        WeatherRegime.CALM,
        WeatherRegime.MEDIUM,
        WeatherRegime.STRONG,
        WeatherRegime.SEVERE
    ]
    
    print(f"Analyzing impact of moving {flight_id} by {time_change} minutes under different weather conditions:")
    
    for weather in weather_regimes:
        impact_card = simulator.analyze_single_flight_change(
            flight_id=flight_id,
            time_change_minutes=time_change,
            airport="BOM",
            analysis_date=date.today(),
            weather_regime=weather
        )
        
        print(f"\n{weather.value.upper()} Weather:")
        print(f"  Delay Impact: {impact_card.delay_change_minutes:.1f} min ({impact_card.delay_direction})")
        print(f"  Capacity Impact: {impact_card.peak_overload_change} ({impact_card.capacity_direction})")
        print(f"  CO2 Impact: {impact_card.co2_change_kg:.1f} kg ({impact_card.co2_direction})")
        print(f"  Recommendation: {impact_card.recommendation[:50]}...")


def demo_impact_card_export(simulator: WhatIfSimulator):
    """Demonstrate impact card export functionality."""
    print_section("Impact Card Export (JSON Format)")
    
    # Create and analyze a scenario
    impact_card = simulator.analyze_single_flight_change(
        flight_id="AI2509",
        time_change_minutes=-10,
        airport="BOM",
        analysis_date=date.today()
    )
    
    # Export to dictionary (JSON-ready format)
    impact_dict = impact_card.to_dict()
    
    print("Impact Card JSON Structure:")
    import json
    print(json.dumps(impact_dict, indent=2))


def demo_edge_cases(simulator: WhatIfSimulator):
    """Demonstrate edge cases and error handling."""
    print_section("Edge Cases and Error Handling")
    
    # Test 1: Non-existent flight
    print("Test 1: Non-existent flight")
    impact_card = simulator.analyze_single_flight_change(
        flight_id="NONEXISTENT123",
        time_change_minutes=10,
        airport="BOM",
        analysis_date=date.today()
    )
    print(f"  Result: {impact_card.recommendation}")
    print(f"  Confidence: {impact_card.confidence_level}")
    
    # Test 2: Zero time change
    print("\nTest 2: Zero time change")
    impact_card = simulator.analyze_single_flight_change(
        flight_id="AI2509",
        time_change_minutes=0,
        airport="BOM",
        analysis_date=date.today()
    )
    print(f"  Delay Impact: {impact_card.delay_change_minutes:.1f} minutes")
    print(f"  Recommendation: {impact_card.recommendation[:50]}...")
    
    # Test 3: Extreme time change
    print("\nTest 3: Extreme time change (+2 hours)")
    impact_card = simulator.analyze_single_flight_change(
        flight_id="AI2509",
        time_change_minutes=120,
        airport="BOM",
        analysis_date=date.today()
    )
    print(f"  Delay Impact: {impact_card.delay_change_minutes:.1f} minutes")
    print(f"  CO2 Impact: {impact_card.co2_change_kg:.1f} kg")
    print(f"  Recommendation: {impact_card.recommendation[:50]}...")


def main():
    """Main demo function."""
    print_header("What-If Simulation System Demo")
    
    print("This demo showcases the what-if simulation capabilities for analyzing")
    print("the impact of flight schedule changes on delays, capacity, and CO2 emissions.")
    
    # Initialize simulator
    print("\nInitializing What-If Simulator...")
    simulator = WhatIfSimulator()
    
    # Note: In a real implementation, we would use actual database data
    print("Note: This demo uses simulated data for demonstration purposes.")
    
    try:
        # Demo CO2 factors
        demo_co2_factors()
        
        # Demo single flight analysis
        demo_single_flight_analysis(simulator)
        
        # Demo scenario simulation
        demo_scenario_simulation(simulator)
        
        # Demo before/after comparison
        demo_before_after_comparison(simulator)
        
        # Demo weather impact
        demo_weather_impact(simulator)
        
        # Demo impact card export
        demo_impact_card_export(simulator)
        
        # Demo edge cases
        demo_edge_cases(simulator)
        
        print_header("Demo Complete")
        print("The what-if simulation system provides comprehensive analysis of")
        print("flight schedule changes with detailed impact metrics and recommendations.")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        print("This may be due to missing dependencies or database connectivity.")
        print("Please ensure all required services are properly configured.")


if __name__ == "__main__":
    main()