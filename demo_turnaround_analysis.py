#!/usr/bin/env python3
"""
Demo script for turnaround time analysis functionality.

This script demonstrates the key features of the turnaround time analysis system:
1. Turnaround time estimation for same-tail operations
2. P90 quantile estimation for turnaround times
3. Taxi time estimation functions (EXOT/EXIN)
4. Validation logic for feasible departure slots
"""

import os
import sys
from datetime import datetime, date, time, timedelta
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.services.turnaround_analysis import TurnaroundAnalysisService
from src.services.database import FlightDatabaseService
from src.models.flight import Flight, Airport, FlightTime


def create_sample_flights():
    """Create sample flights for demonstration."""
    flights = []
    
    # Flight 1: AI2509 BOM-DEL
    flight1 = Flight()
    flight1.flight_id = "demo-ai2509-001"
    flight1.flight_number = "AI2509"
    flight1.aircraft_registration = "VT-EXA"
    flight1.aircraft_type = "A320"
    flight1.origin = Airport(code="BOM", name="Mumbai", city="Mumbai")
    flight1.destination = Airport(code="DEL", name="Delhi", city="Delhi")
    flight1.departure = FlightTime()
    flight1.departure.scheduled = time(10, 30)
    flight1.flight_date = date(2024, 1, 15)
    flights.append(flight1)
    
    # Flight 2: 6E123 DEL-BLR (same aircraft, turnaround at DEL)
    flight2 = Flight()
    flight2.flight_id = "demo-6e123-001"
    flight2.flight_number = "6E123"
    flight2.aircraft_registration = "VT-EXA"  # Same aircraft
    flight2.aircraft_type = "A320"
    flight2.origin = Airport(code="DEL", name="Delhi", city="Delhi")
    flight2.destination = Airport(code="BLR", name="Bangalore", city="Bangalore")
    flight2.departure = FlightTime()
    flight2.departure.scheduled = time(13, 15)  # 2h 45m after AI2509 arrival
    flight2.flight_date = date(2024, 1, 15)
    flights.append(flight2)
    
    # Flight 3: SG456 BOM-HYD (different aircraft)
    flight3 = Flight()
    flight3.flight_id = "demo-sg456-001"
    flight3.flight_number = "SG456"
    flight3.aircraft_registration = "VT-EXB"
    flight3.aircraft_type = "B737"
    flight3.origin = Airport(code="BOM", name="Mumbai", city="Mumbai")
    flight3.destination = Airport(code="HYD", name="Hyderabad", city="Hyderabad")
    flight3.departure = FlightTime()
    flight3.departure.scheduled = time(14, 45)
    flight3.flight_date = date(2024, 1, 15)
    flights.append(flight3)
    
    return flights


def demo_turnaround_estimation():
    """Demonstrate turnaround time estimation."""
    print("=" * 60)
    print("TURNAROUND TIME ESTIMATION DEMO")
    print("=" * 60)
    
    service = TurnaroundAnalysisService()
    
    # Test different aircraft types and airports
    test_cases = [
        ("VT-EXA320", "BOM", "A320 at Mumbai"),
        ("VT-EXB737", "DEL", "B737 at Delhi"),
        ("VT-EXC777", "BLR", "B777 at Bangalore"),
        ("VT-UNKNOWN", "CCU", "Unknown aircraft at Kolkata")
    ]
    
    for aircraft_reg, airport, description in test_cases:
        print(f"\n{description}:")
        print("-" * 40)
        
        estimate = service.estimate_turnaround_time(aircraft_reg, airport)
        
        print(f"Aircraft Registration: {estimate.aircraft_registration}")
        print(f"Airport: {estimate.airport_code}")
        print(f"Aircraft Type: {estimate.aircraft_type}")
        print(f"Route Type: {estimate.typical_route_type}")
        print(f"Sample Size: {estimate.sample_size}")
        print(f"Confidence: {estimate.confidence_level}")
        print(f"Turnaround Times:")
        print(f"  P50 (Median): {estimate.p50_turnaround_minutes:.1f} minutes")
        print(f"  P90 (Planning): {estimate.p90_turnaround_minutes:.1f} minutes")
        print(f"  P95 (Conservative): {estimate.p95_turnaround_minutes:.1f} minutes")
        print(f"  Range: {estimate.min_observed:.1f} - {estimate.max_observed:.1f} minutes")
        
        # Test feasibility check
        arrival_time = datetime(2024, 1, 15, 10, 0)
        departure_time_tight = arrival_time + timedelta(minutes=60)
        departure_time_safe = arrival_time + timedelta(minutes=90)
        
        is_feasible_tight, required_tight = estimate.is_feasible_departure(arrival_time, departure_time_tight)
        is_feasible_safe, required_safe = estimate.is_feasible_departure(arrival_time, departure_time_safe)
        
        print(f"Feasibility Check:")
        print(f"  60-min turnaround: {'✓ Feasible' if is_feasible_tight else '✗ Too tight'}")
        print(f"  90-min turnaround: {'✓ Feasible' if is_feasible_safe else '✗ Too tight'}")


def demo_taxi_time_estimation():
    """Demonstrate taxi time estimation."""
    print("\n" + "=" * 60)
    print("TAXI TIME ESTIMATION DEMO (EXOT/EXIN)")
    print("=" * 60)
    
    service = TurnaroundAnalysisService()
    flights = create_sample_flights()
    
    runways = ["09R", "27L", "14", "32"]
    
    for flight in flights[:2]:  # Test first two flights
        print(f"\n{flight.flight_number} ({flight.origin.code}-{flight.destination.code}):")
        print("-" * 50)
        
        for runway in runways[:2]:  # Test 2 runways per flight
            taxi_estimate = service.predict_taxi_time(flight, runway)
            
            print(f"  Runway {runway}:")
            print(f"    Operation: {taxi_estimate.operation_type}")
            print(f"    Expected: {taxi_estimate.expected_taxi_minutes:.1f} minutes")
            print(f"    P90: {taxi_estimate.p90_taxi_minutes:.1f} minutes")
            print(f"    P95: {taxi_estimate.p95_taxi_minutes:.1f} minutes")
            print(f"    Confidence: {taxi_estimate.confidence_level}")


def demo_departure_slot_validation():
    """Demonstrate departure slot validation."""
    print("\n" + "=" * 60)
    print("DEPARTURE SLOT VALIDATION DEMO")
    print("=" * 60)
    
    service = TurnaroundAnalysisService()
    
    # Create arrival flight (AI2509 arriving at DEL)
    arrival_flight = Flight()
    arrival_flight.flight_id = "arrival-ai2509"
    arrival_flight.flight_number = "AI2509"
    arrival_flight.aircraft_registration = "VT-EXA"
    arrival_flight.aircraft_type = "A320"
    arrival_flight.destination = Airport(code="DEL", name="Delhi", city="Delhi")
    arrival_flight.arrival = FlightTime()
    arrival_flight.arrival.actual = datetime(2024, 1, 15, 12, 30)  # Arrived at 12:30
    
    # Create departure flight (6E123 departing from DEL)
    departure_flight = Flight()
    departure_flight.flight_id = "departure-6e123"
    departure_flight.flight_number = "6E123"
    departure_flight.aircraft_registration = "VT-EXA"  # Same aircraft
    departure_flight.aircraft_type = "A320"
    departure_flight.origin = Airport(code="DEL", name="Delhi", city="Delhi")
    departure_flight.destination = Airport(code="BLR", name="Bangalore", city="Bangalore")
    
    # Test different proposed departure times
    test_scenarios = [
        (datetime(2024, 1, 15, 13, 0), "30-minute turnaround (very tight)"),
        (datetime(2024, 1, 15, 13, 30), "60-minute turnaround (tight)"),
        (datetime(2024, 1, 15, 14, 15), "105-minute turnaround (comfortable)"),
        (datetime(2024, 1, 15, 15, 0), "150-minute turnaround (very safe)")
    ]
    
    for proposed_time, scenario_desc in test_scenarios:
        print(f"\nScenario: {scenario_desc}")
        print("-" * 50)
        
        validation = service.validate_departure_slot(
            departure_flight, 
            proposed_time, 
            arrival_flight
        )
        
        print(f"Flight: {validation.flight_id}")
        print(f"Aircraft: {validation.aircraft_registration}")
        print(f"Airport: {validation.airport_code}")
        print(f"Arrival Time: {validation.arrival_time.strftime('%H:%M')}")
        print(f"Proposed Departure: {validation.proposed_departure_time.strftime('%H:%M')}")
        print(f"Required Turnaround: {validation.required_turnaround_minutes:.0f} minutes")
        print(f"Risk Level: {validation.risk_level.upper()}")
        
        feasibility_p90 = "✓" if validation.is_feasible_p90 else "✗"
        feasibility_p95 = "✓" if validation.is_feasible_p95 else "✗"
        print(f"Feasibility P90: {feasibility_p90}")
        print(f"Feasibility P95: {feasibility_p95}")
        
        if validation.risk_factors:
            print(f"Risk Factors: {', '.join(validation.risk_factors)}")
        
        if validation.recommended_departure_time:
            print(f"Recommended Time: {validation.recommended_departure_time.strftime('%H:%M')}")
            print(f"Additional Buffer: {validation.buffer_minutes:.0f} minutes")
        else:
            print("Recommendation: Current time is acceptable")


def demo_api_responses():
    """Demonstrate API response formats."""
    print("\n" + "=" * 60)
    print("API RESPONSE FORMAT DEMO")
    print("=" * 60)
    
    service = TurnaroundAnalysisService()
    
    # Get turnaround estimate
    estimate = service.estimate_turnaround_time("VT-EXA", "BOM")
    
    print("\nTurnaround Estimate API Response:")
    print("-" * 40)
    import json
    print(json.dumps(estimate.to_dict(), indent=2))
    
    # Get taxi time estimate
    flight = create_sample_flights()[0]
    taxi_estimate = service.predict_taxi_time(flight, "09R")
    
    print("\nTaxi Time Estimate API Response:")
    print("-" * 40)
    print(json.dumps(taxi_estimate.to_dict(), indent=2))


def main():
    """Run all demonstrations."""
    print("AGENTIC FLIGHT SCHEDULER - TURNAROUND TIME ANALYSIS DEMO")
    print("=" * 80)
    print("This demo showcases the turnaround time analysis capabilities:")
    print("• P90 quantile estimation for turnaround times")
    print("• Taxi time estimation (EXOT/EXIN)")
    print("• Departure slot feasibility validation")
    print("• Risk assessment for tight turnarounds")
    
    try:
        demo_turnaround_estimation()
        demo_taxi_time_estimation()
        demo_departure_slot_validation()
        demo_api_responses()
        
        print("\n" + "=" * 80)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nKey Features Demonstrated:")
        print("✓ Turnaround time estimation with P50/P90/P95 percentiles")
        print("✓ Aircraft-specific and airport-specific defaults")
        print("✓ Taxi time prediction for different runways")
        print("✓ Departure slot validation with risk assessment")
        print("✓ Feasibility checks for same-tail operations")
        print("✓ API-ready response formats")
        
        print("\nNext Steps:")
        print("• Integrate with historical flight data for better estimates")
        print("• Add real-time taxi time tracking")
        print("• Implement weather impact on turnaround times")
        print("• Connect to schedule optimization engine")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())