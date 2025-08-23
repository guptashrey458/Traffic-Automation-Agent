#!/usr/bin/env python3
"""
Demo script for delay risk prediction models.

This script demonstrates the delay prediction functionality including:
- Departure delay risk prediction
- Arrival delay risk prediction  
- Turnaround time estimation
- Taxi time prediction
- Model evaluation metrics
"""

import sys
from datetime import datetime, date, time, timedelta
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.services.delay_prediction import (
    DelayRiskPredictor, OperationalContext, DelayRiskLevel, PredictionConfidence
)
from src.models.flight import Flight, Airport, FlightTime
from src.services.database import FlightDatabaseService


def create_sample_flights():
    """Create sample flights for demonstration."""
    flights = []
    
    # Morning rush hour flight - AI2509 BOM to DEL
    flights.append(Flight(
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
    ))
    
    # Peak hour IndiGo flight - 6E123 BOM to BLR
    flights.append(Flight(
        flight_id="6E123_001", 
        flight_number="6E123",
        airline_code="6E",
        origin=Airport.from_string("Mumbai (BOM)"),
        destination=Airport.from_string("Bangalore (BLR)"),
        aircraft_type="A320",
        aircraft_registration="VT-XYZ",
        flight_date=date(2024, 1, 15),
        departure=FlightTime(scheduled=time(8, 15)),
        arrival=FlightTime(scheduled=time(9, 45))
    ))
    
    # Off-peak Vistara flight - UK955 DEL to BOM
    flights.append(Flight(
        flight_id="UK955_001",
        flight_number="UK955", 
        airline_code="UK",
        origin=Airport.from_string("Delhi (DEL)"),
        destination=Airport.from_string("Mumbai (BOM)"),
        aircraft_type="A321",
        aircraft_registration="VT-DEF",
        flight_date=date(2024, 1, 15),
        departure=FlightTime(scheduled=time(14, 30)),
        arrival=FlightTime(scheduled=time(17, 0))
    ))
    
    return flights


def create_operational_contexts():
    """Create different operational contexts for testing."""
    contexts = {}
    
    # Normal conditions
    contexts["normal"] = OperationalContext(
        airport_code="BOM",
        analysis_datetime=datetime(2024, 1, 15, 8, 0),
        current_demand=20,
        runway_capacity=35,
        utilization_rate=0.57,
        weather_regime="calm",
        is_peak_hour=True,
        is_weekend=False,
        recent_avg_delay=8.0,
        cascade_risk_score=0.2
    )
    
    # High congestion
    contexts["congested"] = OperationalContext(
        airport_code="BOM",
        analysis_datetime=datetime(2024, 1, 15, 8, 0),
        current_demand=32,
        runway_capacity=35,
        utilization_rate=0.91,
        weather_regime="calm",
        is_peak_hour=True,
        is_weekend=False,
        recent_avg_delay=25.0,
        cascade_risk_score=0.7
    )
    
    # Severe weather
    contexts["severe_weather"] = OperationalContext(
        airport_code="BOM",
        analysis_datetime=datetime(2024, 1, 15, 8, 0),
        current_demand=18,
        runway_capacity=35,
        utilization_rate=0.51,
        weather_regime="severe",
        visibility_km=2.0,
        wind_speed_kts=35.0,
        is_peak_hour=True,
        is_weekend=False,
        recent_avg_delay=15.0,
        cascade_risk_score=0.4
    )
    
    # Off-peak quiet
    contexts["quiet"] = OperationalContext(
        airport_code="DEL",
        analysis_datetime=datetime(2024, 1, 15, 14, 30),
        current_demand=12,
        runway_capacity=40,
        utilization_rate=0.30,
        weather_regime="calm",
        is_peak_hour=False,
        is_weekend=False,
        recent_avg_delay=3.0,
        cascade_risk_score=0.1
    )
    
    return contexts


def print_prediction_results(flight, context, dep_prediction, arr_prediction):
    """Print formatted prediction results."""
    print(f"\n{'='*80}")
    print(f"FLIGHT: {flight.flight_number} ({flight.origin.code} → {flight.destination.code})")
    print(f"Aircraft: {flight.aircraft_type} | Airline: {flight.airline_code}")
    print(f"Scheduled: {flight.departure.scheduled} → {flight.arrival.scheduled}")
    print(f"Context: {context.weather_regime.title()} weather, {context.utilization_rate:.1%} utilization")
    print(f"{'='*80}")
    
    # Departure prediction
    print(f"\n🛫 DEPARTURE DELAY PREDICTION:")
    print(f"   Risk Level: {dep_prediction.risk_level.value.upper()}")
    print(f"   Delay Probability: {dep_prediction.delay_probability:.1%}")
    print(f"   Expected Delay: {dep_prediction.expected_delay_minutes:.1f} minutes")
    print(f"   Delay Range: {dep_prediction.delay_range_min:.1f} - {dep_prediction.delay_range_max:.1f} minutes")
    print(f"   Confidence: {dep_prediction.confidence.value.upper()} ({dep_prediction.confidence_score:.1%})")
    if dep_prediction.key_risk_factors:
        print(f"   Risk Factors: {', '.join(dep_prediction.key_risk_factors)}")
    
    # Arrival prediction
    print(f"\n🛬 ARRIVAL DELAY PREDICTION:")
    print(f"   Risk Level: {arr_prediction.risk_level.value.upper()}")
    print(f"   Delay Probability: {arr_prediction.delay_probability:.1%}")
    print(f"   Expected Delay: {arr_prediction.expected_delay_minutes:.1f} minutes")
    print(f"   Delay Range: {arr_prediction.delay_range_min:.1f} - {arr_prediction.delay_range_max:.1f} minutes")
    print(f"   Confidence: {arr_prediction.confidence.value.upper()} ({arr_prediction.confidence_score:.1%})")
    if arr_prediction.key_risk_factors:
        print(f"   Risk Factors: {', '.join(arr_prediction.key_risk_factors)}")


def demonstrate_turnaround_analysis(predictor):
    """Demonstrate turnaround time analysis."""
    print(f"\n{'='*80}")
    print("TURNAROUND TIME ANALYSIS")
    print(f"{'='*80}")
    
    aircraft_registrations = ["VT-ABC", "VT-XYZ320", "VT-777DEF"]
    airports = ["BOM", "DEL", "BLR"]
    
    for aircraft in aircraft_registrations:
        for airport in airports:
            estimate = predictor.estimate_turnaround_time(aircraft, airport)
            
            print(f"\n✈️  {aircraft} at {airport}:")
            print(f"   Aircraft Type: {estimate.aircraft_type}")
            print(f"   P50 Turnaround: {estimate.p50_turnaround_minutes:.0f} minutes")
            print(f"   P90 Turnaround: {estimate.p90_turnaround_minutes:.0f} minutes")
            print(f"   P95 Turnaround: {estimate.p95_turnaround_minutes:.0f} minutes")
            print(f"   Route Type: {estimate.typical_route_type}")
            
            # Test feasibility
            arrival_time = datetime(2024, 1, 15, 10, 0)
            tight_departure = arrival_time + timedelta(minutes=60)
            safe_departure = arrival_time + timedelta(minutes=int(estimate.p90_turnaround_minutes))
            
            print(f"   Feasible 60min turnaround: {'✅' if estimate.is_feasible_departure(arrival_time, tight_departure) else '❌'}")
            print(f"   Feasible P90 turnaround: {'✅' if estimate.is_feasible_departure(arrival_time, safe_departure) else '❌'}")


def demonstrate_taxi_time_analysis(predictor):
    """Demonstrate taxi time analysis."""
    print(f"\n{'='*80}")
    print("TAXI TIME ANALYSIS")
    print(f"{'='*80}")
    
    sample_flight = Flight(
        flight_id="TAXI_TEST",
        flight_number="AI123",
        origin=Airport.from_string("Mumbai (BOM)"),
        destination=Airport.from_string("Delhi (DEL)")
    )
    
    runways = ["09L", "09R", "27L", "27R"]
    
    for runway in runways:
        taxi_estimate = predictor.predict_taxi_time(sample_flight, runway)
        
        print(f"\n🛣️  Runway {runway} at {taxi_estimate.airport_code}:")
        print(f"   Operation: {taxi_estimate.operation_type.title()}")
        print(f"   Expected Taxi: {taxi_estimate.expected_taxi_minutes:.1f} minutes")
        print(f"   P90 Taxi: {taxi_estimate.p90_taxi_minutes:.1f} minutes")


def demonstrate_model_info(predictor):
    """Demonstrate model information and capabilities."""
    print(f"\n{'='*80}")
    print("MODEL INFORMATION")
    print(f"{'='*80}")
    
    info = predictor.get_model_info()
    
    print(f"\n📊 Model Status:")
    print(f"   Models Trained: {'✅' if info['model_trained'] else '❌'}")
    print(f"   Training Date: {info['training_date'] or 'Not trained'}")
    print(f"   Scikit-learn Available: {'✅' if info['sklearn_available'] else '❌'}")
    print(f"   Model Directory: {info['model_dir']}")
    
    print(f"\n🤖 Available Models:")
    for model_name, loaded in info['models_loaded'].items():
        status = '✅' if loaded else '❌'
        print(f"   {model_name}: {status}")
    
    print(f"\n🔧 Feature Encoders: {info['feature_encoders']}")
    
    if not info['model_trained']:
        print(f"\n💡 Note: Using fallback heuristic-based predictions")
        print(f"   For ML-based predictions, train models with historical data")


def main():
    """Main demonstration function."""
    print("🚀 DELAY RISK PREDICTION SYSTEM DEMO")
    print("=" * 80)
    
    # Initialize predictor
    print("\n📡 Initializing DelayRiskPredictor...")
    predictor = DelayRiskPredictor(model_dir="demo_models")
    
    # Show model information
    demonstrate_model_info(predictor)
    
    # Create sample data
    flights = create_sample_flights()
    contexts = create_operational_contexts()
    
    # Demonstrate predictions under different conditions
    context_names = ["normal", "congested", "severe_weather", "quiet"]
    
    for context_name in context_names:
        context = contexts[context_name]
        
        print(f"\n\n🌟 SCENARIO: {context_name.upper().replace('_', ' ')} CONDITIONS")
        print(f"Airport: {context.airport_code} | Utilization: {context.utilization_rate:.1%} | Weather: {context.weather_regime}")
        
        for flight in flights:
            # Skip if flight doesn't match context airport
            if context_name == "quiet" and flight.origin.code != "DEL":
                continue
            if context_name != "quiet" and flight.origin.code != "BOM":
                continue
                
            # Make predictions
            dep_prediction = predictor.predict_departure_delay(flight, context)
            arr_prediction = predictor.predict_arrival_delay(flight, context)
            
            # Print results
            print_prediction_results(flight, context, dep_prediction, arr_prediction)
    
    # Demonstrate turnaround analysis
    demonstrate_turnaround_analysis(predictor)
    
    # Demonstrate taxi time analysis
    demonstrate_taxi_time_analysis(predictor)
    
    print(f"\n{'='*80}")
    print("✅ DEMO COMPLETED SUCCESSFULLY!")
    print("💡 This system provides comprehensive delay risk assessment for flight scheduling optimization.")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()