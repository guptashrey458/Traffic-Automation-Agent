#!/usr/bin/env python3
"""
Enhanced What-If Simulation Demo with Real Flight Data

This script demonstrates the what-if simulation capabilities using actual
flight data from the Excel file, showing realistic impact analysis.
"""

import os
import sys
from datetime import datetime, date, time, timedelta
from pathlib import Path
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.services.whatif_simulator import (
    WhatIfSimulator, WhatIfScenario, CO2Factors, ChangeType
)
from src.services.schedule_optimizer import FlightChange
from src.services.analytics import WeatherRegime
from src.services.data_ingestion import DataIngestionService
from src.models.flight import Flight, Airport, FlightTime
from src.models.excel_parser import ExcelFlightParser


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


def load_real_flight_data() -> list:
    """Load real flight data from Excel file."""
    print("Loading real flight data from Excel file...")
    
    excel_file = "429e6e3f-281d-4e4c-b00a-92fb020cb2fcFlight_Data.xlsx"
    
    if not os.path.exists(excel_file):
        print(f"Excel file not found: {excel_file}")
        return []
    
    try:
        # Use the ingestion service to load data
        ingestion_service = DataIngestionService()
        result = ingestion_service.ingest_excel_files([excel_file])
        
        print(f"✅ Loaded {result.valid_flights} valid flights from {result.total_flights} total")
        print(f"   Processing time: {result.processing_time_seconds:.2f} seconds")
        
        if result.errors:
            print(f"   Warnings: {len(result.errors)} parsing issues")
        
        return result.flights
        
    except Exception as e:
        print(f"❌ Error loading Excel data: {e}")
        return []


def create_realistic_simulator_with_data(flights: list) -> WhatIfSimulator:
    """Create a what-if simulator with real flight data."""
    
    # Create a mock database service that returns our real data
    class MockDatabaseWithRealData:
        def __init__(self, flights):
            self.flights = flights
        
        def query_flights_by_date_range(self, start_date, end_date, airport_code=None):
            # Filter flights by date and airport
            filtered_flights = []
            
            for flight in self.flights:
                # Check if flight matches date range
                if flight.flight_date and start_date <= flight.flight_date <= end_date:
                    # Check airport filter
                    if not airport_code or (
                        (flight.origin and flight.origin.code == airport_code) or
                        (flight.destination and flight.destination.code == airport_code)
                    ):
                        # Convert flight to dictionary format expected by simulator
                        flight_dict = {
                            'flight_id': flight.flight_id,
                            'flight_number': flight.flight_number,
                            'airline_code': flight.airline_code,
                            'origin_code': flight.origin.code if flight.origin else None,
                            'origin_name': flight.origin.name if flight.origin else None,
                            'destination_code': flight.destination.code if flight.destination else None,
                            'destination_name': flight.destination.name if flight.destination else None,
                            'std_utc': flight.departure.scheduled.isoformat() if flight.departure.scheduled else None,
                            'atd_utc': flight.departure.actual.isoformat() if flight.departure.actual else None,
                            'dep_delay_min': flight.dep_delay_min,
                            'arr_delay_min': flight.arr_delay_min,
                            'aircraft_type': flight.aircraft_type
                        }
                        filtered_flights.append(flight_dict)
            
            # Return mock result
            class MockResult:
                def __init__(self, data):
                    self.data = data
                    self.row_count = len(data)
                    self.execution_time_ms = 50
            
            return MockResult(filtered_flights)
    
    # Create simulator with real data
    mock_db = MockDatabaseWithRealData(flights)
    simulator = WhatIfSimulator(db_service=mock_db)
    
    return simulator


def analyze_real_flight_impacts(simulator: WhatIfSimulator, flights: list):
    """Analyze real flight schedule changes with actual data."""
    print_section("Real Flight Impact Analysis")
    
    if not flights:
        print("No flight data available for analysis")
        return
    
    # Find some interesting flights to analyze
    delayed_flights = [f for f in flights if f.dep_delay_min and f.dep_delay_min > 15]
    bom_flights = [f for f in flights if f.origin and f.origin.code == "BOM"]
    
    print(f"Dataset Overview:")
    print(f"  Total flights: {len(flights)}")
    print(f"  Delayed flights (>15 min): {len(delayed_flights)}")
    print(f"  BOM departures: {len(bom_flights)}")
    
    if delayed_flights:
        # Analyze the most delayed flight
        most_delayed = max(delayed_flights, key=lambda f: f.dep_delay_min or 0)
        
        print(f"\nAnalyzing most delayed flight:")
        print(f"  Flight: {most_delayed.flight_number}")
        print(f"  Route: {most_delayed.origin.code if most_delayed.origin else 'UNK'} → {most_delayed.destination.code if most_delayed.destination else 'UNK'}")
        print(f"  Current delay: {most_delayed.dep_delay_min} minutes")
        print(f"  Scheduled: {most_delayed.departure.scheduled}")
        
        # Scenario 1: Move this flight 20 minutes earlier
        print(f"\n🔍 Scenario: Move {most_delayed.flight_number} 20 minutes earlier")
        
        airport = most_delayed.origin.code if most_delayed.origin else "BOM"
        analysis_date = most_delayed.flight_date or date.today()
        
        impact_card = simulator.analyze_single_flight_change(
            flight_id=most_delayed.flight_number,
            time_change_minutes=-20,
            airport=airport,
            analysis_date=analysis_date
        )
        
        print_impact_analysis(impact_card)
    
    # Analyze a BOM flight if available
    if bom_flights:
        bom_flight = bom_flights[0]
        print(f"\n🔍 BOM Departure Analysis:")
        print(f"  Flight: {bom_flight.flight_number}")
        print(f"  Destination: {bom_flight.destination.code if bom_flight.destination else 'UNK'}")
        print(f"  Delay: {bom_flight.dep_delay_min or 0} minutes")
        
        impact_card = simulator.analyze_single_flight_change(
            flight_id=bom_flight.flight_number,
            time_change_minutes=-15,
            airport="BOM",
            analysis_date=bom_flight.flight_date or date.today()
        )
        
        print_impact_analysis(impact_card)


def print_impact_analysis(impact_card):
    """Print formatted impact analysis."""
    print(f"\n📊 Impact Analysis:")
    print(f"  Delay Impact: {impact_card.delay_change_minutes:.1f} minutes ({impact_card.delay_direction})")
    print(f"  Capacity Impact: {impact_card.peak_overload_change} overload change ({impact_card.capacity_direction})")
    print(f"  CO2 Impact: {impact_card.co2_change_kg:.1f} kg ({impact_card.co2_direction})")
    print(f"  Fuel Impact: {impact_card.fuel_savings_liters:.1f} liters")
    print(f"  OTP Change: {impact_card.on_time_performance_change:.1f}%")
    print(f"  Fairness Score: {impact_card.fairness_score:.2f}")
    print(f"  Affected Flights: {impact_card.affected_flights_count}")
    print(f"  Recommendation: {impact_card.recommendation}")
    print(f"  Confidence: {impact_card.confidence_level}")


def create_multi_flight_scenario(simulator: WhatIfSimulator, flights: list):
    """Create and analyze a multi-flight optimization scenario."""
    print_section("Multi-Flight Optimization Scenario")
    
    if len(flights) < 3:
        print("Not enough flights for multi-flight scenario")
        return
    
    # Find flights that could benefit from rescheduling
    delayed_flights = [f for f in flights if f.dep_delay_min and f.dep_delay_min > 10][:3]
    
    if not delayed_flights:
        delayed_flights = flights[:3]  # Use first 3 flights as fallback
    
    print(f"Creating scenario with {len(delayed_flights)} flights:")
    
    changes = []
    for i, flight in enumerate(delayed_flights):
        # Stagger the changes: -15, -10, -5 minutes
        time_change = -15 + (i * 5)
        
        original_time = datetime.combine(
            flight.flight_date or date.today(),
            flight.departure.scheduled or time(8, 0)
        )
        new_time = original_time + timedelta(minutes=time_change)
        
        change = FlightChange(
            flight_id=flight.flight_id,
            original_time=original_time,
            new_time=new_time,
            change_type="departure"
        )
        changes.append(change)
        
        print(f"  {flight.flight_number}: {time_change:+d} minutes (delay: {flight.dep_delay_min or 0}min)")
    
    # Create and simulate scenario
    scenario = WhatIfScenario(
        scenario_id="multi_flight_optimization",
        description="Optimize multiple delayed flights",
        changes=changes,
        base_date=delayed_flights[0].flight_date or date.today(),
        airport=delayed_flights[0].origin.code if delayed_flights[0].origin else "BOM"
    )
    
    impact_card = simulator.simulate_scenario(scenario)
    
    print(f"\n📊 Multi-Flight Impact:")
    print(f"  Total Delay Change: {impact_card.delay_change_minutes:.1f} minutes")
    print(f"  Peak Overload Change: {impact_card.peak_overload_change}")
    print(f"  CO2 Impact: {impact_card.co2_change_kg:.1f} kg")
    print(f"  Fuel Impact: {impact_card.fuel_savings_liters:.1f} liters")
    print(f"  OTP Change: {impact_card.on_time_performance_change:.1f}%")
    print(f"  Fairness Score: {impact_card.fairness_score:.2f}")
    print(f"  Recommendation: {impact_card.recommendation}")


def analyze_airport_congestion(flights: list):
    """Analyze airport congestion patterns from real data."""
    print_section("Airport Congestion Analysis")
    
    # Group flights by airport and time
    airport_stats = {}
    
    for flight in flights:
        if not flight.origin or not flight.departure.scheduled:
            continue
        
        airport = flight.origin.code
        hour = flight.departure.scheduled.hour
        
        if airport not in airport_stats:
            airport_stats[airport] = {'total': 0, 'delayed': 0, 'hourly': {}}
        
        airport_stats[airport]['total'] += 1
        
        if flight.dep_delay_min and flight.dep_delay_min > 15:
            airport_stats[airport]['delayed'] += 1
        
        if hour not in airport_stats[airport]['hourly']:
            airport_stats[airport]['hourly'][hour] = 0
        airport_stats[airport]['hourly'][hour] += 1
    
    # Print congestion analysis
    for airport, stats in sorted(airport_stats.items(), key=lambda x: x[1]['total'], reverse=True)[:5]:
        delay_rate = (stats['delayed'] / stats['total']) * 100 if stats['total'] > 0 else 0
        peak_hour = max(stats['hourly'].items(), key=lambda x: x[1]) if stats['hourly'] else (0, 0)
        
        print(f"\n🏢 {airport} Airport:")
        print(f"  Total departures: {stats['total']}")
        print(f"  Delayed flights: {stats['delayed']} ({delay_rate:.1f}%)")
        print(f"  Peak hour: {peak_hour[0]:02d}:00 ({peak_hour[1]} flights)")
        
        # Suggest optimization
        if delay_rate > 20:
            print(f"  🚨 High delay rate - consider peak spreading")
        elif peak_hour[1] > 10:
            print(f"  ⚠️  Peak congestion - monitor capacity")
        else:
            print(f"  ✅ Normal operations")


def demonstrate_weather_impact(simulator: WhatIfSimulator, flights: list):
    """Demonstrate weather impact on what-if analysis."""
    print_section("Weather Impact Analysis")
    
    if not flights:
        print("No flight data for weather analysis")
        return
    
    # Pick a representative flight
    test_flight = flights[0]
    airport = test_flight.origin.code if test_flight.origin else "BOM"
    analysis_date = test_flight.flight_date or date.today()
    
    print(f"Analyzing weather impact for {test_flight.flight_number} at {airport}")
    print(f"Scenario: Move flight 15 minutes earlier under different weather conditions")
    
    weather_conditions = [
        (WeatherRegime.CALM, "Clear skies, light winds"),
        (WeatherRegime.MEDIUM, "Moderate winds, some restrictions"),
        (WeatherRegime.STRONG, "Strong winds, reduced capacity"),
        (WeatherRegime.SEVERE, "Severe weather, minimal operations")
    ]
    
    for weather, description in weather_conditions:
        impact_card = simulator.analyze_single_flight_change(
            flight_id=test_flight.flight_number,
            time_change_minutes=-15,
            airport=airport,
            analysis_date=analysis_date,
            weather_regime=weather
        )
        
        print(f"\n🌤️  {weather.value.upper()} Weather ({description}):")
        print(f"   Delay Impact: {impact_card.delay_change_minutes:.1f} min")
        print(f"   Capacity Impact: {impact_card.peak_overload_change}")
        print(f"   CO2 Impact: {impact_card.co2_change_kg:.1f} kg")
        print(f"   Confidence: {impact_card.confidence_level}")


def main():
    """Main demo function."""
    print_header("Enhanced What-If Simulation with Real Flight Data")
    
    print("This demo uses actual flight data from your Excel file to show")
    print("realistic impact analysis of schedule changes.")
    
    # Load real flight data
    flights = load_real_flight_data()
    
    if not flights:
        print("\n❌ No flight data loaded. Please ensure the Excel file is available.")
        return
    
    # Create enhanced simulator with real data
    print("\nInitializing What-If Simulator with real flight data...")
    simulator = create_realistic_simulator_with_data(flights)
    
    try:
        # Analyze airport congestion patterns
        analyze_airport_congestion(flights)
        
        # Analyze real flight impacts
        analyze_real_flight_impacts(simulator, flights)
        
        # Create multi-flight scenario
        create_multi_flight_scenario(simulator, flights)
        
        # Demonstrate weather impact
        demonstrate_weather_impact(simulator, flights)
        
        print_header("Real Data Demo Complete")
        print("✅ The what-if simulation system now uses actual flight data")
        print("✅ Impact analysis reflects real delay patterns and congestion")
        print("✅ CO2 calculations use actual aircraft types and delays")
        print("✅ Recommendations are based on real operational constraints")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        print("This may be due to data format issues or missing dependencies.")


if __name__ == "__main__":
    main()