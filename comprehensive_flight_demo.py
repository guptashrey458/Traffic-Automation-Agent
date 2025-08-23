#!/usr/bin/env python3
"""
Comprehensive demonstration of flight data access and analysis capabilities
"""

from datetime import datetime, date
from collections import defaultdict
import json

from src.services.data_ingestion import DataIngestionService
from src.services.database import FlightDatabaseService, DatabaseConfig


def main():
    print("🛫 COMPREHENSIVE FLIGHT DATA DEMONSTRATION")
    print("=" * 60)
    
    # Load the Excel data
    excel_file = "429e6e3f-281d-4e4c-b00a-92fb020cb2fcFlight_Data.xlsx"
    ingestion_service = DataIngestionService()
    
    print("📊 Loading Excel flight data...")
    result = ingestion_service.ingest_excel_files([excel_file])
    
    print(f"✅ Successfully loaded {result.total_flights} flights!")
    print(f"   Valid flights: {result.valid_flights}")
    print(f"   Processing time: {result.processing_time_seconds:.2f}s")
    
    flights = result.flights
    
    # === BASIC FLIGHT INFORMATION ===
    print(f"\n📋 BASIC FLIGHT INFORMATION")
    print("-" * 40)
    
    print(f"Total flights in dataset: {len(flights)}")
    
    # Date range analysis
    dates = [f.flight_date for f in flights if f.flight_date]
    if dates:
        print(f"Date range: {min(dates)} to {max(dates)}")
        print(f"Total days covered: {(max(dates) - min(dates)).days + 1}")
    
    # Sample flight details
    sample = flights[0]
    print(f"\n✈️  SAMPLE FLIGHT DETAILS:")
    print(f"   Flight: {sample.flight_number} ({sample.airline_code})")
    print(f"   Route: {sample.origin.name} → {sample.destination.name}")
    print(f"   Date: {sample.flight_date}")
    print(f"   Aircraft: {sample.aircraft_type} ({sample.aircraft_registration})")
    print(f"   Scheduled: {sample.departure.scheduled} → {sample.arrival.scheduled}")
    print(f"   Actual: {sample.departure.actual} → {sample.arrival.actual}")
    print(f"   Delays: Dep {sample.dep_delay_min}min, Arr {sample.arr_delay_min}min")
    print(f"   Status: {sample.status.value}")
    
    # === AIRLINE ANALYSIS ===
    print(f"\n🏢 AIRLINE ANALYSIS")
    print("-" * 25)
    
    airline_stats = defaultdict(lambda: {'flights': 0, 'delays': [], 'routes': set()})
    
    for flight in flights:
        airline = flight.airline_code
        airline_stats[airline]['flights'] += 1
        airline_stats[airline]['routes'].add(flight.get_route_key())
        if flight.dep_delay_min is not None:
            airline_stats[airline]['delays'].append(flight.dep_delay_min)
    
    print("Top airlines by flight count:")
    sorted_airlines = sorted(airline_stats.items(), key=lambda x: x[1]['flights'], reverse=True)
    
    for airline, stats in sorted_airlines[:10]:
        avg_delay = sum(stats['delays']) / len(stats['delays']) if stats['delays'] else 0
        on_time_rate = len([d for d in stats['delays'] if d <= 15]) / len(stats['delays']) * 100 if stats['delays'] else 0
        
        print(f"   {airline}: {stats['flights']} flights, {len(stats['routes'])} routes")
        print(f"        Avg delay: {avg_delay:.1f}min, On-time: {on_time_rate:.1f}%")
    
    # === ROUTE ANALYSIS ===
    print(f"\n🗺️  ROUTE ANALYSIS")
    print("-" * 20)
    
    route_stats = defaultdict(lambda: {'flights': 0, 'delays': [], 'airlines': set()})
    
    for flight in flights:
        route = flight.get_route_key()
        route_stats[route]['flights'] += 1
        route_stats[route]['airlines'].add(flight.airline_code)
        if flight.dep_delay_min is not None:
            route_stats[route]['delays'].append(flight.dep_delay_min)
    
    print("Busiest routes:")
    sorted_routes = sorted(route_stats.items(), key=lambda x: x[1]['flights'], reverse=True)
    
    for route, stats in sorted_routes[:10]:
        avg_delay = sum(stats['delays']) / len(stats['delays']) if stats['delays'] else 0
        print(f"   {route}: {stats['flights']} flights, {len(stats['airlines'])} airlines")
        print(f"           Avg delay: {avg_delay:.1f}min")
    
    # === DELAY ANALYSIS ===
    print(f"\n⏰ DELAY ANALYSIS")
    print("-" * 20)
    
    dep_delays = [f.dep_delay_min for f in flights if f.dep_delay_min is not None]
    arr_delays = [f.arr_delay_min for f in flights if f.arr_delay_min is not None]
    
    if dep_delays:
        print(f"Departure delays:")
        print(f"   Total flights with delay data: {len(dep_delays)}")
        print(f"   Average delay: {sum(dep_delays) / len(dep_delays):.1f} minutes")
        print(f"   Maximum delay: {max(dep_delays)} minutes")
        print(f"   Minimum delay: {min(dep_delays)} minutes")
        
        # Delay categories
        on_time = len([d for d in dep_delays if d <= 15])
        minor_delay = len([d for d in dep_delays if 15 < d <= 60])
        major_delay = len([d for d in dep_delays if d > 60])
        
        print(f"   On-time (≤15min): {on_time} ({on_time/len(dep_delays)*100:.1f}%)")
        print(f"   Minor delay (16-60min): {minor_delay} ({minor_delay/len(dep_delays)*100:.1f}%)")
        print(f"   Major delay (>60min): {major_delay} ({major_delay/len(dep_delays)*100:.1f}%)")
    
    # === AIRCRAFT ANALYSIS ===
    print(f"\n✈️  AIRCRAFT ANALYSIS")
    print("-" * 25)
    
    aircraft_stats = defaultdict(lambda: {'flights': 0, 'airlines': set(), 'delays': []})
    
    for flight in flights:
        aircraft = flight.aircraft_type
        aircraft_stats[aircraft]['flights'] += 1
        aircraft_stats[aircraft]['airlines'].add(flight.airline_code)
        if flight.dep_delay_min is not None:
            aircraft_stats[aircraft]['delays'].append(flight.dep_delay_min)
    
    print("Aircraft types in use:")
    sorted_aircraft = sorted(aircraft_stats.items(), key=lambda x: x[1]['flights'], reverse=True)
    
    for aircraft, stats in sorted_aircraft[:10]:
        avg_delay = sum(stats['delays']) / len(stats['delays']) if stats['delays'] else 0
        print(f"   {aircraft}: {stats['flights']} flights, {len(stats['airlines'])} airlines")
        print(f"           Avg delay: {avg_delay:.1f}min")
    
    # === TIME-BASED ANALYSIS ===
    print(f"\n🕐 TIME-BASED ANALYSIS")
    print("-" * 30)
    
    # Hourly distribution
    hourly_stats = defaultdict(lambda: {'flights': 0, 'delays': []})
    
    for flight in flights:
        if flight.departure.scheduled:
            hour = flight.departure.scheduled.hour
            hourly_stats[hour]['flights'] += 1
            if flight.dep_delay_min is not None:
                hourly_stats[hour]['delays'].append(flight.dep_delay_min)
    
    print("Peak hours (by flight count):")
    sorted_hours = sorted(hourly_stats.items(), key=lambda x: x[1]['flights'], reverse=True)
    
    for hour, stats in sorted_hours[:8]:
        avg_delay = sum(stats['delays']) / len(stats['delays']) if stats['delays'] else 0
        print(f"   {hour:02d}:00-{hour:02d}:59: {stats['flights']} flights, avg delay {avg_delay:.1f}min")
    
    # === AIRPORT ANALYSIS ===
    print(f"\n🏢 AIRPORT ANALYSIS")
    print("-" * 25)
    
    airport_stats = defaultdict(lambda: {'departures': 0, 'arrivals': 0, 'delays': []})
    
    for flight in flights:
        if flight.origin:
            airport_stats[flight.origin.code]['departures'] += 1
            if flight.dep_delay_min is not None:
                airport_stats[flight.origin.code]['delays'].append(flight.dep_delay_min)
        
        if flight.destination:
            airport_stats[flight.destination.code]['arrivals'] += 1
    
    print("Busiest airports:")
    sorted_airports = sorted(airport_stats.items(), 
                           key=lambda x: x[1]['departures'] + x[1]['arrivals'], 
                           reverse=True)
    
    for airport, stats in sorted_airports[:10]:
        total_flights = stats['departures'] + stats['arrivals']
        avg_delay = sum(stats['delays']) / len(stats['delays']) if stats['delays'] else 0
        
        print(f"   {airport}: {total_flights} total ({stats['departures']} dep, {stats['arrivals']} arr)")
        print(f"          Avg departure delay: {avg_delay:.1f}min")
    
    # === DATA QUALITY ASSESSMENT ===
    print(f"\n📊 DATA QUALITY ASSESSMENT")
    print("-" * 35)
    
    total_flights = len(flights)
    
    # Completeness metrics
    flights_with_std = len([f for f in flights if f.departure.scheduled])
    flights_with_atd = len([f for f in flights if f.departure.actual])
    flights_with_sta = len([f for f in flights if f.arrival.scheduled])
    flights_with_ata = len([f for f in flights if f.arrival.actual])
    flights_with_route = len([f for f in flights if f.origin and f.destination])
    flights_with_aircraft = len([f for f in flights if f.aircraft_type and f.aircraft_type != "UNKNOWN"])
    
    print(f"Data completeness:")
    print(f"   Scheduled departure time: {flights_with_std/total_flights*100:.1f}%")
    print(f"   Actual departure time: {flights_with_atd/total_flights*100:.1f}%")
    print(f"   Scheduled arrival time: {flights_with_sta/total_flights*100:.1f}%")
    print(f"   Actual arrival time: {flights_with_ata/total_flights*100:.1f}%")
    print(f"   Route information: {flights_with_route/total_flights*100:.1f}%")
    print(f"   Aircraft information: {flights_with_aircraft/total_flights*100:.1f}%")
    
    # === WHAT WE CAN DO WITH THIS DATA ===
    print(f"\n🚀 WHAT WE CAN DO WITH THIS DATA")
    print("-" * 45)
    
    print("✅ Peak Traffic Analysis:")
    print("   - Identify busiest hours, days, and routes")
    print("   - Analyze traffic patterns by airport")
    print("   - Detect congestion periods")
    
    print("\n✅ Delay Prediction:")
    print("   - Historical delay patterns by airline, route, time")
    print("   - Weather correlation (with external data)")
    print("   - Aircraft type performance analysis")
    
    print("\n✅ Schedule Optimization:")
    print("   - Identify optimal departure times")
    print("   - Route efficiency analysis")
    print("   - Resource allocation recommendations")
    
    print("\n✅ Operational Insights:")
    print("   - Airline performance benchmarking")
    print("   - Airport capacity utilization")
    print("   - On-time performance tracking")
    
    print("\n✅ Advanced Analytics:")
    print("   - Machine learning model training")
    print("   - Real-time monitoring dashboards")
    print("   - Predictive maintenance scheduling")
    
    # === NEXT STEPS ===
    print(f"\n🎯 NEXT STEPS")
    print("-" * 15)
    
    print("1. ✅ Data successfully loaded and validated")
    print("2. 🔄 Store in DuckDB for high-performance queries")
    print("3. 📈 Implement peak traffic analysis algorithms")
    print("4. 🤖 Build delay prediction models")
    print("5. ⚡ Create schedule optimization engine")
    print("6. 📊 Develop real-time monitoring dashboard")
    
    print(f"\n🎉 DEMONSTRATION COMPLETE!")
    print(f"✅ {len(flights)} flights ready for advanced analytics")
    print(f"✅ Rich dataset with {len(set(f.airline_code for f in flights))} airlines")
    print(f"✅ {len(set(f.get_route_key() for f in flights))} unique routes")
    print(f"✅ Complete time-series data for traffic analysis")


if __name__ == "__main__":
    main()