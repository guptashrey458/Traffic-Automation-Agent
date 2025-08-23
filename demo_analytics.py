#!/usr/bin/env python3
"""
Demo script for the Analytics Engine - Peak Traffic Analysis

This script demonstrates the peak traffic analysis capabilities of the 
Agentic Flight Scheduler system.
"""

import sys
import os
from datetime import datetime, date, time
from unittest.mock import Mock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.services.analytics import AnalyticsEngine, WeatherRegime
from src.services.database import FlightDatabaseService, QueryResult


def create_sample_data():
    """Create sample flight data for Mumbai airport during morning rush."""
    base_date = date(2024, 1, 15)
    
    # Simulate a busy morning at Mumbai (BOM)
    flights = [
        # Early morning (6 AM)
        {'flight_id': 'AI101', 'flight_number': 'AI101', 'origin_code': 'BOM', 'destination_code': 'DEL',
         'std_utc': datetime.combine(base_date, time(6, 0)), 'atd_utc': datetime.combine(base_date, time(6, 5)),
         'dep_delay_min': 5, 'arr_delay_min': None},
        
        # Building traffic (7 AM)
        {'flight_id': 'AI201', 'flight_number': 'AI201', 'origin_code': 'BOM', 'destination_code': 'CCU',
         'std_utc': datetime.combine(base_date, time(7, 0)), 'atd_utc': datetime.combine(base_date, time(7, 10)),
         'dep_delay_min': 10, 'arr_delay_min': None},
        {'flight_id': '6E301', 'flight_number': '6E301', 'origin_code': 'BOM', 'destination_code': 'HYD',
         'std_utc': datetime.combine(base_date, time(7, 15)), 'atd_utc': datetime.combine(base_date, time(7, 30)),
         'dep_delay_min': 15, 'arr_delay_min': None},
        
        # Peak hour (8 AM) - Heavy traffic with delays
        {'flight_id': 'AI301', 'flight_number': 'AI301', 'origin_code': 'BOM', 'destination_code': 'DEL',
         'std_utc': datetime.combine(base_date, time(8, 0)), 'atd_utc': datetime.combine(base_date, time(8, 25)),
         'dep_delay_min': 25, 'arr_delay_min': None},
        {'flight_id': '6E401', 'flight_number': '6E401', 'origin_code': 'BOM', 'destination_code': 'BLR',
         'std_utc': datetime.combine(base_date, time(8, 2)), 'atd_utc': datetime.combine(base_date, time(8, 35)),
         'dep_delay_min': 33, 'arr_delay_min': None},
        {'flight_id': 'SG501', 'flight_number': 'SG501', 'origin_code': 'BOM', 'destination_code': 'MAA',
         'std_utc': datetime.combine(base_date, time(8, 4)), 'atd_utc': datetime.combine(base_date, time(8, 40)),
         'dep_delay_min': 36, 'arr_delay_min': None},
        {'flight_id': 'AI401', 'flight_number': 'AI401', 'origin_code': 'BOM', 'destination_code': 'CCU',
         'std_utc': datetime.combine(base_date, time(8, 6)), 'atd_utc': datetime.combine(base_date, time(8, 50)),
         'dep_delay_min': 44, 'arr_delay_min': None},
        {'flight_id': '6E501', 'flight_number': '6E501', 'origin_code': 'BOM', 'destination_code': 'AMD',
         'std_utc': datetime.combine(base_date, time(8, 8)), 'atd_utc': datetime.combine(base_date, time(8, 55)),
         'dep_delay_min': 47, 'arr_delay_min': None},
        
        # Arrivals during peak hour
        {'flight_id': 'AI501', 'flight_number': 'AI501', 'origin_code': 'DEL', 'destination_code': 'BOM',
         'std_utc': datetime.combine(base_date, time(6, 0)), 'atd_utc': datetime.combine(base_date, time(6, 0)),
         'sta_utc': datetime.combine(base_date, time(8, 0)), 'ata_utc': datetime.combine(base_date, time(8, 15)),
         'dep_delay_min': 0, 'arr_delay_min': 15},
        {'flight_id': '6E601', 'flight_number': '6E601', 'origin_code': 'BLR', 'destination_code': 'BOM',
         'std_utc': datetime.combine(base_date, time(6, 30)), 'atd_utc': datetime.combine(base_date, time(6, 30)),
         'sta_utc': datetime.combine(base_date, time(8, 3)), 'ata_utc': datetime.combine(base_date, time(8, 18)),
         'dep_delay_min': 0, 'arr_delay_min': 15},
        {'flight_id': 'SG601', 'flight_number': 'SG601', 'origin_code': 'MAA', 'destination_code': 'BOM',
         'std_utc': datetime.combine(base_date, time(6, 15)), 'atd_utc': datetime.combine(base_date, time(6, 15)),
         'sta_utc': datetime.combine(base_date, time(8, 7)), 'ata_utc': datetime.combine(base_date, time(8, 22)),
         'dep_delay_min': 0, 'arr_delay_min': 15},
        
        # More flights to create overload
        {'flight_id': 'AI601', 'flight_number': 'AI601', 'origin_code': 'BOM', 'destination_code': 'JAI',
         'std_utc': datetime.combine(base_date, time(8, 1)), 'atd_utc': datetime.combine(base_date, time(8, 35)),
         'dep_delay_min': 34, 'arr_delay_min': None},
        {'flight_id': '6E701', 'flight_number': '6E701', 'origin_code': 'BOM', 'destination_code': 'IXC',
         'std_utc': datetime.combine(base_date, time(8, 3)), 'atd_utc': datetime.combine(base_date, time(8, 40)),
         'dep_delay_min': 37, 'arr_delay_min': None},
        {'flight_id': 'UK801', 'flight_number': 'UK801', 'origin_code': 'BOM', 'destination_code': 'GOI',
         'std_utc': datetime.combine(base_date, time(8, 5)), 'atd_utc': datetime.combine(base_date, time(8, 45)),
         'dep_delay_min': 40, 'arr_delay_min': None},
        {'flight_id': 'SG701', 'flight_number': 'SG701', 'origin_code': 'BOM', 'destination_code': 'PNQ',
         'std_utc': datetime.combine(base_date, time(8, 7)), 'atd_utc': datetime.combine(base_date, time(8, 50)),
         'dep_delay_min': 43, 'arr_delay_min': None},
        {'flight_id': 'AI701', 'flight_number': 'AI701', 'origin_code': 'BOM', 'destination_code': 'LKO',
         'std_utc': datetime.combine(base_date, time(8, 9)), 'atd_utc': datetime.combine(base_date, time(8, 55)),
         'dep_delay_min': 46, 'arr_delay_min': None},
    ]
    
    return flights


def print_separator(title):
    """Print a formatted separator."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)


def print_bucket_analysis(bucket):
    """Print detailed analysis of a time bucket."""
    print(f"Time: {bucket.start_time.strftime('%H:%M')}-{bucket.end_time.strftime('%H:%M')}")
    print(f"  Departures: {bucket.scheduled_departures} scheduled, {bucket.actual_departures} actual")
    print(f"  Arrivals: {bucket.scheduled_arrivals} scheduled, {bucket.actual_arrivals} actual")
    print(f"  Total Demand: {bucket.total_demand}")
    print(f"  Capacity: {bucket.capacity}")
    print(f"  Utilization: {bucket.utilization:.1%}")
    print(f"  Overload: {bucket.overload} flights")
    print(f"  Traffic Level: {bucket.traffic_level.value.upper()}")
    print(f"  Average Delay: {bucket.avg_delay:.1f} minutes")
    print(f"  Delayed Flights: {bucket.delayed_flights}")


def main():
    """Main demonstration function."""
    print_separator("AGENTIC FLIGHT SCHEDULER - ANALYTICS ENGINE DEMO")
    
    # Create sample data
    sample_flights = create_sample_data()
    print(f"Created sample dataset with {len(sample_flights)} flights for Mumbai (BOM)")
    print(f"Analysis date: January 15, 2024")
    
    # Create mock database service
    mock_db = Mock(spec=FlightDatabaseService)
    mock_db.query_flights_by_date_range.return_value = QueryResult(
        data=sample_flights,
        row_count=len(sample_flights)
    )
    
    # Initialize analytics engine
    analytics_engine = AnalyticsEngine(db_service=mock_db)
    
    print_separator("PEAK TRAFFIC ANALYSIS - NORMAL CONDITIONS")
    
    # Perform peak analysis under normal weather conditions
    analysis = analytics_engine.analyze_peaks(
        airport="BOM",
        analysis_date=date(2024, 1, 15),
        bucket_minutes=10,
        weather_regime=WeatherRegime.CALM
    )
    
    print(f"Airport: {analysis.airport}")
    print(f"Analysis Date: {analysis.analysis_date}")
    print(f"Bucket Size: {analysis.bucket_minutes} minutes")
    print(f"Weather Regime: {analysis.weather_regime.value}")
    print(f"Peak Hour: {analysis.peak_hour}:00")
    print(f"Peak Demand: {analysis.peak_demand} flights")
    print(f"Average Utilization: {analysis.avg_utilization:.1%}")
    
    # Show peak hour details
    print_separator("PEAK HOUR ANALYSIS (8:00-8:10 AM)")
    
    peak_bucket = None
    for bucket in analysis.time_buckets:
        if bucket.start_time.hour == 8 and bucket.start_time.minute == 0:
            peak_bucket = bucket
            break
    
    if peak_bucket:
        print_bucket_analysis(peak_bucket)
    
    # Show overload windows
    print_separator("OVERLOAD WINDOWS DETECTED")
    
    if analysis.overload_windows:
        for i, window in enumerate(analysis.overload_windows, 1):
            print(f"Window {i}:")
            print(f"  Time: {window.start_time.strftime('%H:%M')} - {window.end_time.strftime('%H:%M')}")
            print(f"  Duration: {window.duration_minutes} minutes")
            print(f"  Peak Overload: {window.peak_overload} flights")
            print(f"  Average Overload: {window.avg_overload:.1f} flights")
            print(f"  Affected Flights: {window.affected_flights}")
            print(f"  Severity: {window.severity.upper()}")
            print(f"  Recommendations:")
            for rec in window.recommendations:
                print(f"    - {rec}")
            print()
    else:
        print("No overload windows detected.")
    
    # Show delay hotspots
    print_separator("DELAY HOTSPOTS")
    
    if analysis.delay_hotspots:
        print("Top delay hotspots:")
        for i, hotspot in enumerate(analysis.delay_hotspots[:5], 1):
            print(f"{i}. {hotspot['time']} - Avg Delay: {hotspot['avg_delay']}min, "
                  f"Delayed Flights: {hotspot['delayed_flights']}, "
                  f"Delay Rate: {hotspot['delay_rate']}%")
    else:
        print("No significant delay hotspots detected.")
    
    # Show recommendations
    print_separator("SYSTEM RECOMMENDATIONS")
    
    for i, recommendation in enumerate(analysis.recommendations, 1):
        print(f"{i}. {recommendation}")
    
    # Weather impact comparison
    print_separator("WEATHER IMPACT ANALYSIS")
    
    # Analyze under strong wind conditions
    stormy_analysis = analytics_engine.analyze_peaks(
        airport="BOM",
        analysis_date=date(2024, 1, 15),
        bucket_minutes=10,
        weather_regime=WeatherRegime.STRONG
    )
    
    # Find peak buckets for comparison
    normal_peak = next(b for b in analysis.time_buckets if b.start_time.hour == 8 and b.start_time.minute == 0)
    stormy_peak = next(b for b in stormy_analysis.time_buckets if b.start_time.hour == 8 and b.start_time.minute == 0)
    
    print("Weather Impact on Peak Hour (8:00-8:10 AM):")
    print(f"  Normal Conditions:")
    print(f"    Capacity: {normal_peak.capacity} flights")
    print(f"    Utilization: {normal_peak.utilization:.1%}")
    print(f"    Overload: {normal_peak.overload} flights")
    print(f"  Strong Wind Conditions:")
    print(f"    Capacity: {stormy_peak.capacity} flights")
    print(f"    Utilization: {stormy_peak.utilization:.1%}")
    print(f"    Overload: {stormy_peak.overload} flights")
    print(f"  Impact:")
    print(f"    Capacity Reduction: {normal_peak.capacity - stormy_peak.capacity} flights")
    print(f"    Additional Overload: {stormy_peak.overload - normal_peak.overload} flights")
    
    # Generate heatmap data sample
    print_separator("HEATMAP DATA SAMPLE")
    
    heatmap_data = analysis.get_heatmap_data()
    
    print("Sample heatmap data for morning hours:")
    print("Time     | Demand | Capacity | Utilization | Overload | Traffic Level")
    print("-" * 70)
    
    for bucket_data in heatmap_data:
        if 6 <= bucket_data["hour"] <= 9:
            print(f"{bucket_data['time']:8} | {bucket_data['demand']:6} | "
                  f"{bucket_data['capacity']:8} | {bucket_data['utilization']:10.1%} | "
                  f"{bucket_data['overload']:8} | {bucket_data['traffic_level']:12}")
    
    print_separator("DEMO COMPLETED")
    print("The Analytics Engine successfully:")
    print("✓ Analyzed peak traffic patterns")
    print("✓ Identified overload windows")
    print("✓ Detected delay hotspots")
    print("✓ Generated actionable recommendations")
    print("✓ Assessed weather impact on capacity")
    print("✓ Created heatmap visualization data")
    print("\nThis demonstrates the core functionality required for Requirements 2.1-2.4")


if __name__ == "__main__":
    main()