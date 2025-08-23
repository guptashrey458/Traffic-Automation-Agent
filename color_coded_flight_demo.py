#!/usr/bin/env python3
"""
Comprehensive demonstration of color-coded flight delay analysis
Replicating Excel's green/yellow/red status indicators in our system
"""

from datetime import date
from src.services.data_ingestion import DataIngestionService
from src.services.database import FlightDatabaseService, DatabaseConfig, DelayStatus


def main():
    print("🚦 COLOR-CODED FLIGHT DELAY ANALYSIS DEMONSTRATION")
    print("=" * 65)
    
    # Setup database with enhanced schema
    config = DatabaseConfig(
        db_path="data/color_coded_flights.duckdb",
        parquet_export_path="data/exports"
    )
    
    # Load and process flight data
    excel_file = "429e6e3f-281d-4e4c-b00a-92fb020cb2fcFlight_Data.xlsx"
    ingestion_service = DataIngestionService()
    db_service = FlightDatabaseService(config)
    
    print("📊 Loading Excel flight data with color-coded delay analysis...")
    
    # Ingest data
    result = ingestion_service.ingest_excel_files([excel_file])
    print(f"✅ Loaded {result.total_flights} flights from Excel")
    
    # Store in enhanced database with delay status
    storage_result = db_service.store_flights(result.flights)
    print(f"✅ Stored {storage_result['stored']} flights with delay status classification")
    
    if storage_result['errors']:
        print(f"⚠️  Storage errors: {len(storage_result['errors'])}")
        for error in storage_result['errors'][:3]:
            print(f"   - {error}")
    
    # === DELAY STATUS OVERVIEW ===
    print(f"\n🚦 DELAY STATUS OVERVIEW")
    print("-" * 30)
    
    delay_summary = db_service.get_delay_status_summary()
    
    if 'error' not in delay_summary:
        print(f"Total flights analyzed: {delay_summary['total_flights']}")
        print()
        
        status_breakdown = delay_summary['status_breakdown']
        
        # Green flights (on-time)
        if 'green' in status_breakdown:
            green_stats = status_breakdown['green']
            print(f"🟩 GREEN (On-time ≤15min):")
            print(f"   Count: {green_stats['count']} flights ({green_stats['percentage']}%)")
            print(f"   Avg delay: {green_stats['avg_delay']} minutes")
            print(f"   Range: {green_stats['min_delay']} to {green_stats['max_delay']} minutes")
        
        # Yellow flights (moderate delay)
        if 'yellow' in status_breakdown:
            yellow_stats = status_breakdown['yellow']
            print(f"\n🟨 YELLOW (Moderate 16-60min):")
            print(f"   Count: {yellow_stats['count']} flights ({yellow_stats['percentage']}%)")
            print(f"   Avg delay: {yellow_stats['avg_delay']} minutes")
            print(f"   Range: {yellow_stats['min_delay']} to {yellow_stats['max_delay']} minutes")
        
        # Red flights (critical delay)
        if 'red' in status_breakdown:
            red_stats = status_breakdown['red']
            print(f"\n🟥 RED (Critical >60min):")
            print(f"   Count: {red_stats['count']} flights ({red_stats['percentage']}%)")
            print(f"   Avg delay: {red_stats['avg_delay']} minutes")
            print(f"   Range: {red_stats['min_delay']} to {red_stats['max_delay']} minutes")
    
    # === QUERY BY STATUS EXAMPLES ===
    print(f"\n🔍 QUERYING BY DELAY STATUS")
    print("-" * 35)
    
    # Query critical delays (red status)
    print("🟥 Critical Delays (Red Status):")
    red_flights = db_service.query_flights_by_delay_status(DelayStatus.CRITICAL)
    print(f"   Found {red_flights.row_count} critical delay flights")
    print(f"   Query time: {red_flights.execution_time_ms:.1f}ms")
    
    if red_flights.data:
        print("   Worst performers:")
        # Sort by delay and show top 5
        sorted_red = sorted(red_flights.data, key=lambda x: x.get('dep_delay_min', 0), reverse=True)
        for i, flight in enumerate(sorted_red[:5], 1):
            print(f"      {i}. {flight['flight_number']} ({flight['airline_code']}): {flight['dep_delay_min']}min")
            print(f"         Route: {flight['route']}, Date: {flight['flight_date']}")
    
    # Query on-time flights (green status)
    print(f"\n🟩 On-time Flights (Green Status):")
    green_flights = db_service.query_flights_by_delay_status(DelayStatus.ON_TIME)
    print(f"   Found {green_flights.row_count} on-time flights")
    print(f"   Query time: {green_flights.execution_time_ms:.1f}ms")
    
    if green_flights.data:
        # Show best performers
        airlines_green = {}
        routes_green = {}
        
        for flight in green_flights.data:
            airline = flight['airline_code']
            route = flight['route']
            airlines_green[airline] = airlines_green.get(airline, 0) + 1
            routes_green[route] = routes_green.get(route, 0) + 1
        
        print("   Best performing airlines (green flights):")
        for airline, count in sorted(airlines_green.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"      {airline}: {count} on-time flights")
        
        print("   Best performing routes (green flights):")
        for route, count in sorted(routes_green.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"      {route}: {count} on-time flights")
    
    # Query moderate delays (yellow status)
    print(f"\n🟨 Moderate Delays (Yellow Status):")
    yellow_flights = db_service.query_flights_by_delay_status(DelayStatus.MODERATE)
    print(f"   Found {yellow_flights.row_count} moderate delay flights")
    print(f"   Query time: {yellow_flights.execution_time_ms:.1f}ms")
    
    # === DATE-SPECIFIC ANALYSIS ===
    print(f"\n📅 DATE-SPECIFIC DELAY STATUS ANALYSIS")
    print("-" * 45)
    
    # Get a sample date from the data
    if result.flights:
        sample_date = result.flights[0].flight_date
        print(f"Analyzing delay status for {sample_date}:")
        
        date_summary = db_service.get_delay_status_summary(sample_date)
        
        if 'error' not in date_summary and date_summary['status_breakdown']:
            for status, stats in date_summary['status_breakdown'].items():
                status_emoji = {"green": "🟩", "yellow": "🟨", "red": "🟥"}.get(status, "⚪")
                print(f"   {status_emoji} {status.upper()}: {stats['count']} flights ({stats['percentage']}%)")
    
    # === OPERATIONAL INSIGHTS ===
    print(f"\n💡 OPERATIONAL INSIGHTS FROM COLOR-CODED DATA")
    print("-" * 55)
    
    print("🎯 Key Performance Indicators:")
    if 'error' not in delay_summary:
        total = delay_summary['total_flights']
        green_pct = delay_summary['status_breakdown'].get('green', {}).get('percentage', 0)
        red_pct = delay_summary['status_breakdown'].get('red', {}).get('percentage', 0)
        
        print(f"   On-time Performance: {green_pct}%")
        print(f"   Critical Delay Rate: {red_pct}%")
        
        # Performance assessment
        if green_pct >= 80:
            assessment = "🌟 EXCELLENT - Industry leading performance"
        elif green_pct >= 70:
            assessment = "✅ GOOD - Above average performance"
        elif green_pct >= 60:
            assessment = "⚠️  AVERAGE - Room for improvement"
        elif green_pct >= 50:
            assessment = "🔶 BELOW AVERAGE - Action needed"
        else:
            assessment = "🚨 POOR - Immediate intervention required"
        
        print(f"   Overall Assessment: {assessment}")
    
    print(f"\n🔧 Actionable Recommendations:")
    print("   🟩 GREEN ZONE (Maintain Excellence):")
    print("      - Document best practices from high-performing routes")
    print("      - Replicate successful operational procedures")
    print("      - Use as benchmarks for other operations")
    
    print(f"\n   🟨 YELLOW ZONE (Quick Wins Available):")
    print("      - Focus improvement efforts here for maximum impact")
    print("      - Implement small buffer times in schedules")
    print("      - Optimize ground operations and turnaround times")
    
    print(f"\n   🟥 RED ZONE (Critical Action Required):")
    print("      - Investigate root causes of major delays")
    print("      - Consider schedule restructuring for problem routes")
    print("      - Implement contingency planning for high-risk periods")
    
    # === EXCEL INTEGRATION ===
    print(f"\n📊 EXCEL INTEGRATION GUIDE")
    print("-" * 30)
    
    print("To replicate this analysis in Excel:")
    print("1. Add a 'Delay Status' column")
    print("2. Use this formula: =IF(K2<=15,\"🟩 Green\",IF(K2<=60,\"🟨 Yellow\",\"🟥 Red\"))")
    print("3. Apply conditional formatting:")
    print("   - Green: Background #90EE90, Text #006400")
    print("   - Yellow: Background #FFFF99, Text #B8860B") 
    print("   - Red: Background #FFB6C1, Text #8B0000")
    
    print(f"\n4. Create pivot tables for analysis:")
    print("   - Rows: Airline, Route, Hour")
    print("   - Values: Count of flights by delay status")
    print("   - This gives you the same insights as our database queries!")
    
    # === EXPORT CAPABILITIES ===
    print(f"\n📤 ENHANCED EXPORT WITH COLOR CODING")
    print("-" * 40)
    
    export_result = db_service.export_to_parquet()
    if export_result['success']:
        print(f"✅ Exported flight data with delay status to Parquet")
        print(f"   File: {export_result['output_path']}")
        print(f"   Includes: delay_status columns for advanced analytics")
        print(f"   Use in: Python pandas, R, Tableau, Power BI, etc.")
    
    print(f"\n🎉 COLOR-CODED DELAY ANALYSIS COMPLETE!")
    print(f"✅ Excel color coding successfully replicated in database")
    print(f"✅ {delay_summary.get('total_flights', 0)} flights classified by delay status")
    print(f"✅ Ready for advanced traffic analysis and optimization")
    
    # Cleanup
    db_service.close()


if __name__ == "__main__":
    main()