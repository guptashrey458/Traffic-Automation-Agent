#!/usr/bin/env python3
"""
Demonstration: Complete Flight Data Access Pipeline
Loading real Excel data into DuckDB and performing various analyses
"""

import os
from datetime import datetime, date
from pathlib import Path

from src.services.data_ingestion import DataIngestionService
from src.services.database import FlightDatabaseService, DatabaseConfig
from src.models.flight import Flight


def main():
    print("🛫 Flight Data Access Pipeline Demonstration")
    print("=" * 60)
    
    # Setup database configuration
    config = DatabaseConfig(
        db_path="data/demo_flights.duckdb",
        parquet_export_path="data/parquet_exports"
    )
    
    # Initialize services
    ingestion_service = DataIngestionService()
    db_service = FlightDatabaseService(config)
    
    # Step 1: Load Excel file
    excel_file = "429e6e3f-281d-4e4c-b00a-92fb020cb2fcFlight_Data.xlsx"
    
    if not os.path.exists(excel_file):
        print(f"❌ Excel file not found: {excel_file}")
        print("Please ensure the Excel file is in the current directory.")
        return
    
    print(f"📊 Loading Excel file: {excel_file}")
    
    try:
        # Ingest Excel data
        ingestion_result = ingestion_service.ingest_excel_files([excel_file])
        
        print(f"✅ Excel Processing Complete!")
        print(f"   📈 Total flights processed: {ingestion_result.total_flights}")
        print(f"   ✅ Valid flights: {ingestion_result.valid_flights}")
        print(f"   ❌ Invalid flights: {ingestion_result.invalid_flights}")
        print(f"   ⏱️  Processing time: {ingestion_result.processing_time_seconds:.2f}s")
        
        if ingestion_result.errors:
            print(f"   ⚠️  Errors encountered: {len(ingestion_result.errors)}")
            for error in ingestion_result.errors[:3]:  # Show first 3 errors
                print(f"      - {error}")
        
        if not ingestion_result.flights:
            print("❌ No valid flights found in Excel file")
            return
        
        print(f"\n🗄️  Storing {len(ingestion_result.flights)} flights in DuckDB...")
        
        # Store in database
        storage_result = db_service.store_flights(ingestion_result.flights)
        
        print(f"✅ Database Storage Complete!")
        print(f"   💾 Flights stored: {storage_result['stored']}")
        print(f"   📊 Success rate: {storage_result['success_rate']:.1%}")
        print(f"   ⏱️  Storage time: {storage_result['processing_time_seconds']:.2f}s")
        
        if storage_result['errors']:
            print(f"   ⚠️  Storage errors: {len(storage_result['errors'])}")
        
        # Step 2: Demonstrate data access capabilities
        print(f"\n🔍 Demonstrating Data Access Capabilities")
        print("-" * 50)
        
        # Get database statistics
        stats = db_service.get_database_stats()
        print(f"📈 Database Overview:")
        print(f"   Total flights: {stats['airline_stats']['total_flights']}")
        print(f"   Unique airlines: {stats['airline_stats']['unique_airlines']}")
        print(f"   Unique routes: {stats['airport_stats']['unique_routes']}")
        print(f"   Date range: {stats['date_range']['min_date']} to {stats['date_range']['max_date']}")
        print(f"   Data quality: {stats['data_quality']['std_completeness_pct']:.1f}% STD completeness")
        
        # Show sample flight data
        print(f"\n✈️  Sample Flight Records:")
        sample_flights = ingestion_result.flights[:3]  # First 3 flights
        for i, flight in enumerate(sample_flights, 1):
            print(f"   {i}. {flight.flight_number} ({flight.airline_code})")
            print(f"      Route: {flight.origin.name} → {flight.destination.name}")
            print(f"      Date: {flight.flight_date}")
            print(f"      STD: {flight.departure.scheduled} | ATD: {flight.departure.actual}")
            if flight.dep_delay_min is not None:
                print(f"      Departure Delay: {flight.dep_delay_min} minutes")
            print(f"      Aircraft: {flight.aircraft_type}")
            print()
        
        # Step 3: Query demonstrations
        print(f"🔎 Query Demonstrations")
        print("-" * 30)
        
        # Find date range in data
        dates = set(f.flight_date for f in ingestion_result.flights if f.flight_date)
        if dates:
            min_date = min(dates)
            max_date = max(dates)
            
            # Query flights for a specific date
            print(f"📅 Querying flights for {min_date}...")
            date_query = db_service.query_flights_by_date_range(min_date, min_date)
            print(f"   Found {date_query.row_count} flights")
            print(f"   Query time: {date_query.execution_time_ms:.1f}ms")
            
            # Show airlines breakdown
            airlines = {}
            for flight_data in date_query.data:
                airline = flight_data.get('airline_code', 'Unknown')
                airlines[airline] = airlines.get(airline, 0) + 1
            
            print(f"   Airlines on {min_date}:")
            for airline, count in sorted(airlines.items(), key=lambda x: x[1], reverse=True):
                print(f"      {airline}: {count} flights")
        
        # Find airports in data
        origins = set(f.origin.code for f in ingestion_result.flights if f.origin)
        if origins:
            main_airport = list(origins)[0]  # Take first airport
            
            print(f"\n🏢 Peak Traffic Analysis for {main_airport}...")
            peak_result = db_service.query_peak_traffic(
                airport_code=main_airport,
                bucket_minutes=60  # 1-hour buckets
            )
            
            print(f"   Found {peak_result.row_count} time buckets")
            print(f"   Query time: {peak_result.execution_time_ms:.1f}ms")
            
            if peak_result.data:
                print(f"   Peak traffic periods:")
                # Sort by flight count and show top 3
                sorted_buckets = sorted(peak_result.data, 
                                      key=lambda x: x.get('flight_count', 0), 
                                      reverse=True)[:3]
                
                for bucket in sorted_buckets:
                    time_bucket = bucket.get('time_bucket', 'Unknown')
                    flight_count = bucket.get('flight_count', 0)
                    delayed_count = bucket.get('delayed_count', 0)
                    traffic_level = bucket.get('traffic_level', 'Unknown')
                    
                    print(f"      {time_bucket}: {flight_count} flights, "
                          f"{delayed_count} delayed ({traffic_level} traffic)")
        
        # Step 4: Export demonstration
        print(f"\n📤 Export Demonstration")
        print("-" * 25)
        
        export_result = db_service.export_to_parquet()
        if export_result['success']:
            print(f"✅ Exported to Parquet successfully!")
            print(f"   📁 File: {export_result['output_path']}")
            print(f"   📊 Rows: {export_result['rows_exported']}")
            print(f"   💾 Size: {export_result['file_size_bytes']:,} bytes")
            print(f"   ⏱️  Time: {export_result['processing_time_seconds']:.2f}s")
        else:
            print(f"❌ Export failed: {export_result.get('error', 'Unknown error')}")
        
        # Step 5: Advanced queries
        print(f"\n🧠 Advanced Analysis Examples")
        print("-" * 35)
        
        # Delay analysis
        delayed_flights = [f for f in ingestion_result.flights 
                          if f.dep_delay_min is not None and f.dep_delay_min > 15]
        
        if delayed_flights:
            avg_delay = sum(f.dep_delay_min for f in delayed_flights) / len(delayed_flights)
            max_delay = max(f.dep_delay_min for f in delayed_flights)
            
            print(f"📊 Delay Analysis:")
            print(f"   Flights with >15min delay: {len(delayed_flights)}")
            print(f"   Average delay: {avg_delay:.1f} minutes")
            print(f"   Maximum delay: {max_delay} minutes")
            print(f"   On-time performance: {((len(ingestion_result.flights) - len(delayed_flights)) / len(ingestion_result.flights) * 100):.1f}%")
        
        # Route analysis
        routes = {}
        for flight in ingestion_result.flights:
            if flight.origin and flight.destination:
                route = f"{flight.origin.code}-{flight.destination.code}"
                routes[route] = routes.get(route, 0) + 1
        
        if routes:
            print(f"\n🗺️  Route Analysis:")
            print(f"   Total unique routes: {len(routes)}")
            print(f"   Top routes:")
            for route, count in sorted(routes.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"      {route}: {count} flights")
        
        print(f"\n🎉 Demonstration Complete!")
        print(f"✅ Successfully loaded and analyzed flight data from Excel")
        print(f"✅ Data is now accessible through high-performance DuckDB queries")
        print(f"✅ Ready for advanced analytics and optimization tasks")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        db_service.close()


if __name__ == "__main__":
    main()