#!/usr/bin/env python3
"""
Debug and demonstrate Excel data access
"""

import pandas as pd
from src.services.data_ingestion import DataIngestionService
from src.models.excel_parser import ExcelFlightParser

def main():
    print("🔍 Debugging Excel Data Access")
    print("=" * 40)
    
    excel_file = "429e6e3f-281d-4e4c-b00a-92fb020cb2fcFlight_Data.xlsx"
    
    # Method 1: Direct pandas access
    print("📊 Method 1: Direct Pandas Access")
    try:
        df = pd.read_excel(excel_file)
        print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
        print(f"📋 Columns: {list(df.columns)}")
        print(f"\n📝 First 3 rows:")
        print(df.head(3).to_string())
        
        print(f"\n📊 Data Summary:")
        print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"   Unique flights: {df['Flight'].nunique()}")
        print(f"   Unique routes: {df['Sector'].nunique()}")
        
    except Exception as e:
        print(f"❌ Pandas access failed: {e}")
    
    # Method 2: Using our parser
    print(f"\n🔧 Method 2: Using Our Excel Parser")
    try:
        parser = ExcelFlightParser()
        batch = parser.parse_excel_file(excel_file)
        
        print(f"✅ Parsed {len(batch.flights)} flights")
        print(f"❌ Errors: {len(batch.errors)}")
        
        if batch.errors:
            print("First few errors:")
            for error in batch.errors[:3]:
                print(f"   - {error}")
        
        if batch.flights:
            print(f"\n✈️  Sample parsed flights:")
            for i, flight in enumerate(batch.flights[:3], 1):
                print(f"   {i}. {flight.flight_number}")
                print(f"      Route: {flight.get_route_key()}")
                print(f"      Date: {flight.flight_date}")
                print(f"      Valid: {flight.is_valid()}")
                print(f"      Raw data keys: {list(flight.raw_data.keys())}")
                print()
        
    except Exception as e:
        print(f"❌ Parser access failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Method 3: Using ingestion service
    print(f"\n🚀 Method 3: Using Ingestion Service")
    try:
        ingestion_service = DataIngestionService()
        result = ingestion_service.ingest_excel_files([excel_file])
        
        print(f"✅ Ingested {result.total_flights} total flights")
        print(f"✅ Valid flights: {result.valid_flights}")
        print(f"❌ Invalid flights: {result.invalid_flights}")
        
        if result.flights:
            print(f"\n📊 Flight Analysis:")
            
            # Airlines
            airlines = {}
            for flight in result.flights:
                airlines[flight.airline_code] = airlines.get(flight.airline_code, 0) + 1
            print(f"   Airlines: {dict(sorted(airlines.items(), key=lambda x: x[1], reverse=True))}")
            
            # Routes
            routes = {}
            for flight in result.flights:
                route = flight.get_route_key()
                routes[route] = routes.get(route, 0) + 1
            top_routes = dict(sorted(routes.items(), key=lambda x: x[1], reverse=True)[:5])
            print(f"   Top routes: {top_routes}")
            
            # Delays
            delays = [f.dep_delay_min for f in result.flights if f.dep_delay_min is not None]
            if delays:
                avg_delay = sum(delays) / len(delays)
                print(f"   Average delay: {avg_delay:.1f} minutes")
                print(f"   Flights with delays: {len([d for d in delays if d > 0])}")
            
            # Sample flight details
            print(f"\n✈️  Detailed Flight Sample:")
            sample = result.flights[0]
            print(f"   Flight ID: {sample.flight_id}")
            print(f"   Flight Number: {sample.flight_number}")
            print(f"   Airline: {sample.airline_code}")
            print(f"   Route: {sample.origin.name} → {sample.destination.name}")
            print(f"   Date: {sample.flight_date}")
            print(f"   STD: {sample.departure.scheduled}")
            print(f"   ATD: {sample.departure.actual}")
            print(f"   STA: {sample.arrival.scheduled}")
            print(f"   ATA: {sample.arrival.actual}")
            print(f"   Dep Delay: {sample.dep_delay_min} min")
            print(f"   Arr Delay: {sample.arr_delay_min} min")
            print(f"   Aircraft: {sample.aircraft_type}")
            print(f"   Status: {sample.status}")
            print(f"   Data Source: {sample.data_source}")
        
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()