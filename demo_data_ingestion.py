#!/usr/bin/env python3
"""Demo script for the Data Ingestion Service."""

import os
from src.services.data_ingestion import DataIngestionService


def main():
    """Demonstrate the data ingestion service with the sample Excel file."""
    print("🛫 Flight Data Ingestion Service Demo")
    print("=" * 50)
    
    # Initialize the service
    service = DataIngestionService()
    
    # Check if the sample Excel file exists
    sample_file = '429e6e3f-281d-4e4c-b00a-92fb020cb2fcFlight_Data.xlsx'
    
    if not os.path.exists(sample_file):
        print(f"❌ Sample file '{sample_file}' not found!")
        print("Please ensure the Excel file is in the current directory.")
        return
    
    print(f"📁 Processing file: {sample_file}")
    print("⏳ Ingesting flight data...")
    
    # Process the Excel file
    result = service.ingest_excel_files([sample_file])
    
    # Display results
    print(f"\n✅ Processing completed in {result.processing_time_seconds:.2f} seconds")
    print(f"📊 Files processed: {result.total_files_processed}")
    print(f"✈️  Total flights found: {result.total_flights}")
    print(f"✅ Valid flights: {result.valid_flights}")
    print(f"❌ Invalid flights: {result.invalid_flights}")
    
    if result.errors:
        print(f"⚠️  Errors encountered: {len(result.errors)}")
        for error in result.errors[:3]:  # Show first 3 errors
            print(f"   - {error}")
    
    # Show sample flights
    if result.flights:
        print(f"\n🔍 Sample flights:")
        for i, flight in enumerate(result.flights[:5]):  # Show first 5 flights
            print(f"   {i+1}. {flight.flight_number} ({flight.airline_code})")
            print(f"      Route: {flight.origin.code} → {flight.destination.code}")
            print(f"      Aircraft: {flight.aircraft_type}")
            print(f"      Date: {flight.flight_date}")
            if flight.dep_delay_min is not None:
                print(f"      Departure delay: {flight.dep_delay_min} minutes")
            if flight.arr_delay_min is not None:
                print(f"      Arrival delay: {flight.arr_delay_min} minutes")
            print()
    
    # Generate and display summary
    print("📈 Generating ingestion summary...")
    summary = service.get_ingestion_summary(result)
    
    print(f"\n📋 Ingestion Summary:")
    print(f"   Status: {summary['status']}")
    print(f"   Data quality rate: {summary['flights']['data_quality_rate']:.1%}")
    
    if 'airlines' in summary:
        print(f"   Airlines found: {len(summary['airlines'])}")
        top_airlines = list(summary['airlines'].items())[:5]
        for airline, count in top_airlines:
            print(f"      - {airline}: {count} flights")
    
    if 'top_routes' in summary:
        print(f"   Top routes:")
        top_routes = list(summary['top_routes'].items())[:5]
        for route, count in top_routes:
            print(f"      - {route}: {count} flights")
    
    if 'delay_analysis' in summary and summary['delay_analysis']:
        print(f"   Delay analysis:")
        if 'departure' in summary['delay_analysis']:
            dep_stats = summary['delay_analysis']['departure']
            print(f"      - Avg departure delay: {dep_stats['avg_delay_min']:.1f} min")
            print(f"      - On-time departure rate: {dep_stats['on_time_rate']:.1%}")
        
        if 'arrival' in summary['delay_analysis']:
            arr_stats = summary['delay_analysis']['arrival']
            print(f"      - Avg arrival delay: {arr_stats['avg_delay_min']:.1f} min")
            print(f"      - On-time arrival rate: {arr_stats['on_time_rate']:.1%}")
    
    # Validate ingestion accuracy
    print(f"\n🔍 Validating ingestion accuracy...")
    validation = service.validate_ingestion_accuracy(result)
    
    print(f"   Data completeness:")
    completeness = validation['data_completeness']
    print(f"      - STD completeness: {completeness['std_completeness']:.1%}")
    print(f"      - ATD completeness: {completeness['atd_completeness']:.1%}")
    print(f"      - STA completeness: {completeness['sta_completeness']:.1%}")
    print(f"      - ATA completeness: {completeness['ata_completeness']:.1%}")
    print(f"      - Route completeness: {completeness['route_completeness']:.1%}")
    
    print(f"   Timestamp validation:")
    ts_validation = validation['timestamp_validation']
    print(f"      - UTC conversion rate: {ts_validation['utc_conversion_rate']:.1%}")
    
    print(f"   Delay calculation:")
    delay_validation = validation['delay_calculation_validation']
    print(f"      - Departure delay calculation rate: {delay_validation['departure_delay_calculation_rate']:.1%}")
    print(f"      - Arrival delay calculation rate: {delay_validation['arrival_delay_calculation_rate']:.1%}")
    
    if validation['data_quality_issues']:
        print(f"   Data quality issues found: {len(validation['data_quality_issues'])}")
        for issue in validation['data_quality_issues'][:3]:  # Show first 3 issues
            print(f"      - {issue}")
    
    print(f"\n🎉 Demo completed successfully!")
    print(f"   The DataIngestionService successfully processed {result.valid_flights} flights")
    print(f"   with IST→UTC conversion, delay calculations, and missing data handling.")


if __name__ == "__main__":
    main()