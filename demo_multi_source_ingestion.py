"""Demo script showcasing multi-source flight data ingestion capabilities."""

import os
import sys
from datetime import date, timedelta
from pathlib import Path
import tempfile

# Add src to path
sys.path.append('src')

from src.services.unified_data_loader import UnifiedDataLoader, UnifiedDataConfig, DataSource
from src.services.data_ingestion import IngestionResult


def create_sample_html_files(temp_dir: str) -> list[str]:
    """Create sample FlightRadar24 HTML files for demo."""
    html_files = []
    
    # Sample HTML content for BOM airport
    bom_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Mumbai (BOM) Airport - Departures - FlightRadar24</title>
    </head>
    <body>
        <div class="airport-header">
            <h1>Mumbai (BOM) - Departures</h1>
            <p>Date: January 15, 2024</p>
        </div>
        
        <table class="flights-table">
            <tr>
                <th>Flight</th>
                <th>Destination</th>
                <th>Aircraft</th>
                <th>STD</th>
                <th>Status</th>
            </tr>
            <tr>
                <td>AI 123</td>
                <td>Delhi (DEL)</td>
                <td>A320</td>
                <td>10:30</td>
                <td>Departed 10:45</td>
            </tr>
            <tr>
                <td>6E 456</td>
                <td>Kolkata (CCU)</td>
                <td>A321</td>
                <td>11:15</td>
                <td>Scheduled</td>
            </tr>
            <tr>
                <td>UK 789</td>
                <td>Bangalore (BLR)</td>
                <td>A320neo</td>
                <td>12:00</td>
                <td>Delayed 12:30</td>
            </tr>
            <tr>
                <td>SG 234</td>
                <td>Chennai (MAA)</td>
                <td>B737</td>
                <td>13:45</td>
                <td>Scheduled</td>
            </tr>
        </table>
    </body>
    </html>
    """
    
    # Sample HTML content for DEL airport
    del_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Delhi (DEL) Airport - Arrivals - FlightRadar24</title>
    </head>
    <body>
        <h1>Delhi (DEL) - Arrivals</h1>
        <p>Date: 2024-01-15</p>
        
        <div class="flights-container">
            <div class="flight-row">
                <span class="flight-number">AI 567</span>
                <span class="route">Mumbai (BOM) → Delhi (DEL)</span>
                <span class="aircraft">A320</span>
                <span class="time">14:30</span>
                <span class="status">Landed 14:25</span>
            </div>
            
            <div class="flight-row">
                <span class="flight-number">6E 890</span>
                <span class="route">Kolkata (CCU) → Delhi (DEL)</span>
                <span class="aircraft">A321</span>
                <span class="time">15:45</span>
                <span class="status">Scheduled</span>
            </div>
            
            <div class="flight-row">
                <span class="flight-number">UK 345</span>
                <span class="route">Bangalore (BLR) → Delhi (DEL)</span>
                <span class="aircraft">A320neo</span>
                <span class="time">16:20</span>
                <span class="status">Delayed 16:45</span>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Create HTML files
    bom_file = os.path.join(temp_dir, "BOM_2024-01-15_departures.html")
    with open(bom_file, 'w', encoding='utf-8') as f:
        f.write(bom_html)
    html_files.append(bom_file)
    
    del_file = os.path.join(temp_dir, "DEL_2024-01-15_arrivals.html")
    with open(del_file, 'w', encoding='utf-8') as f:
        f.write(del_html)
    html_files.append(del_file)
    
    return html_files


def demo_source_availability():
    """Demo checking availability of different data sources."""
    print("=== DATA SOURCE AVAILABILITY DEMO ===\n")
    
    # Create temporary directory for demo files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create sample HTML files
        html_files = create_sample_html_files(temp_dir)
        print(f"Created {len(html_files)} sample HTML files in {temp_dir}")
        
        # Configure unified data loader
        config = UnifiedDataConfig(
            excel_directory="./data",  # May or may not exist
            flightaware_api_key=os.getenv('FLIGHTAWARE_API_KEY'),  # May or may not be set
            fr24_html_directory=temp_dir,  # Our temp directory with sample files
            airport_codes=["BOM", "DEL"]
        )
        
        loader = UnifiedDataLoader(config)
        
        # Check available sources
        sources = loader.get_available_sources()
        
        print("Available data sources:")
        for source_name, source_info in sources.items():
            status = "✅ Available" if source_info['available'] else "❌ Not Available"
            print(f"  {source_name.upper()}: {status}")
            
            if source_name == 'excel':
                print(f"    Directory: {source_info['directory']}")
                print(f"    Files found: {source_info['file_count']}")
                if source_info['sample_files']:
                    print(f"    Sample files: {', '.join(source_info['sample_files'][:2])}")
            
            elif source_name == 'flightaware':
                print(f"    API configured: {source_info['api_configured']}")
                if 'connection_test' in source_info:
                    test_result = source_info['connection_test']
                    print(f"    Connection test: {test_result['status']}")
            
            elif source_name == 'flightradar24':
                print(f"    Directory: {source_info['directory']}")
                print(f"    HTML files found: {source_info['file_count']}")
            
            print()


def demo_flightradar24_ingestion():
    """Demo FlightRadar24 HTML file ingestion."""
    print("=== FLIGHTRADAR24 INGESTION DEMO ===\n")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create sample HTML files
        html_files = create_sample_html_files(temp_dir)
        
        # Configure and load data
        config = UnifiedDataConfig(
            fr24_html_directory=temp_dir,
            airport_codes=["BOM", "DEL"]
        )
        
        loader = UnifiedDataLoader(config)
        
        print("Loading data from FlightRadar24 HTML files...")
        result = loader.load_data(DataSource.FLIGHTRADAR24)
        
        print(f"Processing completed in {result.processing_time_seconds:.2f} seconds")
        print(f"Files processed: {result.total_files_processed}")
        print(f"Total flights found: {result.total_flights}")
        print(f"Valid flights: {result.valid_flights}")
        
        if result.flights:
            print("\nSample flights:")
            for i, flight in enumerate(result.flights[:5]):
                route = f"{flight.origin.code if flight.origin else 'UNK'}-{flight.destination.code if flight.destination else 'UNK'}"
                dep_time = flight.departure.scheduled.strftime('%H:%M') if flight.departure.scheduled else 'N/A'
                print(f"  {i+1}. {flight.flight_number}: {route} @ {dep_time} ({flight.data_source})")
        
        # Validate data quality
        quality_report = loader.validate_data_quality(result)
        print(f"\nData Quality Score: {quality_report['completeness_score']:.2f}")
        print(f"Airport Coverage: {', '.join(quality_report['airport_coverage'])}")
        
        if quality_report['recommendations']:
            print("Recommendations:")
            for rec in quality_report['recommendations']:
                print(f"  - {rec}")


def demo_auto_detection():
    """Demo automatic data source detection."""
    print("=== AUTO-DETECTION DEMO ===\n")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create sample HTML files
        html_files = create_sample_html_files(temp_dir)
        
        # Configure with multiple potential sources
        config = UnifiedDataConfig(
            excel_directory="./data",  # May have Excel files
            flightaware_api_key=os.getenv('FLIGHTAWARE_API_KEY'),  # May be configured
            fr24_html_directory=temp_dir,  # Has our sample HTML files
            airport_codes=["BOM", "DEL"]
        )
        
        loader = UnifiedDataLoader(config)
        
        print("Running auto-detection to find best data source...")
        result = loader.load_data(DataSource.AUTO)
        
        print(f"Auto-detection completed in {result.processing_time_seconds:.2f} seconds")
        print(f"Best source selected based on data availability")
        print(f"Total flights found: {result.total_flights}")
        print(f"Valid flights: {result.valid_flights}")
        
        if result.flights:
            # Analyze data sources used
            source_counts = {}
            for flight in result.flights:
                source = flight.data_source
                source_counts[source] = source_counts.get(source, 0) + 1
            
            print("\nData source distribution:")
            for source, count in source_counts.items():
                print(f"  {source}: {count} flights")


def demo_schema_normalization():
    """Demo data schema normalization across sources."""
    print("=== SCHEMA NORMALIZATION DEMO ===\n")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create sample HTML files
        html_files = create_sample_html_files(temp_dir)
        
        config = UnifiedDataConfig(
            fr24_html_directory=temp_dir,
            airport_codes=["BOM", "DEL"]
        )
        
        loader = UnifiedDataLoader(config)
        
        # Load raw data
        result = loader.load_data(DataSource.FLIGHTRADAR24)
        
        print("Before normalization:")
        if result.flights:
            sample_flight = result.flights[0]
            print(f"  Flight number: '{sample_flight.flight_number}'")
            print(f"  Airline code: '{sample_flight.airline_code}'")
            print(f"  Aircraft type: '{sample_flight.aircraft_type}'")
            print(f"  Origin code: '{sample_flight.origin.code if sample_flight.origin else 'None'}'")
        
        # Normalize data
        normalized_flights = loader.normalize_data_schema(result.flights)
        
        print("\nAfter normalization:")
        if normalized_flights:
            sample_flight = normalized_flights[0]
            print(f"  Flight number: '{sample_flight.flight_number}'")
            print(f"  Airline code: '{sample_flight.airline_code}'")
            print(f"  Aircraft type: '{sample_flight.aircraft_type}'")
            print(f"  Origin code: '{sample_flight.origin.code if sample_flight.origin else 'None'}'")
        
        # Test schema compliance
        print("\nSchema compliance check:")
        for flight in normalized_flights[:3]:
            flight_dict = flight.to_dict()
            required_fields = ['flight_id', 'flight_number', 'origin_code', 'destination_code']
            
            compliance = all(field in flight_dict and flight_dict[field] for field in required_fields)
            status = "✅ Compliant" if compliance else "❌ Non-compliant"
            print(f"  {flight.flight_number}: {status}")


def demo_cli_usage():
    """Demo command-line interface usage."""
    print("=== CLI USAGE DEMO ===\n")
    
    print("The new CLI supports multiple data sources:")
    print()
    
    print("1. Test data source availability:")
    print("   python -m src.cli.data_ingestion_cli --test-sources --html-dir ./fr24_data")
    print()
    
    print("2. Load from FlightRadar24 HTML files:")
    print("   python -m src.cli.data_ingestion_cli --source flightradar24 --html-dir ./fr24_data --airports BOM DEL")
    print()
    
    print("3. Load from FlightAware API (requires API key):")
    print("   python -m src.cli.data_ingestion_cli --source flightaware --api-key YOUR_KEY --start-date 2024-01-15 --end-date 2024-01-21")
    print()
    
    print("4. Auto-detect best source:")
    print("   python -m src.cli.data_ingestion_cli --source auto --excel-dir ./data --html-dir ./fr24_data")
    print()
    
    print("5. Validate data quality:")
    print("   python -m src.cli.data_ingestion_cli --source auto --validate-quality --output-format detailed")
    print()
    
    print("6. Save results to file:")
    print("   python -m src.cli.data_ingestion_cli --source flightradar24 --html-dir ./fr24_data --save-flights flights.json")


def main():
    """Run all demos."""
    print("🛫 MULTI-SOURCE FLIGHT DATA INGESTION DEMO 🛬\n")
    print("This demo showcases the new FlightRadar24/FlightAware data ingestion capabilities")
    print("for hackathon compliance, including unified data loading and schema normalization.\n")
    
    try:
        demo_source_availability()
        print("\n" + "="*60 + "\n")
        
        demo_flightradar24_ingestion()
        print("\n" + "="*60 + "\n")
        
        demo_auto_detection()
        print("\n" + "="*60 + "\n")
        
        demo_schema_normalization()
        print("\n" + "="*60 + "\n")
        
        demo_cli_usage()
        
        print("\n" + "="*60)
        print("✅ DEMO COMPLETED SUCCESSFULLY!")
        print("\nKey Features Demonstrated:")
        print("  • FlightRadar24 HTML parsing")
        print("  • FlightAware API integration")
        print("  • Unified data loading with auto-detection")
        print("  • Schema normalization for pipeline compatibility")
        print("  • Command-line interface for data source switching")
        print("  • Data quality validation and reporting")
        print("\nThe system is now ready for hackathon compliance with multiple data sources!")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()