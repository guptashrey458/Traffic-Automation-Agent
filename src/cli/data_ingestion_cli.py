"""Command-line interface for flight data ingestion with multiple sources."""

import argparse
import os
import sys
from datetime import datetime, date, timedelta
from pathlib import Path
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.services.unified_data_loader import UnifiedDataLoader, UnifiedDataConfig, DataSource
from src.services.data_ingestion import IngestionResult


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Flight Data Ingestion CLI - Support for Excel, FlightAware, and FlightRadar24",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load from Excel files
  python -m src.cli.data_ingestion_cli --source excel --excel-dir ./data --airports BOM DEL
  
  # Load from FlightAware API
  python -m src.cli.data_ingestion_cli --source flightaware --api-key YOUR_KEY --start-date 2024-01-15 --end-date 2024-01-21 --airports BOM DEL
  
  # Load from FlightRadar24 HTML files
  python -m src.cli.data_ingestion_cli --source flightradar24 --html-dir ./fr24_data --airports BOM DEL
  
  # Auto-detect best source
  python -m src.cli.data_ingestion_cli --source auto --excel-dir ./data --html-dir ./fr24_data --airports BOM DEL
  
  # Test data source availability
  python -m src.cli.data_ingestion_cli --test-sources --excel-dir ./data --api-key YOUR_KEY
        """
    )
    
    # Data source selection
    parser.add_argument(
        '--source', 
        choices=['excel', 'flightaware', 'flightradar24', 'auto'],
        default='auto',
        help='Data source to use (default: auto)'
    )
    
    # Excel configuration
    parser.add_argument(
        '--excel-dir',
        type=str,
        help='Directory containing Excel files'
    )
    
    parser.add_argument(
        '--excel-files',
        nargs='+',
        help='Specific Excel files to process'
    )
    
    # FlightAware configuration
    parser.add_argument(
        '--api-key',
        type=str,
        help='FlightAware API key (or set FLIGHTAWARE_API_KEY env var)'
    )
    
    parser.add_argument(
        '--api-url',
        type=str,
        default='https://aeroapi.flightaware.com/aeroapi',
        help='FlightAware API base URL'
    )
    
    # FlightRadar24 configuration
    parser.add_argument(
        '--html-dir',
        type=str,
        help='Directory containing FlightRadar24 HTML files'
    )
    
    parser.add_argument(
        '--html-files',
        nargs='+',
        help='Specific HTML files to process'
    )
    
    # Common parameters
    parser.add_argument(
        '--airports',
        nargs='+',
        default=['BOM', 'DEL'],
        help='Airport codes to process (default: BOM DEL)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date for data collection (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date for data collection (YYYY-MM-DD)'
    )
    
    # Output options
    parser.add_argument(
        '--output-format',
        choices=['json', 'summary', 'detailed'],
        default='summary',
        help='Output format (default: summary)'
    )
    
    parser.add_argument(
        '--output-file',
        type=str,
        help='Save results to file'
    )
    
    parser.add_argument(
        '--save-flights',
        type=str,
        help='Save flight data to JSON file'
    )
    
    # Testing and validation
    parser.add_argument(
        '--test-sources',
        action='store_true',
        help='Test availability of configured data sources'
    )
    
    parser.add_argument(
        '--validate-quality',
        action='store_true',
        help='Validate data quality after ingestion'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser


def parse_date(date_str: str) -> date:
    """Parse date string in YYYY-MM-DD format."""
    try:
        return datetime.strptime(date_str, '%Y-%m-%d').date()
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}. Use YYYY-MM-DD")


def create_config(args) -> UnifiedDataConfig:
    """Create UnifiedDataConfig from command-line arguments."""
    # Get API key from args or environment
    api_key = args.api_key or os.getenv('FLIGHTAWARE_API_KEY')
    
    config = UnifiedDataConfig(
        excel_directory=args.excel_dir,
        flightaware_api_key=api_key,
        flightaware_base_url=args.api_url,
        fr24_html_directory=args.html_dir,
        airport_codes=args.airports
    )
    
    return config


def test_data_sources(loader: UnifiedDataLoader, args) -> None:
    """Test availability of configured data sources."""
    print("Testing data source availability...\n")
    
    sources = loader.get_available_sources()
    
    for source_name, source_info in sources.items():
        print(f"=== {source_name.upper()} ===")
        
        if source_info['available']:
            print("✅ Available")
            
            if source_name == 'excel':
                print(f"   Directory: {source_info['directory']}")
                print(f"   Files found: {source_info['file_count']}")
                if source_info['sample_files']:
                    print(f"   Sample files: {', '.join(source_info['sample_files'])}")
            
            elif source_name == 'flightaware':
                print(f"   API configured: {source_info['api_configured']}")
                print(f"   Base URL: {source_info['base_url']}")
                
                if 'connection_test' in source_info:
                    test_result = source_info['connection_test']
                    if test_result['status'] == 'success':
                        print("   ✅ Connection test passed")
                    else:
                        print(f"   ❌ Connection test failed: {test_result['message']}")
            
            elif source_name == 'flightradar24':
                print(f"   Directory: {source_info['directory']}")
                print(f"   HTML files found: {source_info['file_count']}")
                if source_info['sample_files']:
                    print(f"   Sample files: {', '.join(source_info['sample_files'])}")
        
        else:
            print("❌ Not available")
            
            if source_name == 'excel' and not args.excel_dir:
                print("   Reason: No Excel directory specified")
            elif source_name == 'flightaware' and not source_info['api_configured']:
                print("   Reason: No API key configured")
            elif source_name == 'flightradar24' and not args.html_dir:
                print("   Reason: No HTML directory specified")
        
        print()


def format_output(result: IngestionResult, format_type: str, args) -> str:
    """Format ingestion result for output."""
    if format_type == 'json':
        # Convert result to JSON-serializable format
        result_dict = {
            'total_files_processed': result.total_files_processed,
            'successful_files': result.successful_files,
            'failed_files': result.failed_files,
            'total_flights': result.total_flights,
            'valid_flights': result.valid_flights,
            'invalid_flights': result.invalid_flights,
            'processing_time_seconds': result.processing_time_seconds,
            'errors': result.errors,
            'flights': [flight.to_dict() for flight in result.flights]
        }
        return json.dumps(result_dict, indent=2, default=str)
    
    elif format_type == 'summary':
        output = []
        output.append("=== FLIGHT DATA INGESTION SUMMARY ===")
        output.append(f"Source: {args.source}")
        output.append(f"Airports: {', '.join(args.airports)}")
        
        if args.start_date and args.end_date:
            output.append(f"Date range: {args.start_date} to {args.end_date}")
        
        output.append("")
        output.append("Files processed:")
        output.append(f"  Total: {result.total_files_processed}")
        output.append(f"  Successful: {result.successful_files}")
        output.append(f"  Failed: {len(result.failed_files)}")
        
        output.append("")
        output.append("Flight data:")
        output.append(f"  Total flights: {result.total_flights}")
        output.append(f"  Valid flights: {result.valid_flights}")
        output.append(f"  Invalid flights: {result.invalid_flights}")
        
        if result.valid_flights > 0:
            output.append(f"  Data quality: {result.valid_flights / result.total_flights * 100:.1f}%")
        
        output.append(f"  Processing time: {result.processing_time_seconds:.2f} seconds")
        
        if result.errors:
            output.append("")
            output.append("Errors encountered:")
            for error in result.errors[:5]:  # Show first 5 errors
                output.append(f"  - {error}")
            if len(result.errors) > 5:
                output.append(f"  ... and {len(result.errors) - 5} more errors")
        
        return "\n".join(output)
    
    elif format_type == 'detailed':
        # Include flight-level details
        output = [format_output(result, 'summary', args)]
        
        if result.flights:
            output.append("")
            output.append("=== FLIGHT DETAILS ===")
            
            # Group flights by airline
            airlines = {}
            for flight in result.flights:
                airline = flight.airline_code or 'UNKNOWN'
                if airline not in airlines:
                    airlines[airline] = []
                airlines[airline].append(flight)
            
            for airline, flights in sorted(airlines.items()):
                output.append(f"\n{airline}: {len(flights)} flights")
                
                # Show sample flights
                for flight in flights[:3]:
                    route = f"{flight.origin.code if flight.origin else 'UNK'}-{flight.destination.code if flight.destination else 'UNK'}"
                    dep_time = flight.departure.scheduled.strftime('%H:%M') if flight.departure.scheduled else 'N/A'
                    delay = f"{flight.dep_delay_min}min" if flight.dep_delay_min is not None else 'N/A'
                    output.append(f"  {flight.flight_number}: {route} @ {dep_time} (delay: {delay})")
                
                if len(flights) > 3:
                    output.append(f"  ... and {len(flights) - 3} more flights")
        
        return "\n".join(output)
    
    return str(result)


def main():
    """Main CLI function."""
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        # Create configuration
        config = create_config(args)
        loader = UnifiedDataLoader(config)
        
        # Test sources if requested
        if args.test_sources:
            test_data_sources(loader, args)
            return
        
        # Parse dates if provided
        start_date = None
        end_date = None
        
        if args.start_date:
            start_date = parse_date(args.start_date)
        
        if args.end_date:
            end_date = parse_date(args.end_date)
        
        # Default to last week if no dates provided for API sources
        if args.source in ['flightaware', 'auto'] and not start_date:
            end_date = date.today()
            start_date = end_date - timedelta(days=7)
            if args.verbose:
                print(f"Using default date range: {start_date} to {end_date}")
        
        # Determine file paths
        file_paths = None
        if args.source == 'excel' and args.excel_files:
            file_paths = args.excel_files
        elif args.source == 'flightradar24' and args.html_files:
            file_paths = args.html_files
        
        # Load data
        if args.verbose:
            print(f"Loading data from {args.source} source...")
            print(f"Airports: {', '.join(args.airports)}")
            if start_date and end_date:
                print(f"Date range: {start_date} to {end_date}")
        
        source_enum = DataSource(args.source)
        result = loader.load_data(
            source=source_enum,
            start_date=start_date,
            end_date=end_date,
            file_paths=file_paths,
            airport_codes=args.airports
        )
        
        # Normalize data schema
        result.flights = loader.normalize_data_schema(result.flights)
        
        # Validate data quality if requested
        if args.validate_quality:
            quality_report = loader.validate_data_quality(result)
            print("\n=== DATA QUALITY REPORT ===")
            print(f"Status: {quality_report['status']}")
            if quality_report['status'] == 'success':
                print(f"Completeness score: {quality_report['completeness_score']:.2f}")
                print(f"Airport coverage: {', '.join(quality_report['airport_coverage'])}")
                if quality_report['recommendations']:
                    print("Recommendations:")
                    for rec in quality_report['recommendations']:
                        print(f"  - {rec}")
            print()
        
        # Format and output results
        output = format_output(result, args.output_format, args)
        
        if args.output_file:
            with open(args.output_file, 'w') as f:
                f.write(output)
            print(f"Results saved to {args.output_file}")
        else:
            print(output)
        
        # Save flight data if requested
        if args.save_flights and result.flights:
            flight_data = [flight.to_dict() for flight in result.flights]
            with open(args.save_flights, 'w') as f:
                json.dump(flight_data, f, indent=2, default=str)
            print(f"Flight data saved to {args.save_flights}")
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()