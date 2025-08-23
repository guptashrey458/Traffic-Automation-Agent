"""Test Excel parser with real data."""

import pytest
from src.models.excel_parser import ExcelFlightParser


def test_excel_parser_with_real_data():
    """Test Excel parser with the actual hackathon data."""
    file_path = '429e6e3f-281d-4e4c-b00a-92fb020cb2fcFlight_Data.xlsx'
    
    try:
        parser = ExcelFlightParser()
        batch = parser.parse_excel_file(file_path)
        
        # Get summary
        summary = parser.get_flight_summary(batch)
        
        print("=== Flight Data Summary ===")
        print(f"Total flights processed: {summary['total_flights']}")
        print(f"Valid flights: {summary['valid_flights']}")
        print(f"Invalid flights: {summary['invalid_flights']}")
        print(f"Errors: {summary['error_count']}")
        
        if summary['valid_flights'] > 0:
            print(f"\nAirlines: {summary['airlines']}")
            print(f"Top routes: {summary['top_routes']}")
            print(f"Aircraft types: {summary['aircraft_types']}")
            print(f"Date range: {summary['date_range']}")
            print(f"Time periods: {summary['time_periods']}")
            
            # Show sample flights
            valid_flights = batch.get_valid_flights()
            print(f"\n=== Sample Flights ===")
            for i, flight in enumerate(valid_flights[:5]):
                print(f"{i+1}. {flight.flight_number} ({flight.airline_code}) "
                      f"{flight.origin.code if flight.origin else 'UNK'} -> "
                      f"{flight.destination.code if flight.destination else 'UNK'} "
                      f"on {flight.flight_date} "
                      f"STD: {flight.departure.scheduled} "
                      f"Delay: {flight.dep_delay_min}min")
        
        # Verify we got some valid flights
        assert summary['valid_flights'] > 0, "Should have parsed some valid flights"
        assert len(summary['airlines']) > 0, "Should have identified airlines"
        assert len(summary['top_routes']) > 0, "Should have identified routes"
        
    except FileNotFoundError:
        pytest.skip("Excel file not found - skipping real data test")
    except Exception as e:
        pytest.fail(f"Error testing Excel parser: {e}")


if __name__ == "__main__":
    test_excel_parser_with_real_data()