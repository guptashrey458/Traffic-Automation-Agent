"""Test data models with real Excel data."""

import pytest
import pandas as pd
from src.models.validation import DataValidator
from src.models.flight import FlightDataBatch


def test_real_excel_data_parsing():
    """Test parsing real Excel data from the hackathon file."""
    file_path = '429e6e3f-281d-4e4c-b00a-92fb020cb2fcFlight_Data.xlsx'
    
    try:
        # Read the first sheet
        df = pd.read_excel(file_path, sheet_name='6AM - 9AM')
        
        # Create a batch for processing
        batch = FlightDataBatch(source_file=file_path)
        
        # Process first 10 rows to test
        processed_count = 0
        for idx, row in df.head(10).iterrows():
            row_dict = row.to_dict()
            
            # Skip rows without flight numbers
            if pd.isna(row_dict.get('Flight Number')) or row_dict.get('Flight Number') == '':
                continue
            
            flight = DataValidator.create_flight_from_raw_data(row_dict, "6AM - 9AM")
            if flight:
                batch.add_flight(flight)
                processed_count += 1
        
        # Get statistics
        stats = batch.get_stats()
        print(f"Processed {processed_count} flights from real data")
        print(f"Batch stats: {stats}")
        
        # Verify we processed some flights
        assert stats["total_flights"] > 0
        
        # Check that we have valid flights
        valid_flights = batch.get_valid_flights()
        if len(valid_flights) > 0:
            flight = valid_flights[0]
            print(f"Sample flight: {flight.flight_number} from {flight.origin.code if flight.origin else 'UNK'} to {flight.destination.code if flight.destination else 'UNK'}")
            
            # Verify flight has basic required data
            assert flight.flight_number != ""
            
    except FileNotFoundError:
        pytest.skip("Excel file not found - skipping real data test")
    except Exception as e:
        pytest.fail(f"Error processing real Excel data: {e}")


def test_time_parsing_with_real_formats():
    """Test time parsing with actual formats from the Excel file."""
    
    # Test actual time formats from the data
    test_cases = [
        ("06:00:00", (6, 0, 0)),
        ("06:20:00", (6, 20, 0)),
        ("08:10:00", (8, 10, 0)),
        ("Landed 8:14 AM", "8:14 AM"),
        ("Landed 8:01 AM", "8:01 AM"),
        ("Landed 9:20 AM", "9:20 AM"),
    ]
    
    for time_str, expected in test_cases:
        if time_str.startswith("Landed"):
            # Test ATA parsing
            result = DataValidator.parse_ata_string(time_str)
            assert result is not None, f"Failed to parse ATA: {time_str}"
        else:
            # Test regular time parsing
            result = DataValidator.parse_time_string(time_str)
            assert result is not None, f"Failed to parse time: {time_str}"
            assert (result.hour, result.minute, result.second) == expected


def test_airport_parsing_with_real_formats():
    """Test airport parsing with actual formats from the Excel file."""
    
    test_cases = [
        ("Mumbai (BOM)", "BOM", "Mumbai"),
        ("Chandigarh (IXC)", "IXC", "Chandigarh"),
        ("Delhi (DEL)", "DEL", "Delhi"),
    ]
    
    for airport_str, expected_code, expected_city in test_cases:
        from src.models.flight import Airport
        airport = Airport.from_string(airport_str)
        
        assert airport.code == expected_code
        assert airport.city == expected_city
        assert airport.name == airport_str


if __name__ == "__main__":
    # Run the real data test manually
    test_real_excel_data_parsing()
    test_time_parsing_with_real_formats()
    test_airport_parsing_with_real_formats()
    print("All real data tests passed!")