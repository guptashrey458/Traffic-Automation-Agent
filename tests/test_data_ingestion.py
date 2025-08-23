"""Tests for data ingestion service."""

import pytest
import pandas as pd
from datetime import datetime, date, time
from pathlib import Path
import tempfile
import os
from unittest.mock import Mock, patch

from src.services.data_ingestion import DataIngestionService, IngestionResult
from src.models.flight import Flight, Airport, FlightTime


class TestDataIngestionService:
    """Test suite for DataIngestionService."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = DataIngestionService()
    
    def create_sample_excel_file(self, filename: str, data: list) -> str:
        """Create a sample Excel file for testing."""
        df = pd.DataFrame(data)
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, filename)
        df.to_excel(file_path, index=False)
        return file_path
    
    def test_ingest_single_excel_file_success(self):
        """Test successful ingestion of a single Excel file."""
        # Create sample data
        sample_data = [
            {
                'Flight Number': 'AI2509',
                'Unnamed: 2': '2025-07-25',
                'From': 'Mumbai (BOM)',
                'To': 'Chandigarh (IXC)',
                'Aircraft': 'A20N (VT-EXU)',
                'Flight time': '1:54',
                'STD': '06:00:00',
                'ATD': '06:20:00',
                'STA': '08:10:00',
                'ATA': 'Landed 8:14 AM'
            },
            {
                'Flight Number': '',
                'Unnamed: 2': '2025-07-24',
                'From': 'Mumbai (BOM)',
                'To': 'Chandigarh (IXC)',
                'Aircraft': 'A20N (VT-RTJ)',
                'Flight time': '1:54',
                'STD': '06:00:00',
                'ATD': '06:07:00',
                'STA': '08:10:00',
                'ATA': 'Landed 8:01 AM'
            }
        ]
        
        file_path = self.create_sample_excel_file('test_flights.xlsx', sample_data)
        
        try:
            # Test ingestion
            result = self.service.ingest_excel_files([file_path])
            
            # Assertions
            assert result.total_files_processed == 1
            assert result.successful_files == 1
            assert len(result.failed_files) == 0
            assert result.valid_flights > 0
            assert len(result.flights) > 0
            
            # Check first flight
            flight = result.flights[0]
            assert flight.flight_number == 'AI2509'
            assert flight.origin.code == 'BOM'
            assert flight.destination.code == 'IXC'
            assert flight.aircraft_type == 'A320'  # Normalized from A20N
            
        finally:
            # Clean up
            if os.path.exists(file_path):
                os.remove(file_path)
    
    def test_ingest_multiple_excel_files(self):
        """Test ingestion of multiple Excel files."""
        # Create two sample files
        sample_data_1 = [
            {
                'Flight Number': 'AI2509',
                'Unnamed: 2': '2025-07-25',
                'From': 'Mumbai (BOM)',
                'To': 'Delhi (DEL)',
                'Aircraft': 'A320',
                'STD': '06:00:00',
                'ATD': '06:20:00',
                'STA': '08:10:00',
                'ATA': 'Landed 8:14 AM'
            }
        ]
        
        sample_data_2 = [
            {
                'Flight Number': '6E123',
                'Unnamed: 2': '2025-07-25',
                'From': 'Delhi (DEL)',
                'To': 'Mumbai (BOM)',
                'Aircraft': 'A320',
                'STD': '10:00:00',
                'ATD': '10:15:00',
                'STA': '12:30:00',
                'ATA': 'Landed 12:45 PM'
            }
        ]
        
        file_path_1 = self.create_sample_excel_file('test_flights_1.xlsx', sample_data_1)
        file_path_2 = self.create_sample_excel_file('test_flights_2.xlsx', sample_data_2)
        
        try:
            # Test ingestion
            result = self.service.ingest_excel_files([file_path_1, file_path_2])
            
            # Assertions
            assert result.total_files_processed == 2
            assert result.successful_files == 2
            assert len(result.failed_files) == 0
            assert result.valid_flights >= 2
            
            # Check that we have flights from both files
            flight_numbers = [f.flight_number for f in result.flights]
            assert 'AI2509' in flight_numbers
            assert '6E123' in flight_numbers
            
        finally:
            # Clean up
            for file_path in [file_path_1, file_path_2]:
                if os.path.exists(file_path):
                    os.remove(file_path)
    
    def test_ist_to_utc_conversion(self):
        """Test IST to UTC timestamp conversion."""
        # Create sample data with IST times
        sample_data = [
            {
                'Flight Number': 'AI2509',
                'Unnamed: 2': '2025-07-25',
                'From': 'Mumbai (BOM)',
                'To': 'Delhi (DEL)',
                'Aircraft': 'A320',
                'STD': '06:00:00',  # 6:00 AM IST
                'ATD': '06:20:00',  # 6:20 AM IST
                'STA': '08:10:00',
                'ATA': 'Landed 8:14 AM'
            }
        ]
        
        file_path = self.create_sample_excel_file('test_utc.xlsx', sample_data)
        
        try:
            result = self.service.ingest_excel_files([file_path])
            
            # Check that times are converted to UTC
            flight = result.flights[0]
            
            # IST is UTC+5:30, so 6:20 AM IST should be 12:50 AM UTC
            if flight.departure.actual:
                assert flight.departure.actual.tzinfo is not None
                # The hour should be adjusted for UTC (6:20 IST = 00:50 UTC)
                assert flight.departure.actual.hour == 0
                assert flight.departure.actual.minute == 50
            
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
    
    def test_delay_calculation(self):
        """Test delay calculation logic."""
        # Create sample data with delays
        sample_data = [
            {
                'Flight Number': 'AI2509',
                'Unnamed: 2': '2025-07-25',
                'From': 'Mumbai (BOM)',
                'To': 'Delhi (DEL)',
                'Aircraft': 'A320',
                'STD': '06:00:00',  # Scheduled 6:00 AM
                'ATD': '06:20:00',  # Actual 6:20 AM (20 min delay)
                'STA': '08:10:00',  # Scheduled 8:10 AM
                'ATA': 'Landed 8:30 AM'  # Actual 8:30 AM (20 min delay)
            }
        ]
        
        file_path = self.create_sample_excel_file('test_delays.xlsx', sample_data)
        
        try:
            result = self.service.ingest_excel_files([file_path])
            
            flight = result.flights[0]
            
            # Check delay calculations
            assert flight.dep_delay_min == 20  # 20 minutes departure delay
            assert flight.arr_delay_min == 20  # 20 minutes arrival delay
            
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
    
    def test_missing_data_handling(self):
        """Test handling of missing data with fallbacks."""
        # Create sample data with missing actual times
        sample_data = [
            {
                'Flight Number': 'AI2509',
                'Unnamed: 2': '2025-07-25',
                'From': 'Mumbai (BOM)',
                'To': 'Delhi (DEL)',
                'Aircraft': 'A320',
                'STD': '06:00:00',
                'ATD': '',  # Missing actual departure
                'STA': '08:10:00',
                'ATA': ''   # Missing actual arrival
            }
        ]
        
        file_path = self.create_sample_excel_file('test_missing.xlsx', sample_data)
        
        try:
            result = self.service.ingest_excel_files([file_path])
            
            flight = result.flights[0]
            
            # Check that missing data is handled
            assert flight.departure.actual is not None  # Should be imputed
            assert flight.arrival.actual is not None    # Should be imputed
            assert flight.departure.actual_str == "IMPUTED_FROM_SCHEDULED"
            assert flight.arrival.actual_str == "IMPUTED_FROM_SCHEDULED"
            assert flight.dep_delay_min == 0  # No delay when using scheduled time
            assert flight.arr_delay_min == 0
            
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
    
    def test_aircraft_type_inference(self):
        """Test aircraft type inference from flight numbers."""
        # Test different airline codes
        test_cases = [
            ('6E123', 'A320'),  # IndiGo
            ('AI456', 'A320'),  # Air India
            ('SG789', 'B737'),  # SpiceJet
            ('UK101', 'A320'),  # Vistara
            ('G8202', 'A320'),  # GoAir
            ('XX999', 'UNKNOWN')  # Unknown airline
        ]
        
        for flight_number, expected_aircraft in test_cases:
            inferred = self.service._infer_aircraft_type(flight_number)
            assert inferred == expected_aircraft
    
    def test_file_not_found_handling(self):
        """Test handling of non-existent files."""
        non_existent_file = '/path/to/non/existent/file.xlsx'
        
        result = self.service.ingest_excel_files([non_existent_file])
        
        assert result.total_files_processed == 1
        assert result.successful_files == 0
        assert len(result.failed_files) == 1
        assert non_existent_file in result.failed_files
        assert len(result.errors) > 0
        assert "File not found" in result.errors[0]
    
    def test_ingestion_summary_generation(self):
        """Test generation of ingestion summary."""
        # Create sample data
        sample_data = [
            {
                'Flight Number': 'AI2509',
                'Unnamed: 2': '2025-07-25',
                'From': 'Mumbai (BOM)',
                'To': 'Delhi (DEL)',
                'Aircraft': 'A320',
                'STD': '06:00:00',
                'ATD': '06:20:00',
                'STA': '08:10:00',
                'ATA': 'Landed 8:30 AM'
            },
            {
                'Flight Number': '6E123',
                'Unnamed: 2': '2025-07-25',
                'From': 'Delhi (DEL)',
                'To': 'Mumbai (BOM)',
                'Aircraft': 'A320',
                'STD': '10:00:00',
                'ATD': '10:05:00',
                'STA': '12:30:00',
                'ATA': 'Landed 12:35 PM'
            }
        ]
        
        file_path = self.create_sample_excel_file('test_summary.xlsx', sample_data)
        
        try:
            result = self.service.ingest_excel_files([file_path])
            summary = self.service.get_ingestion_summary(result)
            
            # Check summary structure
            assert summary['status'] == 'success'
            assert 'processing_time_seconds' in summary
            assert 'files' in summary
            assert 'flights' in summary
            assert 'airlines' in summary
            assert 'top_routes' in summary
            assert 'aircraft_types' in summary
            assert 'date_range' in summary
            assert 'delay_analysis' in summary
            
            # Check specific values
            assert summary['files']['total_processed'] == 1
            assert summary['files']['successful'] == 1
            assert summary['flights']['valid'] >= 2
            assert 'AI' in summary['airlines']
            assert '6E' in summary['airlines']
            
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
    
    def test_validation_accuracy_check(self):
        """Test validation of ingestion accuracy."""
        # Create sample data with various completeness levels
        sample_data = [
            {
                'Flight Number': 'AI2509',
                'Unnamed: 2': '2025-07-25',
                'From': 'Mumbai (BOM)',
                'To': 'Delhi (DEL)',
                'Aircraft': 'A320',
                'STD': '06:00:00',
                'ATD': '06:20:00',
                'STA': '08:10:00',
                'ATA': 'Landed 8:30 AM'
            },
            {
                'Flight Number': '6E123',
                'Unnamed: 2': '2025-07-25',
                'From': 'Delhi (DEL)',
                'To': 'Mumbai (BOM)',
                'Aircraft': 'A320',
                'STD': '10:00:00',
                'ATD': '',  # Missing ATD
                'STA': '12:30:00',
                'ATA': ''   # Missing ATA
            }
        ]
        
        file_path = self.create_sample_excel_file('test_validation.xlsx', sample_data)
        
        try:
            result = self.service.ingest_excel_files([file_path])
            validation = self.service.validate_ingestion_accuracy(result)
            
            # Check validation structure
            assert 'data_completeness' in validation
            assert 'timestamp_validation' in validation
            assert 'delay_calculation_validation' in validation
            assert 'data_quality_issues' in validation
            
            # Check completeness rates (after imputation, all should be complete)
            completeness = validation['data_completeness']
            assert completeness['std_completeness'] == 1.0  # All flights have STD
            assert completeness['atd_completeness'] == 1.0   # All have ATD (after imputation)
            assert completeness['route_completeness'] == 1.0 # All have routes
            
            # Check that data quality issues are detected for imputed data
            assert len(validation['data_quality_issues']) > 0
            imputed_issues = [issue for issue in validation['data_quality_issues'] if 'Imputed' in issue]
            assert len(imputed_issues) > 0  # Should have imputation warnings
            
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
    
    def test_empty_file_handling(self):
        """Test handling of empty Excel files."""
        # Create empty DataFrame
        empty_data = []
        file_path = self.create_sample_excel_file('test_empty.xlsx', empty_data)
        
        try:
            result = self.service.ingest_excel_files([file_path])
            
            # Should handle gracefully
            assert result.total_files_processed == 1
            assert result.valid_flights == 0
            assert len(result.flights) == 0
            
            summary = self.service.get_ingestion_summary(result)
            assert summary['status'] == 'no_data'
            
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
    
    def test_malformed_data_handling(self):
        """Test handling of malformed Excel data."""
        # Create data with various malformed entries
        malformed_data = [
            {
                'Flight Number': 'AI2509',
                'Unnamed: 2': 'invalid_date',
                'From': '',  # Empty origin
                'To': 'Delhi (DEL)',
                'Aircraft': 'A320',
                'STD': 'invalid_time',
                'ATD': '25:99:00',  # Invalid time
                'STA': '08:10:00',
                'ATA': 'Not a valid ATA format'
            }
        ]
        
        file_path = self.create_sample_excel_file('test_malformed.xlsx', malformed_data)
        
        try:
            result = self.service.ingest_excel_files([file_path])
            
            # Should handle malformed data gracefully
            assert result.total_files_processed == 1
            # May have some flights but with data quality issues
            
            validation = self.service.validate_ingestion_accuracy(result)
            assert len(validation['data_quality_issues']) > 0
            
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)


@pytest.fixture
def sample_excel_data():
    """Fixture providing sample Excel data for tests."""
    return [
        {
            'Flight Number': 'AI2509',
            'Unnamed: 2': '2025-07-25',
            'From': 'Mumbai (BOM)',
            'To': 'Chandigarh (IXC)',
            'Aircraft': 'A20N (VT-EXU)',
            'Flight time': '1:54',
            'STD': '06:00:00',
            'ATD': '06:20:00',
            'STA': '08:10:00',
            'ATA': 'Landed 8:14 AM'
        },
        {
            'Flight Number': '',
            'Unnamed: 2': '2025-07-24',
            'From': 'Mumbai (BOM)',
            'To': 'Chandigarh (IXC)',
            'Aircraft': 'A20N (VT-RTJ)',
            'Flight time': '1:54',
            'STD': '06:00:00',
            'ATD': '06:07:00',
            'STA': '08:10:00',
            'ATA': 'Landed 8:01 AM'
        }
    ]


def test_integration_with_real_excel_file():
    """Integration test with the actual Excel file in the repository."""
    service = DataIngestionService()
    
    # Test with the actual file if it exists
    real_file_path = '429e6e3f-281d-4e4c-b00a-92fb020cb2fcFlight_Data.xlsx'
    
    if os.path.exists(real_file_path):
        result = service.ingest_excel_files([real_file_path])
        
        # Basic assertions for real data
        assert result.total_files_processed == 1
        assert result.successful_files == 1
        assert result.valid_flights > 0
        
        # Generate summary
        summary = service.get_ingestion_summary(result)
        assert summary['status'] == 'success'
        
        # Validate accuracy
        validation = service.validate_ingestion_accuracy(result)
        assert validation['data_completeness']['std_completeness'] > 0.5
        
        print(f"Processed {result.valid_flights} valid flights from real data")
        print(f"Airlines found: {list(summary['airlines'].keys())}")
        print(f"Top routes: {list(summary['top_routes'].keys())[:5]}")
    else:
        pytest.skip("Real Excel file not found for integration test")