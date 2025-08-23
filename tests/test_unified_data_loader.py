"""Tests for unified data loader with multiple sources."""

import pytest
import os
import tempfile
from datetime import date, datetime, time
from unittest.mock import Mock, patch, MagicMock
import json

from src.services.unified_data_loader import (
    UnifiedDataLoader, UnifiedDataConfig, DataSource
)
from src.services.data_ingestion import IngestionResult
from src.models.flight import Flight, FlightTime, Airport


class TestUnifiedDataConfig:
    """Test UnifiedDataConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = UnifiedDataConfig()
        
        assert config.airport_codes == ["BOM", "DEL"]
        assert config.excel_file_patterns == ["*.xlsx", "*.xls", "*Flight_Data*"]
        assert config.default_timezone == "Asia/Kolkata"
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = UnifiedDataConfig(
            excel_directory="/data/excel",
            flightaware_api_key="test_key",
            fr24_html_directory="/data/html",
            airport_codes=["BOM", "DEL", "CCU"]
        )
        
        assert config.excel_directory == "/data/excel"
        assert config.flightaware_api_key == "test_key"
        assert config.fr24_html_directory == "/data/html"
        assert config.airport_codes == ["BOM", "DEL", "CCU"]


class TestUnifiedDataLoader:
    """Test UnifiedDataLoader class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def sample_config(self, temp_dir):
        """Create sample configuration for testing."""
        return UnifiedDataConfig(
            excel_directory=temp_dir,
            flightaware_api_key="test_api_key",
            fr24_html_directory=temp_dir,
            airport_codes=["BOM", "DEL"]
        )
    
    @pytest.fixture
    def loader(self, sample_config):
        """Create UnifiedDataLoader instance."""
        return UnifiedDataLoader(sample_config)
    
    def test_initialization(self, loader):
        """Test loader initialization."""
        assert loader.excel_service is not None
        assert loader.flightaware_service is not None  # API key provided
        assert loader.fr24_service is not None  # HTML directory provided
    
    def test_initialization_without_api_key(self, temp_dir):
        """Test initialization without FlightAware API key."""
        config = UnifiedDataConfig(
            excel_directory=temp_dir,
            fr24_html_directory=temp_dir
        )
        loader = UnifiedDataLoader(config)
        
        assert loader.excel_service is not None
        assert loader.flightaware_service is None  # No API key
        assert loader.fr24_service is not None
    
    def test_load_excel_data(self, loader):
        """Test loading Excel data."""
        # Mock the excel service method directly
        mock_result = IngestionResult()
        mock_result.valid_flights = 10
        
        with patch.object(loader.excel_service, 'ingest_excel_files', return_value=mock_result):
            # Test with specific file paths
            file_paths = ["/path/to/file1.xlsx", "/path/to/file2.xlsx"]
            result = loader.load_data(DataSource.EXCEL, file_paths=file_paths)
            
            assert result.valid_flights == 10
    
    def test_load_flightaware_data(self, loader):
        """Test loading FlightAware data."""
        # Mock FlightAware service method directly
        mock_result = IngestionResult()
        mock_result.valid_flights = 15
        
        with patch.object(loader.flightaware_service, 'ingest_airport_schedules', return_value=mock_result):
            start_date = date(2024, 1, 15)
            end_date = date(2024, 1, 21)
            
            result = loader.load_data(
                DataSource.FLIGHTAWARE,
                start_date=start_date,
                end_date=end_date
            )
            
            assert result.valid_flights == 15
    
    def test_load_flightaware_without_service(self, temp_dir):
        """Test loading FlightAware data without configured service."""
        config = UnifiedDataConfig(excel_directory=temp_dir)
        loader = UnifiedDataLoader(config)
        
        with pytest.raises(ValueError, match="FlightAware API key not configured"):
            loader.load_data(DataSource.FLIGHTAWARE)
    
    def test_load_flightaware_without_dates(self, loader):
        """Test loading FlightAware data without required dates."""
        with pytest.raises(ValueError, match="Start and end dates required"):
            loader.load_data(DataSource.FLIGHTAWARE)
    
    def test_load_fr24_data(self, loader):
        """Test loading FlightRadar24 data."""
        # Mock FR24 service method directly
        mock_result = IngestionResult()
        mock_result.valid_flights = 8
        
        with patch.object(loader.fr24_service, 'ingest_html_files', return_value=mock_result):
            file_paths = ["/path/to/file1.html", "/path/to/file2.html"]
            result = loader.load_data(DataSource.FLIGHTRADAR24, file_paths=file_paths)
            
            assert result.valid_flights == 8
    
    def test_load_unsupported_source(self, loader):
        """Test loading from unsupported source."""
        with pytest.raises(ValueError, match="Unsupported data source"):
            # This would require creating a new enum value, so we'll mock it
            loader.load_data("unsupported_source")
    
    def test_normalize_data_schema(self, loader):
        """Test data schema normalization."""
        # Create sample flights with various data quality issues
        flights = [
            Flight(
                flight_number="AI 123",  # Has space
                airline_code="",  # Missing airline code
                aircraft_type="",  # Missing aircraft type
                origin=Airport(code="BOM", name="Mumbai", city="Mumbai"),
                destination=Airport(code="DEL", name="Delhi", city="Delhi"),
                flight_date=date(2024, 1, 15),
                departure=FlightTime(
                    scheduled=time(10, 30),
                    actual=datetime(2024, 1, 15, 10, 45)
                )
            ),
            Flight(
                flight_number="6E2345",  # No space
                origin=Airport(code="", name="Unknown", city="Unknown"),  # Missing code
                destination=Airport(code="DEL", name="Delhi", city="Delhi")
            )
        ]
        
        normalized = loader.normalize_data_schema(flights)
        
        # Check first flight
        assert normalized[0].flight_number == "AI123"  # Space removed
        assert normalized[0].airline_code == "AI"  # Extracted from flight number
        assert normalized[0].aircraft_type == "A320"  # Inferred from airline
        assert normalized[0].dep_delay_min == 15  # Calculated delay
        
        # Check second flight
        assert normalized[1].flight_number == "6E2345"
        assert normalized[1].airline_code == "6E"
        assert normalized[1].origin.code == "UNK"  # Fixed missing code
    
    def test_validate_data_quality(self, loader):
        """Test data quality validation."""
        # Create sample result with mixed quality data
        result = IngestionResult()
        
        # Good quality flight
        good_flight = Flight(
            flight_number="AI123",
            airline_code="AI",
            aircraft_type="A320",
            origin=Airport(code="BOM", name="Mumbai", city="Mumbai"),
            destination=Airport(code="DEL", name="Delhi", city="Delhi"),
            departure=FlightTime(scheduled=time(10, 30)),
            arrival=FlightTime(scheduled=time(12, 30)),
            data_source="excel"
        )
        
        # Poor quality flight
        poor_flight = Flight(
            flight_number="UK456",
            aircraft_type="UNKNOWN",
            data_source="flightaware"
        )
        
        result.flights = [good_flight, poor_flight]
        
        quality_report = loader.validate_data_quality(result)
        
        assert quality_report["status"] == "success"
        assert quality_report["total_flights"] == 2
        assert quality_report["completeness_score"] < 1.0  # Not perfect due to poor flight
        assert "excel" in quality_report["source_distribution"]
        assert "flightaware" in quality_report["source_distribution"]
    
    def test_validate_data_quality_no_data(self, loader):
        """Test data quality validation with no data."""
        result = IngestionResult()
        
        quality_report = loader.validate_data_quality(result)
        
        assert quality_report["status"] == "no_data"
        assert "recommendations" in quality_report
    
    def test_discover_excel_files(self, loader):
        """Test Excel file discovery."""
        from pathlib import Path
        
        # Mock Path class and its methods
        with patch('pathlib.Path') as mock_path_class:
            mock_path_instance = Mock()
            mock_path_class.return_value = mock_path_instance
            mock_path_instance.exists.return_value = True
            
            # Mock glob to return sample files
            mock_file1 = Mock()
            mock_file1.is_file.return_value = True
            mock_file1.__str__ = lambda: "/data/file1.xlsx"
            
            mock_file2 = Mock()
            mock_file2.is_file.return_value = True
            mock_file2.__str__ = lambda: "/data/file2.xlsx"
            
            mock_path_instance.glob.return_value = [mock_file1, mock_file2]
            
            files = loader._discover_excel_files()
            
            assert len(files) == 2
            assert "/data/file1.xlsx" in files
            assert "/data/file2.xlsx" in files
    
    def test_discover_html_files(self, loader):
        """Test HTML file discovery."""
        from pathlib import Path
        
        # Mock Path class and its methods
        with patch('pathlib.Path') as mock_path_class:
            mock_path_instance = Mock()
            mock_path_class.return_value = mock_path_instance
            mock_path_instance.exists.return_value = True
            
            # Mock glob to return sample files
            mock_file1 = Mock()
            mock_file1.is_file.return_value = True
            mock_file1.__str__ = lambda: "/data/file1.html"
            
            mock_file2 = Mock()
            mock_file2.is_file.return_value = True
            mock_file2.__str__ = lambda: "/data/file2.html"
            
            mock_path_instance.glob.return_value = [mock_file1, mock_file2]
            
            files = loader._discover_html_files()
            
            assert len(files) == 2
            assert "/data/file1.html" in files
            assert "/data/file2.html" in files
    
    def test_get_available_sources(self, loader):
        """Test getting available sources information."""
        with patch.object(loader, '_discover_excel_files', return_value=["/data/test.xlsx"]):
            with patch.object(loader, '_discover_html_files', return_value=["/data/test.html"]):
                sources = loader.get_available_sources()
        
        assert "excel" in sources
        assert "flightaware" in sources
        assert "flightradar24" in sources
        
        assert sources["excel"]["available"] is True
        assert sources["excel"]["file_count"] == 1
        
        assert sources["flightaware"]["available"] is True
        assert sources["flightaware"]["api_configured"] is True
        
        assert sources["flightradar24"]["available"] is True
        assert sources["flightradar24"]["file_count"] == 1
    
    def test_infer_aircraft_type(self, loader):
        """Test aircraft type inference."""
        assert loader._infer_aircraft_type("AI123") == "A320"
        assert loader._infer_aircraft_type("6E456") == "A320"
        assert loader._infer_aircraft_type("SG789") == "B737"
        assert loader._infer_aircraft_type("UK012") == "A320"
        assert loader._infer_aircraft_type("G8345") == "A320"
        assert loader._infer_aircraft_type("XX999") == "UNKNOWN"
        assert loader._infer_aircraft_type("") == "UNKNOWN"


class TestDataSourceEnum:
    """Test DataSource enum."""
    
    def test_enum_values(self):
        """Test enum values."""
        assert DataSource.EXCEL.value == "excel"
        assert DataSource.FLIGHTAWARE.value == "flightaware"
        assert DataSource.FLIGHTRADAR24.value == "flightradar24"
        assert DataSource.AUTO.value == "auto"


@pytest.mark.integration
class TestUnifiedDataLoaderIntegration:
    """Integration tests for UnifiedDataLoader."""
    
    def test_auto_detection_with_excel_data(self):
        """Test auto-detection with Excel data available."""
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a mock Excel file
            excel_file = os.path.join(temp_dir, "test_flight_data.xlsx")
            with open(excel_file, 'w') as f:
                f.write("mock excel content")
            
            config = UnifiedDataConfig(excel_directory=temp_dir)
            loader = UnifiedDataLoader(config)
            
            # Mock the Excel service to return some data
            with patch.object(loader.excel_service, 'ingest_excel_files') as mock_ingest:
                mock_result = IngestionResult()
                mock_result.valid_flights = 5
                mock_ingest.return_value = mock_result
                
                result = loader.load_data(DataSource.AUTO)
                
                assert result.valid_flights == 5
    
    def test_schema_compliance(self):
        """Test that normalized data complies with existing pipeline schema."""
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config = UnifiedDataConfig(excel_directory=temp_dir)
            loader = UnifiedDataLoader(config)
            
            # Create flights with various data sources
            flights = [
                Flight(
                    flight_number="AI123",
                    data_source="excel",
                    origin=Airport(code="BOM", name="Mumbai", city="Mumbai"),
                    destination=Airport(code="DEL", name="Delhi", city="Delhi"),
                    flight_date=date(2024, 1, 15),
                    departure=FlightTime(
                        scheduled=time(10, 30),
                        actual=datetime(2024, 1, 15, 10, 45)
                    )
                ),
                Flight(
                    flight_number="6E456",
                    data_source="flightaware",
                    origin=Airport(code="DEL", name="Delhi", city="Delhi"),
                    destination=Airport(code="BOM", name="Mumbai", city="Mumbai")
                )
            ]
            
            normalized = loader.normalize_data_schema(flights)
            
            # Verify schema compliance
            for flight in normalized:
                # Required fields for existing pipeline
                assert flight.flight_number  # flight_no in schema
                assert flight.origin and flight.origin.code  # origin airport
                assert flight.destination and flight.destination.code  # destination airport
                assert flight.airline_code  # extracted from flight number
                assert flight.aircraft_type  # inferred or set
                
                # Verify flight.to_dict() works (used by existing services)
                flight_dict = flight.to_dict()
                assert "flight_id" in flight_dict
                assert "flight_number" in flight_dict
                assert "origin_code" in flight_dict
                assert "destination_code" in flight_dict