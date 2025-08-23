"""Tests for FlightRadar24 HTML parser ingestion service."""

import pytest
import tempfile
import os
from datetime import date, time
from pathlib import Path

from src.services.flightradar24_ingestion import (
    FlightRadar24IngestionService, FlightRadar24Config
)
from src.models.flight import Flight


class TestFlightRadar24Config:
    """Test FlightRadar24Config class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = FlightRadar24Config(data_directory="/data/html")
        
        assert config.data_directory == "/data/html"
        assert config.airport_codes is None
        assert config.date_format == "%Y-%m-%d"
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = FlightRadar24Config(
            data_directory="/custom/path",
            airport_codes=["BOM", "DEL"],
            date_format="%d-%m-%Y"
        )
        
        assert config.data_directory == "/custom/path"
        assert config.airport_codes == ["BOM", "DEL"]
        assert config.date_format == "%d-%m-%Y"


class TestFlightRadar24IngestionService:
    """Test FlightRadar24IngestionService class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def config(self, temp_dir):
        """Create test configuration."""
        return FlightRadar24Config(data_directory=temp_dir)
    
    @pytest.fixture
    def service(self, config):
        """Create FlightRadar24 ingestion service."""
        return FlightRadar24IngestionService(config)
    
    def test_initialization(self, service):
        """Test service initialization."""
        assert service.config is not None
        assert service.ist_timezone is not None
        assert service.utc_timezone is not None
    
    def test_extract_metadata_from_filename(self, service):
        """Test extracting metadata from filename."""
        # Test various filename patterns
        test_cases = [
            ("BOM_2024-01-15_departures.html", "BOM", date(2024, 1, 15)),
            ("DEL_20240116_arrivals.html", "DEL", date(2024, 1, 16)),
            ("flightradar24_CCU_20240117.html", "CCU", date(2024, 1, 17)),
            ("mumbai_flights_BOM_2024-01-18.html", "BOM", date(2024, 1, 18)),
        ]
        
        for filename, expected_airport, expected_date in test_cases:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup("<html></html>", 'html.parser')
            
            airport, flight_date = service._extract_metadata(filename, soup)
            
            assert airport == expected_airport
            assert flight_date == expected_date
    
    def test_extract_metadata_from_html_content(self, service):
        """Test extracting metadata from HTML content."""
        html_content = """
        <html>
            <head><title>BOM Airport Flights - 2024-01-15</title></head>
            <body>
                <h1>Mumbai Airport Departures</h1>
                <p>Date: 2024-01-15</p>
            </body>
        </html>
        """
        
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        airport, flight_date = service._extract_metadata("unknown_file.html", soup)
        
        assert airport == "BOM"
        assert flight_date == date(2024, 1, 15)
    
    def test_parse_time_string(self, service):
        """Test parsing various time string formats."""
        test_cases = [
            ("14:30", time(14, 30)),
            ("2:30 PM", time(14, 30)),
            ("10:15 AM", time(10, 15)),
            ("23:45", time(23, 45)),
            ("invalid", None),
            ("", None)
        ]
        
        for time_str, expected in test_cases:
            result = service._parse_time_string(time_str)
            assert result == expected
    
    def test_parse_flight_table(self, service):
        """Test parsing flight data from HTML table."""
        html_content = """
        <html>
            <body>
                <table>
                    <tr>
                        <th>Flight</th>
                        <th>Route</th>
                        <th>Aircraft</th>
                        <th>Departure</th>
                        <th>Status</th>
                    </tr>
                    <tr>
                        <td>AI 123</td>
                        <td>DEL</td>
                        <td>A320</td>
                        <td>14:30</td>
                        <td>Departed</td>
                    </tr>
                    <tr>
                        <td>6E456</td>
                        <td>CCU</td>
                        <td>A321</td>
                        <td>15:45</td>
                        <td>Scheduled</td>
                    </tr>
                </table>
            </body>
        </html>
        """
        
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        flights = service._parse_flight_table(soup, "BOM", date(2024, 1, 15))
        
        assert len(flights) == 2
        
        # Check first flight
        assert flights[0].flight_number == "AI123"
        assert flights[0].aircraft_type == "A320"
        assert flights[0].departure.scheduled == time(14, 30)
        
        # Check second flight
        assert flights[1].flight_number == "6E456"
        assert flights[1].aircraft_type == "A321"
        assert flights[1].departure.scheduled == time(15, 45)
    
    def test_parse_flight_divs(self, service):
        """Test parsing flight data from div structures."""
        html_content = """
        <html>
            <body>
                <div class="flight-row">
                    <span>AI 789</span>
                    <span>BOM → DEL</span>
                    <span>16:20</span>
                    <span>18:30</span>
                </div>
                <div class="flight-item">
                    <p>Flight UK 234 from CCU at 12:15</p>
                </div>
            </body>
        </html>
        """
        
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        flights = service._parse_flight_divs(soup, "BOM", date(2024, 1, 15))
        
        assert len(flights) >= 1  # At least one flight should be parsed
        
        # Check that flight numbers are extracted
        flight_numbers = [f.flight_number for f in flights]
        assert "AI789" in flight_numbers or "UK234" in flight_numbers
    
    def test_process_html_file(self, service, temp_dir):
        """Test processing a complete HTML file."""
        # Create a test HTML file
        html_content = """
        <!DOCTYPE html>
        <html>
            <head><title>BOM Departures - 2024-01-15</title></head>
            <body>
                <h1>Mumbai Airport Departures</h1>
                <table>
                    <tr>
                        <th>Flight</th>
                        <th>Destination</th>
                        <th>Aircraft</th>
                        <th>STD</th>
                        <th>Status</th>
                    </tr>
                    <tr>
                        <td>AI 123</td>
                        <td>DEL</td>
                        <td>A320</td>
                        <td>10:30</td>
                        <td>Departed</td>
                    </tr>
                    <tr>
                        <td>6E 456</td>
                        <td>CCU</td>
                        <td>A321</td>
                        <td>11:45</td>
                        <td>Scheduled</td>
                    </tr>
                </table>
            </body>
        </html>
        """
        
        html_file = os.path.join(temp_dir, "BOM_2024-01-15_departures.html")
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        batch = service._process_html_file(html_file)
        
        assert len(batch.flights) == 2
        assert batch.source_file == html_file
        assert len(batch.errors) == 0
        
        # Verify flight data
        flights = batch.flights
        assert flights[0].flight_number == "AI123"
        assert flights[0].origin.code == "BOM"
        assert flights[0].departure.scheduled == time(10, 30)
        
        assert flights[1].flight_number == "6E456"
        assert flights[1].origin.code == "BOM"
        assert flights[1].departure.scheduled == time(11, 45)
    
    def test_ingest_html_files(self, service, temp_dir):
        """Test ingesting multiple HTML files."""
        # Create test HTML files
        html_files = []
        
        for i, airport in enumerate(["BOM", "DEL"]):
            html_content = f"""
            <html>
                <head><title>{airport} Flights</title></head>
                <body>
                    <table>
                        <tr><th>Flight</th><th>Route</th><th>Time</th></tr>
                        <tr><td>AI {100 + i}</td><td>CCU</td><td>10:30</td></tr>
                    </table>
                </body>
            </html>
            """
            
            html_file = os.path.join(temp_dir, f"{airport}_flights.html")
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            html_files.append(html_file)
        
        result = service.ingest_html_files(html_files)
        
        assert result.total_files_processed == 2
        assert result.successful_files == 2
        assert len(result.failed_files) == 0
        assert result.total_flights >= 2  # At least 2 flights
    
    def test_ingest_html_files_with_missing_file(self, service):
        """Test ingesting HTML files with missing file."""
        html_files = ["/nonexistent/file.html"]
        
        result = service.ingest_html_files(html_files)
        
        assert result.total_files_processed == 1
        assert result.successful_files == 0
        assert len(result.failed_files) == 1
        assert "File not found" in result.errors[0]
    
    def test_find_html_files(self, service, temp_dir):
        """Test finding HTML files by airport and date range."""
        # Create test files with various naming patterns
        test_files = [
            "BOM_2024-01-15_departures.html",
            "BOM_2024-01-16_arrivals.html",
            "DEL_2024-01-15_departures.html",
            "flightradar24_BOM_20240117.html",
            "other_file.txt"  # Should be ignored
        ]
        
        for filename in test_files:
            file_path = os.path.join(temp_dir, filename)
            with open(file_path, 'w') as f:
                f.write("<html></html>")
        
        # Update service config to use temp directory
        service.config.data_directory = temp_dir
        
        start_date = date(2024, 1, 15)
        end_date = date(2024, 1, 17)
        
        html_files = service._find_html_files("BOM", start_date, end_date)
        
        # Should find BOM files within date range
        assert len(html_files) >= 2  # At least the BOM files
        
        # Verify only HTML files are included
        for file_path in html_files:
            assert file_path.endswith('.html')
            assert 'BOM' in os.path.basename(file_path)
    
    def test_validate_html_structure(self, service, temp_dir):
        """Test HTML structure validation."""
        # Create a valid HTML file
        valid_html = """
        <html>
            <body>
                <table>
                    <tr><th>Flight</th><th>Route</th></tr>
                    <tr><td>AI 123</td><td>BOM-DEL</td></tr>
                </table>
            </body>
        </html>
        """
        
        valid_file = os.path.join(temp_dir, "valid.html")
        with open(valid_file, 'w', encoding='utf-8') as f:
            f.write(valid_html)
        
        result = service.validate_html_structure(valid_file)
        
        assert result["status"] == "valid"
        assert result["has_tables"] is True
        assert result["flight_numbers_found"] >= 1
        assert "AI123" in result["sample_flight_numbers"] or "AI 123" in result["sample_flight_numbers"]
    
    def test_validate_html_structure_invalid(self, service, temp_dir):
        """Test HTML structure validation with invalid file."""
        # Create an invalid HTML file (no flight data)
        invalid_html = """
        <html>
            <body>
                <p>This is just text with no flight data.</p>
            </body>
        </html>
        """
        
        invalid_file = os.path.join(temp_dir, "invalid.html")
        with open(invalid_file, 'w', encoding='utf-8') as f:
            f.write(invalid_html)
        
        result = service.validate_html_structure(invalid_file)
        
        assert result["status"] == "invalid"
        assert result["flight_numbers_found"] == 0
    
    def test_ingest_airport_directory(self, service, temp_dir):
        """Test ingesting all files for an airport directory."""
        # Create test files for BOM airport
        test_files = [
            ("BOM_2024-01-15_departures.html", "AI 123"),
            ("BOM_2024-01-16_arrivals.html", "6E 456"),
        ]
        
        for filename, flight_no in test_files:
            html_content = f"""
            <html>
                <body>
                    <table>
                        <tr><th>Flight</th><th>Route</th></tr>
                        <tr><td>{flight_no}</td><td>DEL</td></tr>
                    </table>
                </body>
            </html>
            """
            
            file_path = os.path.join(temp_dir, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
        
        # Update service config
        service.config.data_directory = temp_dir
        
        start_date = date(2024, 1, 15)
        end_date = date(2024, 1, 16)
        
        result = service.ingest_airport_directory("BOM", start_date, end_date)
        
        assert result.total_flights >= 2
        assert result.successful_files >= 2


@pytest.mark.integration
class TestFlightRadar24Integration:
    """Integration tests for FlightRadar24 service."""
    
    def test_real_html_file_parsing(self, tmp_path):
        """Test parsing a realistic HTML file structure."""
        # Create a more realistic HTML structure similar to FlightRadar24
        realistic_html = """
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
            
            <div class="flights-container">
                <div class="flight-row" data-flight="AI123">
                    <span class="flight-number">AI 123</span>
                    <span class="route">Mumbai (BOM) → Delhi (DEL)</span>
                    <span class="aircraft">A320 (VT-EXU)</span>
                    <span class="time">10:30</span>
                    <span class="status">Departed 10:45</span>
                </div>
                
                <div class="flight-row" data-flight="6E456">
                    <span class="flight-number">6E 456</span>
                    <span class="route">Mumbai (BOM) → Kolkata (CCU)</span>
                    <span class="aircraft">A321 (VT-IND)</span>
                    <span class="time">11:15</span>
                    <span class="status">Scheduled</span>
                </div>
                
                <div class="flight-row" data-flight="UK789">
                    <span class="flight-number">UK 789</span>
                    <span class="route">Mumbai (BOM) → Bangalore (BLR)</span>
                    <span class="aircraft">A320neo (VT-TNQ)</span>
                    <span class="time">12:00</span>
                    <span class="status">Delayed 12:30</span>
                </div>
            </div>
        </body>
        </html>
        """
        
        html_file = tmp_path / "BOM_2024-01-15_realistic.html"
        html_file.write_text(realistic_html, encoding='utf-8')
        
        config = FlightRadar24Config(data_directory=str(tmp_path))
        service = FlightRadar24IngestionService(config)
        
        result = service.ingest_html_files([str(html_file)])
        
        # Should successfully parse the realistic structure
        assert result.total_files_processed == 1
        assert result.successful_files == 1
        assert result.total_flights >= 3  # Should find at least 3 flights
        
        # Verify flight data quality
        flights = result.flights
        flight_numbers = [f.flight_number for f in flights]
        
        # Should extract flight numbers correctly
        expected_flights = ["AI123", "6E456", "UK789"]
        for expected in expected_flights:
            assert any(expected in fn for fn in flight_numbers), f"Flight {expected} not found in {flight_numbers}"