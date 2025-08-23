"""Tests for FlightAware API ingestion service."""

import pytest
from datetime import date, datetime
from unittest.mock import Mock, patch, MagicMock
import requests

from src.services.flightaware_ingestion import (
    FlightAwareIngestionService, FlightAwareConfig
)
from src.models.flight import Flight


class TestFlightAwareConfig:
    """Test FlightAwareConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = FlightAwareConfig(api_key="test_key")
        
        assert config.api_key == "test_key"
        assert config.base_url == "https://aeroapi.flightaware.com/aeroapi"
        assert config.timeout == 30
        assert config.max_retries == 3
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = FlightAwareConfig(
            api_key="custom_key",
            base_url="https://custom.api.com",
            timeout=60,
            max_retries=5
        )
        
        assert config.api_key == "custom_key"
        assert config.base_url == "https://custom.api.com"
        assert config.timeout == 60
        assert config.max_retries == 5


class TestFlightAwareIngestionService:
    """Test FlightAwareIngestionService class."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return FlightAwareConfig(api_key="test_api_key")
    
    @pytest.fixture
    def service(self, config):
        """Create FlightAware ingestion service."""
        return FlightAwareIngestionService(config)
    
    def test_initialization(self, service):
        """Test service initialization."""
        assert service.config.api_key == "test_api_key"
        assert service.session.headers['x-apikey'] == "test_api_key"
        assert service.session.headers['Accept'] == "application/json"
    
    @patch('requests.Session.get')
    def test_make_api_request_success(self, mock_get, service):
        """Test successful API request."""
        # Mock successful response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"test": "data"}
        mock_get.return_value = mock_response
        
        result = service._make_api_request("http://test.com", {"param": "value"})
        
        assert result == {"test": "data"}
        mock_get.assert_called_once_with(
            "http://test.com",
            params={"param": "value"},
            timeout=30
        )
    
    @patch('requests.Session.get')
    def test_make_api_request_retry(self, mock_get, service):
        """Test API request with retry logic."""
        # Mock first request fails, second succeeds
        mock_response_fail = Mock()
        mock_response_fail.raise_for_status.side_effect = requests.exceptions.RequestException("Network error")
        
        mock_response_success = Mock()
        mock_response_success.raise_for_status.return_value = None
        mock_response_success.json.return_value = {"retry": "success"}
        
        mock_get.side_effect = [mock_response_fail, mock_response_success]
        
        with patch('time.sleep'):  # Mock sleep to speed up test
            result = service._make_api_request("http://test.com", {})
        
        assert result == {"retry": "success"}
        assert mock_get.call_count == 2
    
    @patch('requests.Session.get')
    def test_make_api_request_max_retries(self, mock_get, service):
        """Test API request exceeding max retries."""
        # Mock all requests fail
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.RequestException("Persistent error")
        mock_get.return_value = mock_response
        
        with patch('time.sleep'):  # Mock sleep to speed up test
            with pytest.raises(requests.exceptions.RequestException):
                service._make_api_request("http://test.com", {})
        
        assert mock_get.call_count == 3  # max_retries
    
    def test_parse_flightaware_departure(self, service):
        """Test parsing FlightAware departure data."""
        flight_data = {
            "ident": "AI123",
            "destination": {"code": "DEL"},
            "aircraft_type": "A320",
            "scheduled_departure": "2024-01-15T10:30:00Z",
            "actual_departure": "2024-01-15T10:45:00Z",
            "scheduled_arrival": "2024-01-15T12:30:00Z",
            "actual_arrival": "2024-01-15T12:50:00Z"
        }
        
        flight = service._parse_flightaware_departure(flight_data, "BOM")
        
        assert flight is not None
        assert flight.flight_number == "AI123"
        assert flight.data_source == "flightaware"
        assert flight.origin.code == "BOM"
        assert flight.destination.code == "DEL"
        assert flight.aircraft_type == "A320"
        assert flight.departure.scheduled.hour == 10
        assert flight.departure.scheduled.minute == 30
        assert flight.departure.actual.hour == 10
        assert flight.departure.actual.minute == 45
    
    def test_parse_flightaware_departure_missing_data(self, service):
        """Test parsing FlightAware departure data with missing fields."""
        flight_data = {
            "ident": "UK456",
            "destination": {"code": "CCU"}
        }
        
        flight = service._parse_flightaware_departure(flight_data, "BOM")
        
        assert flight is not None
        assert flight.flight_number == "UK456"
        assert flight.origin.code == "BOM"
        assert flight.destination.code == "CCU"
        assert flight.aircraft_type == ""  # Missing in data
        assert flight.departure.scheduled is None
    
    def test_parse_flightaware_departure_invalid_data(self, service):
        """Test parsing invalid FlightAware departure data."""
        flight_data = {}  # Missing required 'ident' field
        
        flight = service._parse_flightaware_departure(flight_data, "BOM")
        
        assert flight is None
    
    def test_parse_flightaware_arrival(self, service):
        """Test parsing FlightAware arrival data."""
        flight_data = {
            "ident": "6E789",
            "origin": {"code": "DEL"},
            "aircraft_type": "A320",
            "scheduled_departure": "2024-01-15T08:00:00Z",
            "actual_departure": "2024-01-15T08:15:00Z",
            "scheduled_arrival": "2024-01-15T10:00:00Z",
            "actual_arrival": "2024-01-15T10:20:00Z"
        }
        
        flight = service._parse_flightaware_arrival(flight_data, "BOM")
        
        assert flight is not None
        assert flight.flight_number == "6E789"
        assert flight.data_source == "flightaware"
        assert flight.origin.code == "DEL"
        assert flight.destination.code == "BOM"
        assert flight.aircraft_type == "A320"
    
    @patch.object(FlightAwareIngestionService, '_make_api_request')
    def test_get_airport_departures(self, mock_api_request, service):
        """Test getting airport departures."""
        # Mock API response
        mock_api_request.return_value = {
            "departures": [
                {
                    "ident": "AI123",
                    "destination": {"code": "DEL"},
                    "scheduled_departure": "2024-01-15T10:30:00Z"
                },
                {
                    "ident": "6E456",
                    "destination": {"code": "CCU"},
                    "scheduled_departure": "2024-01-15T11:00:00Z"
                }
            ]
        }
        
        start_date = date(2024, 1, 15)
        end_date = date(2024, 1, 15)
        
        batch = service._get_airport_departures("BOM", start_date, end_date)
        
        assert len(batch.flights) == 2
        assert batch.flights[0].flight_number == "AI123"
        assert batch.flights[1].flight_number == "6E456"
        assert batch.source_file == "FlightAware:Departures:BOM"
    
    @patch.object(FlightAwareIngestionService, '_make_api_request')
    def test_get_airport_arrivals(self, mock_api_request, service):
        """Test getting airport arrivals."""
        # Mock API response
        mock_api_request.return_value = {
            "arrivals": [
                {
                    "ident": "SG789",
                    "origin": {"code": "DEL"},
                    "scheduled_arrival": "2024-01-15T14:30:00Z"
                }
            ]
        }
        
        start_date = date(2024, 1, 15)
        end_date = date(2024, 1, 15)
        
        batch = service._get_airport_arrivals("BOM", start_date, end_date)
        
        assert len(batch.flights) == 1
        assert batch.flights[0].flight_number == "SG789"
        assert batch.source_file == "FlightAware:Arrivals:BOM"
    
    @patch.object(FlightAwareIngestionService, '_get_airport_departures')
    @patch.object(FlightAwareIngestionService, '_get_airport_arrivals')
    def test_ingest_airport_schedules(self, mock_arrivals, mock_departures, service):
        """Test ingesting airport schedules."""
        # Mock departure and arrival batches
        from src.models.flight import FlightDataBatch
        
        dep_batch = FlightDataBatch()
        dep_batch.add_flight(Flight(flight_number="AI123"))
        mock_departures.return_value = dep_batch
        
        arr_batch = FlightDataBatch()
        arr_batch.add_flight(Flight(flight_number="6E456"))
        mock_arrivals.return_value = arr_batch
        
        start_date = date(2024, 1, 15)
        end_date = date(2024, 1, 16)
        airport_codes = ["BOM", "DEL"]
        
        result = service.ingest_airport_schedules(airport_codes, start_date, end_date)
        
        assert result.total_files_processed == 2
        assert result.successful_files == 2
        assert len(result.failed_files) == 0
        assert result.total_flights == 4  # 2 airports × 2 flights each
    
    @patch.object(FlightAwareIngestionService, '_make_api_request')
    def test_test_connection_success(self, mock_api_request, service):
        """Test successful connection test."""
        mock_api_request.return_value = {"airport": "BOM", "name": "Mumbai"}
        
        result = service.test_connection()
        
        assert result["status"] == "success"
        assert "airport_info" in result
    
    @patch.object(FlightAwareIngestionService, '_make_api_request')
    def test_test_connection_failure(self, mock_api_request, service):
        """Test failed connection test."""
        mock_api_request.return_value = None
        
        result = service.test_connection()
        
        assert result["status"] == "error"
        assert "Failed to connect" in result["message"]
    
    @patch.object(FlightAwareIngestionService, '_make_api_request')
    def test_test_connection_exception(self, mock_api_request, service):
        """Test connection test with exception."""
        mock_api_request.side_effect = Exception("Network error")
        
        result = service.test_connection()
        
        assert result["status"] == "error"
        assert "Network error" in result["message"]


@pytest.mark.integration
class TestFlightAwareIntegration:
    """Integration tests for FlightAware service."""
    
    @pytest.mark.skip(reason="Requires real API key")
    def test_real_api_connection(self):
        """Test connection to real FlightAware API."""
        # This test requires a real API key and should be run manually
        import os
        api_key = os.getenv('FLIGHTAWARE_API_KEY')
        
        if not api_key:
            pytest.skip("FLIGHTAWARE_API_KEY environment variable not set")
        
        config = FlightAwareConfig(api_key=api_key)
        service = FlightAwareIngestionService(config)
        
        result = service.test_connection()
        assert result["status"] == "success"
    
    @pytest.mark.skip(reason="Requires real API key")
    def test_real_data_ingestion(self):
        """Test ingestion of real data from FlightAware API."""
        import os
        api_key = os.getenv('FLIGHTAWARE_API_KEY')
        
        if not api_key:
            pytest.skip("FLIGHTAWARE_API_KEY environment variable not set")
        
        config = FlightAwareConfig(api_key=api_key)
        service = FlightAwareIngestionService(config)
        
        # Test with a small date range
        start_date = date.today()
        end_date = start_date
        
        result = service.ingest_airport_schedules(["BOM"], start_date, end_date)
        
        assert result.total_files_processed > 0
        # Note: May have 0 flights if no flights scheduled for today