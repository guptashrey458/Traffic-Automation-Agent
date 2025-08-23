"""Test data models and validation."""

import pytest
from datetime import datetime, time, date
from src.models.flight import Flight, Airport, FlightTime, FlightStatus, FlightDataBatch
from src.models.validation import DataValidator


class TestAirport:
    """Test Airport model."""
    
    def test_from_string_with_code(self):
        """Test parsing airport string with code in parentheses."""
        airport = Airport.from_string("Mumbai (BOM)")
        assert airport.code == "BOM"
        assert airport.name == "Mumbai (BOM)"
        assert airport.city == "Mumbai"
    
    def test_from_string_without_code(self):
        """Test parsing airport string without code."""
        airport = Airport.from_string("Mumbai")
        assert airport.code == "MUM"
        assert airport.name == "Mumbai"
        assert airport.city == "Mumbai"
    
    def test_from_string_empty(self):
        """Test parsing empty airport string."""
        airport = Airport.from_string("")
        assert airport.code == "UNK"
        assert airport.name == "Unknown"
        assert airport.city == "Unknown"


class TestFlightTime:
    """Test FlightTime model."""
    
    def test_delay_calculation(self):
        """Test delay calculation."""
        scheduled = time(8, 0)  # 8:00 AM
        actual = datetime(2025, 7, 25, 8, 15)  # 8:15 AM
        
        flight_time = FlightTime(scheduled=scheduled, actual=actual)
        delay = flight_time.get_delay_minutes()
        
        assert delay == 15
    
    def test_delay_calculation_no_data(self):
        """Test delay calculation with missing data."""
        flight_time = FlightTime(scheduled=time(8, 0))
        delay = flight_time.get_delay_minutes()
        
        assert delay is None


class TestFlight:
    """Test Flight model."""
    
    def test_flight_creation(self):
        """Test basic flight creation."""
        origin = Airport.from_string("Mumbai (BOM)")
        destination = Airport.from_string("Delhi (DEL)")
        
        flight = Flight(
            flight_number="AI2509",
            origin=origin,
            destination=destination,
            departure=FlightTime(scheduled=time(6, 0)),
            arrival=FlightTime(scheduled=time(8, 10))
        )
        
        assert flight.flight_number == "AI2509"
        assert flight.airline_code == "AI"
        assert flight.origin.code == "BOM"
        assert flight.destination.code == "DEL"
        assert flight.is_valid()
    
    def test_route_key(self):
        """Test route key generation."""
        origin = Airport.from_string("Mumbai (BOM)")
        destination = Airport.from_string("Delhi (DEL)")
        
        flight = Flight(origin=origin, destination=destination)
        assert flight.get_route_key() == "BOM-DEL"
    
    def test_to_dict(self):
        """Test flight to dictionary conversion."""
        origin = Airport.from_string("Mumbai (BOM)")
        destination = Airport.from_string("Delhi (DEL)")
        
        flight = Flight(
            flight_number="AI2509",
            origin=origin,
            destination=destination,
            flight_date=date(2025, 7, 25),
            departure=FlightTime(scheduled=time(6, 0)),
            arrival=FlightTime(scheduled=time(8, 10))
        )
        
        flight_dict = flight.to_dict()
        
        assert flight_dict["flight_number"] == "AI2509"
        assert flight_dict["airline_code"] == "AI"
        assert flight_dict["origin_code"] == "BOM"
        assert flight_dict["destination_code"] == "DEL"
        assert flight_dict["route"] == "BOM-DEL"
        assert flight_dict["flight_date"] == "2025-07-25"


class TestDataValidator:
    """Test DataValidator utilities."""
    
    def test_normalize_flight_number(self):
        """Test flight number normalization."""
        assert DataValidator.normalize_flight_number("AI 2509") == "AI2509"
        assert DataValidator.normalize_flight_number("  ai2509  ") == "AI2509"
        assert DataValidator.normalize_flight_number("AI-2509") == "AI2509"
        assert DataValidator.normalize_flight_number("") == ""
    
    def test_parse_time_string(self):
        """Test time string parsing."""
        assert DataValidator.parse_time_string("06:00:00") == time(6, 0, 0)
        assert DataValidator.parse_time_string("06:00") == time(6, 0, 0)
        assert DataValidator.parse_time_string("0600") == time(6, 0, 0)
        assert DataValidator.parse_time_string("invalid") is None
        assert DataValidator.parse_time_string("") is None
    
    def test_parse_ata_string(self):
        """Test ATA string parsing."""
        flight_date = date(2025, 7, 25)
        result = DataValidator.parse_ata_string("Landed 8:14 AM", flight_date)
        
        assert result is not None
        assert result.hour == 8
        assert result.minute == 14
        assert result.date() == flight_date
        
        # Test PM time
        result = DataValidator.parse_ata_string("Landed 2:30 PM", flight_date)
        assert result.hour == 14
        assert result.minute == 30
        
        # Test invalid format
        assert DataValidator.parse_ata_string("Invalid format") is None
    
    def test_parse_date_string(self):
        """Test date string parsing."""
        assert DataValidator.parse_date_string("25 Jul 2025") == date(2025, 7, 25)
        assert DataValidator.parse_date_string("2025-07-25") == date(2025, 7, 25)
        assert DataValidator.parse_date_string("25/07/2025") == date(2025, 7, 25)
        assert DataValidator.parse_date_string("\n25 Jul 2025") == date(2025, 7, 25)
        assert DataValidator.parse_date_string("invalid") is None
    
    def test_normalize_aircraft_type(self):
        """Test aircraft type normalization."""
        assert DataValidator.normalize_aircraft_type("Airbus A320") == "A320"
        assert DataValidator.normalize_aircraft_type("Boeing 737-800") == "B738"
        assert DataValidator.normalize_aircraft_type("A320NEO") == "A320"
        assert DataValidator.normalize_aircraft_type("") == "UNKNOWN"
    
    def test_validate_flight_data(self):
        """Test flight data validation."""
        valid_data = {
            'flight_number': 'AI2509',
            'from': 'Mumbai (BOM)',
            'to': 'Delhi (DEL)',
            'STD': time(6, 0),
            'ATD': time(6, 15),
            'STA': time(8, 10)
        }
        
        is_valid, errors = DataValidator.validate_flight_data(valid_data)
        assert is_valid
        assert len(errors) == 0
        
        # Test invalid data
        invalid_data = {
            'flight_number': '',  # Missing flight number
            'from': '',           # Missing origin
            'to': 'Delhi (DEL)',
            'STD': 'invalid_time'  # Invalid time format
        }
        
        is_valid, errors = DataValidator.validate_flight_data(invalid_data)
        assert not is_valid
        assert len(errors) > 0
    
    def test_create_flight_from_raw_data(self):
        """Test creating flight from raw Excel data."""
        raw_data = {
            'Flight Number': 'AI2509',
            'From': 'Mumbai (BOM)',
            'To': 'Chandigarh (IXC)',
            'Aircraft': 'A320',
            'STD': time(6, 0),
            'ATD': time(6, 20),
            'STA': time(8, 10),
            'ATA': 'Landed 8:14 AM',
            'Date': '25 Jul 2025',
            'Flight time': '2h 10m'
        }
        
        flight = DataValidator.create_flight_from_raw_data(raw_data, "6AM - 9AM")
        
        assert flight is not None
        assert flight.flight_number == "AI2509"
        assert flight.airline_code == "AI"
        assert flight.origin.code == "BOM"
        assert flight.destination.code == "IXC"
        assert flight.aircraft_type == "A320"
        assert flight.time_period == "6AM - 9AM"
        assert flight.is_valid()


class TestFlightDataBatch:
    """Test FlightDataBatch model."""
    
    def test_batch_operations(self):
        """Test batch operations."""
        batch = FlightDataBatch(source_file="test.xlsx")
        
        # Add valid flight
        valid_flight = Flight(
            flight_number="AI2509",
            origin=Airport.from_string("Mumbai (BOM)"),
            destination=Airport.from_string("Delhi (DEL)"),
            departure=FlightTime(scheduled=time(6, 0))
        )
        batch.add_flight(valid_flight)
        
        # Add invalid flight
        invalid_flight = Flight()  # No required data
        batch.add_flight(invalid_flight)
        
        # Add error
        batch.add_error("Test error")
        
        # Test statistics
        stats = batch.get_stats()
        assert stats["total_flights"] == 2
        assert stats["valid_flights"] == 1
        assert stats["invalid_flights"] == 1
        assert stats["error_count"] == 1
        
        # Test valid flights retrieval
        valid_flights = batch.get_valid_flights()
        assert len(valid_flights) == 1
        assert valid_flights[0].flight_number == "AI2509"