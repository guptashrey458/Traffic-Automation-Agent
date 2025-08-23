"""Data validation and normalization utilities."""

import re
from datetime import datetime, time, date
from typing import Optional, Dict, Any, List, Tuple
import pytz
from .flight import Flight, Airport, FlightTime, FlightStatus


class DataValidator:
    """Data validation and normalization utilities."""
    
    # Common airline codes mapping
    AIRLINE_CODES = {
        "AI": "Air India",
        "6E": "IndiGo", 
        "SG": "SpiceJet",
        "UK": "Vistara",
        "G8": "GoAir",
        "I5": "AirAsia India",
        "9W": "Jet Airways",
    }
    
    # Common airport codes mapping
    AIRPORT_CODES = {
        "BOM": "Mumbai",
        "DEL": "Delhi", 
        "BLR": "Bangalore",
        "MAA": "Chennai",
        "CCU": "Kolkata",
        "HYD": "Hyderabad",
        "PNQ": "Pune",
        "GOI": "Goa",
        "IXC": "Chandigarh",
        "AMD": "Ahmedabad",
        "COK": "Kochi",
        "TRV": "Trivandrum",
        "JAI": "Jaipur",
        "IXB": "Bagdogra",
        "GAU": "Guwahati",
    }
    
    @staticmethod
    def normalize_flight_number(flight_number: str) -> str:
        """Normalize flight number format."""
        if not flight_number:
            return ""
        
        # Remove extra whitespace and convert to uppercase
        normalized = flight_number.strip().upper()
        
        # Remove any non-alphanumeric characters except spaces
        normalized = re.sub(r'[^A-Z0-9\s]', '', normalized)
        
        # Remove spaces
        normalized = normalized.replace(' ', '')
        
        return normalized
    
    @staticmethod
    def normalize_airport_string(airport_str: str) -> str:
        """Normalize airport string format."""
        if not airport_str:
            return ""
        
        # Clean up the string
        normalized = airport_str.strip()
        
        # Handle common variations
        normalized = normalized.replace("(", " (").replace(")", ") ")
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    @staticmethod
    def parse_time_string(time_str: str) -> Optional[time]:
        """Parse various time string formats to time object."""
        if not time_str or str(time_str).strip() == "":
            return None
        
        time_str = str(time_str).strip()
        
        # Handle different time formats
        time_patterns = [
            r'(\d{1,2}):(\d{2}):(\d{2})',  # HH:MM:SS
            r'(\d{1,2}):(\d{2})',          # HH:MM
            r'(\d{1,2})(\d{2})',           # HHMM
        ]
        
        for pattern in time_patterns:
            match = re.match(pattern, time_str)
            if match:
                groups = match.groups()
                hour = int(groups[0])
                minute = int(groups[1])
                second = int(groups[2]) if len(groups) > 2 else 0
                
                # Validate time components
                if 0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59:
                    return time(hour, minute, second)
        
        return None
    
    @staticmethod
    def parse_ata_string(ata_str: str, flight_date: Optional[date] = None) -> Optional[datetime]:
        """Parse ATA string like 'Landed 8:14 AM' to datetime."""
        if not ata_str or str(ata_str).strip() == "":
            return None
        
        ata_str = str(ata_str).strip()
        
        # Pattern for "Landed HH:MM AM/PM"
        pattern = r'Landed\s+(\d{1,2}):(\d{2})\s+(AM|PM)'
        match = re.search(pattern, ata_str, re.IGNORECASE)
        
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2))
            am_pm = match.group(3).upper()
            
            # Convert to 24-hour format
            if am_pm == "PM" and hour != 12:
                hour += 12
            elif am_pm == "AM" and hour == 12:
                hour = 0
            
            # Use provided date or current date
            if flight_date:
                return datetime.combine(flight_date, time(hour, minute))
            else:
                return datetime.combine(date.today(), time(hour, minute))
        
        return None
    
    @staticmethod
    def parse_date_string(date_str: str) -> Optional[date]:
        """Parse various date string formats."""
        if not date_str or str(date_str).strip() == "":
            return None
        
        date_str = str(date_str).strip()
        
        # Remove newlines and extra whitespace
        date_str = re.sub(r'\s+', ' ', date_str.replace('\n', ' '))
        
        # Common date patterns
        date_patterns = [
            r'(\d{1,2})\s+(\w{3})\s+(\d{4})',      # "25 Jul 2025"
            r'(\d{4})-(\d{1,2})-(\d{1,2})',        # "2025-07-25"
            r'(\d{1,2})/(\d{1,2})/(\d{4})',        # "25/07/2025"
            r'(\d{1,2})-(\d{1,2})-(\d{4})',        # "25-07-2025"
        ]
        
        month_names = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }
        
        for pattern in date_patterns:
            match = re.search(pattern, date_str, re.IGNORECASE)
            if match:
                groups = match.groups()
                
                if len(groups) == 3:
                    if pattern.startswith(r'(\d{1,2})\s+(\w{3})'):  # "25 Jul 2025"
                        day = int(groups[0])
                        month = month_names.get(groups[1].lower(), 0)
                        year = int(groups[2])
                    elif pattern.startswith(r'(\d{4})'):  # "2025-07-25"
                        year = int(groups[0])
                        month = int(groups[1])
                        day = int(groups[2])
                    else:  # "25/07/2025" or "25-07-2025"
                        day = int(groups[0])
                        month = int(groups[1])
                        year = int(groups[2])
                    
                    try:
                        return date(year, month, day)
                    except ValueError:
                        continue
        
        return None
    
    @staticmethod
    def convert_ist_to_utc(dt: datetime) -> datetime:
        """Convert IST datetime to UTC."""
        ist = pytz.timezone('Asia/Kolkata')
        utc = pytz.UTC
        
        # Localize to IST if naive
        if dt.tzinfo is None:
            dt = ist.localize(dt)
        
        # Convert to UTC
        return dt.astimezone(utc)
    
    @staticmethod
    def validate_flight_data(flight_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate flight data dictionary and return validation results."""
        errors = []
        
        # Required fields
        required_fields = ['flight_number', 'from', 'to']
        for field in required_fields:
            if not flight_data.get(field):
                errors.append(f"Missing required field: {field}")
        
        # Flight number format
        flight_number = flight_data.get('flight_number', '')
        if flight_number and not re.match(r'^[A-Z0-9]{2,8}$', flight_number.upper().replace(' ', '')):
            errors.append(f"Invalid flight number format: {flight_number}")
        
        # Time validation
        time_fields = ['STD', 'ATD', 'STA']
        for field in time_fields:
            time_val = flight_data.get(field)
            if time_val and not isinstance(time_val, time):
                parsed_time = DataValidator.parse_time_string(str(time_val))
                if parsed_time is None:
                    errors.append(f"Invalid time format for {field}: {time_val}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def normalize_aircraft_type(aircraft_str: str) -> str:
        """Normalize aircraft type string."""
        if not aircraft_str:
            return "UNKNOWN"
        
        aircraft_str = aircraft_str.strip().upper()
        
        # Common aircraft type mappings
        aircraft_mappings = {
            "AIRBUS A320": "A320",
            "AIRBUS A321": "A321", 
            "BOEING 737": "B737",
            "BOEING 737-800": "B738",
            "BOEING 777": "B777",
            "BOEING 787": "B787",
            "A320NEO": "A320",
            "A321NEO": "A321",
            "A20N": "A320",  # A320neo variant
            "A21N": "A321",  # A321neo variant
            "B737-800": "B738",
            "B737-900": "B739",
        }
        
        # Check for exact matches first
        if aircraft_str in aircraft_mappings:
            return aircraft_mappings[aircraft_str]
        
        # Check for partial matches
        for key, value in aircraft_mappings.items():
            if key in aircraft_str or aircraft_str in key:
                return value
        
        # Extract common patterns
        if "A20N" in aircraft_str or "A320NEO" in aircraft_str:
            return "A320"
        elif "A21N" in aircraft_str or "A321NEO" in aircraft_str:
            return "A321"
        elif "A320" in aircraft_str:
            return "A320"
        elif "A321" in aircraft_str:
            return "A321"
        elif "737" in aircraft_str:
            return "B737"
        elif "777" in aircraft_str:
            return "B777"
        elif "787" in aircraft_str:
            return "B787"
        
        return aircraft_str[:10]  # Truncate long strings
    
    @staticmethod
    def create_flight_from_raw_data(raw_data: Dict[str, Any], time_period: str = "") -> Optional[Flight]:
        """Create a Flight object from raw Excel data."""
        try:
            # Normalize flight number
            flight_number = DataValidator.normalize_flight_number(
                str(raw_data.get('Flight Number', ''))
            )
            
            if not flight_number:
                return None
            
            # Parse airports
            origin_str = DataValidator.normalize_airport_string(
                str(raw_data.get('From', ''))
            )
            destination_str = DataValidator.normalize_airport_string(
                str(raw_data.get('To', ''))
            )
            
            origin = Airport.from_string(origin_str) if origin_str else None
            destination = Airport.from_string(destination_str) if destination_str else None
            
            # Parse times
            std = DataValidator.parse_time_string(str(raw_data.get('STD', '')))
            atd = DataValidator.parse_time_string(str(raw_data.get('ATD', '')))
            sta = DataValidator.parse_time_string(str(raw_data.get('STA', '')))
            
            # Parse date
            flight_date = None
            date_fields = ['Date', 'Unnamed: 2']
            for field in date_fields:
                if field in raw_data and raw_data[field]:
                    flight_date = DataValidator.parse_date_string(str(raw_data[field]))
                    if flight_date:
                        break
            
            # Parse ATA
            ata_str = str(raw_data.get('ATA', ''))
            ata = DataValidator.parse_ata_string(ata_str, flight_date)
            
            # Parse ATD to datetime if available
            atd_datetime = None
            if atd and flight_date:
                atd_datetime = datetime.combine(flight_date, atd)
            
            # Create flight times
            departure = FlightTime(
                scheduled=std,
                actual=atd_datetime,
                actual_str=str(raw_data.get('ATD', '')) if raw_data.get('ATD') else None
            )
            
            arrival = FlightTime(
                scheduled=sta,
                actual=ata,
                actual_str=ata_str
            )
            
            # Normalize aircraft type
            aircraft_type = DataValidator.normalize_aircraft_type(
                str(raw_data.get('Aircraft', ''))
            )
            
            # Create flight object
            flight = Flight(
                flight_number=flight_number,
                origin=origin,
                destination=destination,
                aircraft_type=aircraft_type,
                flight_date=flight_date,
                departure=departure,
                arrival=arrival,
                flight_duration=str(raw_data.get('Flight time', '')),
                time_period=time_period,
                raw_data=raw_data
            )
            
            return flight
            
        except Exception as e:
            # Log error but don't raise - return None for invalid data
            return None