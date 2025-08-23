"""Flight data models and validation."""

from dataclasses import dataclass, field
from datetime import datetime, time, date
from typing import Optional, List, Dict, Any
from enum import Enum
import uuid


class FlightStatus(Enum):
    """Flight status enumeration."""
    SCHEDULED = "scheduled"
    DEPARTED = "departed"
    ARRIVED = "arrived"
    CANCELLED = "cancelled"
    DELAYED = "delayed"


class AircraftType(Enum):
    """Common aircraft types."""
    A320 = "A320"
    A321 = "A321"
    B737 = "B737"
    B738 = "B738"
    B777 = "B777"
    B787 = "B787"
    OTHER = "OTHER"


@dataclass
class Airport:
    """Airport information."""
    code: str  # IATA code like "BOM", "DEL"
    name: str  # Full name like "Mumbai (BOM)"
    city: str  # City name
    timezone: str = "Asia/Kolkata"  # Default to IST
    
    @classmethod
    def from_string(cls, airport_str: str) -> "Airport":
        """Parse airport from string like 'Mumbai (BOM)'."""
        if not airport_str or airport_str.strip() == "":
            return cls(code="UNK", name="Unknown", city="Unknown")
        
        # Extract code from parentheses
        if "(" in airport_str and ")" in airport_str:
            name = airport_str.strip()
            code_start = airport_str.rfind("(") + 1
            code_end = airport_str.rfind(")")
            code = airport_str[code_start:code_end].strip()
            city = airport_str[:airport_str.rfind("(")].strip()
        else:
            # Fallback if format is different
            name = airport_str.strip()
            code = airport_str.strip()[:3].upper()
            city = airport_str.strip()
        
        return cls(code=code, name=name, city=city)


@dataclass
class FlightTime:
    """Flight time information with validation."""
    scheduled: Optional[time] = None
    actual: Optional[datetime] = None
    actual_str: Optional[str] = None  # For parsing "Landed 8:14 AM" format
    
    def get_delay_minutes(self) -> Optional[int]:
        """Calculate delay in minutes if both scheduled and actual are available."""
        if not self.scheduled or not self.actual:
            return None
        
        # Convert scheduled time to datetime for comparison
        # Assuming same date as actual for delay calculation
        scheduled_dt = datetime.combine(self.actual.date(), self.scheduled)
        
        # Handle day rollover
        if scheduled_dt > self.actual:
            # If scheduled time is later than actual, it might be next day
            from datetime import timedelta
            scheduled_dt -= timedelta(days=1)
        
        delay = (self.actual - scheduled_dt).total_seconds() / 60
        return int(delay)


@dataclass
class Flight:
    """Core flight data model."""
    
    # Identifiers
    flight_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    flight_number: str = ""
    airline_code: str = ""
    
    # Route information
    origin: Optional[Airport] = None
    destination: Optional[Airport] = None
    
    # Aircraft information
    aircraft_type: str = ""
    aircraft_registration: str = ""
    
    # Timing information
    flight_date: Optional[date] = None
    departure: FlightTime = field(default_factory=FlightTime)
    arrival: FlightTime = field(default_factory=FlightTime)
    
    # Operational data
    flight_duration: Optional[str] = None  # "2h 10m" format
    status: FlightStatus = FlightStatus.SCHEDULED
    
    # Calculated fields
    dep_delay_min: Optional[int] = None
    arr_delay_min: Optional[int] = None
    
    # Metadata
    data_source: str = "excel"
    time_period: str = ""  # "6AM - 9AM", "9AM - 12PM", etc.
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization validation and calculations."""
        # Extract airline code from flight number
        if self.flight_number and not self.airline_code:
            # Extract first 2-3 letters from flight number (e.g., "AI2509" -> "AI", "6E123" -> "6E")
            airline_code = ""
            for char in self.flight_number:
                if char.isalpha() or char.isdigit():
                    airline_code += char
                    if len(airline_code) >= 2 and not char.isalpha():
                        break
                else:
                    break
            # Take first 2 characters for standard airline codes
            self.airline_code = airline_code[:2] if len(airline_code) >= 2 else airline_code
        
        # Calculate delays
        if self.departure.scheduled and self.departure.actual:
            self.dep_delay_min = self.departure.get_delay_minutes()
        
        if self.arrival.scheduled and self.arrival.actual:
            self.arr_delay_min = self.arrival.get_delay_minutes()
        
        # Determine status based on available data
        if self.arrival.actual or self.arrival.actual_str:
            self.status = FlightStatus.ARRIVED
        elif self.departure.actual:
            self.status = FlightStatus.DEPARTED
        elif self.dep_delay_min and self.dep_delay_min > 15:
            self.status = FlightStatus.DELAYED
    
    def is_valid(self) -> bool:
        """Check if flight data is valid for analysis."""
        return (
            bool(self.flight_number) and
            self.origin is not None and
            self.destination is not None and
            self.departure.scheduled is not None
        )
    
    def get_route_key(self) -> str:
        """Get route identifier for grouping."""
        if not self.origin or not self.destination:
            return "UNKNOWN-UNKNOWN"
        return f"{self.origin.code}-{self.destination.code}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert flight to dictionary for storage/API."""
        return {
            "flight_id": self.flight_id,
            "flight_number": self.flight_number,
            "airline_code": self.airline_code,
            "origin_code": self.origin.code if self.origin else None,
            "origin_name": self.origin.name if self.origin else None,
            "destination_code": self.destination.code if self.destination else None,
            "destination_name": self.destination.name if self.destination else None,
            "aircraft_type": self.aircraft_type,
            "flight_date": self.flight_date.isoformat() if self.flight_date else None,
            "std": self.departure.scheduled.isoformat() if self.departure.scheduled else None,
            "atd": self.departure.actual.isoformat() if self.departure.actual else None,
            "sta": self.arrival.scheduled.isoformat() if self.arrival.scheduled else None,
            "ata": self.arrival.actual.isoformat() if self.arrival.actual else None,
            "dep_delay_min": self.dep_delay_min,
            "arr_delay_min": self.arr_delay_min,
            "status": self.status.value,
            "time_period": self.time_period,
            "route": self.get_route_key(),
        }


@dataclass
class FlightDataBatch:
    """Batch of flight data for processing."""
    flights: List[Flight] = field(default_factory=list)
    source_file: str = ""
    processing_timestamp: datetime = field(default_factory=datetime.now)
    errors: List[str] = field(default_factory=list)
    
    def add_flight(self, flight: Flight) -> None:
        """Add a flight to the batch."""
        self.flights.append(flight)
    
    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)
    
    def get_valid_flights(self) -> List[Flight]:
        """Get only valid flights from the batch."""
        return [f for f in self.flights if f.is_valid()]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batch processing statistics."""
        valid_flights = self.get_valid_flights()
        return {
            "total_flights": len(self.flights),
            "valid_flights": len(valid_flights),
            "invalid_flights": len(self.flights) - len(valid_flights),
            "error_count": len(self.errors),
            "airlines": list(set(f.airline_code for f in valid_flights if f.airline_code)),
            "routes": list(set(f.get_route_key() for f in valid_flights)),
            "processing_timestamp": self.processing_timestamp.isoformat(),
        }