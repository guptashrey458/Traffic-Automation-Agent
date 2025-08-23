"""Turnaround time analysis service for same-tail operations and taxi time estimation."""

import os
import warnings
from datetime import datetime, date, time, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from pathlib import Path

from ..models.flight import Flight
from .database import FlightDatabaseService, QueryResult


class TurnaroundType(Enum):
    """Types of turnaround operations."""
    DOMESTIC_DOMESTIC = "domestic_domestic"
    DOMESTIC_INTERNATIONAL = "domestic_international"
    INTERNATIONAL_DOMESTIC = "international_domestic"
    INTERNATIONAL_INTERNATIONAL = "international_international"


class TaxiPhase(Enum):
    """Taxi operation phases."""
    TAXI_OUT = "taxi_out"  # EXOT - Expected Taxi Out Time
    TAXI_IN = "taxi_in"    # EXIN - Expected Taxi In Time


@dataclass
class TurnaroundEstimate:
    """Turnaround time estimation for same-tail operations."""
    aircraft_registration: str
    airport_code: str
    
    # Turnaround time estimates (in minutes)
    p50_turnaround_minutes: float  # Median turnaround time
    p90_turnaround_minutes: float  # 90th percentile (planning buffer)
    p95_turnaround_minutes: float  # 95th percentile (conservative buffer)
    
    # Historical context
    sample_size: int
    min_observed: float
    max_observed: float
    
    # Operational factors
    aircraft_type: str
    typical_route_type: str  # "domestic", "international", "mixed"
    turnaround_type: TurnaroundType
    
    # Validation metadata
    confidence_level: str = "medium"  # "low", "medium", "high"
    last_updated: datetime = field(default_factory=datetime.now)
    
    def is_feasible_departure(self, arrival_time: datetime, 
                            departure_time: datetime, 
                            buffer_percentile: int = 90) -> Tuple[bool, float]:
        """
        Check if departure time is feasible given arrival time.
        
        Args:
            arrival_time: Actual or scheduled arrival time
            departure_time: Proposed departure time
            buffer_percentile: Percentile to use for feasibility (90 or 95)
            
        Returns:
            Tuple of (is_feasible, required_turnaround_minutes)
        """
        required_turnaround = (departure_time - arrival_time).total_seconds() / 60
        
        if buffer_percentile == 95:
            min_required = self.p95_turnaround_minutes
        else:
            min_required = self.p90_turnaround_minutes
        
        return required_turnaround >= min_required, required_turnaround
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "aircraft_registration": self.aircraft_registration,
            "airport_code": self.airport_code,
            "turnaround_estimates": {
                "p50_minutes": round(self.p50_turnaround_minutes, 1),
                "p90_minutes": round(self.p90_turnaround_minutes, 1),
                "p95_minutes": round(self.p95_turnaround_minutes, 1)
            },
            "historical_context": {
                "sample_size": self.sample_size,
                "min_observed": round(self.min_observed, 1),
                "max_observed": round(self.max_observed, 1)
            },
            "operational_factors": {
                "aircraft_type": self.aircraft_type,
                "typical_route_type": self.typical_route_type,
                "turnaround_type": self.turnaround_type.value
            },
            "metadata": {
                "confidence_level": self.confidence_level,
                "last_updated": self.last_updated.isoformat()
            }
        }


@dataclass
class TaxiTimeEstimate:
    """Taxi time estimation for EXOT/EXIN calculations."""
    airport_code: str
    runway: str
    operation_type: str  # "departure" or "arrival"
    
    # Taxi time estimates (in minutes)
    expected_taxi_minutes: float
    p90_taxi_minutes: float
    p95_taxi_minutes: float
    
    # Factors affecting taxi time
    terminal_distance: str = "unknown"  # "near", "medium", "far"
    congestion_factor: float = 1.0
    weather_impact: float = 1.0
    
    # Historical context
    sample_size: int = 0
    confidence_level: str = "medium"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "airport_code": self.airport_code,
            "runway": self.runway,
            "operation_type": self.operation_type,
            "taxi_estimates": {
                "expected_minutes": round(self.expected_taxi_minutes, 1),
                "p90_minutes": round(self.p90_taxi_minutes, 1),
                "p95_minutes": round(self.p95_taxi_minutes, 1)
            },
            "factors": {
                "terminal_distance": self.terminal_distance,
                "congestion_factor": self.congestion_factor,
                "weather_impact": self.weather_impact
            },
            "metadata": {
                "sample_size": self.sample_size,
                "confidence_level": self.confidence_level
            }
        }


@dataclass
class TurnaroundValidation:
    """Validation result for feasible departure slots."""
    flight_id: str
    aircraft_registration: str
    airport_code: str
    
    # Timing information
    arrival_time: datetime
    proposed_departure_time: datetime
    required_turnaround_minutes: float
    
    # Validation results
    is_feasible_p90: bool
    is_feasible_p95: bool
    turnaround_estimate: TurnaroundEstimate
    
    # Risk assessment
    risk_level: str  # "low", "medium", "high", "critical"
    risk_factors: List[str] = field(default_factory=list)
    
    # Recommendations
    recommended_departure_time: Optional[datetime] = None
    buffer_minutes: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "flight_id": self.flight_id,
            "aircraft_registration": self.aircraft_registration,
            "airport_code": self.airport_code,
            "timing": {
                "arrival_time": self.arrival_time.isoformat(),
                "proposed_departure_time": self.proposed_departure_time.isoformat(),
                "required_turnaround_minutes": round(self.required_turnaround_minutes, 1)
            },
            "validation": {
                "is_feasible_p90": self.is_feasible_p90,
                "is_feasible_p95": self.is_feasible_p95
            },
            "risk_assessment": {
                "risk_level": self.risk_level,
                "risk_factors": self.risk_factors
            },
            "recommendations": {
                "recommended_departure_time": self.recommended_departure_time.isoformat() if self.recommended_departure_time else None,
                "buffer_minutes": round(self.buffer_minutes, 1)
            }
        }


class TurnaroundAnalysisService:
    """Service for analyzing turnaround times and validating departure slots."""
    
    def __init__(self, db_service: Optional[FlightDatabaseService] = None):
        """
        Initialize the turnaround analysis service.
        
        Args:
            db_service: Database service for historical data
        """
        self.db_service = db_service or FlightDatabaseService()
        
        # Default turnaround times by aircraft type and route type (in minutes)
        self.default_turnaround_times = {
            "domestic": {
                "A320": {"p50": 45, "p90": 75, "p95": 90},
                "A321": {"p50": 50, "p90": 80, "p95": 95},
                "B737": {"p50": 45, "p90": 75, "p95": 90},
                "B738": {"p50": 45, "p90": 75, "p95": 90},
                "default": {"p50": 45, "p90": 75, "p95": 90}
            },
            "international": {
                "A320": {"p50": 90, "p90": 150, "p95": 180},
                "A321": {"p50": 95, "p90": 155, "p95": 185},
                "B737": {"p50": 90, "p90": 150, "p95": 180},
                "B738": {"p50": 90, "p90": 150, "p95": 180},
                "B777": {"p50": 120, "p90": 180, "p95": 210},
                "B787": {"p50": 110, "p90": 170, "p95": 200},
                "default": {"p50": 90, "p90": 150, "p95": 180}
            }
        }
        
        # Default taxi times by airport and operation type (in minutes)
        self.default_taxi_times = {
            "BOM": {  # Mumbai
                "departure": {"expected": 15, "p90": 25, "p95": 30},
                "arrival": {"expected": 12, "p90": 20, "p95": 25}
            },
            "DEL": {  # Delhi
                "departure": {"expected": 18, "p90": 28, "p95": 35},
                "arrival": {"expected": 15, "p90": 25, "p95": 30}
            },
            "BLR": {  # Bangalore
                "departure": {"expected": 12, "p90": 20, "p95": 25},
                "arrival": {"expected": 10, "p90": 18, "p95": 22}
            },
            "default": {
                "departure": {"expected": 15, "p90": 25, "p95": 30},
                "arrival": {"expected": 12, "p90": 20, "p95": 25}
            }
        }
    
    def estimate_turnaround_time(self, aircraft_registration: str, 
                               airport_code: str) -> TurnaroundEstimate:
        """
        Estimate turnaround time for same-tail operations.
        
        Args:
            aircraft_registration: Aircraft tail number
            airport_code: Airport where turnaround occurs
            
        Returns:
            TurnaroundEstimate with P90 and other percentiles
        """
        try:
            # Query historical turnaround data
            turnaround_data = self._get_turnaround_data(aircraft_registration, airport_code)
            
            if turnaround_data and len(turnaround_data) >= 5:  # Minimum sample size
                return self._calculate_historical_turnaround(turnaround_data, aircraft_registration, airport_code)
            else:
                # Use default estimates based on aircraft type and typical routes
                return self._default_turnaround_estimate(aircraft_registration, airport_code)
                
        except Exception as e:
            print(f"Error estimating turnaround time: {e}")
            return self._default_turnaround_estimate(aircraft_registration, airport_code)
    
    def predict_taxi_time(self, flight: Flight, runway: str) -> TaxiTimeEstimate:
        """
        Predict taxi time for EXOT/EXIN calculations.
        
        Args:
            flight: Flight object
            runway: Runway identifier
            
        Returns:
            TaxiTimeEstimate for the operation
        """
        # Determine operation type and airport
        if flight.origin and flight.origin.code:
            operation_type = "departure"
            airport_code = flight.origin.code
        elif flight.destination and flight.destination.code:
            operation_type = "arrival"
            airport_code = flight.destination.code
        else:
            # Fallback
            operation_type = "departure"
            airport_code = "UNKNOWN"
        
        try:
            # Query historical taxi time data
            taxi_data = self._get_taxi_time_data(airport_code, runway, operation_type)
            
            if taxi_data and len(taxi_data) >= 3:
                return self._calculate_historical_taxi_time(taxi_data, airport_code, runway, operation_type)
            else:
                # Use default estimates
                return self._default_taxi_estimate(airport_code, runway, operation_type)
                
        except Exception as e:
            print(f"Error predicting taxi time: {e}")
            return self._default_taxi_estimate(airport_code, runway, operation_type)
    
    def validate_departure_slot(self, flight: Flight, 
                              proposed_departure_time: datetime,
                              arrival_flight: Optional[Flight] = None) -> TurnaroundValidation:
        """
        Validate if a proposed departure slot is feasible for same-tail operations.
        
        Args:
            flight: Departure flight to validate
            proposed_departure_time: Proposed departure time
            arrival_flight: Previous arrival flight (if known)
            
        Returns:
            TurnaroundValidation with feasibility assessment
        """
        if not flight.aircraft_registration:
            # Cannot validate without aircraft registration
            return self._create_unknown_validation(flight, proposed_departure_time)
        
        # Find the previous arrival for this aircraft
        if not arrival_flight:
            arrival_flight = self._find_previous_arrival(flight.aircraft_registration, 
                                                       proposed_departure_time,
                                                       flight.origin.code if flight.origin else None)
        
        if not arrival_flight or not arrival_flight.arrival.actual:
            # No previous arrival found or no actual arrival time
            return self._create_no_arrival_validation(flight, proposed_departure_time)
        
        # Get turnaround estimate
        airport_code = flight.origin.code if flight.origin else "UNKNOWN"
        turnaround_estimate = self.estimate_turnaround_time(flight.aircraft_registration, airport_code)
        
        # Calculate required turnaround time
        arrival_time = arrival_flight.arrival.actual
        required_turnaround = (proposed_departure_time - arrival_time).total_seconds() / 60
        
        # Check feasibility
        is_feasible_p90, _ = turnaround_estimate.is_feasible_departure(arrival_time, proposed_departure_time, 90)
        is_feasible_p95, _ = turnaround_estimate.is_feasible_departure(arrival_time, proposed_departure_time, 95)
        
        # Assess risk level
        risk_level, risk_factors = self._assess_turnaround_risk(required_turnaround, turnaround_estimate)
        
        # Generate recommendations
        recommended_departure_time = None
        buffer_minutes = 0.0
        
        if not is_feasible_p90:
            # Recommend departure time based on P90 turnaround
            recommended_departure_time = arrival_time + timedelta(minutes=turnaround_estimate.p90_turnaround_minutes)
            buffer_minutes = turnaround_estimate.p90_turnaround_minutes - required_turnaround
        
        return TurnaroundValidation(
            flight_id=flight.flight_id,
            aircraft_registration=flight.aircraft_registration,
            airport_code=airport_code,
            arrival_time=arrival_time,
            proposed_departure_time=proposed_departure_time,
            required_turnaround_minutes=required_turnaround,
            is_feasible_p90=is_feasible_p90,
            is_feasible_p95=is_feasible_p95,
            turnaround_estimate=turnaround_estimate,
            risk_level=risk_level,
            risk_factors=risk_factors,
            recommended_departure_time=recommended_departure_time,
            buffer_minutes=buffer_minutes
        )
    
    def _get_turnaround_data(self, aircraft_registration: str, 
                           airport_code: str) -> List[Dict[str, Any]]:
        """Query historical turnaround data for an aircraft at an airport."""
        try:
            conn = self.db_service.connect()
            
            # Query for same-tail turnarounds at the airport
            query = """
            WITH turnarounds AS (
                SELECT 
                    arr.aircraft_registration,
                    arr.ata_utc as arrival_time,
                    dep.std_utc as departure_time,
                    dep.aircraft_type,
                    arr.destination_code as turnaround_airport,
                    dep.origin_code,
                    CASE 
                        WHEN arr.route LIKE '%-%' AND LENGTH(arr.route) <= 7 THEN 'domestic'
                        ELSE 'international'
                    END as arrival_route_type,
                    CASE 
                        WHEN dep.route LIKE '%-%' AND LENGTH(dep.route) <= 7 THEN 'domestic'
                        ELSE 'international'
                    END as departure_route_type,
                    (EXTRACT(EPOCH FROM dep.std_utc - arr.ata_utc) / 60) as turnaround_minutes
                FROM flights arr
                JOIN flights dep ON (
                    arr.aircraft_registration = dep.aircraft_registration
                    AND arr.destination_code = dep.origin_code
                    AND dep.std_utc > arr.ata_utc
                    AND dep.std_utc <= arr.ata_utc + INTERVAL '24 hours'
                )
                WHERE arr.aircraft_registration = ?
                    AND arr.destination_code = ?
                    AND arr.ata_utc IS NOT NULL
                    AND dep.std_utc IS NOT NULL
                    AND (EXTRACT(EPOCH FROM dep.std_utc - arr.ata_utc) / 60) BETWEEN 30 AND 720
            )
            SELECT * FROM turnarounds
            ORDER BY arrival_time DESC
            LIMIT 50
            """
            
            result = conn.execute(query, [aircraft_registration, airport_code]).fetchall()
            columns = [desc[0] for desc in conn.description]
            
            return [dict(zip(columns, row)) for row in result]
            
        except Exception as e:
            print(f"Error querying turnaround data: {e}")
            return []
    
    def _get_taxi_time_data(self, airport_code: str, runway: str, 
                          operation_type: str) -> List[Dict[str, Any]]:
        """Query historical taxi time data for an airport and runway."""
        try:
            conn = self.db_service.connect()
            
            # For now, return empty list as we don't have taxi time data in the schema
            # This would be enhanced with actual taxi time tracking
            return []
            
        except Exception as e:
            print(f"Error querying taxi time data: {e}")
            return []
    
    def _calculate_historical_turnaround(self, turnaround_data: List[Dict[str, Any]], 
                                       aircraft_registration: str, 
                                       airport_code: str) -> TurnaroundEstimate:
        """Calculate turnaround estimates from historical data."""
        turnaround_times = [t['turnaround_minutes'] for t in turnaround_data if t['turnaround_minutes'] > 0]
        
        if not turnaround_times:
            return self._default_turnaround_estimate(aircraft_registration, airport_code)
        
        # Calculate percentiles
        p50 = np.percentile(turnaround_times, 50)
        p90 = np.percentile(turnaround_times, 90)
        p95 = np.percentile(turnaround_times, 95)
        
        # Determine aircraft type and route type
        aircraft_type = turnaround_data[0].get('aircraft_type', 'UNKNOWN')
        
        # Determine typical route type
        arrival_types = [t.get('arrival_route_type', 'domestic') for t in turnaround_data]
        departure_types = [t.get('departure_route_type', 'domestic') for t in turnaround_data]
        
        # Classify turnaround type
        if 'international' in arrival_types or 'international' in departure_types:
            if 'domestic' in arrival_types and 'domestic' in departure_types:
                typical_route_type = "mixed"
                turnaround_type = TurnaroundType.DOMESTIC_INTERNATIONAL
            else:
                typical_route_type = "international"
                turnaround_type = TurnaroundType.INTERNATIONAL_INTERNATIONAL
        else:
            typical_route_type = "domestic"
            turnaround_type = TurnaroundType.DOMESTIC_DOMESTIC
        
        # Determine confidence level
        confidence_level = "high" if len(turnaround_times) >= 20 else "medium" if len(turnaround_times) >= 10 else "low"
        
        return TurnaroundEstimate(
            aircraft_registration=aircraft_registration,
            airport_code=airport_code,
            p50_turnaround_minutes=p50,
            p90_turnaround_minutes=p90,
            p95_turnaround_minutes=p95,
            sample_size=len(turnaround_times),
            min_observed=min(turnaround_times),
            max_observed=max(turnaround_times),
            aircraft_type=aircraft_type,
            typical_route_type=typical_route_type,
            turnaround_type=turnaround_type,
            confidence_level=confidence_level
        )
    
    def _default_turnaround_estimate(self, aircraft_registration: str, 
                                   airport_code: str) -> TurnaroundEstimate:
        """Generate default turnaround estimate when no historical data is available."""
        # Try to extract aircraft type from registration
        aircraft_type = self._extract_aircraft_type_from_registration(aircraft_registration)
        
        # Assume domestic operations as default
        route_type = "domestic"
        defaults = self.default_turnaround_times[route_type].get(aircraft_type, 
                                                               self.default_turnaround_times[route_type]["default"])
        
        return TurnaroundEstimate(
            aircraft_registration=aircraft_registration,
            airport_code=airport_code,
            p50_turnaround_minutes=defaults["p50"],
            p90_turnaround_minutes=defaults["p90"],
            p95_turnaround_minutes=defaults["p95"],
            sample_size=0,
            min_observed=defaults["p50"],
            max_observed=defaults["p95"],
            aircraft_type=aircraft_type,
            typical_route_type=route_type,
            turnaround_type=TurnaroundType.DOMESTIC_DOMESTIC,
            confidence_level="low"
        )
    
    def _calculate_historical_taxi_time(self, taxi_data: List[Dict[str, Any]], 
                                      airport_code: str, runway: str, 
                                      operation_type: str) -> TaxiTimeEstimate:
        """Calculate taxi time estimates from historical data."""
        taxi_times = [t['taxi_minutes'] for t in taxi_data if t['taxi_minutes'] > 0]
        
        if not taxi_times:
            return self._default_taxi_estimate(airport_code, runway, operation_type)
        
        expected_taxi = np.mean(taxi_times)
        p90_taxi = np.percentile(taxi_times, 90)
        p95_taxi = np.percentile(taxi_times, 95)
        
        return TaxiTimeEstimate(
            airport_code=airport_code,
            runway=runway,
            operation_type=operation_type,
            expected_taxi_minutes=expected_taxi,
            p90_taxi_minutes=p90_taxi,
            p95_taxi_minutes=p95_taxi,
            sample_size=len(taxi_times),
            confidence_level="high" if len(taxi_times) >= 20 else "medium"
        )
    
    def _default_taxi_estimate(self, airport_code: str, runway: str, 
                             operation_type: str) -> TaxiTimeEstimate:
        """Generate default taxi time estimate."""
        defaults = self.default_taxi_times.get(airport_code, self.default_taxi_times["default"])
        operation_defaults = defaults[operation_type]
        
        return TaxiTimeEstimate(
            airport_code=airport_code,
            runway=runway,
            operation_type=operation_type,
            expected_taxi_minutes=operation_defaults["expected"],
            p90_taxi_minutes=operation_defaults["p90"],
            p95_taxi_minutes=operation_defaults["p95"],
            sample_size=0,
            confidence_level="low"
        )
    
    def _find_previous_arrival(self, aircraft_registration: str, 
                             departure_time: datetime, 
                             airport_code: Optional[str]) -> Optional[Flight]:
        """Find the previous arrival flight for the same aircraft."""
        try:
            conn = self.db_service.connect()
            
            # Look for arrivals in the 24 hours before the departure
            query = """
            SELECT * FROM flights
            WHERE aircraft_registration = ?
                AND destination_code = ?
                AND ata_utc IS NOT NULL
                AND ata_utc < ?
                AND ata_utc >= ?
            ORDER BY ata_utc DESC
            LIMIT 1
            """
            
            start_time = departure_time - timedelta(hours=24)
            params = [aircraft_registration, airport_code, departure_time, start_time]
            
            result = conn.execute(query, params).fetchone()
            
            if result:
                columns = [desc[0] for desc in conn.description]
                flight_data = dict(zip(columns, result))
                
                # Convert to Flight object (simplified)
                from ..models.flight import Airport, FlightTime
                
                flight = Flight()
                flight.flight_id = flight_data['flight_id']
                flight.flight_number = flight_data['flight_number']
                flight.aircraft_registration = flight_data['aircraft_registration']
                flight.aircraft_type = flight_data['aircraft_type']
                
                if flight_data['destination_code']:
                    flight.destination = Airport(
                        code=flight_data['destination_code'],
                        name=flight_data['destination_name'] or flight_data['destination_code'],
                        city=flight_data['destination_name'] or flight_data['destination_code']
                    )
                
                flight.arrival = FlightTime()
                flight.arrival.actual = flight_data['ata_utc']
                
                return flight
            
            return None
            
        except Exception as e:
            print(f"Error finding previous arrival: {e}")
            return None
    
    def _assess_turnaround_risk(self, required_turnaround: float, 
                              turnaround_estimate: TurnaroundEstimate) -> Tuple[str, List[str]]:
        """Assess risk level for turnaround time."""
        risk_factors = []
        
        if required_turnaround < turnaround_estimate.p50_turnaround_minutes:
            risk_level = "critical"
            risk_factors.append("Below median turnaround time")
        elif required_turnaround < turnaround_estimate.p90_turnaround_minutes:
            risk_level = "high"
            risk_factors.append("Below P90 turnaround time")
        elif required_turnaround < turnaround_estimate.p95_turnaround_minutes:
            risk_level = "medium"
            risk_factors.append("Below P95 turnaround time")
        else:
            risk_level = "low"
        
        # Additional risk factors
        if turnaround_estimate.sample_size < 5:
            risk_factors.append("Limited historical data")
        
        if turnaround_estimate.confidence_level == "low":
            risk_factors.append("Low confidence in estimates")
        
        return risk_level, risk_factors
    
    def _extract_aircraft_type_from_registration(self, registration: str) -> str:
        """Extract aircraft type from registration if possible."""
        if not registration:
            return "UNKNOWN"
        
        # Common patterns in Indian aircraft registrations
        # This is a simplified approach - real implementation would use aircraft database
        registration_upper = registration.upper()
        
        if "320" in registration_upper:
            return "A320"
        elif "321" in registration_upper:
            return "A321"
        elif "737" in registration_upper or "738" in registration_upper:
            return "B737"
        elif "777" in registration_upper:
            return "B777"
        elif "787" in registration_upper:
            return "B787"
        else:
            return "UNKNOWN"
    
    def _create_unknown_validation(self, flight: Flight, 
                                 proposed_departure_time: datetime) -> TurnaroundValidation:
        """Create validation result when aircraft registration is unknown."""
        return TurnaroundValidation(
            flight_id=flight.flight_id,
            aircraft_registration="UNKNOWN",
            airport_code=flight.origin.code if flight.origin else "UNKNOWN",
            arrival_time=proposed_departure_time - timedelta(hours=2),  # Dummy arrival
            proposed_departure_time=proposed_departure_time,
            required_turnaround_minutes=120,  # Dummy value
            is_feasible_p90=True,  # Assume feasible when unknown
            is_feasible_p95=True,
            turnaround_estimate=self._default_turnaround_estimate("UNKNOWN", 
                                                                flight.origin.code if flight.origin else "UNKNOWN"),
            risk_level="medium",
            risk_factors=["Unknown aircraft registration"]
        )
    
    def _create_no_arrival_validation(self, flight: Flight, 
                                    proposed_departure_time: datetime) -> TurnaroundValidation:
        """Create validation result when no previous arrival is found."""
        airport_code = flight.origin.code if flight.origin else "UNKNOWN"
        turnaround_estimate = self.estimate_turnaround_time(flight.aircraft_registration, airport_code)
        
        return TurnaroundValidation(
            flight_id=flight.flight_id,
            aircraft_registration=flight.aircraft_registration,
            airport_code=airport_code,
            arrival_time=proposed_departure_time - timedelta(hours=2),  # Dummy arrival
            proposed_departure_time=proposed_departure_time,
            required_turnaround_minutes=120,  # Dummy value
            is_feasible_p90=True,  # Assume feasible when no previous arrival
            is_feasible_p95=True,
            turnaround_estimate=turnaround_estimate,
            risk_level="medium",
            risk_factors=["No previous arrival flight found"]
        )