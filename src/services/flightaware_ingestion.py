"""FlightAware AeroAPI integration for official schedule data ingestion."""

import os
import requests
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional
import pytz
from dataclasses import dataclass

from ..models.flight import Flight, FlightTime, Airport, FlightDataBatch
from ..services.data_ingestion import IngestionResult


@dataclass
class FlightAwareConfig:
    """Configuration for FlightAware AeroAPI."""
    api_key: str
    base_url: str = "https://aeroapi.flightaware.com/aeroapi"
    timeout: int = 30
    max_retries: int = 3


class FlightAwareIngestionService:
    """Service for ingesting flight data from FlightAware AeroAPI."""
    
    def __init__(self, config: FlightAwareConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'x-apikey': config.api_key,
            'Accept': 'application/json'
        })
        self.ist_timezone = pytz.timezone('Asia/Kolkata')
        self.utc_timezone = pytz.UTC
    
    def ingest_airport_schedules(self, airport_codes: List[str], 
                               start_date: date, end_date: date) -> IngestionResult:
        """
        Ingest flight schedules for specified airports and date range.
        
        Args:
            airport_codes: List of IATA airport codes (e.g., ['BOM', 'DEL'])
            start_date: Start date for schedule data
            end_date: End date for schedule data
            
        Returns:
            IngestionResult with processed flight data
        """
        start_time = datetime.now()
        result = IngestionResult()
        
        for airport_code in airport_codes:
            try:
                # Get departures
                dep_batch = self._get_airport_departures(airport_code, start_date, end_date)
                result.add_batch(dep_batch)
                
                # Get arrivals
                arr_batch = self._get_airport_arrivals(airport_code, start_date, end_date)
                result.add_batch(arr_batch)
                
                result.successful_files += 1
                
            except Exception as e:
                result.failed_files.append(f"FlightAware:{airport_code}")
                result.errors.append(f"Error fetching {airport_code}: {str(e)}")
        
        result.total_files_processed = len(airport_codes)
        result.processing_time_seconds = (datetime.now() - start_time).total_seconds()
        
        return result
    
    def _get_airport_departures(self, airport_code: str, 
                              start_date: date, end_date: date) -> FlightDataBatch:
        """Get departure flights for an airport."""
        batch = FlightDataBatch(source_file=f"FlightAware:Departures:{airport_code}")
        
        current_date = start_date
        while current_date <= end_date:
            try:
                # FlightAware API endpoint for departures
                url = f"{self.config.base_url}/airports/{airport_code}/flights/departures"
                params = {
                    'start': current_date.isoformat(),
                    'end': (current_date + timedelta(days=1)).isoformat(),
                    'max_pages': 10,  # Limit to avoid rate limits
                    'type': 'Airline'  # Only commercial flights
                }
                
                response = self._make_api_request(url, params)
                if response:
                    flights_data = response.get('departures', [])
                    for flight_data in flights_data:
                        flight = self._parse_flightaware_departure(flight_data, airport_code)
                        if flight:
                            batch.add_flight(flight)
                
            except Exception as e:
                batch.add_error(f"Error fetching departures for {airport_code} on {current_date}: {str(e)}")
            
            current_date += timedelta(days=1)
        
        return batch
    
    def _get_airport_arrivals(self, airport_code: str, 
                            start_date: date, end_date: date) -> FlightDataBatch:
        """Get arrival flights for an airport."""
        batch = FlightDataBatch(source_file=f"FlightAware:Arrivals:{airport_code}")
        
        current_date = start_date
        while current_date <= end_date:
            try:
                # FlightAware API endpoint for arrivals
                url = f"{self.config.base_url}/airports/{airport_code}/flights/arrivals"
                params = {
                    'start': current_date.isoformat(),
                    'end': (current_date + timedelta(days=1)).isoformat(),
                    'max_pages': 10,  # Limit to avoid rate limits
                    'type': 'Airline'  # Only commercial flights
                }
                
                response = self._make_api_request(url, params)
                if response:
                    flights_data = response.get('arrivals', [])
                    for flight_data in flights_data:
                        flight = self._parse_flightaware_arrival(flight_data, airport_code)
                        if flight:
                            batch.add_flight(flight)
                
            except Exception as e:
                batch.add_error(f"Error fetching arrivals for {airport_code} on {current_date}: {str(e)}")
            
            current_date += timedelta(days=1)
        
        return batch
    
    def _make_api_request(self, url: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make API request with retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                response = self.session.get(
                    url, 
                    params=params, 
                    timeout=self.config.timeout
                )
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                if attempt == self.config.max_retries - 1:
                    raise e
                # Wait before retry (exponential backoff)
                import time
                time.sleep(2 ** attempt)
        
        return None
    
    def _parse_flightaware_departure(self, flight_data: Dict[str, Any], 
                                   airport_code: str) -> Optional[Flight]:
        """Parse FlightAware departure data into Flight object."""
        try:
            # Extract basic flight information
            ident = flight_data.get('ident', '')
            if not ident:
                return None
            
            # Create flight object
            flight = Flight()
            flight.flight_number = ident
            flight.data_source = "flightaware"
            flight.raw_data = flight_data
            
            # Set origin (departure airport)
            flight.origin = Airport(
                code=airport_code,
                name=f"{airport_code} Airport",
                city=airport_code
            )
            
            # Set destination
            destination_code = flight_data.get('destination', {}).get('code', '')
            if destination_code:
                flight.destination = Airport(
                    code=destination_code,
                    name=f"{destination_code} Airport",
                    city=destination_code
                )
            
            # Parse aircraft information
            aircraft_type = flight_data.get('aircraft_type', '')
            if aircraft_type:
                flight.aircraft_type = aircraft_type
            
            # Parse scheduled departure time
            scheduled_departure = flight_data.get('scheduled_departure')
            if scheduled_departure:
                scheduled_dt = datetime.fromisoformat(scheduled_departure.replace('Z', '+00:00'))
                flight.departure.scheduled = scheduled_dt.time()
                flight.flight_date = scheduled_dt.date()
            
            # Parse actual departure time
            actual_departure = flight_data.get('actual_departure')
            if actual_departure:
                actual_dt = datetime.fromisoformat(actual_departure.replace('Z', '+00:00'))
                flight.departure.actual = actual_dt
            
            # Parse scheduled arrival time
            scheduled_arrival = flight_data.get('scheduled_arrival')
            if scheduled_arrival:
                scheduled_dt = datetime.fromisoformat(scheduled_arrival.replace('Z', '+00:00'))
                flight.arrival.scheduled = scheduled_dt.time()
            
            # Parse actual arrival time
            actual_arrival = flight_data.get('actual_arrival')
            if actual_arrival:
                actual_dt = datetime.fromisoformat(actual_arrival.replace('Z', '+00:00'))
                flight.arrival.actual = actual_dt
            
            return flight
            
        except Exception as e:
            return None
    
    def _parse_flightaware_arrival(self, flight_data: Dict[str, Any], 
                                 airport_code: str) -> Optional[Flight]:
        """Parse FlightAware arrival data into Flight object."""
        try:
            # Extract basic flight information
            ident = flight_data.get('ident', '')
            if not ident:
                return None
            
            # Create flight object
            flight = Flight()
            flight.flight_number = ident
            flight.data_source = "flightaware"
            flight.raw_data = flight_data
            
            # Set destination (arrival airport)
            flight.destination = Airport(
                code=airport_code,
                name=f"{airport_code} Airport",
                city=airport_code
            )
            
            # Set origin
            origin_code = flight_data.get('origin', {}).get('code', '')
            if origin_code:
                flight.origin = Airport(
                    code=origin_code,
                    name=f"{origin_code} Airport",
                    city=origin_code
                )
            
            # Parse aircraft information
            aircraft_type = flight_data.get('aircraft_type', '')
            if aircraft_type:
                flight.aircraft_type = aircraft_type
            
            # Parse scheduled departure time
            scheduled_departure = flight_data.get('scheduled_departure')
            if scheduled_departure:
                scheduled_dt = datetime.fromisoformat(scheduled_departure.replace('Z', '+00:00'))
                flight.departure.scheduled = scheduled_dt.time()
                flight.flight_date = scheduled_dt.date()
            
            # Parse actual departure time
            actual_departure = flight_data.get('actual_departure')
            if actual_departure:
                actual_dt = datetime.fromisoformat(actual_departure.replace('Z', '+00:00'))
                flight.departure.actual = actual_dt
            
            # Parse scheduled arrival time
            scheduled_arrival = flight_data.get('scheduled_arrival')
            if scheduled_arrival:
                scheduled_dt = datetime.fromisoformat(scheduled_arrival.replace('Z', '+00:00'))
                flight.arrival.scheduled = scheduled_dt.time()
            
            # Parse actual arrival time
            actual_arrival = flight_data.get('actual_arrival')
            if actual_arrival:
                actual_dt = datetime.fromisoformat(actual_arrival.replace('Z', '+00:00'))
                flight.arrival.actual = actual_dt
            
            return flight
            
        except Exception as e:
            return None
    
    def test_connection(self) -> Dict[str, Any]:
        """Test FlightAware API connection."""
        try:
            url = f"{self.config.base_url}/airports/BOM"
            response = self._make_api_request(url, {})
            
            if response:
                return {
                    "status": "success",
                    "message": "FlightAware API connection successful",
                    "airport_info": response
                }
            else:
                return {
                    "status": "error",
                    "message": "Failed to connect to FlightAware API"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"FlightAware API connection failed: {str(e)}"
            }