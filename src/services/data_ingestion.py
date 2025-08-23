"""Data ingestion service for processing multiple Excel files with flight data."""

import os
import pandas as pd
from datetime import datetime, date, time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pytz
from dataclasses import dataclass, field

from ..models.flight import Flight, FlightDataBatch, Airport, FlightTime
from ..models.validation import DataValidator
from ..models.excel_parser import ExcelFlightParser


@dataclass
class IngestionResult:
    """Result of data ingestion process."""
    total_files_processed: int = 0
    successful_files: int = 0
    failed_files: List[str] = field(default_factory=list)
    total_flights: int = 0
    valid_flights: int = 0
    invalid_flights: int = 0
    errors: List[str] = field(default_factory=list)
    processing_time_seconds: float = 0.0
    flights: List[Flight] = field(default_factory=list)
    
    def add_batch(self, batch: FlightDataBatch) -> None:
        """Add a batch result to the overall result."""
        self.total_flights += len(batch.flights)
        valid_batch_flights = batch.get_valid_flights()
        self.valid_flights += len(valid_batch_flights)
        self.invalid_flights += len(batch.flights) - len(valid_batch_flights)
        self.errors.extend(batch.errors)
        self.flights.extend(valid_batch_flights)


class DataIngestionService:
    """Service for ingesting flight data from multiple Excel files."""
    
    def __init__(self):
        self.parser = ExcelFlightParser()
        self.ist_timezone = pytz.timezone('Asia/Kolkata')
        self.utc_timezone = pytz.UTC
    
    def ingest_excel_files(self, file_paths: List[str]) -> IngestionResult:
        """
        Process multiple Excel files and return consolidated flight data.
        
        Args:
            file_paths: List of paths to Excel files
            
        Returns:
            IngestionResult with processed flight data and statistics
        """
        start_time = datetime.now()
        result = IngestionResult()
        result.total_files_processed = len(file_paths)
        
        for file_path in file_paths:
            try:
                if not os.path.exists(file_path):
                    result.failed_files.append(file_path)
                    result.errors.append(f"File not found: {file_path}")
                    continue
                
                # Process single file
                batch = self._process_single_file(file_path)
                result.add_batch(batch)
                result.successful_files += 1
                
            except Exception as e:
                result.failed_files.append(file_path)
                result.errors.append(f"Error processing {file_path}: {str(e)}")
        
        # Convert IST to UTC for all flights
        self._convert_timestamps_to_utc(result.flights)
        
        # Calculate delays for all flights
        self._calculate_delays(result.flights)
        
        # Handle missing data
        self._handle_missing_data(result.flights)
        
        result.processing_time_seconds = (datetime.now() - start_time).total_seconds()
        
        return result
    
    def _process_single_file(self, file_path: str) -> FlightDataBatch:
        """Process a single Excel file and return flight batch."""
        try:
            # Use existing parser to handle complex Excel structures
            batch = self.parser.parse_excel_file(file_path)
            
            # Additional processing for each flight in the batch
            for flight in batch.flights:
                self._enrich_flight_data(flight, file_path)
            
            return batch
            
        except Exception as e:
            batch = FlightDataBatch(source_file=file_path)
            batch.add_error(f"Failed to process file: {str(e)}")
            return batch
    
    def _enrich_flight_data(self, flight: Flight, source_file: str) -> None:
        """Enrich flight data with additional processing."""
        # Set data source
        flight.data_source = f"excel:{os.path.basename(source_file)}"
        
        # Extract aircraft registration from raw data if available
        if 'Aircraft' in flight.raw_data:
            aircraft_str = str(flight.raw_data['Aircraft'])
            # Extract registration (e.g., "A20N (VT-EXU)" -> "VT-EXU")
            import re
            reg_match = re.search(r'\(([A-Z0-9-]+)\)', aircraft_str)
            if reg_match:
                flight.aircraft_registration = reg_match.group(1)
    
    def _convert_timestamps_to_utc(self, flights: List[Flight]) -> None:
        """Convert IST timestamps to UTC for all flights."""
        for flight in flights:
            # Convert departure times
            if flight.departure.actual and flight.flight_date:
                # Combine date and time, localize to IST, then convert to UTC
                ist_datetime = self.ist_timezone.localize(flight.departure.actual)
                flight.departure.actual = ist_datetime.astimezone(self.utc_timezone)
            
            # Convert arrival times
            if flight.arrival.actual and flight.flight_date:
                ist_datetime = self.ist_timezone.localize(flight.arrival.actual)
                flight.arrival.actual = ist_datetime.astimezone(self.utc_timezone)
            
            # Convert scheduled times to UTC as well (combine with flight date)
            if flight.departure.scheduled and flight.flight_date:
                scheduled_dt = datetime.combine(flight.flight_date, flight.departure.scheduled)
                ist_scheduled = self.ist_timezone.localize(scheduled_dt)
                utc_scheduled = ist_scheduled.astimezone(self.utc_timezone)
                flight.departure.scheduled = utc_scheduled.time()
            
            if flight.arrival.scheduled and flight.flight_date:
                scheduled_dt = datetime.combine(flight.flight_date, flight.arrival.scheduled)
                ist_scheduled = self.ist_timezone.localize(scheduled_dt)
                utc_scheduled = ist_scheduled.astimezone(self.utc_timezone)
                flight.arrival.scheduled = utc_scheduled.time()
    
    def _calculate_delays(self, flights: List[Flight]) -> None:
        """Calculate departure and arrival delays for all flights."""
        for flight in flights:
            # Calculate departure delay
            if flight.departure.scheduled and flight.departure.actual and flight.flight_date:
                # Create scheduled datetime in UTC
                scheduled_dt = datetime.combine(flight.flight_date, flight.departure.scheduled)
                if scheduled_dt.tzinfo is None:
                    scheduled_dt = self.utc_timezone.localize(scheduled_dt)
                
                # Calculate delay in minutes
                delay_seconds = (flight.departure.actual - scheduled_dt).total_seconds()
                flight.dep_delay_min = int(delay_seconds / 60)
            
            # Calculate arrival delay
            if flight.arrival.scheduled and flight.arrival.actual and flight.flight_date:
                # Create scheduled datetime in UTC
                scheduled_dt = datetime.combine(flight.flight_date, flight.arrival.scheduled)
                if scheduled_dt.tzinfo is None:
                    scheduled_dt = self.utc_timezone.localize(scheduled_dt)
                
                # Calculate delay in minutes
                delay_seconds = (flight.arrival.actual - scheduled_dt).total_seconds()
                flight.arr_delay_min = int(delay_seconds / 60)
    
    def _handle_missing_data(self, flights: List[Flight]) -> None:
        """Handle missing data with fallback strategies."""
        for flight in flights:
            # Check if actual departure time is missing (None, NaN, or empty string in raw data)
            has_atd = flight.departure.actual is not None
            if not has_atd and 'ATD' in flight.raw_data:
                import pandas as pd
                atd_raw = flight.raw_data['ATD']
                has_atd = atd_raw and not pd.isna(atd_raw) and str(atd_raw).strip() != ""
            
            # If actual departure time is missing, use scheduled time
            if not has_atd and flight.departure.scheduled and flight.flight_date:
                scheduled_dt = datetime.combine(flight.flight_date, flight.departure.scheduled)
                flight.departure.actual = self.utc_timezone.localize(scheduled_dt)
                flight.departure.actual_str = "IMPUTED_FROM_SCHEDULED"
                flight.dep_delay_min = 0  # No delay if using scheduled time
            
            # Check if actual arrival time is missing
            has_ata = flight.arrival.actual is not None
            if not has_ata and 'ATA' in flight.raw_data:
                import pandas as pd
                ata_raw = flight.raw_data['ATA']
                has_ata = ata_raw and not pd.isna(ata_raw) and str(ata_raw).strip() != ""
            
            # If actual arrival time is missing, use scheduled time
            if not has_ata and flight.arrival.scheduled and flight.flight_date:
                scheduled_dt = datetime.combine(flight.flight_date, flight.arrival.scheduled)
                flight.arrival.actual = self.utc_timezone.localize(scheduled_dt)
                flight.arrival.actual_str = "IMPUTED_FROM_SCHEDULED"
                flight.arr_delay_min = 0  # No delay if using scheduled time
            
            # If aircraft type is missing, try to infer from flight number
            if not flight.aircraft_type or flight.aircraft_type == "UNKNOWN":
                flight.aircraft_type = self._infer_aircraft_type(flight.flight_number)
            
            # If origin/destination is missing, mark as unknown
            if not flight.origin:
                flight.origin = Airport(code="UNK", name="Unknown Origin", city="Unknown")
            
            if not flight.destination:
                flight.destination = Airport(code="UNK", name="Unknown Destination", city="Unknown")
    
    def _infer_aircraft_type(self, flight_number: str) -> str:
        """Infer aircraft type from flight number patterns."""
        if not flight_number:
            return "UNKNOWN"
        
        # Common patterns for Indian airlines
        airline_aircraft_map = {
            "6E": "A320",  # IndiGo primarily uses A320 family
            "AI": "A320",  # Air India mixed fleet, default to A320
            "SG": "B737",  # SpiceJet uses Boeing 737
            "UK": "A320",  # Vistara uses A320 family
            "G8": "A320",  # GoAir uses A320 family
        }
        
        # Extract airline code (first 2 characters, can include digits like "6E")
        airline_code = ""
        for char in flight_number:
            if char.isalpha() or char.isdigit():
                airline_code += char
                if len(airline_code) >= 2:
                    break
            else:
                break
        
        return airline_aircraft_map.get(airline_code, "UNKNOWN")
    
    def get_ingestion_summary(self, result: IngestionResult) -> Dict[str, Any]:
        """Generate comprehensive summary of ingestion results."""
        if not result.flights:
            return {
                "status": "no_data",
                "message": "No valid flights found",
                "files_processed": result.total_files_processed,
                "errors": result.errors
            }
        
        # Analyze flights
        airlines = {}
        routes = {}
        aircraft_types = {}
        dates = set()
        delay_stats = {"dep_delays": [], "arr_delays": []}
        
        for flight in result.flights:
            # Count airlines
            if flight.airline_code:
                airlines[flight.airline_code] = airlines.get(flight.airline_code, 0) + 1
            
            # Count routes
            route = flight.get_route_key()
            routes[route] = routes.get(route, 0) + 1
            
            # Count aircraft types
            if flight.aircraft_type:
                aircraft_types[flight.aircraft_type] = aircraft_types.get(flight.aircraft_type, 0) + 1
            
            # Collect dates
            if flight.flight_date:
                dates.add(flight.flight_date)
            
            # Collect delay statistics
            if flight.dep_delay_min is not None:
                delay_stats["dep_delays"].append(flight.dep_delay_min)
            if flight.arr_delay_min is not None:
                delay_stats["arr_delays"].append(flight.arr_delay_min)
        
        # Calculate delay statistics
        dep_delays = delay_stats["dep_delays"]
        arr_delays = delay_stats["arr_delays"]
        
        delay_summary = {}
        if dep_delays:
            delay_summary["departure"] = {
                "avg_delay_min": sum(dep_delays) / len(dep_delays),
                "max_delay_min": max(dep_delays),
                "min_delay_min": min(dep_delays),
                "flights_delayed": len([d for d in dep_delays if d > 15]),
                "on_time_rate": len([d for d in dep_delays if d <= 15]) / len(dep_delays)
            }
        
        if arr_delays:
            delay_summary["arrival"] = {
                "avg_delay_min": sum(arr_delays) / len(arr_delays),
                "max_delay_min": max(arr_delays),
                "min_delay_min": min(arr_delays),
                "flights_delayed": len([d for d in arr_delays if d > 15]),
                "on_time_rate": len([d for d in arr_delays if d <= 15]) / len(arr_delays)
            }
        
        return {
            "status": "success",
            "processing_time_seconds": result.processing_time_seconds,
            "files": {
                "total_processed": result.total_files_processed,
                "successful": result.successful_files,
                "failed": len(result.failed_files),
                "failed_files": result.failed_files
            },
            "flights": {
                "total": result.total_flights,
                "valid": result.valid_flights,
                "invalid": result.invalid_flights,
                "data_quality_rate": result.valid_flights / result.total_flights if result.total_flights > 0 else 0
            },
            "airlines": dict(sorted(airlines.items(), key=lambda x: x[1], reverse=True)),
            "top_routes": dict(sorted(routes.items(), key=lambda x: x[1], reverse=True)[:10]),
            "aircraft_types": dict(sorted(aircraft_types.items(), key=lambda x: x[1], reverse=True)),
            "date_range": {
                "start": min(dates).isoformat() if dates else None,
                "end": max(dates).isoformat() if dates else None,
                "unique_dates": len(dates),
                "total_days": (max(dates) - min(dates)).days + 1 if dates else 0
            },
            "delay_analysis": delay_summary,
            "errors": result.errors[:10],  # Show first 10 errors
            "total_errors": len(result.errors)
        }
    
    def validate_ingestion_accuracy(self, result: IngestionResult) -> Dict[str, Any]:
        """Validate the accuracy of the ingestion process."""
        validation_results = {
            "data_completeness": {},
            "timestamp_validation": {},
            "delay_calculation_validation": {},
            "data_quality_issues": []
        }
        
        if not result.flights:
            validation_results["data_quality_issues"].append("No flights to validate")
            return validation_results
        
        # Data completeness checks
        total_flights = len(result.flights)
        flights_with_std = len([f for f in result.flights if f.departure.scheduled])
        flights_with_atd = len([f for f in result.flights if f.departure.actual])
        flights_with_sta = len([f for f in result.flights if f.arrival.scheduled])
        flights_with_ata = len([f for f in result.flights if f.arrival.actual])
        
        validation_results["data_completeness"] = {
            "std_completeness": flights_with_std / total_flights,
            "atd_completeness": flights_with_atd / total_flights,
            "sta_completeness": flights_with_sta / total_flights,
            "ata_completeness": flights_with_ata / total_flights,
            "route_completeness": len([f for f in result.flights if f.origin and f.destination]) / total_flights
        }
        
        # Timestamp validation
        utc_flights = len([f for f in result.flights if f.departure.actual and f.departure.actual.tzinfo])
        validation_results["timestamp_validation"] = {
            "utc_conversion_rate": utc_flights / flights_with_atd if flights_with_atd > 0 else 0,
            "timezone_aware_flights": utc_flights
        }
        
        # Delay calculation validation
        flights_with_dep_delay = len([f for f in result.flights if f.dep_delay_min is not None])
        flights_with_arr_delay = len([f for f in result.flights if f.arr_delay_min is not None])
        
        validation_results["delay_calculation_validation"] = {
            "departure_delay_calculation_rate": flights_with_dep_delay / total_flights,
            "arrival_delay_calculation_rate": flights_with_arr_delay / total_flights,
            "flights_with_both_delays": len([f for f in result.flights if f.dep_delay_min is not None and f.arr_delay_min is not None])
        }
        
        # Identify data quality issues
        for flight in result.flights[:100]:  # Check first 100 flights for performance
            if not flight.is_valid():
                validation_results["data_quality_issues"].append(f"Invalid flight: {flight.flight_number}")
            
            if flight.departure.actual_str == "IMPUTED_FROM_SCHEDULED":
                validation_results["data_quality_issues"].append(f"Imputed departure time for: {flight.flight_number}")
            
            if flight.arrival.actual_str == "IMPUTED_FROM_SCHEDULED":
                validation_results["data_quality_issues"].append(f"Imputed arrival time for: {flight.flight_number}")
        
        return validation_results