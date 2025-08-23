"""Unified data loader that works with Excel, FlightAware, and FlightRadar24 sources."""

import os
from datetime import date, datetime
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from dataclasses import dataclass
import pytz

from ..services.data_ingestion import DataIngestionService, IngestionResult
from ..services.flightaware_ingestion import FlightAwareIngestionService, FlightAwareConfig
from ..services.flightradar24_ingestion import FlightRadar24IngestionService, FlightRadar24Config
from ..models.flight import Flight


class DataSource(Enum):
    """Supported data sources."""
    EXCEL = "excel"
    FLIGHTAWARE = "flightaware"
    FLIGHTRADAR24 = "flightradar24"
    AUTO = "auto"  # Automatically detect source


@dataclass
class UnifiedDataConfig:
    """Configuration for unified data loading."""
    # Excel configuration
    excel_directory: Optional[str] = None
    excel_file_patterns: List[str] = None
    
    # FlightAware configuration
    flightaware_api_key: Optional[str] = None
    flightaware_base_url: str = "https://aeroapi.flightaware.com/aeroapi"
    
    # FlightRadar24 configuration
    fr24_html_directory: Optional[str] = None
    
    # Common configuration
    airport_codes: List[str] = None
    default_timezone: str = "Asia/Kolkata"
    
    def __post_init__(self):
        if self.airport_codes is None:
            self.airport_codes = ["BOM", "DEL"]  # Default to Mumbai and Delhi
        
        if self.excel_file_patterns is None:
            self.excel_file_patterns = ["*.xlsx", "*.xls", "*Flight_Data*"]


class UnifiedDataLoader:
    """Unified data loader supporting multiple data sources."""
    
    def __init__(self, config: UnifiedDataConfig):
        self.config = config
        self.timezone = pytz.timezone(config.default_timezone)
        
        # Initialize individual services
        self.excel_service = DataIngestionService()
        
        # Initialize FlightAware service if API key is provided
        self.flightaware_service = None
        if config.flightaware_api_key:
            fa_config = FlightAwareConfig(api_key=config.flightaware_api_key)
            self.flightaware_service = FlightAwareIngestionService(fa_config)
        
        # Initialize FlightRadar24 service if HTML directory is provided
        self.fr24_service = None
        if config.fr24_html_directory:
            fr24_config = FlightRadar24Config(data_directory=config.fr24_html_directory)
            self.fr24_service = FlightRadar24IngestionService(fr24_config)
    
    def load_data(self, source: DataSource, 
                  start_date: Optional[date] = None,
                  end_date: Optional[date] = None,
                  file_paths: Optional[List[str]] = None,
                  airport_codes: Optional[List[str]] = None) -> IngestionResult:
        """
        Load flight data from specified source.
        
        Args:
            source: Data source to use
            start_date: Start date for data collection (for API sources)
            end_date: End date for data collection (for API sources)
            file_paths: Specific file paths to process (for file-based sources)
            airport_codes: Airport codes to collect data for
            
        Returns:
            IngestionResult with processed flight data
        """
        if airport_codes is None:
            airport_codes = self.config.airport_codes
        
        if source == DataSource.EXCEL:
            return self._load_excel_data(file_paths)
        
        elif source == DataSource.FLIGHTAWARE:
            if not self.flightaware_service:
                raise ValueError("FlightAware API key not configured")
            if not start_date or not end_date:
                raise ValueError("Start and end dates required for FlightAware data")
            return self.flightaware_service.ingest_airport_schedules(
                airport_codes, start_date, end_date
            )
        
        elif source == DataSource.FLIGHTRADAR24:
            if not self.fr24_service:
                raise ValueError("FlightRadar24 HTML directory not configured")
            return self._load_fr24_data(airport_codes, start_date, end_date, file_paths)
        
        elif source == DataSource.AUTO:
            return self._load_auto_detect(start_date, end_date, file_paths, airport_codes)
        
        else:
            raise ValueError(f"Unsupported data source: {source}")
    
    def _load_excel_data(self, file_paths: Optional[List[str]]) -> IngestionResult:
        """Load data from Excel files."""
        if file_paths:
            return self.excel_service.ingest_excel_files(file_paths)
        
        # Auto-discover Excel files
        if not self.config.excel_directory:
            raise ValueError("Excel directory not configured and no file paths provided")
        
        excel_files = self._discover_excel_files()
        return self.excel_service.ingest_excel_files(excel_files)
    
    def _load_fr24_data(self, airport_codes: List[str], 
                       start_date: Optional[date], end_date: Optional[date],
                       file_paths: Optional[List[str]]) -> IngestionResult:
        """Load data from FlightRadar24 HTML files."""
        if file_paths:
            return self.fr24_service.ingest_html_files(file_paths)
        
        # Load by airport and date range
        if start_date and end_date:
            combined_result = IngestionResult()
            for airport_code in airport_codes:
                result = self.fr24_service.ingest_airport_directory(
                    airport_code, start_date, end_date
                )
                combined_result.total_files_processed += result.total_files_processed
                combined_result.successful_files += result.successful_files
                combined_result.failed_files.extend(result.failed_files)
                combined_result.total_flights += result.total_flights
                combined_result.valid_flights += result.valid_flights
                combined_result.invalid_flights += result.invalid_flights
                combined_result.errors.extend(result.errors)
                combined_result.flights.extend(result.flights)
                combined_result.processing_time_seconds += result.processing_time_seconds
            
            return combined_result
        
        # Auto-discover HTML files
        html_files = self._discover_html_files()
        return self.fr24_service.ingest_html_files(html_files)
    
    def _load_auto_detect(self, start_date: Optional[date], end_date: Optional[date],
                         file_paths: Optional[List[str]], 
                         airport_codes: List[str]) -> IngestionResult:
        """Auto-detect and load from available sources."""
        results = []
        
        # Try FlightAware first (most reliable)
        if self.flightaware_service and start_date and end_date:
            try:
                fa_result = self.flightaware_service.ingest_airport_schedules(
                    airport_codes, start_date, end_date
                )
                if fa_result.valid_flights > 0:
                    results.append(("FlightAware", fa_result))
            except Exception as e:
                print(f"FlightAware ingestion failed: {e}")
        
        # Try Excel files
        try:
            excel_result = self._load_excel_data(file_paths)
            if excel_result.valid_flights > 0:
                results.append(("Excel", excel_result))
        except Exception as e:
            print(f"Excel ingestion failed: {e}")
        
        # Try FlightRadar24 HTML files
        if self.fr24_service:
            try:
                fr24_result = self._load_fr24_data(airport_codes, start_date, end_date, None)
                if fr24_result.valid_flights > 0:
                    results.append(("FlightRadar24", fr24_result))
            except Exception as e:
                print(f"FlightRadar24 ingestion failed: {e}")
        
        # Combine results
        if not results:
            return IngestionResult()  # Empty result
        
        # Return the result with the most valid flights
        best_source, best_result = max(results, key=lambda x: x[1].valid_flights)
        print(f"Auto-detection selected: {best_source} with {best_result.valid_flights} valid flights")
        
        return best_result
    
    def _discover_excel_files(self) -> List[str]:
        """Discover Excel files in configured directory."""
        excel_files = []
        
        if not self.config.excel_directory or not os.path.exists(self.config.excel_directory):
            return excel_files
        
        from pathlib import Path
        data_dir = Path(self.config.excel_directory)
        
        for pattern in self.config.excel_file_patterns:
            for file_path in data_dir.glob(pattern):
                if file_path.is_file():
                    excel_files.append(str(file_path))
        
        return excel_files
    
    def _discover_html_files(self) -> List[str]:
        """Discover HTML files in configured directory."""
        html_files = []
        
        if not self.config.fr24_html_directory or not os.path.exists(self.config.fr24_html_directory):
            return html_files
        
        from pathlib import Path
        data_dir = Path(self.config.fr24_html_directory)
        
        # Look for HTML files
        for file_path in data_dir.glob("*.html"):
            if file_path.is_file():
                html_files.append(str(file_path))
        
        return html_files
    
    def get_available_sources(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available data sources."""
        sources = {}
        
        # Check Excel availability
        excel_files = self._discover_excel_files()
        sources["excel"] = {
            "available": len(excel_files) > 0,
            "file_count": len(excel_files),
            "directory": self.config.excel_directory,
            "sample_files": excel_files[:3]
        }
        
        # Check FlightAware availability
        sources["flightaware"] = {
            "available": self.flightaware_service is not None,
            "api_configured": self.config.flightaware_api_key is not None,
            "base_url": self.config.flightaware_base_url
        }
        
        # Test FlightAware connection if available
        if self.flightaware_service:
            try:
                test_result = self.flightaware_service.test_connection()
                sources["flightaware"]["connection_test"] = test_result
            except Exception as e:
                sources["flightaware"]["connection_test"] = {
                    "status": "error",
                    "message": str(e)
                }
        
        # Check FlightRadar24 availability
        html_files = self._discover_html_files()
        sources["flightradar24"] = {
            "available": len(html_files) > 0,
            "file_count": len(html_files),
            "directory": self.config.fr24_html_directory,
            "sample_files": html_files[:3]
        }
        
        return sources
    
    def normalize_data_schema(self, flights: List[Flight]) -> List[Flight]:
        """
        Normalize flight data to match existing pipeline schema.
        Ensures compatibility with existing analytics and optimization services.
        """
        normalized_flights = []
        
        for flight in flights:
            # Ensure all required fields are present
            if not flight.flight_number:
                continue
            
            # Normalize flight number format (remove spaces, ensure uppercase)
            flight.flight_number = flight.flight_number.replace(' ', '').upper()
            
            # Ensure airline code is extracted
            if not flight.airline_code and flight.flight_number:
                # Extract airline code from flight number
                airline_code = ""
                for char in flight.flight_number:
                    if char.isalpha() or char.isdigit():
                        airline_code += char
                        if len(airline_code) >= 2 and not char.isalpha():
                            break
                    else:
                        break
                flight.airline_code = airline_code[:2] if len(airline_code) >= 2 else airline_code
            
            # Ensure airports have proper codes
            if flight.origin and not flight.origin.code:
                flight.origin.code = "UNK"
            if flight.destination and not flight.destination.code:
                flight.destination.code = "UNK"
            
            # Ensure aircraft type is set
            if not flight.aircraft_type:
                flight.aircraft_type = self._infer_aircraft_type(flight.flight_number)
            
            # Calculate delays if not already calculated
            if flight.departure.scheduled and flight.departure.actual and flight.dep_delay_min is None:
                scheduled_dt = datetime.combine(flight.flight_date, flight.departure.scheduled)
                if scheduled_dt.tzinfo is None:
                    scheduled_dt = pytz.UTC.localize(scheduled_dt)
                
                delay_seconds = (flight.departure.actual - scheduled_dt).total_seconds()
                flight.dep_delay_min = int(delay_seconds / 60)
            
            if flight.arrival.scheduled and flight.arrival.actual and flight.arr_delay_min is None:
                scheduled_dt = datetime.combine(flight.flight_date, flight.arrival.scheduled)
                if scheduled_dt.tzinfo is None:
                    scheduled_dt = pytz.UTC.localize(scheduled_dt)
                
                delay_seconds = (flight.arrival.actual - scheduled_dt).total_seconds()
                flight.arr_delay_min = int(delay_seconds / 60)
            
            # Ensure timezone-aware timestamps
            if flight.departure.actual and flight.departure.actual.tzinfo is None:
                flight.departure.actual = pytz.UTC.localize(flight.departure.actual)
            
            if flight.arrival.actual and flight.arrival.actual.tzinfo is None:
                flight.arrival.actual = pytz.UTC.localize(flight.arrival.actual)
            
            normalized_flights.append(flight)
        
        return normalized_flights
    
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
        
        # Extract airline code
        airline_code = ""
        for char in flight_number:
            if char.isalpha() or char.isdigit():
                airline_code += char
                if len(airline_code) >= 2:
                    break
            else:
                break
        
        return airline_aircraft_map.get(airline_code, "UNKNOWN")
    
    def validate_data_quality(self, result: IngestionResult) -> Dict[str, Any]:
        """Validate data quality across all sources."""
        if not result.flights:
            return {
                "status": "no_data",
                "message": "No flights found",
                "recommendations": ["Check data source configuration", "Verify date ranges"]
            }
        
        # Analyze data completeness
        total_flights = len(result.flights)
        flights_with_times = len([f for f in result.flights 
                                if f.departure.scheduled and f.arrival.scheduled])
        flights_with_routes = len([f for f in result.flights 
                                 if f.origin and f.destination])
        flights_with_aircraft = len([f for f in result.flights 
                                   if f.aircraft_type and f.aircraft_type != "UNKNOWN"])
        
        # Analyze data sources
        source_distribution = {}
        for flight in result.flights:
            source = flight.data_source
            source_distribution[source] = source_distribution.get(source, 0) + 1
        
        # Analyze airports
        airports = set()
        for flight in result.flights:
            if flight.origin:
                airports.add(flight.origin.code)
            if flight.destination:
                airports.add(flight.destination.code)
        
        # Quality score calculation
        completeness_score = (flights_with_times + flights_with_routes + flights_with_aircraft) / (3 * total_flights)
        
        recommendations = []
        if completeness_score < 0.8:
            recommendations.append("Data completeness is below 80% - consider using multiple sources")
        if len(source_distribution) == 1:
            recommendations.append("Consider using multiple data sources for better coverage")
        if len(airports) < 2:
            recommendations.append("Limited airport coverage - verify airport codes configuration")
        
        return {
            "status": "success",
            "total_flights": total_flights,
            "completeness_score": completeness_score,
            "data_completeness": {
                "flights_with_times": flights_with_times,
                "flights_with_routes": flights_with_routes,
                "flights_with_aircraft": flights_with_aircraft,
                "time_completeness_rate": flights_with_times / total_flights,
                "route_completeness_rate": flights_with_routes / total_flights,
                "aircraft_completeness_rate": flights_with_aircraft / total_flights
            },
            "source_distribution": source_distribution,
            "airport_coverage": list(airports),
            "recommendations": recommendations
        }