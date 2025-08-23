"""Excel data parsing utilities for complex flight data structures."""

import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, date
from .flight import Flight, FlightDataBatch
from .validation import DataValidator


class ExcelFlightParser:
    """Parser for complex Excel flight data structures."""
    
    def __init__(self):
        self.current_flight_number = None
        self.current_flight_data = {}
    
    def parse_excel_file(self, file_path: str) -> FlightDataBatch:
        """Parse Excel file and return batch of flights."""
        batch = FlightDataBatch(source_file=file_path)
        
        try:
            # Get all sheet names
            xl_file = pd.ExcelFile(file_path)
            
            for sheet_name in xl_file.sheet_names:
                sheet_flights = self._parse_sheet(file_path, sheet_name)
                for flight in sheet_flights:
                    batch.add_flight(flight)
                    
        except Exception as e:
            batch.add_error(f"Error reading Excel file: {str(e)}")
        
        return batch
    
    def _parse_sheet(self, file_path: str, sheet_name: str) -> List[Flight]:
        """Parse a single sheet and return list of flights."""
        flights = []
        
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            flights = self._parse_dataframe(df, sheet_name)
            
        except Exception as e:
            print(f"Error parsing sheet {sheet_name}: {e}")
        
        return flights
    
    def _parse_dataframe(self, df: pd.DataFrame, time_period: str) -> List[Flight]:
        """Parse DataFrame with complex flight data structure."""
        flights = []
        current_flight_number = None
        
        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            
            # Check if this row has a flight number
            flight_number = self._extract_flight_number(row_dict.get('Flight Number'))
            
            if flight_number:
                # This row contains a flight number
                current_flight_number = flight_number
                
                # Check if this row also has complete flight data
                if self._has_complete_flight_data(row_dict):
                    flight = self._create_flight_from_row(row_dict, current_flight_number, time_period)
                    if flight:
                        flights.append(flight)
            
            elif current_flight_number and self._has_complete_flight_data(row_dict):
                # This row has flight data for the current flight number
                flight = self._create_flight_from_row(row_dict, current_flight_number, time_period)
                if flight:
                    flights.append(flight)
        
        return flights
    
    def _extract_flight_number(self, flight_number_value: Any) -> Optional[str]:
        """Extract and validate flight number from cell value."""
        if pd.isna(flight_number_value):
            return None
        
        flight_number = str(flight_number_value).strip()
        
        # Skip empty strings or just spaces
        if not flight_number or flight_number.isspace():
            return None
        
        # Normalize flight number
        normalized = DataValidator.normalize_flight_number(flight_number)
        
        # Validate it looks like a flight number (letters + numbers)
        if len(normalized) >= 3 and any(c.isalpha() for c in normalized) and any(c.isdigit() for c in normalized):
            return normalized
        
        return None
    
    def _has_complete_flight_data(self, row_dict: Dict[str, Any]) -> bool:
        """Check if row has complete flight data (route and times)."""
        required_fields = ['From', 'To', 'STD']
        
        for field in required_fields:
            value = row_dict.get(field)
            if pd.isna(value) or (isinstance(value, str) and value.strip() == ""):
                return False
        
        return True
    
    def _create_flight_from_row(self, row_dict: Dict[str, Any], flight_number: str, time_period: str) -> Optional[Flight]:
        """Create flight object from row data and flight number."""
        try:
            # Add flight number to row data
            enhanced_row_dict = row_dict.copy()
            enhanced_row_dict['Flight Number'] = flight_number
            
            # Handle pandas Timestamp for date fields
            for date_field in ['Date', 'Unnamed: 2']:
                if date_field in enhanced_row_dict:
                    value = enhanced_row_dict[date_field]
                    if isinstance(value, pd.Timestamp):
                        enhanced_row_dict[date_field] = value.strftime('%Y-%m-%d')
            
            # Create flight using existing validation logic
            flight = DataValidator.create_flight_from_raw_data(enhanced_row_dict, time_period)
            
            return flight
            
        except Exception as e:
            print(f"Error creating flight from row: {e}")
            return None
    
    def get_flight_summary(self, batch: FlightDataBatch) -> Dict[str, Any]:
        """Get summary statistics for parsed flights."""
        valid_flights = batch.get_valid_flights()
        
        if not valid_flights:
            return {
                "total_flights": len(batch.flights),
                "valid_flights": 0,
                "summary": "No valid flights found"
            }
        
        # Analyze flights
        airlines = {}
        routes = {}
        aircraft_types = {}
        dates = set()
        
        for flight in valid_flights:
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
        
        return {
            "total_flights": len(batch.flights),
            "valid_flights": len(valid_flights),
            "invalid_flights": len(batch.flights) - len(valid_flights),
            "error_count": len(batch.errors),
            "airlines": dict(sorted(airlines.items(), key=lambda x: x[1], reverse=True)),
            "top_routes": dict(sorted(routes.items(), key=lambda x: x[1], reverse=True)[:10]),
            "aircraft_types": dict(sorted(aircraft_types.items(), key=lambda x: x[1], reverse=True)),
            "date_range": {
                "start": min(dates).isoformat() if dates else None,
                "end": max(dates).isoformat() if dates else None,
                "unique_dates": len(dates)
            },
            "time_periods": list(set(f.time_period for f in valid_flights if f.time_period))
        }