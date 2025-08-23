"""FlightRadar24 HTML parser for saved page data extraction."""

import os
import re
from datetime import datetime, date, time, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
import pytz
from dataclasses import dataclass
from bs4 import BeautifulSoup

from ..models.flight import Flight, FlightTime, Airport, FlightDataBatch
from ..services.data_ingestion import IngestionResult


@dataclass
class FlightRadar24Config:
    """Configuration for FlightRadar24 HTML parsing."""
    data_directory: str
    airport_codes: List[str] = None
    date_format: str = "%Y-%m-%d"


class FlightRadar24IngestionService:
    """Service for ingesting flight data from saved FlightRadar24 HTML pages."""
    
    def __init__(self, config: FlightRadar24Config):
        self.config = config
        self.ist_timezone = pytz.timezone('Asia/Kolkata')
        self.utc_timezone = pytz.UTC
    
    def ingest_html_files(self, html_file_paths: List[str]) -> IngestionResult:
        """
        Process multiple FlightRadar24 HTML files and return consolidated flight data.
        
        Args:
            html_file_paths: List of paths to HTML files
            
        Returns:
            IngestionResult with processed flight data
        """
        start_time = datetime.now()
        result = IngestionResult()
        result.total_files_processed = len(html_file_paths)
        
        for file_path in html_file_paths:
            try:
                if not os.path.exists(file_path):
                    result.failed_files.append(file_path)
                    result.errors.append(f"File not found: {file_path}")
                    continue
                
                # Process single HTML file
                batch = self._process_html_file(file_path)
                result.add_batch(batch)
                result.successful_files += 1
                
            except Exception as e:
                result.failed_files.append(file_path)
                result.errors.append(f"Error processing {file_path}: {str(e)}")
        
        result.processing_time_seconds = (datetime.now() - start_time).total_seconds()
        return result
    
    def ingest_airport_directory(self, airport_code: str, 
                                start_date: date, end_date: date) -> IngestionResult:
        """
        Ingest all HTML files for a specific airport and date range.
        
        Args:
            airport_code: IATA airport code (e.g., 'BOM', 'DEL')
            start_date: Start date for data collection
            end_date: End date for data collection
            
        Returns:
            IngestionResult with processed flight data
        """
        html_files = self._find_html_files(airport_code, start_date, end_date)
        return self.ingest_html_files(html_files)
    
    def _find_html_files(self, airport_code: str, 
                        start_date: date, end_date: date) -> List[str]:
        """Find HTML files for airport and date range."""
        html_files = []
        data_dir = Path(self.config.data_directory)
        
        if not data_dir.exists():
            return html_files
        
        # Look for files matching patterns like:
        # - BOM_2024-01-15_departures.html
        # - DEL_2024-01-15_arrivals.html
        # - flightradar24_BOM_20240115.html
        
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            date_str_compact = current_date.strftime("%Y%m%d")
            
            # Common file patterns
            patterns = [
                f"{airport_code}_{date_str}_*.html",
                f"*{airport_code}*{date_str}*.html",
                f"*{airport_code}*{date_str_compact}*.html",
                f"flightradar24_{airport_code}_{date_str_compact}.html"
            ]
            
            for pattern in patterns:
                for file_path in data_dir.glob(pattern):
                    if file_path.is_file():
                        html_files.append(str(file_path))
            
            current_date += timedelta(days=1)
        
        return list(set(html_files))  # Remove duplicates
    
    def _process_html_file(self, file_path: str) -> FlightDataBatch:
        """Process a single FlightRadar24 HTML file."""
        batch = FlightDataBatch(source_file=file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract airport code and date from filename or content
            airport_code, flight_date = self._extract_metadata(file_path, soup)
            
            # Parse flight table data
            flights = self._parse_flight_table(soup, airport_code, flight_date)
            
            for flight in flights:
                batch.add_flight(flight)
                
        except Exception as e:
            batch.add_error(f"Error parsing HTML file {file_path}: {str(e)}")
        
        return batch
    
    def _extract_metadata(self, file_path: str, soup: BeautifulSoup) -> tuple[str, Optional[date]]:
        """Extract airport code and date from filename or HTML content."""
        # Try to extract from filename first
        filename = os.path.basename(file_path)
        
        # Pattern matching for airport codes
        airport_match = re.search(r'([A-Z]{3})', filename.upper())
        airport_code = airport_match.group(1) if airport_match else "UNK"
        
        # Pattern matching for dates
        date_match = re.search(r'(\d{4}[-_]?\d{2}[-_]?\d{2})', filename)
        flight_date = None
        
        if date_match:
            date_str = date_match.group(1).replace('_', '-')
            try:
                if len(date_str) == 8:  # YYYYMMDD format
                    flight_date = datetime.strptime(date_str, "%Y%m%d").date()
                else:  # YYYY-MM-DD format
                    flight_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            except ValueError:
                pass
        
        # Try to extract from HTML content if not found in filename
        if airport_code == "UNK" or flight_date is None:
            # Look for airport information in page title or headers
            title = soup.find('title')
            if title:
                title_text = title.get_text()
                airport_match = re.search(r'([A-Z]{3})', title_text)
                if airport_match:
                    airport_code = airport_match.group(1)
            
            # Look for date information in the page
            date_elements = soup.find_all(text=re.compile(r'\d{4}[-/]\d{2}[-/]\d{2}'))
            if date_elements and flight_date is None:
                for date_text in date_elements:
                    date_match = re.search(r'(\d{4}[-/]\d{2}[-/]\d{2})', date_text)
                    if date_match:
                        try:
                            flight_date = datetime.strptime(
                                date_match.group(1).replace('/', '-'), 
                                "%Y-%m-%d"
                            ).date()
                            break
                        except ValueError:
                            continue
        
        return airport_code, flight_date
    
    def _parse_flight_table(self, soup: BeautifulSoup, airport_code: str, 
                          flight_date: Optional[date]) -> List[Flight]:
        """Parse flight data from HTML table."""
        flights = []
        
        # Look for flight data tables - FlightRadar24 uses various table structures
        tables = soup.find_all('table')
        
        for table in tables:
            # Check if this looks like a flight data table
            headers = table.find_all('th')
            if not headers:
                continue
            
            header_texts = [th.get_text().strip().lower() for th in headers]
            
            # Look for common flight table headers
            flight_indicators = ['flight', 'airline', 'aircraft', 'departure', 'arrival', 'status']
            if not any(indicator in ' '.join(header_texts) for indicator in flight_indicators):
                continue
            
            # Parse table rows
            rows = table.find_all('tr')[1:]  # Skip header row
            
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) < 4:  # Need at least flight, route, time, status
                    continue
                
                flight = self._parse_flight_row(cells, airport_code, flight_date, header_texts)
                if flight:
                    flights.append(flight)
        
        # If no table found, try to parse from div/span structures
        if not flights:
            flights = self._parse_flight_divs(soup, airport_code, flight_date)
        
        return flights
    
    def _parse_flight_row(self, cells: List, airport_code: str, 
                         flight_date: Optional[date], headers: List[str]) -> Optional[Flight]:
        """Parse a single flight row from table."""
        try:
            if len(cells) < 4:
                return None
            
            flight = Flight()
            flight.data_source = "flightradar24"
            flight.flight_date = flight_date or date.today()
            
            # Extract cell texts
            cell_texts = [cell.get_text().strip() for cell in cells]
            
            # Try to identify columns based on content patterns
            for i, text in enumerate(cell_texts):
                # Flight number pattern (e.g., "AI 123", "6E2345")
                if re.match(r'^[A-Z0-9]{2,3}\s*\d{3,4}$', text.replace(' ', '')):
                    flight.flight_number = text.replace(' ', '')
                
                # Time pattern (e.g., "14:30", "2:30 PM")
                elif re.match(r'^\d{1,2}:\d{2}(\s*(AM|PM))?$', text, re.IGNORECASE):
                    time_obj = self._parse_time_string(text)
                    if time_obj:
                        # Determine if this is departure or arrival based on context
                        if not flight.departure.scheduled:
                            flight.departure.scheduled = time_obj
                        elif not flight.arrival.scheduled:
                            flight.arrival.scheduled = time_obj
                
                # Airport code pattern (e.g., "BOM", "DEL")
                elif re.match(r'^[A-Z]{3}$', text):
                    if text != airport_code:
                        # This is the other airport (origin or destination)
                        other_airport = Airport(code=text, name=f"{text} Airport", city=text)
                        if not flight.origin:
                            flight.origin = other_airport
                        elif not flight.destination:
                            flight.destination = other_airport
                
                # Aircraft type pattern (e.g., "A320", "B737")
                elif re.match(r'^[AB]\d{3}[A-Z]?$', text):
                    flight.aircraft_type = text
                
                # Status information
                elif text.lower() in ['departed', 'arrived', 'delayed', 'cancelled', 'scheduled']:
                    # Update flight status based on text
                    pass
            
            # Set the known airport as origin or destination
            known_airport = Airport(code=airport_code, name=f"{airport_code} Airport", city=airport_code)
            
            if not flight.origin and not flight.destination:
                # Assume this airport is origin for departures, destination for arrivals
                # This is a simplification - in practice, we'd need more context
                flight.origin = known_airport
            elif flight.origin and not flight.destination:
                flight.destination = known_airport
            elif not flight.origin and flight.destination:
                flight.origin = known_airport
            
            # Validate minimum required fields
            if flight.flight_number and (flight.origin or flight.destination):
                return flight
            
            return None
            
        except Exception as e:
            return None
    
    def _parse_flight_divs(self, soup: BeautifulSoup, airport_code: str, 
                          flight_date: Optional[date]) -> List[Flight]:
        """Parse flight data from div/span structures when no table is found."""
        flights = []
        
        # Look for flight information in div structures
        # This is a fallback method for different HTML layouts
        flight_divs = soup.find_all('div', class_=re.compile(r'flight|row|item', re.IGNORECASE))
        
        for div in flight_divs:
            text_content = div.get_text()
            
            # Look for flight number patterns
            flight_match = re.search(r'([A-Z0-9]{2,3}\s*\d{3,4})', text_content)
            if not flight_match:
                continue
            
            flight = Flight()
            flight.flight_number = flight_match.group(1).replace(' ', '')
            flight.data_source = "flightradar24"
            flight.flight_date = flight_date or date.today()
            
            # Look for time patterns
            time_matches = re.findall(r'\d{1,2}:\d{2}(?:\s*[AP]M)?', text_content, re.IGNORECASE)
            if len(time_matches) >= 1:
                flight.departure.scheduled = self._parse_time_string(time_matches[0])
            if len(time_matches) >= 2:
                flight.arrival.scheduled = self._parse_time_string(time_matches[1])
            
            # Look for airport codes
            airport_matches = re.findall(r'([A-Z]{3})', text_content)
            other_airports = [code for code in airport_matches if code != airport_code]
            
            if other_airports:
                other_airport = Airport(
                    code=other_airports[0], 
                    name=f"{other_airports[0]} Airport", 
                    city=other_airports[0]
                )
                flight.origin = Airport(code=airport_code, name=f"{airport_code} Airport", city=airport_code)
                flight.destination = other_airport
            
            if flight.flight_number and (flight.origin or flight.destination):
                flights.append(flight)
        
        return flights
    
    def _parse_time_string(self, time_str: str) -> Optional[time]:
        """Parse time string into time object."""
        try:
            time_str = time_str.strip()
            
            # Handle AM/PM format
            if re.search(r'[AP]M', time_str, re.IGNORECASE):
                return datetime.strptime(time_str.upper(), '%I:%M %p').time()
            
            # Handle 24-hour format
            elif ':' in time_str:
                return datetime.strptime(time_str, '%H:%M').time()
            
            return None
            
        except ValueError:
            return None
    
    def validate_html_structure(self, file_path: str) -> Dict[str, Any]:
        """Validate HTML file structure for flight data extraction."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Check for common FlightRadar24 elements
            has_tables = len(soup.find_all('table')) > 0
            has_flight_divs = len(soup.find_all('div', class_=re.compile(r'flight', re.IGNORECASE))) > 0
            
            # Look for flight number patterns
            text_content = soup.get_text()
            flight_patterns = re.findall(r'([A-Z0-9]{2,3}\s*\d{3,4})', text_content)
            
            # Look for time patterns
            time_patterns = re.findall(r'\d{1,2}:\d{2}(?:\s*[AP]M)?', text_content, re.IGNORECASE)
            
            # Look for airport codes
            airport_patterns = re.findall(r'([A-Z]{3})', text_content)
            
            return {
                "status": "valid" if (has_tables or has_flight_divs) and flight_patterns else "invalid",
                "has_tables": has_tables,
                "has_flight_divs": has_flight_divs,
                "flight_numbers_found": len(flight_patterns),
                "time_patterns_found": len(time_patterns),
                "airport_codes_found": len(set(airport_patterns)),
                "sample_flight_numbers": flight_patterns[:5],
                "sample_times": time_patterns[:5],
                "sample_airports": list(set(airport_patterns))[:5]
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error validating HTML file: {str(e)}"
            }