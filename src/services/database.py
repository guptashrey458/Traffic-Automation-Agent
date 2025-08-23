"""DuckDB database service for flight data storage and querying."""

import os
import duckdb
import json
from datetime import datetime, date
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import pandas as pd
from dataclasses import dataclass, field

from ..models.flight import Flight, FlightDataBatch
from enum import Enum


class DelayStatus(Enum):
    """Delay status based on color coding system"""
    ON_TIME = "green"      # ≤ 15 minutes
    MODERATE = "yellow"    # 16-60 minutes  
    CRITICAL = "red"       # > 60 minutes


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    db_path: str = "data/flights.duckdb"
    parquet_export_path: str = "data/parquet"
    enable_wal: bool = True
    memory_limit: str = "2GB"
    threads: int = 4


@dataclass
class QueryResult:
    """Result of a database query."""
    data: List[Dict[str, Any]] = field(default_factory=list)
    row_count: int = 0
    execution_time_ms: float = 0.0
    query: str = ""
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert query result to pandas DataFrame."""
        return pd.DataFrame(self.data)


class FlightDatabaseService:
    """Service for managing flight data in DuckDB."""
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self.connection: Optional[duckdb.DuckDBPyConnection] = None
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """Ensure required directories exist."""
        os.makedirs(os.path.dirname(self.config.db_path), exist_ok=True)
        os.makedirs(self.config.parquet_export_path, exist_ok=True)
    
    def connect(self) -> duckdb.DuckDBPyConnection:
        """Establish connection to DuckDB database."""
        if self.connection is None:
            self.connection = duckdb.connect(self.config.db_path)
            self._configure_database()
            self._create_schema()
        return self.connection
    
    def _configure_database(self) -> None:
        """Configure DuckDB settings for optimal performance."""
        if not self.connection:
            return
        
        # Configure memory and performance settings
        self.connection.execute(f"SET memory_limit='{self.config.memory_limit}'")
        self.connection.execute(f"SET threads={self.config.threads}")
        
        # Enable Write-Ahead Logging for better concurrency
        if self.config.enable_wal:
            self.connection.execute("PRAGMA enable_checkpoint_on_shutdown")
        
        # Configure for time-series workloads
        self.connection.execute("SET enable_progress_bar=false")
        self.connection.execute("SET preserve_insertion_order=false")
    
    def _create_schema(self) -> None:
        """Create optimized schema for flight data."""
        if not self.connection:
            return
        
        # Create main flights table with optimized schema
        create_flights_table = """
        CREATE TABLE IF NOT EXISTS flights (
            flight_id VARCHAR PRIMARY KEY,
            flight_number VARCHAR NOT NULL,
            airline_code VARCHAR(3),
            
            -- Route information
            origin_code VARCHAR(3) NOT NULL,
            origin_name VARCHAR(100),
            destination_code VARCHAR(3) NOT NULL,
            destination_name VARCHAR(100),
            route VARCHAR(7) NOT NULL,  -- "BOM-DEL" format
            
            -- Aircraft information
            aircraft_type VARCHAR(10),
            aircraft_registration VARCHAR(20),
            
            -- Timing information (all in UTC)
            flight_date DATE NOT NULL,
            std_utc TIMESTAMP,  -- Scheduled Time of Departure
            atd_utc TIMESTAMP,  -- Actual Time of Departure
            sta_utc TIMESTAMP,  -- Scheduled Time of Arrival
            ata_utc TIMESTAMP,  -- Actual Time of Arrival
            
            -- Calculated delays (in minutes)
            dep_delay_min INTEGER,
            arr_delay_min INTEGER,
            
            -- Delay status indicators (green/yellow/red)
            dep_delay_status VARCHAR(10),
            arr_delay_status VARCHAR(10),
            
            -- Operational data
            flight_duration_min INTEGER,
            status VARCHAR(20) DEFAULT 'scheduled',
            time_period VARCHAR(20),
            
            -- Metadata
            data_source VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
            -- Raw data for debugging (JSON)
            raw_data VARCHAR
        )
        """
        
        self.connection.execute(create_flights_table)
        
        # Create indexes for fast time-based queries
        self._create_indexes()
        
        # Create views for common queries
        self._create_views()
    
    def _create_indexes(self) -> None:
        """Create indexes for optimal query performance."""
        if not self.connection:
            return
        
        indexes = [
            # Time-based indexes for peak analysis
            "CREATE INDEX IF NOT EXISTS idx_flights_date ON flights(flight_date)",
            "CREATE INDEX IF NOT EXISTS idx_flights_std_utc ON flights(std_utc)",
            "CREATE INDEX IF NOT EXISTS idx_flights_atd_utc ON flights(atd_utc)",
            "CREATE INDEX IF NOT EXISTS idx_flights_sta_utc ON flights(sta_utc)",
            "CREATE INDEX IF NOT EXISTS idx_flights_ata_utc ON flights(ata_utc)",
            
            # Route and airline indexes
            "CREATE INDEX IF NOT EXISTS idx_flights_route ON flights(route)",
            "CREATE INDEX IF NOT EXISTS idx_flights_origin ON flights(origin_code)",
            "CREATE INDEX IF NOT EXISTS idx_flights_destination ON flights(destination_code)",
            "CREATE INDEX IF NOT EXISTS idx_flights_airline ON flights(airline_code)",
            
            # Aircraft and operational indexes
            "CREATE INDEX IF NOT EXISTS idx_flights_aircraft_type ON flights(aircraft_type)",
            "CREATE INDEX IF NOT EXISTS idx_flights_aircraft_reg ON flights(aircraft_registration)",
            "CREATE INDEX IF NOT EXISTS idx_flights_status ON flights(status)",
            
            # Delay analysis indexes
            "CREATE INDEX IF NOT EXISTS idx_flights_dep_delay ON flights(dep_delay_min)",
            "CREATE INDEX IF NOT EXISTS idx_flights_arr_delay ON flights(arr_delay_min)",
            
            # Composite indexes for common query patterns
            "CREATE INDEX IF NOT EXISTS idx_flights_date_origin ON flights(flight_date, origin_code)",
            "CREATE INDEX IF NOT EXISTS idx_flights_date_route ON flights(flight_date, route)",
            "CREATE INDEX IF NOT EXISTS idx_flights_airline_date ON flights(airline_code, flight_date)",
        ]
        
        for index_sql in indexes:
            try:
                self.connection.execute(index_sql)
            except Exception as e:
                print(f"Warning: Could not create index: {e}")
    
    def _create_views(self) -> None:
        """Create views for common analytical queries."""
        if not self.connection:
            return
        
        # View for daily flight statistics
        daily_stats_view = """
        CREATE OR REPLACE VIEW daily_flight_stats AS
        SELECT 
            flight_date,
            origin_code,
            destination_code,
            route,
            COUNT(*) as total_flights,
            COUNT(CASE WHEN atd_utc IS NOT NULL THEN 1 END) as departed_flights,
            COUNT(CASE WHEN ata_utc IS NOT NULL THEN 1 END) as arrived_flights,
            AVG(dep_delay_min) as avg_dep_delay,
            AVG(arr_delay_min) as avg_arr_delay,
            COUNT(CASE WHEN dep_delay_min > 15 THEN 1 END) as delayed_departures,
            COUNT(CASE WHEN arr_delay_min > 15 THEN 1 END) as delayed_arrivals,
            MIN(std_utc) as first_departure,
            MAX(std_utc) as last_departure
        FROM flights
        WHERE flight_date IS NOT NULL
        GROUP BY flight_date, origin_code, destination_code, route
        """
        
        # View for hourly traffic patterns
        hourly_traffic_view = """
        CREATE OR REPLACE VIEW hourly_traffic_patterns AS
        SELECT 
            flight_date,
            origin_code,
            EXTRACT(hour FROM std_utc) as departure_hour,
            COUNT(*) as scheduled_departures,
            COUNT(CASE WHEN atd_utc IS NOT NULL THEN 1 END) as actual_departures,
            AVG(dep_delay_min) as avg_delay,
            COUNT(CASE WHEN dep_delay_min > 15 THEN 1 END) as delayed_count
        FROM flights
        WHERE std_utc IS NOT NULL
        GROUP BY flight_date, origin_code, EXTRACT(hour FROM std_utc)
        ORDER BY flight_date, origin_code, departure_hour
        """
        
        # View for airline performance
        airline_performance_view = """
        CREATE OR REPLACE VIEW airline_performance AS
        SELECT 
            airline_code,
            COUNT(*) as total_flights,
            AVG(dep_delay_min) as avg_dep_delay,
            AVG(arr_delay_min) as avg_arr_delay,
            COUNT(CASE WHEN dep_delay_min > 15 THEN 1 END) * 100.0 / COUNT(*) as dep_delay_rate,
            COUNT(CASE WHEN arr_delay_min > 15 THEN 1 END) * 100.0 / COUNT(*) as arr_delay_rate,
            COUNT(DISTINCT route) as routes_served,
            COUNT(DISTINCT aircraft_type) as aircraft_types_used
        FROM flights
        WHERE airline_code IS NOT NULL
        GROUP BY airline_code
        ORDER BY total_flights DESC
        """
        
        views = [daily_stats_view, hourly_traffic_view, airline_performance_view]
        
        for view_sql in views:
            try:
                self.connection.execute(view_sql)
            except Exception as e:
                print(f"Warning: Could not create view: {e}")
    
    def store_flights(self, flights: List[Flight]) -> Dict[str, Any]:
        """
        Store flight data in the database.
        
        Args:
            flights: List of Flight objects to store
            
        Returns:
            Dictionary with storage statistics
        """
        if not flights:
            return {"stored": 0, "errors": [], "message": "No flights to store"}
        
        conn = self.connect()
        start_time = datetime.now()
        stored_count = 0
        errors = []
        
        try:
            # Convert flights to records for bulk insert
            records = []
            for flight in flights:
                try:
                    record = self._flight_to_record(flight)
                    records.append(record)
                except Exception as e:
                    errors.append(f"Error converting flight {flight.flight_id}: {str(e)}")
            
            if records:
                # Use DuckDB's efficient bulk insert with upsert
                df = pd.DataFrame(records)
                
                # Delete existing records with same flight_ids to handle updates
                flight_ids = [r['flight_id'] for r in records]
                if flight_ids:
                    placeholders = ','.join(['?' for _ in flight_ids])
                    conn.execute(f"DELETE FROM flights WHERE flight_id IN ({placeholders})", flight_ids)
                
                # Insert new records
                conn.register('flight_data', df)
                conn.execute("""
                    INSERT INTO flights SELECT * FROM flight_data
                """)
                
                stored_count = len(records)
            
        except Exception as e:
            errors.append(f"Database error: {str(e)}")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "stored": stored_count,
            "errors": errors,
            "processing_time_seconds": processing_time,
            "total_attempted": len(flights),
            "success_rate": stored_count / len(flights) if flights else 0
        }
    
    def _flight_to_record(self, flight: Flight) -> Dict[str, Any]:
        """Convert Flight object to database record."""
        return {
            "flight_id": flight.flight_id,
            "flight_number": flight.flight_number,
            "airline_code": flight.airline_code,
            "origin_code": flight.origin.code if flight.origin else None,
            "origin_name": flight.origin.name if flight.origin else None,
            "destination_code": flight.destination.code if flight.destination else None,
            "destination_name": flight.destination.name if flight.destination else None,
            "route": flight.get_route_key(),
            "aircraft_type": flight.aircraft_type,
            "aircraft_registration": flight.aircraft_registration,
            "flight_date": flight.flight_date,
            "std_utc": datetime.combine(flight.flight_date, flight.departure.scheduled) if flight.flight_date and flight.departure.scheduled else None,
            "atd_utc": flight.departure.actual,
            "sta_utc": datetime.combine(flight.flight_date, flight.arrival.scheduled) if flight.flight_date and flight.arrival.scheduled else None,
            "ata_utc": flight.arrival.actual,
            "dep_delay_min": flight.dep_delay_min,
            "arr_delay_min": flight.arr_delay_min,
            "dep_delay_status": self._classify_delay_status(flight.dep_delay_min),
            "arr_delay_status": self._classify_delay_status(flight.arr_delay_min),
            "flight_duration_min": self._parse_duration_to_minutes(flight.flight_duration),
            "status": flight.status.value,
            "time_period": flight.time_period,
            "data_source": flight.data_source,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "raw_data": json.dumps(flight.raw_data) if flight.raw_data else None
        }
    
    def _classify_delay_status(self, delay_minutes: Optional[int]) -> Optional[str]:
        """Classify delay into green/yellow/red status"""
        if delay_minutes is None:
            return None
        
        if delay_minutes <= 15:
            return DelayStatus.ON_TIME.value
        elif delay_minutes <= 60:
            return DelayStatus.MODERATE.value
        else:
            return DelayStatus.CRITICAL.value
    
    def _parse_duration_to_minutes(self, duration_str: Optional[str]) -> Optional[int]:
        """Parse duration string like '2h 10m' to minutes."""
        if not duration_str:
            return None
        
        try:
            import re
            # Match patterns like "2h 10m", "1h", "45m"
            hours_match = re.search(r'(\d+)h', duration_str)
            minutes_match = re.search(r'(\d+)m', duration_str)
            
            hours = int(hours_match.group(1)) if hours_match else 0
            minutes = int(minutes_match.group(1)) if minutes_match else 0
            
            return hours * 60 + minutes
        except:
            return None
    
    def query_flights_by_date_range(self, start_date: date, end_date: date, 
                                   airport_code: Optional[str] = None) -> QueryResult:
        """Query flights within a date range, optionally filtered by airport."""
        conn = self.connect()
        start_time = datetime.now()
        
        base_query = """
        SELECT * FROM flights 
        WHERE flight_date BETWEEN ? AND ?
        """
        params = [start_date, end_date]
        
        if airport_code:
            base_query += " AND (origin_code = ? OR destination_code = ?)"
            params.extend([airport_code, airport_code])
        
        base_query += " ORDER BY flight_date, std_utc"
        
        try:
            result = conn.execute(base_query, params).fetchall()
            columns = [desc[0] for desc in conn.description]
            data = [dict(zip(columns, row)) for row in result]
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return QueryResult(
                data=data,
                row_count=len(data),
                execution_time_ms=execution_time,
                query=base_query
            )
        except Exception as e:
            return QueryResult(
                data=[],
                row_count=0,
                execution_time_ms=0,
                query=f"ERROR: {str(e)}"
            )
    
    def query_peak_traffic(self, airport_code: str, date_filter: Optional[date] = None,
                          bucket_minutes: int = 10) -> QueryResult:
        """Query peak traffic patterns for an airport."""
        conn = self.connect()
        start_time = datetime.now()
        
        # Create time buckets for analysis
        query = f"""
        WITH time_buckets AS (
            SELECT 
                flight_date,
                origin_code,
                -- Create {bucket_minutes}-minute buckets
                DATE_TRUNC('minute', std_utc) - 
                INTERVAL (EXTRACT(minute FROM std_utc)::INTEGER % {bucket_minutes}) MINUTE as time_bucket,
                COUNT(*) as flight_count,
                COUNT(CASE WHEN dep_delay_min > 15 THEN 1 END) as delayed_count,
                AVG(dep_delay_min) as avg_delay
            FROM flights
            WHERE origin_code = ?
        """
        
        params = [airport_code]
        
        if date_filter:
            query += " AND flight_date = ?"
            params.append(date_filter)
        
        query += f"""
            AND std_utc IS NOT NULL
            GROUP BY flight_date, origin_code, time_bucket
        )
        SELECT 
            flight_date,
            time_bucket,
            flight_count,
            delayed_count,
            avg_delay,
            CASE 
                WHEN flight_count > 10 THEN 'HIGH'
                WHEN flight_count > 5 THEN 'MEDIUM'
                ELSE 'LOW'
            END as traffic_level
        FROM time_buckets
        ORDER BY flight_date, time_bucket
        """
        
        try:
            result = conn.execute(query, params).fetchall()
            columns = [desc[0] for desc in conn.description]
            data = [dict(zip(columns, row)) for row in result]
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return QueryResult(
                data=data,
                row_count=len(data),
                execution_time_ms=execution_time,
                query=query
            )
        except Exception as e:
            return QueryResult(
                data=[],
                row_count=0,
                execution_time_ms=0,
                query=f"ERROR: {str(e)}"
            )
    
    def export_to_parquet(self, table_name: str = "flights", 
                         date_filter: Optional[date] = None) -> Dict[str, Any]:
        """Export flight data to Parquet format for analytics."""
        conn = self.connect()
        start_time = datetime.now()
        
        try:
            # Determine output filename
            if date_filter:
                filename = f"{table_name}_{date_filter.strftime('%Y%m%d')}.parquet"
            else:
                filename = f"{table_name}_full.parquet"
            
            output_path = os.path.join(self.config.parquet_export_path, filename)
            
            # Build export query
            if date_filter:
                query = f"COPY (SELECT * FROM {table_name} WHERE flight_date = ?) TO '{output_path}' (FORMAT PARQUET)"
                conn.execute(query, [date_filter])
            else:
                query = f"COPY (SELECT * FROM {table_name}) TO '{output_path}' (FORMAT PARQUET)"
                conn.execute(query)
            
            # Get file size and row count
            file_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
            
            # Count rows exported
            if date_filter:
                count_query = f"SELECT COUNT(*) FROM {table_name} WHERE flight_date = ?"
                row_count = conn.execute(count_query, [date_filter]).fetchone()[0]
            else:
                count_query = f"SELECT COUNT(*) FROM {table_name}"
                row_count = conn.execute(count_query).fetchone()[0]
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": True,
                "output_path": output_path,
                "file_size_bytes": file_size,
                "rows_exported": row_count,
                "processing_time_seconds": processing_time,
                "table_name": table_name
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "processing_time_seconds": (datetime.now() - start_time).total_seconds()
            }
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics and health information."""
        conn = self.connect()
        
        try:
            # Basic table statistics
            stats = {}
            
            # Flight count by status
            status_counts = conn.execute("""
                SELECT status, COUNT(*) as count 
                FROM flights 
                GROUP BY status
            """).fetchall()
            stats["flight_status_counts"] = dict(status_counts)
            
            # Date range
            date_range = conn.execute("""
                SELECT MIN(flight_date) as min_date, MAX(flight_date) as max_date,
                       COUNT(DISTINCT flight_date) as unique_dates
                FROM flights
            """).fetchone()
            stats["date_range"] = {
                "min_date": date_range[0],
                "max_date": date_range[1],
                "unique_dates": date_range[2]
            }
            
            # Airport statistics
            airport_stats = conn.execute("""
                SELECT 
                    COUNT(DISTINCT origin_code) as unique_origins,
                    COUNT(DISTINCT destination_code) as unique_destinations,
                    COUNT(DISTINCT route) as unique_routes
                FROM flights
            """).fetchone()
            stats["airport_stats"] = {
                "unique_origins": airport_stats[0],
                "unique_destinations": airport_stats[1],
                "unique_routes": airport_stats[2]
            }
            
            # Airline statistics
            airline_stats = conn.execute("""
                SELECT COUNT(DISTINCT airline_code) as unique_airlines,
                       COUNT(*) as total_flights
                FROM flights
            """).fetchone()
            stats["airline_stats"] = {
                "unique_airlines": airline_stats[0],
                "total_flights": airline_stats[1]
            }
            
            # Data quality metrics
            quality_stats = conn.execute("""
                SELECT 
                    COUNT(CASE WHEN std_utc IS NOT NULL THEN 1 END) * 100.0 / COUNT(*) as std_completeness,
                    COUNT(CASE WHEN atd_utc IS NOT NULL THEN 1 END) * 100.0 / COUNT(*) as atd_completeness,
                    COUNT(CASE WHEN sta_utc IS NOT NULL THEN 1 END) * 100.0 / COUNT(*) as sta_completeness,
                    COUNT(CASE WHEN ata_utc IS NOT NULL THEN 1 END) * 100.0 / COUNT(*) as ata_completeness,
                    AVG(dep_delay_min) as avg_dep_delay,
                    AVG(arr_delay_min) as avg_arr_delay
                FROM flights
            """).fetchone()
            stats["data_quality"] = {
                "std_completeness_pct": round(quality_stats[0], 2) if quality_stats[0] else 0,
                "atd_completeness_pct": round(quality_stats[1], 2) if quality_stats[1] else 0,
                "sta_completeness_pct": round(quality_stats[2], 2) if quality_stats[2] else 0,
                "ata_completeness_pct": round(quality_stats[3], 2) if quality_stats[3] else 0,
                "avg_dep_delay_min": round(quality_stats[4], 2) if quality_stats[4] else 0,
                "avg_arr_delay_min": round(quality_stats[5], 2) if quality_stats[5] else 0
            }
            
            return stats
            
        except Exception as e:
            return {"error": str(e)}
    
    def query_flights_by_delay_status(self, status: DelayStatus, 
                                      date_filter: Optional[date] = None) -> QueryResult:
        """Query flights by delay status (green/yellow/red)."""
        conn = self.connect()
        start_time = datetime.now()
        
        base_query = """
        SELECT * FROM flights 
        WHERE dep_delay_status = ?
        """
        params = [status.value]
        
        if date_filter:
            base_query += " AND flight_date = ?"
            params.append(date_filter)
        
        base_query += " ORDER BY flight_date, std_utc"
        
        try:
            result = conn.execute(base_query, params).fetchall()
            columns = [desc[0] for desc in conn.description]
            data = [dict(zip(columns, row)) for row in result]
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return QueryResult(
                data=data,
                row_count=len(data),
                execution_time_ms=execution_time,
                query=base_query
            )
        except Exception as e:
            return QueryResult(
                data=[],
                row_count=0,
                execution_time_ms=0,
                query=f"ERROR: {str(e)}"
            )
    
    def get_delay_status_summary(self, date_filter: Optional[date] = None) -> Dict[str, Any]:
        """Get summary of delay status distribution."""
        conn = self.connect()
        
        base_query = """
        SELECT 
            dep_delay_status,
            COUNT(*) as count,
            AVG(dep_delay_min) as avg_delay,
            MIN(dep_delay_min) as min_delay,
            MAX(dep_delay_min) as max_delay
        FROM flights 
        WHERE dep_delay_status IS NOT NULL
        """
        params = []
        
        if date_filter:
            base_query += " AND flight_date = ?"
            params.append(date_filter)
        
        base_query += " GROUP BY dep_delay_status ORDER BY dep_delay_status"
        
        try:
            result = conn.execute(base_query, params).fetchall()
            
            summary = {}
            total_flights = 0
            
            for row in result:
                status = row[0]
                count = row[1]
                avg_delay = row[2]
                min_delay = row[3]
                max_delay = row[4]
                
                summary[status] = {
                    'count': count,
                    'avg_delay': round(avg_delay, 1) if avg_delay else 0,
                    'min_delay': min_delay,
                    'max_delay': max_delay
                }
                total_flights += count
            
            # Add percentages
            for status in summary:
                summary[status]['percentage'] = round(summary[status]['count'] / total_flights * 100, 1)
            
            return {
                'status_breakdown': summary,
                'total_flights': total_flights,
                'date_filter': date_filter.isoformat() if date_filter else None
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def close(self) -> None:
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None