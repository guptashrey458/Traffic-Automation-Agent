"""Analytics engine for peak traffic analysis and demand forecasting."""

import os
from datetime import datetime, date, time, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from collections import defaultdict

from ..models.flight import Flight
from .database import FlightDatabaseService, QueryResult


class WeatherRegime(Enum):
    """Weather conditions affecting runway capacity."""
    CALM = "calm"           # Good visibility, light winds
    MEDIUM = "medium"       # Moderate winds, some restrictions
    STRONG = "strong"       # Strong winds, reduced capacity
    SEVERE = "severe"       # Severe weather, minimal operations


class TrafficLevel(Enum):
    """Traffic intensity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TimeBucket:
    """Time bucket for traffic analysis."""
    start_time: datetime
    end_time: datetime
    bucket_minutes: int
    scheduled_departures: int = 0
    actual_departures: int = 0
    scheduled_arrivals: int = 0
    actual_arrivals: int = 0
    total_demand: int = 0
    capacity: int = 0
    utilization: float = 0.0
    overload: int = 0
    avg_delay: float = 0.0
    delayed_flights: int = 0
    traffic_level: TrafficLevel = TrafficLevel.LOW
    
    def __post_init__(self):
        """Calculate derived metrics."""
        self.total_demand = self.scheduled_departures + self.scheduled_arrivals
        if self.capacity > 0:
            self.utilization = self.total_demand / self.capacity
            self.overload = max(0, self.total_demand - self.capacity)
        
        # Classify traffic level
        if self.utilization >= 1.2:
            self.traffic_level = TrafficLevel.CRITICAL
        elif self.utilization >= 1.0:
            self.traffic_level = TrafficLevel.HIGH
        elif self.utilization >= 0.7:
            self.traffic_level = TrafficLevel.MEDIUM
        else:
            self.traffic_level = TrafficLevel.LOW


@dataclass
class OverloadWindow:
    """Period of capacity overload."""
    start_time: datetime
    end_time: datetime
    duration_minutes: int
    peak_overload: int
    avg_overload: float
    affected_flights: int
    severity: str = ""  # "minor", "moderate", "severe"
    recommendations: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Classify severity and generate recommendations."""
        if self.peak_overload >= 10:
            self.severity = "severe"
            self.recommendations = [
                "Consider ground delay program",
                "Implement arrival metering",
                "Coordinate with adjacent airports"
            ]
        elif self.peak_overload >= 5:
            self.severity = "moderate"
            self.recommendations = [
                "Optimize runway usage",
                "Adjust departure spacing",
                "Monitor for cascade delays"
            ]
        else:
            self.severity = "minor"
            self.recommendations = [
                "Monitor traffic flow",
                "Prepare for potential delays"
            ]


@dataclass
class PeakAnalysis:
    """Complete peak traffic analysis results."""
    airport: str
    analysis_date: date
    bucket_minutes: int
    time_buckets: List[TimeBucket] = field(default_factory=list)
    overload_windows: List[OverloadWindow] = field(default_factory=list)
    peak_hour: Optional[int] = None
    peak_demand: int = 0
    total_capacity: int = 0
    avg_utilization: float = 0.0
    delay_hotspots: List[Dict[str, Any]] = field(default_factory=list)
    weather_regime: WeatherRegime = WeatherRegime.CALM
    recommendations: List[str] = field(default_factory=list)
    
    def get_heatmap_data(self) -> List[Dict[str, Any]]:
        """Generate heatmap data structure for visualization."""
        heatmap_data = []
        
        for bucket in self.time_buckets:
            heatmap_data.append({
                "time": bucket.start_time.strftime("%H:%M"),
                "hour": bucket.start_time.hour,
                "minute": bucket.start_time.minute,
                "demand": bucket.total_demand,
                "capacity": bucket.capacity,
                "utilization": round(bucket.utilization, 2),
                "overload": bucket.overload,
                "traffic_level": bucket.traffic_level.value,
                "avg_delay": round(bucket.avg_delay, 1),
                "delayed_flights": bucket.delayed_flights,
                "color": self._get_traffic_color(bucket.traffic_level, bucket.utilization)
            })
        
        return heatmap_data
    
    def _get_traffic_color(self, level: TrafficLevel, utilization: float) -> str:
        """Get color code for traffic visualization."""
        if level == TrafficLevel.CRITICAL or utilization >= 1.2:
            return "#FF4444"  # Red
        elif level == TrafficLevel.HIGH or utilization >= 1.0:
            return "#FF8800"  # Orange
        elif level == TrafficLevel.MEDIUM or utilization >= 0.7:
            return "#FFDD00"  # Yellow
        else:
            return "#44AA44"  # Green


@dataclass
class CapacityConfig:
    """Airport capacity configuration."""
    airport_code: str
    runway_capacity_per_hour: Dict[str, int] = field(default_factory=dict)
    weather_adjustments: Dict[WeatherRegime, float] = field(default_factory=dict)
    time_of_day_adjustments: Dict[int, float] = field(default_factory=dict)  # hour -> multiplier
    curfew_hours: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        """Set default capacity values if not provided."""
        if not self.runway_capacity_per_hour:
            # Default capacity for major Indian airports
            if self.airport_code in ["BOM", "DEL"]:
                self.runway_capacity_per_hour = {
                    "departures": 30,  # 30 departures per hour
                    "arrivals": 30,    # 30 arrivals per hour
                    "total": 50        # Combined operations (not additive due to runway conflicts)
                }
            else:
                self.runway_capacity_per_hour = {
                    "departures": 20,
                    "arrivals": 20,
                    "total": 35
                }
        
        if not self.weather_adjustments:
            self.weather_adjustments = {
                WeatherRegime.CALM: 1.0,      # 100% capacity
                WeatherRegime.MEDIUM: 0.85,   # 85% capacity
                WeatherRegime.STRONG: 0.65,   # 65% capacity
                WeatherRegime.SEVERE: 0.3     # 30% capacity
            }
        
        if not self.time_of_day_adjustments:
            # Reduced capacity during night hours
            self.time_of_day_adjustments = {
                0: 0.5, 1: 0.3, 2: 0.3, 3: 0.3, 4: 0.3, 5: 0.5,  # Night hours
                6: 0.8, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0,  # Morning
                12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0,  # Afternoon
                18: 1.0, 19: 1.0, 20: 1.0, 21: 0.8, 22: 0.6, 23: 0.5   # Evening/Night
            }
        
        if not self.curfew_hours and self.airport_code in ["BOM", "DEL"]:
            # Common curfew hours for major airports
            self.curfew_hours = [1, 2, 3, 4, 5]  # 1 AM to 5 AM
    
    def get_capacity(self, hour: int, weather: WeatherRegime = WeatherRegime.CALM,
                    operation_type: str = "total") -> int:
        """Get adjusted capacity for specific conditions."""
        base_capacity = self.runway_capacity_per_hour.get(operation_type, 35)
        
        # Apply weather adjustment
        weather_factor = self.weather_adjustments.get(weather, 1.0)
        
        # Apply time-of-day adjustment
        time_factor = self.time_of_day_adjustments.get(hour, 1.0)
        
        # Apply curfew restrictions
        if hour in self.curfew_hours:
            time_factor *= 0.1  # Severely restricted during curfew
        
        adjusted_capacity = int(base_capacity * weather_factor * time_factor)
        return max(1, adjusted_capacity)  # Minimum capacity of 1


class AnalyticsEngine:
    """Engine for peak traffic analysis and demand forecasting."""
    
    def __init__(self, db_service: Optional[FlightDatabaseService] = None):
        self.db_service = db_service or FlightDatabaseService()
        self.capacity_configs: Dict[str, CapacityConfig] = {}
        self._load_default_capacity_configs()
    
    def _load_default_capacity_configs(self) -> None:
        """Load default capacity configurations for major airports."""
        # Mumbai (BOM) - Busiest airport in India
        self.capacity_configs["BOM"] = CapacityConfig(
            airport_code="BOM",
            runway_capacity_per_hour={
                "departures": 35,
                "arrivals": 35,
                "total": 60  # Two intersecting runways
            }
        )
        
        # Delhi (DEL) - Major hub with multiple runways
        self.capacity_configs["DEL"] = CapacityConfig(
            airport_code="DEL",
            runway_capacity_per_hour={
                "departures": 40,
                "arrivals": 40,
                "total": 70  # Three parallel runways
            }
        )
        
        # Default configuration for other airports
        self.capacity_configs["DEFAULT"] = CapacityConfig(
            airport_code="DEFAULT",
            runway_capacity_per_hour={
                "departures": 20,
                "arrivals": 20,
                "total": 35
            }
        )
    
    def analyze_peaks(self, airport: str, analysis_date: date, 
                     bucket_minutes: int = 10,
                     weather_regime: WeatherRegime = WeatherRegime.CALM) -> PeakAnalysis:
        """
        Analyze peak traffic patterns for a specific airport and date.
        
        Args:
            airport: Airport code (e.g., "BOM", "DEL")
            analysis_date: Date to analyze
            bucket_minutes: Time bucket size in minutes (5, 10, 15, 30)
            weather_regime: Weather conditions affecting capacity
            
        Returns:
            PeakAnalysis object with complete analysis results
        """
        # Get flight data for the specified date and airport
        flight_data = self._get_flight_data(airport, analysis_date)
        
        if not flight_data:
            return PeakAnalysis(
                airport=airport,
                analysis_date=analysis_date,
                bucket_minutes=bucket_minutes,
                weather_regime=weather_regime,
                recommendations=["No flight data available for analysis"]
            )
        
        # Create time buckets
        time_buckets = self._create_time_buckets(
            flight_data, analysis_date, bucket_minutes, airport, weather_regime
        )
        
        # Identify overload windows
        overload_windows = self._identify_overload_windows(time_buckets)
        
        # Calculate peak metrics
        peak_hour, peak_demand = self._find_peak_hour(time_buckets)
        
        # Calculate average utilization
        total_capacity = sum(bucket.capacity for bucket in time_buckets)
        total_demand = sum(bucket.total_demand for bucket in time_buckets)
        avg_utilization = total_demand / total_capacity if total_capacity > 0 else 0
        
        # Identify delay hotspots
        delay_hotspots = self._identify_delay_hotspots(time_buckets)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            time_buckets, overload_windows, weather_regime
        )
        
        return PeakAnalysis(
            airport=airport,
            analysis_date=analysis_date,
            bucket_minutes=bucket_minutes,
            time_buckets=time_buckets,
            overload_windows=overload_windows,
            peak_hour=peak_hour,
            peak_demand=peak_demand,
            total_capacity=total_capacity,
            avg_utilization=avg_utilization,
            delay_hotspots=delay_hotspots,
            weather_regime=weather_regime,
            recommendations=recommendations
        )
    
    def _get_flight_data(self, airport: str, analysis_date: date) -> List[Dict[str, Any]]:
        """Get flight data for analysis."""
        try:
            result = self.db_service.query_flights_by_date_range(
                start_date=analysis_date,
                end_date=analysis_date,
                airport_code=airport
            )
            return result.data
        except Exception as e:
            print(f"Error fetching flight data: {e}")
            return []
    
    def _create_time_buckets(self, flight_data: List[Dict[str, Any]], 
                           analysis_date: date, bucket_minutes: int,
                           airport: str, weather_regime: WeatherRegime) -> List[TimeBucket]:
        """Create time buckets and populate with flight data."""
        buckets = []
        
        # Create buckets for the entire day
        start_of_day = datetime.combine(analysis_date, time(0, 0))
        current_time = start_of_day
        
        # Get capacity configuration
        capacity_config = self.capacity_configs.get(airport, self.capacity_configs["DEFAULT"])
        
        while current_time.date() == analysis_date:
            bucket_end = current_time + timedelta(minutes=bucket_minutes)
            
            # Get capacity for this time bucket
            capacity = capacity_config.get_capacity(
                hour=current_time.hour,
                weather=weather_regime,
                operation_type="total"
            )
            # Scale capacity to bucket size
            bucket_capacity = int(capacity * bucket_minutes / 60)
            
            bucket = TimeBucket(
                start_time=current_time,
                end_time=bucket_end,
                bucket_minutes=bucket_minutes,
                capacity=bucket_capacity
            )
            
            # Count flights in this bucket
            self._populate_bucket_with_flights(bucket, flight_data, airport)
            
            buckets.append(bucket)
            current_time = bucket_end
        
        return buckets
    
    def _populate_bucket_with_flights(self, bucket: TimeBucket, 
                                    flight_data: List[Dict[str, Any]], 
                                    airport: str) -> None:
        """Populate a time bucket with flight counts and delay metrics."""
        departures = []
        arrivals = []
        delays = []
        
        for flight in flight_data:
            # Check departures
            if (flight.get('origin_code') == airport and 
                flight.get('std_utc')):
                
                std_utc = flight['std_utc']
                if isinstance(std_utc, str):
                    std_utc = datetime.fromisoformat(std_utc.replace('Z', '+00:00'))
                
                if bucket.start_time <= std_utc < bucket.end_time:
                    bucket.scheduled_departures += 1
                    if flight.get('atd_utc'):
                        bucket.actual_departures += 1
                    
                    # Track delays
                    if flight.get('dep_delay_min') is not None:
                        delays.append(flight['dep_delay_min'])
                        if flight['dep_delay_min'] > 15:
                            bucket.delayed_flights += 1
            
            # Check arrivals
            if (flight.get('destination_code') == airport and 
                flight.get('sta_utc')):
                
                sta_utc = flight['sta_utc']
                if isinstance(sta_utc, str):
                    sta_utc = datetime.fromisoformat(sta_utc.replace('Z', '+00:00'))
                
                if bucket.start_time <= sta_utc < bucket.end_time:
                    bucket.scheduled_arrivals += 1
                    if flight.get('ata_utc'):
                        bucket.actual_arrivals += 1
                    
                    # Track arrival delays
                    if flight.get('arr_delay_min') is not None:
                        delays.append(flight['arr_delay_min'])
                        if flight['arr_delay_min'] > 15:
                            bucket.delayed_flights += 1
        
        # Calculate average delay
        if delays:
            bucket.avg_delay = sum(delays) / len(delays)
        
        # Trigger post-init calculations
        bucket.__post_init__()
    
    def _identify_overload_windows(self, time_buckets: List[TimeBucket]) -> List[OverloadWindow]:
        """Identify periods of sustained capacity overload."""
        overload_windows = []
        current_window = None
        
        for bucket in time_buckets:
            if bucket.overload > 0:
                if current_window is None:
                    # Start new overload window
                    current_window = {
                        'start_time': bucket.start_time,
                        'end_time': bucket.end_time,
                        'overloads': [bucket.overload],
                        'affected_flights': bucket.total_demand
                    }
                else:
                    # Extend current window
                    current_window['end_time'] = bucket.end_time
                    current_window['overloads'].append(bucket.overload)
                    current_window['affected_flights'] += bucket.total_demand
            else:
                if current_window is not None:
                    # End current window and create OverloadWindow object
                    duration = (current_window['end_time'] - current_window['start_time']).total_seconds() / 60
                    
                    overload_window = OverloadWindow(
                        start_time=current_window['start_time'],
                        end_time=current_window['end_time'],
                        duration_minutes=int(duration),
                        peak_overload=max(current_window['overloads']),
                        avg_overload=sum(current_window['overloads']) / len(current_window['overloads']),
                        affected_flights=current_window['affected_flights']
                    )
                    
                    overload_windows.append(overload_window)
                    current_window = None
        
        # Handle case where overload continues to end of day
        if current_window is not None:
            duration = (current_window['end_time'] - current_window['start_time']).total_seconds() / 60
            
            overload_window = OverloadWindow(
                start_time=current_window['start_time'],
                end_time=current_window['end_time'],
                duration_minutes=int(duration),
                peak_overload=max(current_window['overloads']),
                avg_overload=sum(current_window['overloads']) / len(current_window['overloads']),
                affected_flights=current_window['affected_flights']
            )
            
            overload_windows.append(overload_window)
        
        return overload_windows
    
    def _find_peak_hour(self, time_buckets: List[TimeBucket]) -> Tuple[Optional[int], int]:
        """Find the hour with peak demand."""
        hourly_demand = defaultdict(int)
        
        for bucket in time_buckets:
            hour = bucket.start_time.hour
            hourly_demand[hour] += bucket.total_demand
        
        if not hourly_demand:
            return None, 0
        
        peak_hour = max(hourly_demand.keys(), key=lambda h: hourly_demand[h])
        peak_demand = hourly_demand[peak_hour]
        
        return peak_hour, peak_demand
    
    def _identify_delay_hotspots(self, time_buckets: List[TimeBucket]) -> List[Dict[str, Any]]:
        """Identify time periods with high delay rates."""
        hotspots = []
        
        for bucket in time_buckets:
            if bucket.avg_delay > 20 or bucket.delayed_flights > 5:
                hotspots.append({
                    "time": bucket.start_time.strftime("%H:%M"),
                    "avg_delay": round(bucket.avg_delay, 1),
                    "delayed_flights": bucket.delayed_flights,
                    "total_flights": bucket.total_demand,
                    "delay_rate": round(bucket.delayed_flights / bucket.total_demand * 100, 1) if bucket.total_demand > 0 else 0,
                    "severity": "high" if bucket.avg_delay > 30 else "moderate"
                })
        
        # Sort by delay severity
        hotspots.sort(key=lambda x: x["avg_delay"], reverse=True)
        
        return hotspots[:10]  # Return top 10 hotspots
    
    def _generate_recommendations(self, time_buckets: List[TimeBucket], 
                                overload_windows: List[OverloadWindow],
                                weather_regime: WeatherRegime) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Check for overload issues
        if overload_windows:
            severe_overloads = [w for w in overload_windows if w.severity == "severe"]
            if severe_overloads:
                recommendations.append("Implement ground delay program during peak overload periods")
                recommendations.append("Consider arrival metering to manage demand")
            
            moderate_overloads = [w for w in overload_windows if w.severity == "moderate"]
            if moderate_overloads:
                recommendations.append("Optimize runway usage and departure sequencing")
                recommendations.append("Monitor for potential cascade delays")
        
        # Check utilization patterns
        high_util_buckets = [b for b in time_buckets if b.utilization > 0.9]
        if len(high_util_buckets) > 6:  # More than 1 hour of high utilization
            recommendations.append("Consider spreading traffic across longer time windows")
            recommendations.append("Coordinate with airlines for voluntary schedule adjustments")
        
        # Weather-specific recommendations
        if weather_regime in [WeatherRegime.STRONG, WeatherRegime.SEVERE]:
            recommendations.append("Activate weather contingency procedures")
            recommendations.append("Increase spacing between aircraft operations")
            recommendations.append("Prepare for extended ground delays")
        
        # Delay hotspot recommendations
        delay_buckets = [b for b in time_buckets if b.avg_delay > 20]
        if delay_buckets:
            recommendations.append("Focus delay mitigation efforts on identified hotspot periods")
            recommendations.append("Review turnaround procedures during high-delay windows")
        
        # General optimization recommendations
        if not recommendations:
            recommendations.append("Current traffic patterns are within normal operating parameters")
            recommendations.append("Continue monitoring for emerging congestion patterns")
        
        return recommendations
    
    def generate_demand_heatmap(self, airport: str, start_date: date, end_date: date,
                              bucket_minutes: int = 10) -> Dict[str, Any]:
        """Generate demand heatmap data for visualization."""
        heatmap_data = []
        date_range = []
        
        current_date = start_date
        while current_date <= end_date:
            analysis = self.analyze_peaks(airport, current_date, bucket_minutes)
            
            daily_heatmap = analysis.get_heatmap_data()
            for bucket_data in daily_heatmap:
                bucket_data['date'] = current_date.isoformat()
                heatmap_data.append(bucket_data)
            
            date_range.append(current_date.isoformat())
            current_date += timedelta(days=1)
        
        # Calculate summary statistics
        total_demand = sum(bucket['demand'] for bucket in heatmap_data)
        avg_utilization = sum(bucket['utilization'] for bucket in heatmap_data) / len(heatmap_data) if heatmap_data else 0
        peak_utilization = max(bucket['utilization'] for bucket in heatmap_data) if heatmap_data else 0
        
        return {
            "airport": airport,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "bucket_minutes": bucket_minutes,
            "heatmap_data": heatmap_data,
            "date_range": date_range,
            "summary": {
                "total_demand": total_demand,
                "avg_utilization": round(avg_utilization, 2),
                "peak_utilization": round(peak_utilization, 2),
                "total_buckets": len(heatmap_data)
            }
        }
    
    def update_capacity_config(self, airport: str, config: CapacityConfig) -> None:
        """Update capacity configuration for an airport."""
        self.capacity_configs[airport] = config
    
    def get_capacity_config(self, airport: str) -> CapacityConfig:
        """Get capacity configuration for an airport."""
        return self.capacity_configs.get(airport, self.capacity_configs["DEFAULT"])