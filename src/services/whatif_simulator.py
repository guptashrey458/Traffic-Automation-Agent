"""What-if simulation system for analyzing flight schedule changes."""

import os
from datetime import datetime, date, time, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from collections import defaultdict

from ..models.flight import Flight
from .database import FlightDatabaseService
from .analytics import AnalyticsEngine, PeakAnalysis, WeatherRegime
from .schedule_optimizer import (
    ScheduleOptimizer, Schedule, FlightChange, DelayMetrics, 
    ImpactAnalysis, Constraints, ObjectiveWeights
)


class ChangeType(Enum):
    """Types of flight changes for what-if analysis."""
    TIME_SHIFT = "time_shift"           # Move flight by X minutes
    RUNWAY_CHANGE = "runway_change"     # Change assigned runway
    CANCELLATION = "cancellation"       # Cancel flight
    AIRCRAFT_SWAP = "aircraft_swap"     # Change aircraft type


@dataclass
class CO2Factors:
    """CO2 emission factors for different operations."""
    # CO2 per minute of delay (kg) - based on fuel burn during ground operations
    delay_factor_kg_per_min: float = 2.5
    
    # CO2 per minute of taxi time (kg) - based on single engine taxi
    taxi_factor_kg_per_min: float = 8.0
    
    # CO2 savings from optimized routing (kg per flight)
    routing_optimization_kg: float = 15.0
    
    # Aircraft-specific factors (kg CO2 per minute of operation)
    aircraft_factors: Dict[str, float] = field(default_factory=lambda: {
        "A320": 12.0, "A321": 14.0, "B737": 11.0, "B738": 11.5,
        "B777": 35.0, "B787": 28.0, "DEFAULT": 15.0
    })
    
    def get_aircraft_factor(self, aircraft_type: str) -> float:
        """Get CO2 factor for specific aircraft type."""
        return self.aircraft_factors.get(aircraft_type, self.aircraft_factors["DEFAULT"])


@dataclass
class WhatIfScenario:
    """Represents a what-if scenario with proposed changes."""
    scenario_id: str
    description: str
    changes: List[FlightChange]
    base_date: date
    airport: str
    weather_regime: WeatherRegime = WeatherRegime.CALM
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ImpactCard:
    """Impact card showing before/after metrics for visualization."""
    scenario_id: str
    scenario_description: str
    
    # Delay impact
    delay_change_minutes: float
    delay_direction: str  # "improvement", "degradation", "neutral"
    affected_flights_count: int
    
    # Capacity impact
    peak_overload_change: int
    capacity_direction: str  # "improvement", "degradation", "neutral"
    new_overload_windows: int
    
    # Environmental impact
    co2_change_kg: float
    co2_direction: str  # "improvement", "degradation", "neutral"
    fuel_savings_liters: float
    
    # Operational metrics
    on_time_performance_change: float  # Percentage point change
    fairness_score: float  # 0-1 scale
    constraint_violations: int
    
    # Overall recommendation
    recommendation: str
    confidence_level: str  # "high", "medium", "low"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert impact card to dictionary for API/visualization."""
        return {
            "scenario": {
                "id": self.scenario_id,
                "description": self.scenario_description
            },
            "delay_impact": {
                "change_minutes": round(self.delay_change_minutes, 1),
                "direction": self.delay_direction,
                "affected_flights": self.affected_flights_count
            },
            "capacity_impact": {
                "overload_change": self.peak_overload_change,
                "direction": self.capacity_direction,
                "new_overload_windows": self.new_overload_windows
            },
            "environmental_impact": {
                "co2_change_kg": round(self.co2_change_kg, 1),
                "direction": self.co2_direction,
                "fuel_savings_liters": round(self.fuel_savings_liters, 1)
            },
            "operational_metrics": {
                "otp_change_percent": round(self.on_time_performance_change, 1),
                "fairness_score": round(self.fairness_score, 2),
                "constraint_violations": self.constraint_violations
            },
            "recommendation": {
                "text": self.recommendation,
                "confidence": self.confidence_level
            }
        }


@dataclass
class BeforeAfterComparison:
    """Detailed before/after comparison metrics."""
    before_metrics: DelayMetrics
    after_metrics: DelayMetrics
    before_peak_analysis: PeakAnalysis
    after_peak_analysis: PeakAnalysis
    
    # Detailed changes
    delay_improvements: List[Dict[str, Any]]
    delay_degradations: List[Dict[str, Any]]
    capacity_changes: List[Dict[str, Any]]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of before/after comparison."""
        improvements = self.after_metrics.improvement_over(self.before_metrics)
        
        return {
            "delay_metrics": {
                "before": {
                    "total_delay": self.before_metrics.total_delay_minutes,
                    "avg_delay": self.before_metrics.avg_delay_minutes,
                    "otp": self.before_metrics.on_time_performance
                },
                "after": {
                    "total_delay": self.after_metrics.total_delay_minutes,
                    "avg_delay": self.after_metrics.avg_delay_minutes,
                    "otp": self.after_metrics.on_time_performance
                },
                "improvements": improvements
            },
            "capacity_metrics": {
                "before_overloads": len(self.before_peak_analysis.overload_windows),
                "after_overloads": len(self.after_peak_analysis.overload_windows),
                "peak_utilization_change": (
                    self.after_peak_analysis.avg_utilization - 
                    self.before_peak_analysis.avg_utilization
                )
            },
            "detailed_changes": {
                "improvements": len(self.delay_improvements),
                "degradations": len(self.delay_degradations),
                "capacity_changes": len(self.capacity_changes)
            }
        }


class WhatIfSimulator:
    """What-if simulation system for analyzing flight schedule changes."""
    
    def __init__(self, 
                 db_service: Optional[FlightDatabaseService] = None,
                 analytics_engine: Optional[AnalyticsEngine] = None,
                 schedule_optimizer: Optional[ScheduleOptimizer] = None):
        """
        Initialize the what-if simulator.
        
        Args:
            db_service: Database service for flight data
            analytics_engine: Analytics engine for peak analysis
            schedule_optimizer: Schedule optimizer for impact analysis
        """
        self.db_service = db_service or FlightDatabaseService()
        self.analytics_engine = analytics_engine or AnalyticsEngine()
        self.schedule_optimizer = schedule_optimizer or ScheduleOptimizer()
        
        # CO2 calculation factors
        self.co2_factors = CO2Factors()
        
        # Simulation settings
        self.max_simulation_time_seconds = 30  # Quick response for what-if
        self.confidence_threshold = 0.8  # For recommendation confidence
    
    def analyze_single_flight_change(self, 
                                   flight_id: str,
                                   time_change_minutes: int,
                                   airport: str,
                                   analysis_date: date,
                                   weather_regime: WeatherRegime = WeatherRegime.CALM) -> ImpactCard:
        """
        Analyze the impact of changing a single flight's time.
        
        Args:
            flight_id: ID of flight to change
            time_change_minutes: Minutes to shift flight (+/- for later/earlier)
            airport: Airport code for analysis
            analysis_date: Date to analyze
            weather_regime: Weather conditions
            
        Returns:
            ImpactCard with detailed impact analysis
        """
        # Get base schedule
        base_schedule = self._get_base_schedule(airport, analysis_date)
        
        if not base_schedule.flights:
            return self._create_empty_impact_card(
                f"move_{flight_id}_{time_change_minutes}m",
                f"Move flight {flight_id} by {time_change_minutes} minutes",
                "No flight data available for analysis"
            )
        
        # Find the target flight
        target_flight = None
        for flight in base_schedule.flights:
            if flight.flight_id == flight_id or flight.flight_number == flight_id:
                target_flight = flight
                break
        
        if not target_flight:
            return self._create_empty_impact_card(
                f"move_{flight_id}_{time_change_minutes}m",
                f"Move flight {flight_id} by {time_change_minutes} minutes",
                f"Flight {flight_id} not found in schedule"
            )
        
        # Create flight change
        original_time = datetime.combine(
            analysis_date,
            target_flight.departure.scheduled or time(12, 0)
        )
        new_time = original_time + timedelta(minutes=time_change_minutes)
        
        flight_change = FlightChange(
            flight_id=target_flight.flight_id,
            original_time=original_time,
            new_time=new_time,
            change_type="departure"
        )
        
        # Run what-if analysis
        scenario = WhatIfScenario(
            scenario_id=f"move_{flight_id}_{time_change_minutes}m",
            description=f"Move flight {flight_id} by {time_change_minutes} minutes",
            changes=[flight_change],
            base_date=analysis_date,
            airport=airport,
            weather_regime=weather_regime
        )
        
        return self.simulate_scenario(scenario)
    
    def simulate_scenario(self, scenario: WhatIfScenario) -> ImpactCard:
        """
        Simulate a complete what-if scenario.
        
        Args:
            scenario: What-if scenario to simulate
            
        Returns:
            ImpactCard with impact analysis results
        """
        # Get base schedule and analysis
        base_schedule = self._get_base_schedule(scenario.airport, scenario.base_date)
        base_analysis = self.analytics_engine.analyze_peaks(
            scenario.airport, scenario.base_date, weather_regime=scenario.weather_regime
        )
        
        # Apply changes to create modified schedule
        modified_schedule = self._apply_changes(base_schedule, scenario.changes)
        
        # Run impact analysis using schedule optimizer
        impact_analysis = self.schedule_optimizer.what_if_analysis(
            base_schedule, scenario.changes
        )
        
        # Run peak analysis on modified schedule
        modified_analysis = self._analyze_modified_schedule_peaks(
            modified_schedule, scenario.airport, scenario.base_date, scenario.weather_regime
        )
        
        # Calculate detailed impacts
        delay_impact = self._calculate_delay_impact(base_schedule, modified_schedule)
        capacity_impact = self._calculate_capacity_impact(base_analysis, modified_analysis)
        co2_impact = self._calculate_co2_impact(scenario.changes, base_schedule)
        fairness_score = self._calculate_fairness_score(scenario.changes, base_schedule)
        
        # Generate recommendation
        recommendation, confidence = self._generate_recommendation(
            delay_impact, capacity_impact, co2_impact, fairness_score
        )
        
        return ImpactCard(
            scenario_id=scenario.scenario_id,
            scenario_description=scenario.description,
            delay_change_minutes=delay_impact["total_change"],
            delay_direction=delay_impact["direction"],
            affected_flights_count=len(scenario.changes),
            peak_overload_change=capacity_impact["overload_change"],
            capacity_direction=capacity_impact["direction"],
            new_overload_windows=capacity_impact["new_windows"],
            co2_change_kg=co2_impact["total_kg"],
            co2_direction=co2_impact["direction"],
            fuel_savings_liters=co2_impact["fuel_liters"],
            on_time_performance_change=delay_impact["otp_change"],
            fairness_score=fairness_score,
            constraint_violations=len(impact_analysis.affected_flights),
            recommendation=recommendation,
            confidence_level=confidence
        )
    
    def compare_before_after(self, scenario: WhatIfScenario) -> BeforeAfterComparison:
        """
        Generate detailed before/after comparison.
        
        Args:
            scenario: What-if scenario to analyze
            
        Returns:
            BeforeAfterComparison with detailed metrics
        """
        # Get base schedule and analysis
        base_schedule = self._get_base_schedule(scenario.airport, scenario.base_date)
        base_analysis = self.analytics_engine.analyze_peaks(
            scenario.airport, scenario.base_date, weather_regime=scenario.weather_regime
        )
        
        # Apply changes
        modified_schedule = self._apply_changes(base_schedule, scenario.changes)
        modified_analysis = self._analyze_modified_schedule_peaks(
            modified_schedule, scenario.airport, scenario.base_date, scenario.weather_regime
        )
        
        # Calculate detailed changes
        delay_improvements = self._identify_delay_improvements(base_schedule, modified_schedule)
        delay_degradations = self._identify_delay_degradations(base_schedule, modified_schedule)
        capacity_changes = self._identify_capacity_changes(base_analysis, modified_analysis)
        
        return BeforeAfterComparison(
            before_metrics=base_schedule.get_delay_metrics(),
            after_metrics=modified_schedule.get_delay_metrics(),
            before_peak_analysis=base_analysis,
            after_peak_analysis=modified_analysis,
            delay_improvements=delay_improvements,
            delay_degradations=delay_degradations,
            capacity_changes=capacity_changes
        )
    
    def _get_base_schedule(self, airport: str, analysis_date: date) -> Schedule:
        """Get base schedule for analysis."""
        try:
            result = self.db_service.query_flights_by_date_range(
                start_date=analysis_date,
                end_date=analysis_date,
                airport_code=airport
            )
            
            flights = []
            for flight_data in result.data:
                flight = self._convert_dict_to_flight(flight_data)
                if flight and flight.is_valid():
                    flights.append(flight)
            
            return Schedule(flights=flights, schedule_date=analysis_date)
            
        except Exception as e:
            print(f"Error getting base schedule: {e}")
            return Schedule(schedule_date=analysis_date)
    
    def _convert_dict_to_flight(self, flight_data: Dict[str, Any]) -> Optional[Flight]:
        """Convert flight data dictionary to Flight object."""
        try:
            from ..models.flight import Airport, FlightTime
            
            flight = Flight()
            flight.flight_id = flight_data.get('flight_id', '')
            flight.flight_number = flight_data.get('flight_number', '')
            flight.airline_code = flight_data.get('airline_code', '')
            
            # Set origin and destination
            if flight_data.get('origin_code'):
                flight.origin = Airport(
                    code=flight_data['origin_code'],
                    name=flight_data.get('origin_name', ''),
                    city=flight_data.get('origin_name', '').split('(')[0].strip()
                )
            
            if flight_data.get('destination_code'):
                flight.destination = Airport(
                    code=flight_data['destination_code'],
                    name=flight_data.get('destination_name', ''),
                    city=flight_data.get('destination_name', '').split('(')[0].strip()
                )
            
            # Set timing information
            if flight_data.get('std_utc'):
                std_utc = flight_data['std_utc']
                if isinstance(std_utc, str):
                    try:
                        # Handle different time formats
                        if 'T' in std_utc:
                            std_utc = datetime.fromisoformat(std_utc.replace('Z', '+00:00'))
                            flight.departure.scheduled = std_utc.time()
                            flight.flight_date = std_utc.date()
                        else:
                            # Handle time-only format like "04:05:00"
                            time_parts = std_utc.split(':')
                            if len(time_parts) >= 2:
                                hour = int(time_parts[0])
                                minute = int(time_parts[1])
                                second = int(time_parts[2]) if len(time_parts) > 2 else 0
                                flight.departure.scheduled = time(hour, minute, second)
                                flight.flight_date = date.today()  # Use today as default
                    except ValueError:
                        pass  # Skip invalid time formats
                else:
                    flight.departure.scheduled = std_utc.time()
                    flight.flight_date = std_utc.date()
            
            if flight_data.get('atd_utc'):
                atd_utc = flight_data['atd_utc']
                if isinstance(atd_utc, str):
                    atd_utc = datetime.fromisoformat(atd_utc.replace('Z', '+00:00'))
                # Convert to naive datetime for compatibility
                flight.departure.actual = atd_utc.replace(tzinfo=None)
            
            # Set delay information
            flight.dep_delay_min = flight_data.get('dep_delay_min')
            flight.arr_delay_min = flight_data.get('arr_delay_min')
            
            return flight
            
        except Exception as e:
            print(f"Error converting flight data: {e}")
            return None
    
    def _apply_changes(self, base_schedule: Schedule, changes: List[FlightChange]) -> Schedule:
        """Apply changes to create modified schedule."""
        modified_flights = []
        
        for flight in base_schedule.flights:
            modified_flight = flight  # Copy flight
            
            # Apply any changes for this flight
            for change in changes:
                if (change.flight_id == flight.flight_id or 
                    change.flight_id == flight.flight_number):
                    
                    if change.change_type == "departure":
                        modified_flight.departure.scheduled = change.new_time.time()
                        # Recalculate delay if actual time exists
                        if modified_flight.departure.actual:
                            modified_flight.dep_delay_min = modified_flight.departure.get_delay_minutes()
                    elif change.change_type == "arrival":
                        modified_flight.arrival.scheduled = change.new_time.time()
                        if modified_flight.arrival.actual:
                            modified_flight.arr_delay_min = modified_flight.arrival.get_delay_minutes()
            
            modified_flights.append(modified_flight)
        
        return Schedule(flights=modified_flights, schedule_date=base_schedule.schedule_date)
    
    def _analyze_modified_schedule_peaks(self, modified_schedule: Schedule, 
                                       airport: str, analysis_date: date,
                                       weather_regime: WeatherRegime) -> PeakAnalysis:
        """Analyze peaks for modified schedule."""
        # This is a simplified implementation
        # In a full implementation, we would need to update the database temporarily
        # or modify the analytics engine to work with in-memory schedules
        
        # For now, return the original analysis as a placeholder
        return self.analytics_engine.analyze_peaks(airport, analysis_date, weather_regime=weather_regime)
    
    def _calculate_delay_impact(self, base_schedule: Schedule, 
                              modified_schedule: Schedule) -> Dict[str, Any]:
        """Calculate delay impact metrics."""
        base_metrics = base_schedule.get_delay_metrics()
        modified_metrics = modified_schedule.get_delay_metrics()
        
        total_change = modified_metrics.total_delay_minutes - base_metrics.total_delay_minutes
        otp_change = modified_metrics.on_time_performance - base_metrics.on_time_performance
        
        direction = "neutral"
        if total_change < -5:
            direction = "improvement"
        elif total_change > 5:
            direction = "degradation"
        
        return {
            "total_change": total_change,
            "otp_change": otp_change,
            "direction": direction,
            "base_total": base_metrics.total_delay_minutes,
            "modified_total": modified_metrics.total_delay_minutes
        }
    
    def _calculate_capacity_impact(self, base_analysis: PeakAnalysis, 
                                 modified_analysis: PeakAnalysis) -> Dict[str, Any]:
        """Calculate capacity impact metrics."""
        base_overloads = len(base_analysis.overload_windows)
        modified_overloads = len(modified_analysis.overload_windows)
        
        overload_change = modified_overloads - base_overloads
        new_windows = max(0, overload_change)
        
        direction = "neutral"
        if overload_change < 0:
            direction = "improvement"
        elif overload_change > 0:
            direction = "degradation"
        
        return {
            "overload_change": overload_change,
            "new_windows": new_windows,
            "direction": direction,
            "base_overloads": base_overloads,
            "modified_overloads": modified_overloads
        }
    
    def _calculate_co2_impact(self, changes: List[FlightChange], 
                            base_schedule: Schedule) -> Dict[str, Any]:
        """Calculate CO2 impact of changes."""
        total_co2_kg = 0.0
        
        for change in changes:
            # Find the flight
            flight = None
            for f in base_schedule.flights:
                if f.flight_id == change.flight_id or f.flight_number == change.flight_id:
                    flight = f
                    break
            
            if not flight:
                continue
            
            # Get aircraft-specific factor
            aircraft_factor = self.co2_factors.get_aircraft_factor(flight.aircraft_type)
            
            # Calculate CO2 impact based on time change
            time_delta_minutes = abs(change.time_delta_minutes)
            
            if change.time_delta_minutes < 0:
                # Earlier departure - potential fuel savings from reduced delay
                co2_savings = time_delta_minutes * self.co2_factors.delay_factor_kg_per_min
                total_co2_kg -= co2_savings
            else:
                # Later departure - potential additional fuel burn
                co2_increase = time_delta_minutes * self.co2_factors.delay_factor_kg_per_min
                total_co2_kg += co2_increase
            
            # Add routing optimization benefits for significant changes
            if time_delta_minutes > 15:
                total_co2_kg -= self.co2_factors.routing_optimization_kg
        
        # Convert to fuel liters (approximate: 1 liter jet fuel = 2.5 kg CO2)
        fuel_liters = abs(total_co2_kg) / 2.5
        
        direction = "neutral"
        if total_co2_kg < -10:
            direction = "improvement"
        elif total_co2_kg > 10:
            direction = "degradation"
        
        return {
            "total_kg": total_co2_kg,
            "fuel_liters": fuel_liters,
            "direction": direction
        }
    
    def _calculate_fairness_score(self, changes: List[FlightChange], 
                                base_schedule: Schedule) -> float:
        """Calculate fairness score for changes (0-1 scale)."""
        if not changes:
            return 1.0
        
        # Simple fairness metric based on distribution of impacts
        time_deltas = [abs(change.time_delta_minutes) for change in changes]
        
        if not time_deltas:
            return 1.0
        
        # Fairness is higher when impacts are more evenly distributed
        mean_delta = sum(time_deltas) / len(time_deltas)
        variance = sum((delta - mean_delta) ** 2 for delta in time_deltas) / len(time_deltas)
        
        # Normalize to 0-1 scale (lower variance = higher fairness)
        fairness = max(0.0, 1.0 - (variance / 100))  # Assuming max reasonable variance of 100
        
        return min(1.0, fairness)
    
    def _generate_recommendation(self, delay_impact: Dict[str, Any], 
                               capacity_impact: Dict[str, Any],
                               co2_impact: Dict[str, Any], 
                               fairness_score: float) -> Tuple[str, str]:
        """Generate recommendation and confidence level."""
        # Score each dimension
        delay_score = 0
        if delay_impact["direction"] == "improvement":
            delay_score = 2
        elif delay_impact["direction"] == "neutral":
            delay_score = 1
        
        capacity_score = 0
        if capacity_impact["direction"] == "improvement":
            capacity_score = 2
        elif capacity_impact["direction"] == "neutral":
            capacity_score = 1
        
        co2_score = 0
        if co2_impact["direction"] == "improvement":
            co2_score = 2
        elif co2_impact["direction"] == "neutral":
            co2_score = 1
        
        fairness_score_scaled = int(fairness_score * 2)  # 0-2 scale
        
        total_score = delay_score + capacity_score + co2_score + fairness_score_scaled
        max_score = 8
        
        # Generate recommendation
        if total_score >= 7:
            recommendation = "Highly recommended - significant positive impact across all metrics"
            confidence = "high"
        elif total_score >= 5:
            recommendation = "Recommended - net positive impact with acceptable trade-offs"
            confidence = "medium"
        elif total_score >= 3:
            recommendation = "Consider carefully - mixed impact, evaluate specific priorities"
            confidence = "medium"
        else:
            recommendation = "Not recommended - negative impact outweighs benefits"
            confidence = "high"
        
        # Adjust confidence based on data quality
        if delay_impact.get("base_total", 0) < 10:  # Low baseline delay
            confidence = "low"
        
        return recommendation, confidence
    
    def _identify_delay_improvements(self, base_schedule: Schedule, 
                                   modified_schedule: Schedule) -> List[Dict[str, Any]]:
        """Identify specific delay improvements."""
        improvements = []
        
        base_flights = {f.flight_id: f for f in base_schedule.flights}
        
        for modified_flight in modified_schedule.flights:
            base_flight = base_flights.get(modified_flight.flight_id)
            if not base_flight:
                continue
            
            base_delay = base_flight.dep_delay_min or 0
            modified_delay = modified_flight.dep_delay_min or 0
            
            if modified_delay < base_delay - 5:  # Significant improvement
                improvements.append({
                    "flight_id": modified_flight.flight_id,
                    "flight_number": modified_flight.flight_number,
                    "base_delay": base_delay,
                    "modified_delay": modified_delay,
                    "improvement_minutes": base_delay - modified_delay
                })
        
        return improvements
    
    def _identify_delay_degradations(self, base_schedule: Schedule, 
                                   modified_schedule: Schedule) -> List[Dict[str, Any]]:
        """Identify specific delay degradations."""
        degradations = []
        
        base_flights = {f.flight_id: f for f in base_schedule.flights}
        
        for modified_flight in modified_schedule.flights:
            base_flight = base_flights.get(modified_flight.flight_id)
            if not base_flight:
                continue
            
            base_delay = base_flight.dep_delay_min or 0
            modified_delay = modified_flight.dep_delay_min or 0
            
            if modified_delay > base_delay + 5:  # Significant degradation
                degradations.append({
                    "flight_id": modified_flight.flight_id,
                    "flight_number": modified_flight.flight_number,
                    "base_delay": base_delay,
                    "modified_delay": modified_delay,
                    "degradation_minutes": modified_delay - base_delay
                })
        
        return degradations
    
    def _identify_capacity_changes(self, base_analysis: PeakAnalysis, 
                                 modified_analysis: PeakAnalysis) -> List[Dict[str, Any]]:
        """Identify specific capacity changes."""
        changes = []
        
        # Compare overload windows
        base_windows = len(base_analysis.overload_windows)
        modified_windows = len(modified_analysis.overload_windows)
        
        if base_windows != modified_windows:
            changes.append({
                "type": "overload_windows",
                "base_count": base_windows,
                "modified_count": modified_windows,
                "change": modified_windows - base_windows
            })
        
        # Compare peak utilization
        base_util = base_analysis.avg_utilization
        modified_util = modified_analysis.avg_utilization
        
        if abs(modified_util - base_util) > 0.05:  # 5% change threshold
            changes.append({
                "type": "peak_utilization",
                "base_utilization": base_util,
                "modified_utilization": modified_util,
                "change": modified_util - base_util
            })
        
        return changes
    
    def _create_empty_impact_card(self, scenario_id: str, description: str, 
                                error_message: str) -> ImpactCard:
        """Create empty impact card for error cases."""
        return ImpactCard(
            scenario_id=scenario_id,
            scenario_description=description,
            delay_change_minutes=0.0,
            delay_direction="neutral",
            affected_flights_count=0,
            peak_overload_change=0,
            capacity_direction="neutral",
            new_overload_windows=0,
            co2_change_kg=0.0,
            co2_direction="neutral",
            fuel_savings_liters=0.0,
            on_time_performance_change=0.0,
            fairness_score=1.0,
            constraint_violations=0,
            recommendation=error_message,
            confidence_level="low"
        )