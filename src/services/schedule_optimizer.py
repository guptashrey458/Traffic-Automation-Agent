"""Schedule optimization engine using constraint-based algorithms."""

import os
import warnings
from datetime import datetime, date, time, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from collections import defaultdict

# Suppress optimization solver warnings
warnings.filterwarnings('ignore', category=UserWarning, module='ortools')

# Temporarily disable OR-Tools due to compatibility issues
ORTOOLS_AVAILABLE = False
print("Info: Using fallback optimization methods (OR-Tools disabled for compatibility)")

from ..models.flight import Flight
from .database import FlightDatabaseService
from .analytics import AnalyticsEngine, WeatherRegime
from .delay_prediction import DelayRiskPredictor, OperationalContext


class OptimizationObjective(Enum):
    """Optimization objectives with weights."""
    MINIMIZE_DELAYS = "minimize_delays"
    MINIMIZE_TAXI_TIME = "minimize_taxi_time"
    MINIMIZE_RUNWAY_CHANGES = "minimize_runway_changes"
    MAXIMIZE_FAIRNESS = "maximize_fairness"
    MINIMIZE_CURFEW_VIOLATIONS = "minimize_curfew_violations"


class ConstraintType(Enum):
    """Types of operational constraints."""
    TURNAROUND_TIME = "turnaround_time"
    RUNWAY_CAPACITY = "runway_capacity"
    WAKE_TURBULENCE = "wake_turbulence"
    WEATHER_CAPACITY = "weather_capacity"
    CURFEW_HOURS = "curfew_hours"
    MINIMUM_SEPARATION = "minimum_separation"


@dataclass
class ObjectiveWeights:
    """Weights for multi-objective optimization."""
    delay_weight: float = 1.0
    taxi_weight: float = 0.3
    runway_change_weight: float = 0.2
    fairness_weight: float = 0.4
    curfew_weight: float = 2.0
    
    def normalize(self) -> "ObjectiveWeights":
        """Normalize weights to sum to 1.0."""
        total = (self.delay_weight + self.taxi_weight + self.runway_change_weight + 
                self.fairness_weight + self.curfew_weight)
        
        if total == 0:
            return ObjectiveWeights()
        
        return ObjectiveWeights(
            delay_weight=self.delay_weight / total,
            taxi_weight=self.taxi_weight / total,
            runway_change_weight=self.runway_change_weight / total,
            fairness_weight=self.fairness_weight / total,
            curfew_weight=self.curfew_weight / total
        )


@dataclass
class TimeSlot:
    """Time slot for runway operations."""
    runway: str
    timestamp: datetime
    capacity: int
    current_demand: int = 0
    weather_regime: WeatherRegime = WeatherRegime.CALM
    is_curfew: bool = False
    
    @property
    def utilization(self) -> float:
        """Calculate utilization rate."""
        return self.current_demand / self.capacity if self.capacity > 0 else 0.0
    
    @property
    def available_capacity(self) -> int:
        """Get available capacity."""
        return max(0, self.capacity - self.current_demand)


@dataclass
class FlightChange:
    """Represents a change to a flight's schedule."""
    flight_id: str
    original_time: datetime
    new_time: datetime
    change_type: str  # "departure" or "arrival"
    runway_change: bool = False
    original_runway: Optional[str] = None
    new_runway: Optional[str] = None
    cost_impact: float = 0.0
    delay_impact: float = 0.0
    
    @property
    def time_delta_minutes(self) -> float:
        """Get time change in minutes."""
        return (self.new_time - self.original_time).total_seconds() / 60


@dataclass
class ConstraintViolation:
    """Represents a constraint violation."""
    constraint_type: ConstraintType
    flight_id: str
    description: str
    severity: str  # "minor", "major", "critical"
    suggested_fix: Optional[str] = None


@dataclass
class DelayMetrics:
    """Delay performance metrics."""
    total_delay_minutes: float
    avg_delay_minutes: float
    p95_delay_minutes: float
    delayed_flights_count: int
    on_time_performance: float  # Percentage of flights <= 15min delay
    
    def improvement_over(self, baseline: "DelayMetrics") -> Dict[str, float]:
        """Calculate improvement metrics over baseline."""
        return {
            "total_delay_reduction": baseline.total_delay_minutes - self.total_delay_minutes,
            "avg_delay_reduction": baseline.avg_delay_minutes - self.avg_delay_minutes,
            "p95_delay_reduction": baseline.p95_delay_minutes - self.p95_delay_minutes,
            "otp_improvement": self.on_time_performance - baseline.on_time_performance
        }


@dataclass
class Schedule:
    """Flight schedule representation."""
    flights: List[Flight] = field(default_factory=list)
    time_slots: Dict[str, List[TimeSlot]] = field(default_factory=dict)  # runway -> slots
    schedule_date: Optional[date] = None
    
    def get_delay_metrics(self) -> DelayMetrics:
        """Calculate delay metrics for the schedule."""
        delays = []
        delayed_count = 0
        
        for flight in self.flights:
            if flight.dep_delay_min is not None:
                delays.append(flight.dep_delay_min)
                if flight.dep_delay_min > 15:
                    delayed_count += 1
        
        if not delays:
            return DelayMetrics(0, 0, 0, 0, 100.0)
        
        total_delay = sum(delays)
        avg_delay = total_delay / len(delays)
        p95_delay = np.percentile(delays, 95)
        otp = ((len(delays) - delayed_count) / len(delays)) * 100
        
        return DelayMetrics(
            total_delay_minutes=total_delay,
            avg_delay_minutes=avg_delay,
            p95_delay_minutes=p95_delay,
            delayed_flights_count=delayed_count,
            on_time_performance=otp
        )


@dataclass
class Constraints:
    """Operational constraints for optimization."""
    # Turnaround constraints
    min_turnaround_minutes: Dict[str, float] = field(default_factory=dict)  # aircraft_type -> minutes
    
    # Runway constraints
    runway_capacity: Dict[str, int] = field(default_factory=dict)  # runway -> capacity per hour
    weather_capacity_reduction: Dict[WeatherRegime, float] = field(default_factory=dict)
    
    # Wake turbulence separation (minutes)
    wake_separation: Dict[Tuple[str, str], float] = field(default_factory=dict)  # (leading, following) -> minutes
    
    # Curfew constraints
    curfew_hours: List[int] = field(default_factory=list)  # Hours when operations are restricted
    curfew_penalty: float = 100.0  # High penalty for curfew violations
    
    # Minimum separation between operations (minutes)
    min_separation_minutes: float = 2.0
    
    def __post_init__(self):
        """Set default constraint values."""
        if not self.min_turnaround_minutes:
            self.min_turnaround_minutes = {
                "A320": 45, "A321": 50, "B737": 45, "B738": 45,
                "B777": 90, "B787": 75, "DEFAULT": 60
            }
        
        if not self.runway_capacity:
            self.runway_capacity = {"DEFAULT": 30}  # 30 operations per hour
        
        if not self.weather_capacity_reduction:
            self.weather_capacity_reduction = {
                WeatherRegime.CALM: 1.0,
                WeatherRegime.MEDIUM: 0.85,
                WeatherRegime.STRONG: 0.65,
                WeatherRegime.SEVERE: 0.3
            }
        
        if not self.wake_separation:
            # Simplified wake turbulence separation (minutes)
            self.wake_separation = {
                ("HEAVY", "MEDIUM"): 3.0,
                ("HEAVY", "LIGHT"): 4.0,
                ("MEDIUM", "LIGHT"): 2.0,
                ("DEFAULT", "DEFAULT"): 2.0
            }
        
        if not self.curfew_hours:
            self.curfew_hours = [1, 2, 3, 4, 5]  # 1 AM to 5 AM


@dataclass
class OptimizationResult:
    """Result of schedule optimization."""
    original_schedule: Schedule
    optimized_schedule: Schedule
    cost_reduction: float
    delay_improvement: DelayMetrics
    affected_flights: List[FlightChange]
    constraint_violations: List[ConstraintViolation]
    optimization_time_seconds: float
    solver_status: str
    objective_value: float
    
    def get_summary(self) -> Dict[str, Any]:
        """Get optimization summary."""
        original_metrics = self.original_schedule.get_delay_metrics()
        optimized_metrics = self.optimized_schedule.get_delay_metrics()
        improvements = optimized_metrics.improvement_over(original_metrics)
        
        return {
            "flights_affected": len(self.affected_flights),
            "total_delay_reduction_minutes": improvements["total_delay_reduction"],
            "avg_delay_reduction_minutes": improvements["avg_delay_reduction"],
            "otp_improvement_percent": improvements["otp_improvement"],
            "constraint_violations": len(self.constraint_violations),
            "optimization_time_seconds": self.optimization_time_seconds,
            "solver_status": self.solver_status,
            "cost_reduction": self.cost_reduction
        }


@dataclass
class ImpactAnalysis:
    """What-if analysis results."""
    delay_delta: float
    peak_overload_change: int
    co2_impact: float
    fairness_score: float
    affected_flights: List[str]
    before_metrics: DelayMetrics
    after_metrics: DelayMetrics
    
    def get_impact_card(self) -> Dict[str, Any]:
        """Generate impact card for visualization."""
        return {
            "delay_impact": {
                "change_minutes": round(self.delay_delta, 1),
                "direction": "improvement" if self.delay_delta < 0 else "degradation"
            },
            "capacity_impact": {
                "overload_change": self.peak_overload_change,
                "direction": "improvement" if self.peak_overload_change < 0 else "degradation"
            },
            "environmental_impact": {
                "co2_change_kg": round(self.co2_impact, 1),
                "direction": "improvement" if self.co2_impact < 0 else "degradation"
            },
            "fairness_score": round(self.fairness_score, 2),
            "affected_flights_count": len(self.affected_flights),
            "overall_recommendation": self._get_recommendation()
        }
    
    def _get_recommendation(self) -> str:
        """Get overall recommendation based on impact analysis."""
        if self.delay_delta < -5 and self.peak_overload_change <= 0:
            return "Highly recommended - significant improvement"
        elif self.delay_delta < 0 and self.peak_overload_change <= 0:
            return "Recommended - net positive impact"
        elif abs(self.delay_delta) < 2 and abs(self.peak_overload_change) <= 1:
            return "Neutral - minimal impact"
        elif self.delay_delta > 5 or self.peak_overload_change > 2:
            return "Not recommended - negative impact"
        else:
            return "Consider carefully - mixed impact"


class ScheduleOptimizer:
    """Schedule optimization engine using constraint-based algorithms."""
    
    def __init__(self, db_service: Optional[FlightDatabaseService] = None,
                 analytics_engine: Optional[AnalyticsEngine] = None,
                 delay_predictor: Optional[DelayRiskPredictor] = None):
        """
        Initialize the schedule optimizer.
        
        Args:
            db_service: Database service for flight data
            analytics_engine: Analytics engine for peak analysis
            delay_predictor: Delay prediction service
        """
        self.db_service = db_service or FlightDatabaseService()
        self.analytics_engine = analytics_engine or AnalyticsEngine()
        self.delay_predictor = delay_predictor or DelayRiskPredictor()
        
        # Optimization settings
        self.max_optimization_time_seconds = 300  # 5 minutes max
        self.solution_time_limit_seconds = 60     # 1 minute for quick solutions
        
    def optimize_schedule(self, flights: List[Flight], constraints: Constraints,
                         weights: ObjectiveWeights = None,
                         weather_regime: WeatherRegime = WeatherRegime.CALM) -> OptimizationResult:
        """
        Optimize flight schedule using constraint-based algorithms.
        
        Args:
            flights: List of flights to optimize
            constraints: Operational constraints
            weights: Objective weights for multi-objective optimization
            weather_regime: Current weather conditions
            
        Returns:
            OptimizationResult with optimized schedule and metrics
        """
        start_time = datetime.now()
        
        if weights is None:
            weights = ObjectiveWeights()
        weights = weights.normalize()
        
        # Create original schedule
        original_schedule = Schedule(flights=flights.copy())
        
        if not ORTOOLS_AVAILABLE:
            return self._fallback_optimization(original_schedule, constraints, weights)
        
        try:
            # Use CP-SAT solver for constraint satisfaction
            optimized_schedule, flight_changes, violations = self._solve_with_cp_sat(
                flights, constraints, weights, weather_regime
            )
            
            solver_status = "OPTIMAL"
            
        except Exception as e:
            print(f"CP-SAT optimization failed: {e}")
            # Fallback to min-cost flow
            try:
                optimized_schedule, flight_changes, violations = self._solve_with_min_cost_flow(
                    flights, constraints, weights, weather_regime
                )
                solver_status = "FEASIBLE"
            except Exception as e2:
                print(f"Min-cost flow optimization failed: {e2}")
                return self._fallback_optimization(original_schedule, constraints, weights)
        
        # Calculate metrics
        original_metrics = original_schedule.get_delay_metrics()
        optimized_metrics = optimized_schedule.get_delay_metrics()
        delay_improvement = optimized_metrics.improvement_over(original_metrics)
        
        # Calculate cost reduction
        cost_reduction = self._calculate_cost_reduction(flight_changes, weights)
        
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        return OptimizationResult(
            original_schedule=original_schedule,
            optimized_schedule=optimized_schedule,
            cost_reduction=cost_reduction,
            delay_improvement=optimized_metrics,
            affected_flights=flight_changes,
            constraint_violations=violations,
            optimization_time_seconds=optimization_time,
            solver_status=solver_status,
            objective_value=cost_reduction
        )
    
    def what_if_analysis(self, base_schedule: Schedule, 
                        changes: List[FlightChange]) -> ImpactAnalysis:
        """
        Analyze the impact of proposed schedule changes.
        
        Args:
            base_schedule: Current schedule
            changes: Proposed flight changes
            
        Returns:
            ImpactAnalysis with before/after metrics
        """
        # Create modified schedule
        modified_flights = base_schedule.flights.copy()
        affected_flight_ids = []
        
        for change in changes:
            affected_flight_ids.append(change.flight_id)
            
            # Find and modify the flight
            for flight in modified_flights:
                if flight.flight_id == change.flight_id:
                    if change.change_type == "departure":
                        flight.departure.scheduled = change.new_time.time()
                        # Recalculate delay if actual time exists
                        if flight.departure.actual:
                            flight.dep_delay_min = flight.departure.get_delay_minutes()
                    elif change.change_type == "arrival":
                        flight.arrival.scheduled = change.new_time.time()
                        if flight.arrival.actual:
                            flight.arr_delay_min = flight.arrival.get_delay_minutes()
                    break
        
        modified_schedule = Schedule(flights=modified_flights)
        
        # Calculate metrics
        before_metrics = base_schedule.get_delay_metrics()
        after_metrics = modified_schedule.get_delay_metrics()
        
        # Calculate impacts
        delay_delta = after_metrics.total_delay_minutes - before_metrics.total_delay_minutes
        
        # Estimate peak overload change (simplified)
        peak_overload_change = self._estimate_peak_overload_change(base_schedule, changes)
        
        # Estimate CO2 impact (simplified)
        co2_impact = self._estimate_co2_impact(changes)
        
        # Calculate fairness score
        fairness_score = self._calculate_fairness_score(changes)
        
        return ImpactAnalysis(
            delay_delta=delay_delta,
            peak_overload_change=peak_overload_change,
            co2_impact=co2_impact,
            fairness_score=fairness_score,
            affected_flights=affected_flight_ids,
            before_metrics=before_metrics,
            after_metrics=after_metrics
        )
    
    def validate_constraints(self, schedule: Schedule, 
                           constraints: Constraints) -> List[ConstraintViolation]:
        """
        Validate schedule against operational constraints.
        
        Args:
            schedule: Schedule to validate
            constraints: Constraints to check
            
        Returns:
            List of constraint violations
        """
        violations = []
        
        # Check turnaround time constraints
        violations.extend(self._check_turnaround_constraints(schedule, constraints))
        
        # Check runway capacity constraints
        violations.extend(self._check_capacity_constraints(schedule, constraints))
        
        # Check curfew constraints
        violations.extend(self._check_curfew_constraints(schedule, constraints))
        
        # Check wake turbulence separation
        violations.extend(self._check_wake_separation(schedule, constraints))
        
        return violations
    
    def _solve_with_cp_sat(self, flights: List[Flight], constraints: Constraints,
                          weights: ObjectiveWeights, weather_regime: WeatherRegime) -> Tuple[Schedule, List[FlightChange], List[ConstraintViolation]]:
        """Solve optimization using CP-SAT solver."""
        model = cp_model.CpModel()
        
        # Create time slots (5-minute intervals for the day)
        time_slots = self._create_time_slots(flights, weather_regime, constraints)
        
        # Decision variables: flight -> time slot assignment
        flight_vars = {}
        for i, flight in enumerate(flights):
            flight_vars[flight.flight_id] = {}
            for j, slot in enumerate(time_slots):
                flight_vars[flight.flight_id][j] = model.NewBoolVar(f'flight_{i}_slot_{j}')
        
        # Constraints: Each flight assigned to exactly one slot
        for flight in flights:
            model.Add(sum(flight_vars[flight.flight_id][j] for j in range(len(time_slots))) == 1)
        
        # Capacity constraints: Each slot has limited capacity
        for j, slot in enumerate(time_slots):
            model.Add(
                sum(flight_vars[flight.flight_id][j] for flight in flights) <= slot.capacity
            )
        
        # Objective: Minimize weighted cost
        objective_terms = []
        
        for i, flight in enumerate(flights):
            for j, slot in enumerate(time_slots):
                # Calculate cost for assigning this flight to this slot
                cost = self._calculate_assignment_cost(flight, slot, weights)
                objective_terms.append(cost * flight_vars[flight.flight_id][j])
        
        model.Minimize(sum(objective_terms))
        
        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.solution_time_limit_seconds
        status = solver.Solve(model)
        
        if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            raise Exception(f"CP-SAT solver failed with status: {status}")
        
        # Extract solution
        optimized_flights = []
        flight_changes = []
        
        for flight in flights:
            new_flight = flight  # Copy flight
            
            for j, slot in enumerate(time_slots):
                if solver.Value(flight_vars[flight.flight_id][j]) == 1:
                    # Flight assigned to this slot
                    original_time = datetime.combine(flight.flight_date or date.today(), 
                                                   flight.departure.scheduled or time(12, 0))
                    
                    if abs((slot.timestamp - original_time).total_seconds()) > 300:  # 5+ minute change
                        change = FlightChange(
                            flight_id=flight.flight_id,
                            original_time=original_time,
                            new_time=slot.timestamp,
                            change_type="departure"
                        )
                        flight_changes.append(change)
                        
                        # Update flight time
                        new_flight.departure.scheduled = slot.timestamp.time()
                    
                    optimized_flights.append(new_flight)
                    break
        
        optimized_schedule = Schedule(flights=optimized_flights)
        violations = self.validate_constraints(optimized_schedule, constraints)
        
        return optimized_schedule, flight_changes, violations
    
    def _solve_with_min_cost_flow(self, flights: List[Flight], constraints: Constraints,
                                 weights: ObjectiveWeights, weather_regime: WeatherRegime) -> Tuple[Schedule, List[FlightChange], List[ConstraintViolation]]:
        """Solve optimization using min-cost flow algorithm."""
        # Create network graph
        G = nx.DiGraph()
        
        # Add source and sink nodes
        G.add_node("source", demand=-len(flights))
        G.add_node("sink", demand=len(flights))
        
        # Create time slots
        time_slots = self._create_time_slots(flights, weather_regime, constraints)
        
        # Add slot nodes
        for i, slot in enumerate(time_slots):
            slot_id = f"slot_{i}"
            G.add_node(slot_id, demand=0)
            
            # Connect sink to slot with capacity constraint
            G.add_edge(slot_id, "sink", capacity=slot.capacity, weight=0)
        
        # Add flight nodes and edges
        for flight in flights:
            flight_id = f"flight_{flight.flight_id}"
            G.add_node(flight_id, demand=0)
            
            # Connect source to flight
            G.add_edge("source", flight_id, capacity=1, weight=0)
            
            # Connect flight to compatible slots
            for i, slot in enumerate(time_slots):
                slot_id = f"slot_{i}"
                cost = self._calculate_assignment_cost(flight, slot, weights)
                G.add_edge(flight_id, slot_id, capacity=1, weight=int(cost * 100))  # Scale for integer weights
        
        # Solve min-cost flow
        try:
            flow_cost, flow_dict = nx.network_simplex(G)
        except nx.NetworkXUnfeasible:
            raise Exception("Min-cost flow problem is infeasible")
        
        # Extract solution
        optimized_flights = []
        flight_changes = []
        
        for flight in flights:
            flight_id = f"flight_{flight.flight_id}"
            new_flight = flight  # Copy flight
            
            for i, slot in enumerate(time_slots):
                slot_id = f"slot_{i}"
                
                if flow_dict.get(flight_id, {}).get(slot_id, 0) > 0:
                    # Flight assigned to this slot
                    original_time = datetime.combine(flight.flight_date or date.today(),
                                                   flight.departure.scheduled or time(12, 0))
                    
                    if abs((slot.timestamp - original_time).total_seconds()) > 300:  # 5+ minute change
                        change = FlightChange(
                            flight_id=flight.flight_id,
                            original_time=original_time,
                            new_time=slot.timestamp,
                            change_type="departure"
                        )
                        flight_changes.append(change)
                        
                        # Update flight time
                        new_flight.departure.scheduled = slot.timestamp.time()
                    
                    optimized_flights.append(new_flight)
                    break
        
        optimized_schedule = Schedule(flights=optimized_flights)
        violations = self.validate_constraints(optimized_schedule, constraints)
        
        return optimized_schedule, flight_changes, violations
    
    def _create_time_slots(self, flights: List[Flight], weather_regime: WeatherRegime,
                          constraints: Constraints) -> List[TimeSlot]:
        """Create time slots for optimization."""
        if not flights:
            return []
        
        # Determine time range
        flight_times = []
        for flight in flights:
            if flight.departure.scheduled:
                flight_time = datetime.combine(flight.flight_date or date.today(),
                                             flight.departure.scheduled)
                flight_times.append(flight_time)
        
        if not flight_times:
            return []
        
        start_time = min(flight_times) - timedelta(hours=2)  # 2 hours before earliest
        end_time = max(flight_times) + timedelta(hours=2)    # 2 hours after latest
        
        # Create 5-minute slots
        slots = []
        current_time = start_time.replace(minute=(start_time.minute // 5) * 5, second=0, microsecond=0)
        
        while current_time <= end_time:
            # Determine runway (simplified - use default)
            runway = "DEFAULT"
            
            # Get base capacity
            base_capacity = constraints.runway_capacity.get(runway, 30)
            
            # Apply weather reduction
            weather_factor = constraints.weather_capacity_reduction.get(weather_regime, 1.0)
            
            # Scale to 5-minute slot (capacity is per hour)
            slot_capacity = max(1, int(base_capacity * weather_factor * (5/60)))
            
            # Check if curfew hour
            is_curfew = current_time.hour in constraints.curfew_hours
            if is_curfew:
                slot_capacity = max(1, slot_capacity // 4)  # Severely reduced during curfew
            
            slot = TimeSlot(
                runway=runway,
                timestamp=current_time,
                capacity=slot_capacity,
                weather_regime=weather_regime,
                is_curfew=is_curfew
            )
            
            slots.append(slot)
            current_time += timedelta(minutes=5)
        
        return slots
    
    def _calculate_assignment_cost(self, flight: Flight, slot: TimeSlot, 
                                  weights: ObjectiveWeights) -> float:
        """Calculate cost of assigning a flight to a time slot."""
        cost = 0.0
        
        # Original scheduled time
        original_time = datetime.combine(flight.flight_date or date.today(),
                                       flight.departure.scheduled or time(12, 0))
        
        # Time change penalty
        time_delta_minutes = abs((slot.timestamp - original_time).total_seconds() / 60)
        delay_cost = time_delta_minutes * weights.delay_weight
        cost += delay_cost
        
        # Curfew penalty
        if slot.is_curfew:
            cost += 100 * weights.curfew_weight
        
        # Capacity utilization penalty (prefer less congested slots)
        if slot.utilization > 0.8:
            cost += 20 * weights.delay_weight
        
        # Weather penalty
        if slot.weather_regime in [WeatherRegime.STRONG, WeatherRegime.SEVERE]:
            cost += 10 * weights.delay_weight
        
        return cost
    
    def _calculate_cost_reduction(self, flight_changes: List[FlightChange],
                                 weights: ObjectiveWeights) -> float:
        """Calculate total cost reduction from optimization."""
        total_reduction = 0.0
        
        for change in flight_changes:
            # Positive reduction for improvements, negative for degradations
            if abs(change.time_delta_minutes) < 15:  # Small improvements
                total_reduction += 5.0
            elif change.time_delta_minutes < 0:  # Earlier departure (improvement)
                total_reduction += abs(change.time_delta_minutes) * 0.5
            else:  # Later departure (degradation)
                total_reduction -= change.time_delta_minutes * 0.3
        
        return total_reduction
    
    def _fallback_optimization(self, original_schedule: Schedule, constraints: Constraints,
                              weights: ObjectiveWeights) -> OptimizationResult:
        """Fallback optimization using simple heuristics."""
        print("Using fallback optimization (heuristic-based)")
        
        # Simple heuristic: spread flights more evenly
        optimized_flights = original_schedule.flights.copy()
        flight_changes = []
        
        # Sort flights by scheduled time
        optimized_flights.sort(key=lambda f: f.departure.scheduled or time(12, 0))
        
        # Apply small adjustments to reduce clustering
        for i, flight in enumerate(optimized_flights):
            if i > 0 and flight.departure.scheduled:
                prev_flight = optimized_flights[i-1]
                if prev_flight.departure.scheduled:
                    # Check if flights are too close
                    time_diff = (datetime.combine(date.today(), flight.departure.scheduled) -
                               datetime.combine(date.today(), prev_flight.departure.scheduled)).total_seconds() / 60
                    
                    if 0 < time_diff < 5:  # Less than 5 minutes apart
                        # Adjust current flight by 5 minutes
                        new_time = datetime.combine(date.today(), flight.departure.scheduled) + timedelta(minutes=5)
                        
                        change = FlightChange(
                            flight_id=flight.flight_id,
                            original_time=datetime.combine(date.today(), flight.departure.scheduled),
                            new_time=new_time,
                            change_type="departure"
                        )
                        flight_changes.append(change)
                        
                        flight.departure.scheduled = new_time.time()
        
        optimized_schedule = Schedule(flights=optimized_flights)
        violations = self.validate_constraints(optimized_schedule, constraints)
        
        return OptimizationResult(
            original_schedule=original_schedule,
            optimized_schedule=optimized_schedule,
            cost_reduction=len(flight_changes) * 2.0,  # Simple cost reduction estimate
            delay_improvement=optimized_schedule.get_delay_metrics(),
            affected_flights=flight_changes,
            constraint_violations=violations,
            optimization_time_seconds=1.0,
            solver_status="HEURISTIC",
            objective_value=len(flight_changes) * 2.0
        )
    
    def _check_turnaround_constraints(self, schedule: Schedule, 
                                    constraints: Constraints) -> List[ConstraintViolation]:
        """Check turnaround time constraints."""
        violations = []
        
        # Group flights by aircraft registration
        aircraft_flights = defaultdict(list)
        for flight in schedule.flights:
            if flight.aircraft_registration:
                aircraft_flights[flight.aircraft_registration].append(flight)
        
        for aircraft, flights in aircraft_flights.items():
            # Sort by departure time
            flights.sort(key=lambda f: f.departure.scheduled or time(0, 0))
            
            for i in range(len(flights) - 1):
                current_flight = flights[i]
                next_flight = flights[i + 1]
                
                # Check if this is a turnaround (arrival -> departure at same airport)
                if (current_flight.destination and next_flight.origin and
                    current_flight.destination.code == next_flight.origin.code):
                    
                    # Calculate turnaround time
                    arrival_time = datetime.combine(current_flight.flight_date or date.today(),
                                                  current_flight.arrival.scheduled or time(12, 0))
                    departure_time = datetime.combine(next_flight.flight_date or date.today(),
                                                    next_flight.departure.scheduled or time(12, 0))
                    
                    turnaround_minutes = (departure_time - arrival_time).total_seconds() / 60
                    
                    # Get minimum turnaround requirement
                    aircraft_type = current_flight.aircraft_type or "DEFAULT"
                    min_turnaround = constraints.min_turnaround_minutes.get(aircraft_type,
                                                                          constraints.min_turnaround_minutes.get("DEFAULT", 60))
                    
                    if turnaround_minutes < min_turnaround:
                        violations.append(ConstraintViolation(
                            constraint_type=ConstraintType.TURNAROUND_TIME,
                            flight_id=next_flight.flight_id,
                            description=f"Turnaround time {turnaround_minutes:.1f}min < required {min_turnaround}min",
                            severity="major",
                            suggested_fix=f"Delay departure by {min_turnaround - turnaround_minutes:.1f} minutes"
                        ))
        
        return violations
    
    def _check_capacity_constraints(self, schedule: Schedule, 
                                  constraints: Constraints) -> List[ConstraintViolation]:
        """Check runway capacity constraints."""
        violations = []
        
        # Group flights by hour and runway
        hourly_demand = defaultdict(int)
        
        for flight in schedule.flights:
            if flight.departure.scheduled:
                hour_key = f"{flight.departure.scheduled.hour}"
                hourly_demand[hour_key] += 1
        
        # Check capacity violations
        default_capacity = constraints.runway_capacity.get("DEFAULT", 30)
        
        for hour_key, demand in hourly_demand.items():
            if demand > default_capacity:
                violations.append(ConstraintViolation(
                    constraint_type=ConstraintType.RUNWAY_CAPACITY,
                    flight_id="MULTIPLE",
                    description=f"Hour {hour_key}: demand {demand} > capacity {default_capacity}",
                    severity="major",
                    suggested_fix="Redistribute flights to adjacent hours"
                ))
        
        return violations
    
    def _check_curfew_constraints(self, schedule: Schedule, 
                                constraints: Constraints) -> List[ConstraintViolation]:
        """Check curfew constraints."""
        violations = []
        
        for flight in schedule.flights:
            if flight.departure.scheduled and flight.departure.scheduled.hour in constraints.curfew_hours:
                violations.append(ConstraintViolation(
                    constraint_type=ConstraintType.CURFEW_HOURS,
                    flight_id=flight.flight_id,
                    description=f"Flight scheduled during curfew hour {flight.departure.scheduled.hour}",
                    severity="critical",
                    suggested_fix="Reschedule outside curfew hours"
                ))
        
        return violations
    
    def _check_wake_separation(self, schedule: Schedule, 
                             constraints: Constraints) -> List[ConstraintViolation]:
        """Check wake turbulence separation constraints."""
        violations = []
        
        # Sort flights by departure time
        sorted_flights = sorted(schedule.flights, 
                              key=lambda f: f.departure.scheduled or time(0, 0))
        
        for i in range(len(sorted_flights) - 1):
            current_flight = sorted_flights[i]
            next_flight = sorted_flights[i + 1]
            
            if (current_flight.departure.scheduled and next_flight.departure.scheduled):
                time_diff = (datetime.combine(date.today(), next_flight.departure.scheduled) -
                           datetime.combine(date.today(), current_flight.departure.scheduled)).total_seconds() / 60
                
                # Get wake categories (simplified)
                current_wake = self._get_wake_category(current_flight.aircraft_type)
                next_wake = self._get_wake_category(next_flight.aircraft_type)
                
                required_separation = constraints.wake_separation.get((current_wake, next_wake),
                                                                    constraints.min_separation_minutes)
                
                if 0 < time_diff < required_separation:
                    violations.append(ConstraintViolation(
                        constraint_type=ConstraintType.WAKE_TURBULENCE,
                        flight_id=next_flight.flight_id,
                        description=f"Wake separation {time_diff:.1f}min < required {required_separation}min",
                        severity="minor",
                        suggested_fix=f"Increase separation by {required_separation - time_diff:.1f} minutes"
                    ))
        
        return violations
    
    def _get_wake_category(self, aircraft_type: str) -> str:
        """Get wake turbulence category for aircraft type."""
        if not aircraft_type:
            return "DEFAULT"
        
        heavy_aircraft = ["B777", "B787", "A330", "A340", "B747"]
        if any(ac in aircraft_type.upper() for ac in heavy_aircraft):
            return "HEAVY"
        
        light_aircraft = ["ATR", "DHC", "CRJ"]
        if any(ac in aircraft_type.upper() for ac in light_aircraft):
            return "LIGHT"
        
        return "MEDIUM"  # Default for A320, B737, etc.
    
    def _estimate_peak_overload_change(self, base_schedule: Schedule, 
                                     changes: List[FlightChange]) -> int:
        """Estimate change in peak overload from flight changes."""
        # Simplified estimation - count flights moved to/from peak hours
        peak_hours = [7, 8, 9, 17, 18, 19]  # Typical peak hours
        overload_change = 0
        
        for change in changes:
            original_hour = change.original_time.hour
            new_hour = change.new_time.hour
            
            # Moving from peak to non-peak reduces overload
            if original_hour in peak_hours and new_hour not in peak_hours:
                overload_change -= 1
            # Moving from non-peak to peak increases overload
            elif original_hour not in peak_hours and new_hour in peak_hours:
                overload_change += 1
        
        return overload_change
    
    def _estimate_co2_impact(self, changes: List[FlightChange]) -> float:
        """Estimate CO2 impact of flight changes."""
        # Simplified estimation: earlier departures reduce taxi time and fuel burn
        co2_impact = 0.0
        
        for change in changes:
            time_delta_minutes = change.time_delta_minutes
            
            # Earlier departure reduces CO2 (less taxi time in congestion)
            if time_delta_minutes < 0:
                co2_impact += time_delta_minutes * 0.5  # kg CO2 per minute saved
            # Later departure increases CO2
            else:
                co2_impact += time_delta_minutes * 0.3  # kg CO2 per minute added
        
        return co2_impact
    
    def _calculate_fairness_score(self, changes: List[FlightChange]) -> float:
        """Calculate fairness score for flight changes."""
        if not changes:
            return 1.0
        
        # Calculate distribution of delays/improvements
        time_deltas = [change.time_delta_minutes for change in changes]
        
        # Fairness is higher when changes are more evenly distributed
        std_dev = np.std(time_deltas) if len(time_deltas) > 1 else 0
        
        # Normalize to 0-1 scale (lower std dev = higher fairness)
        fairness_score = max(0, 1 - (std_dev / 30))  # 30 minutes as reference
        
        return fairness_score    

    def get_constraints(self, airport: str, date: Optional[date] = None) -> Constraints:
        """
        Get operational constraints for a specific airport.
        
        Args:
            airport: Airport code (e.g., "BOM", "DEL")
            date: Optional date for time-specific constraints
            
        Returns:
            Constraints object with operational rules and limits
        """
        # Airport-specific constraints
        airport_constraints = {
            "BOM": {
                "runway_capacity": {"09": 30, "27": 25, "DEFAULT": 28},
                "curfew_hours": [23, 0, 1, 2, 3, 4, 5],
                "weather_capacity_factors": {
                    WeatherRegime.CALM: 1.0,
                    WeatherRegime.MEDIUM: 0.8,
                    WeatherRegime.STRONG: 0.6,
                    WeatherRegime.SEVERE: 0.3
                }
            },
            "DEL": {
                "runway_capacity": {"09L": 35, "09R": 35, "27L": 32, "27R": 32, "DEFAULT": 34},
                "curfew_hours": [23, 0, 1, 2, 3, 4, 5],
                "weather_capacity_factors": {
                    WeatherRegime.CALM: 1.0,
                    WeatherRegime.MEDIUM: 0.85,
                    WeatherRegime.STRONG: 0.7,
                    WeatherRegime.SEVERE: 0.4
                }
            },
            "DEFAULT": {
                "runway_capacity": {"DEFAULT": 25},
                "curfew_hours": [23, 0, 1, 2, 3, 4, 5],
                "weather_capacity_factors": {
                    WeatherRegime.CALM: 1.0,
                    WeatherRegime.MEDIUM: 0.8,
                    WeatherRegime.STRONG: 0.6,
                    WeatherRegime.SEVERE: 0.3
                }
            }
        }
        
        # Get airport-specific or default constraints
        config = airport_constraints.get(airport, airport_constraints["DEFAULT"])
        
        # Create constraints object
        constraints = Constraints(
            min_turnaround_minutes={
                "A320": 45, "A321": 50, "B737": 45, "B738": 45,
                "B777": 90, "B787": 75, "DEFAULT": 60
            },
            runway_capacity=config["runway_capacity"],
            wake_separation={
                ("HEAVY", "LIGHT"): 3,
                ("HEAVY", "MEDIUM"): 2,
                ("MEDIUM", "LIGHT"): 2,
                ("DEFAULT", "DEFAULT"): 1
            },
            min_separation_minutes=1,
            curfew_hours=config["curfew_hours"],
            weather_capacity_factors=config["weather_capacity_factors"]
        )
        
        # Add date-specific constraints if needed
        if date:
            # Could add special event constraints, seasonal adjustments, etc.
            pass
        
        return constraints
    
    def health_check(self) -> Dict[str, Any]:
        """Check the health of the schedule optimizer service."""
        return {
            "status": "healthy",
            "ortools_available": ORTOOLS_AVAILABLE,
            "fallback_mode": not ORTOOLS_AVAILABLE,
            "timestamp": datetime.now().isoformat()
        }