"""Weighted graph-based schedule optimization engine."""

import warnings
from datetime import datetime, date, time, timedelta
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict
import heapq
import importlib.util

# Suppress optimization solver warnings
warnings.filterwarnings('ignore', category=UserWarning, module='ortools')

# Check for OR-Tools availability
ORTOOLS_AVAILABLE = False
try:
    # Only import when actually needed to avoid bus errors
    import importlib
    cp_model_spec = importlib.util.find_spec("ortools.sat.python.cp_model")
    if cp_model_spec is not None:
        ORTOOLS_AVAILABLE = True
except Exception:
    ORTOOLS_AVAILABLE = False

if not ORTOOLS_AVAILABLE:
    print("Info: OR-Tools not available, using heuristic fallback methods")

from ..models.flight import Flight
from .analytics import WeatherRegime


class SolverType(Enum):
    """Available solver types."""
    CP_SAT = "cp_sat"
    MIN_COST_FLOW = "min_cost_flow"
    HEURISTIC = "heuristic"


@dataclass
class ObjectiveWeights:
    """Weights for multi-objective optimization with runway-dependent factors."""
    delay_weight: float = 1.0
    taxi_weight: float = 0.3
    fairness_weight: float = 0.4
    co2_weight: float = 0.2
    curfew_weight: float = 2.0
    
    # Runway-dependent weight multipliers
    runway_multipliers: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize default runway multipliers."""
        if not self.runway_multipliers:
            self.runway_multipliers = {
                "09L": {"taxi_weight": 1.0, "co2_weight": 1.0},
                "09R": {"taxi_weight": 1.2, "co2_weight": 1.1},  # Longer taxi
                "27L": {"taxi_weight": 0.9, "co2_weight": 0.9},   # Shorter taxi
                "27R": {"taxi_weight": 1.1, "co2_weight": 1.0},
                "DEFAULT": {"taxi_weight": 1.0, "co2_weight": 1.0}
            }
    
    def get_runway_adjusted_weights(self, runway: str) -> "ObjectiveWeights":
        """Get weights adjusted for specific runway."""
        multipliers = self.runway_multipliers.get(runway, self.runway_multipliers["DEFAULT"])
        
        return ObjectiveWeights(
            delay_weight=self.delay_weight,
            taxi_weight=self.taxi_weight * multipliers.get("taxi_weight", 1.0),
            fairness_weight=self.fairness_weight,
            co2_weight=self.co2_weight * multipliers.get("co2_weight", 1.0),
            curfew_weight=self.curfew_weight,
            runway_multipliers=self.runway_multipliers
        )
    
    def normalize(self) -> "ObjectiveWeights":
        """Normalize weights to sum to 1.0."""
        total = (self.delay_weight + self.taxi_weight + self.fairness_weight + 
                self.co2_weight + self.curfew_weight)
        
        if total == 0:
            return ObjectiveWeights()
        
        return ObjectiveWeights(
            delay_weight=self.delay_weight / total,
            taxi_weight=self.taxi_weight / total,
            fairness_weight=self.fairness_weight / total,
            co2_weight=self.co2_weight / total,
            curfew_weight=self.curfew_weight / total,
            runway_multipliers=self.runway_multipliers
        )


@dataclass
class SlotNode:
    """Represents a runway-time slot in the bipartite graph."""
    slot_id: str
    runway: str
    timestamp: datetime
    capacity: int
    current_demand: int = 0
    weather_adjusted_capacity: int = 0
    is_curfew: bool = False
    weather_regime: WeatherRegime = WeatherRegime.CALM
    
    def __post_init__(self):
        """Initialize weather-adjusted capacity."""
        if self.weather_adjusted_capacity == 0:
            self.weather_adjusted_capacity = self.capacity
    
    @property
    def utilization(self) -> float:
        """Calculate current utilization rate."""
        return self.current_demand / self.weather_adjusted_capacity if self.weather_adjusted_capacity > 0 else 0.0
    
    @property
    def available_capacity(self) -> int:
        """Get available capacity."""
        return max(0, self.weather_adjusted_capacity - self.current_demand)
    
    def can_accommodate(self, additional_demand: int = 1) -> bool:
        """Check if slot can accommodate additional demand."""
        return self.current_demand + additional_demand <= self.weather_adjusted_capacity


@dataclass
class FlightNode:
    """Represents a flight in the bipartite graph."""
    flight_id: str
    original_slot: datetime
    constraints: List[str] = field(default_factory=list)
    priority_score: float = 1.0
    airline: str = ""
    aircraft_type: str = ""
    
    def add_constraint(self, constraint: str) -> None:
        """Add a constraint to the flight."""
        if constraint not in self.constraints:
            self.constraints.append(constraint)


@dataclass
class WeightedEdge:
    """Represents an edge between flight and slot with cost breakdown."""
    flight_id: str
    slot_id: str
    cost: float
    cost_breakdown: Dict[str, float] = field(default_factory=dict)
    feasible: bool = True
    constraint_violations: List[str] = field(default_factory=list)
    
    def add_violation(self, violation: str) -> None:
        """Add a constraint violation."""
        if violation not in self.constraint_violations:
            self.constraint_violations.append(violation)
            self.feasible = False


@dataclass
class GraphConstraint:
    """Represents a constraint in the graph."""
    constraint_id: str
    constraint_type: str  # "turnaround", "wake", "curfew", "capacity"
    affected_flights: List[str]
    description: str
    severity: str = "medium"  # "low", "medium", "high", "critical"


@dataclass
class BipartiteGraph:
    """Bipartite graph modeling flights ↔ runway-time slots."""
    flight_nodes: List[FlightNode] = field(default_factory=list)
    slot_nodes: List[SlotNode] = field(default_factory=list)
    edges: List[WeightedEdge] = field(default_factory=list)
    constraints: List[GraphConstraint] = field(default_factory=list)
    
    # Internal mappings for efficient lookup
    _flight_map: Dict[str, FlightNode] = field(default_factory=dict, init=False)
    _slot_map: Dict[str, SlotNode] = field(default_factory=dict, init=False)
    _edge_map: Dict[Tuple[str, str], WeightedEdge] = field(default_factory=dict, init=False)
    
    def __post_init__(self):
        """Build internal mappings."""
        self._rebuild_mappings()
    
    def _rebuild_mappings(self):
        """Rebuild internal mappings for efficient lookup."""
        self._flight_map = {node.flight_id: node for node in self.flight_nodes}
        self._slot_map = {node.slot_id: node for node in self.slot_nodes}
        self._edge_map = {(edge.flight_id, edge.slot_id): edge for edge in self.edges}
    
    def add_flight_node(self, flight_node: FlightNode) -> None:
        """Add a flight node to the graph."""
        self.flight_nodes.append(flight_node)
        self._flight_map[flight_node.flight_id] = flight_node
    
    def add_slot_node(self, slot_node: SlotNode) -> None:
        """Add a slot node to the graph."""
        self.slot_nodes.append(slot_node)
        self._slot_map[slot_node.slot_id] = slot_node
    
    def add_edge(self, edge: WeightedEdge) -> None:
        """Add an edge to the graph."""
        self.edges.append(edge)
        self._edge_map[(edge.flight_id, edge.slot_id)] = edge
    
    def get_flight_node(self, flight_id: str) -> Optional[FlightNode]:
        """Get flight node by ID."""
        return self._flight_map.get(flight_id)
    
    def get_slot_node(self, slot_id: str) -> Optional[SlotNode]:
        """Get slot node by ID."""
        return self._slot_map.get(slot_id)
    
    def get_edge(self, flight_id: str, slot_id: str) -> Optional[WeightedEdge]:
        """Get edge between flight and slot."""
        return self._edge_map.get((flight_id, slot_id))
    
    def get_feasible_edges(self) -> List[WeightedEdge]:
        """Get all feasible edges."""
        return [edge for edge in self.edges if edge.feasible]
    
    def get_flight_edges(self, flight_id: str) -> List[WeightedEdge]:
        """Get all edges for a specific flight."""
        return [edge for edge in self.edges if edge.flight_id == flight_id]
    
    def get_slot_edges(self, slot_id: str) -> List[WeightedEdge]:
        """Get all edges for a specific slot."""
        return [edge for edge in self.edges if edge.slot_id == slot_id]


@dataclass
class OptimizationResult:
    """Result of weighted graph optimization."""
    assignment: Dict[str, str]  # flight_id -> slot_id
    total_cost: float
    cost_breakdown: Dict[str, float]
    solver_status: str
    optimization_time_seconds: float
    constraint_violations: List[str] = field(default_factory=list)
    infeasible_flights: List[str] = field(default_factory=list)
    
    def get_assignment_for_flight(self, flight_id: str) -> Optional[str]:
        """Get assigned slot for a flight."""
        return self.assignment.get(flight_id)
    
    def get_flights_for_slot(self, slot_id: str) -> List[str]:
        """Get flights assigned to a slot."""
        return [flight_id for flight_id, assigned_slot in self.assignment.items() 
                if assigned_slot == slot_id]


class WeightedGraphOptimizer:
    """Weighted graph-based schedule optimization engine."""
    
    def __init__(self):
        """Initialize the optimizer."""
        self.max_optimization_time_seconds = 300  # 5 minutes max
        self.solution_time_limit_seconds = 60     # 1 minute for quick solutions
        
        # Constraint parameters
        self.min_turnaround_minutes = {
            "A320": 45, "A321": 50, "B737": 45, "B738": 45,
            "B777": 90, "B787": 75, "DEFAULT": 60
        }
        
        self.wake_separation_minutes = {
            ("HEAVY", "LIGHT"): 4.0,
            ("HEAVY", "MEDIUM"): 3.0,
            ("MEDIUM", "LIGHT"): 2.0,
            ("DEFAULT", "DEFAULT"): 2.0
        }
        
        self.curfew_hours = [23, 0, 1, 2, 3, 4, 5]
        
        # Weather capacity reduction factors
        self.weather_capacity_factors = {
            WeatherRegime.CALM: 1.0,
            WeatherRegime.MEDIUM: 0.85,
            WeatherRegime.STRONG: 0.65,
            WeatherRegime.SEVERE: 0.3
        }
    
    def build_feasibility_graph(self, flights: List[Flight], 
                               time_window_hours: int = 4,
                               slot_interval_minutes: int = 5) -> BipartiteGraph:
        """
        Build bipartite graph modeling flights ↔ runway-time slots.
        
        Args:
            flights: List of flights to optimize
            time_window_hours: Time window around original schedule (±hours)
            slot_interval_minutes: Time slot granularity in minutes
            
        Returns:
            BipartiteGraph with flights and slots
        """
        graph = BipartiteGraph()
        
        if not flights:
            return graph
        
        # Create flight nodes
        for flight in flights:
            flight_node = FlightNode(
                flight_id=flight.flight_id,
                original_slot=self._get_flight_datetime(flight),
                airline=flight.airline_code,
                aircraft_type=flight.aircraft_type
            )
            
            # Add flight-specific constraints
            if flight.aircraft_type in ["B777", "B787", "A330"]:
                flight_node.add_constraint("heavy_aircraft")
            
            graph.add_flight_node(flight_node)
        
        # Create time slots
        slot_nodes = self._create_time_slots(flights, time_window_hours, slot_interval_minutes)
        for slot_node in slot_nodes:
            graph.add_slot_node(slot_node)
        
        # Create edges with feasibility checking
        for flight_node in graph.flight_nodes:
            for slot_node in graph.slot_nodes:
                edge = self._create_edge(flight_node, slot_node, flights)
                if edge:
                    graph.add_edge(edge)
        
        # Add graph constraints
        graph.constraints = self._generate_graph_constraints(flights)
        
        return graph
    
    def calculate_edge_costs(self, flight_node: FlightNode, slot_node: SlotNode, 
                           weights: ObjectiveWeights) -> Tuple[float, Dict[str, float]]:
        """
        Calculate edge cost with runway-dependent weights.
        
        Args:
            flight_node: Flight node
            slot_node: Slot node
            weights: Objective weights
            
        Returns:
            Tuple of (total_cost, cost_breakdown)
        """
        # Get runway-adjusted weights
        runway_weights = weights.get_runway_adjusted_weights(slot_node.runway)
        
        cost_breakdown = {}
        
        # 1. Delay cost (time difference from original schedule)
        time_delta_minutes = abs((slot_node.timestamp - flight_node.original_slot).total_seconds() / 60)
        delay_cost = time_delta_minutes * runway_weights.delay_weight
        cost_breakdown["delay"] = delay_cost
        
        # 2. Taxi cost (runway-dependent)
        taxi_cost = self._calculate_taxi_cost(flight_node, slot_node) * runway_weights.taxi_weight
        cost_breakdown["taxi"] = taxi_cost
        
        # 3. Fairness cost (prefer balanced distribution)
        fairness_cost = self._calculate_fairness_cost(flight_node, slot_node) * runway_weights.fairness_weight
        cost_breakdown["fairness"] = fairness_cost
        
        # 4. CO₂ cost (environmental impact)
        co2_cost = self._calculate_co2_cost(flight_node, slot_node) * runway_weights.co2_weight
        cost_breakdown["co2"] = co2_cost
        
        # 5. Curfew cost (high penalty for curfew violations)
        curfew_cost = 0.0
        if slot_node.is_curfew:
            curfew_cost = 100.0 * runway_weights.curfew_weight
        cost_breakdown["curfew"] = curfew_cost
        
        total_cost = sum(cost_breakdown.values())
        
        return total_cost, cost_breakdown
    
    def solve_assignment(self, graph: BipartiteGraph, 
                        weights: ObjectiveWeights = None,
                        solver: str = "cp_sat") -> OptimizationResult:
        """
        Solve flight-to-slot assignment using specified solver.
        
        Args:
            graph: Bipartite graph to solve
            weights: Objective weights
            solver: Solver type ("cp_sat", "min_cost_flow", "heuristic")
            
        Returns:
            OptimizationResult with assignment and metrics
        """
        start_time = datetime.now()
        
        if weights is None:
            weights = ObjectiveWeights().normalize()
        
        # Try solvers in order of preference (avoid CP-SAT due to bus errors)
        if solver == "auto":
            solver_order = ["min_cost_flow", "heuristic"]
        elif solver == "cp_sat":
            # Force fallback to avoid bus errors
            solver_order = ["min_cost_flow", "heuristic"]
        else:
            solver_order = [solver]
        
        for solver_type in solver_order:
            try:
                if solver_type == "cp_sat" and ORTOOLS_AVAILABLE:
                    result = self._solve_with_cp_sat(graph, weights)
                elif solver_type == "min_cost_flow":
                    result = self._solve_with_min_cost_flow(graph, weights)
                else:
                    result = self._solve_with_heuristic(graph, weights)
                
                result.optimization_time_seconds = (datetime.now() - start_time).total_seconds()
                return result
                
            except Exception as e:
                print(f"Solver {solver_type} failed: {e}")
                continue
        
        # If all solvers fail, return empty result
        return OptimizationResult(
            assignment={},
            total_cost=float('inf'),
            cost_breakdown={},
            solver_status="FAILED",
            optimization_time_seconds=(datetime.now() - start_time).total_seconds()
        )
    
    def validate_runway_constraints(self, assignment: Dict[str, str], 
                                   graph: BipartiteGraph) -> List[str]:
        """
        Validate runway constraints for the assignment.
        
        Args:
            assignment: Flight-to-slot assignment
            graph: Bipartite graph
            
        Returns:
            List of constraint violations
        """
        violations = []
        
        # Check capacity constraints
        slot_demand = defaultdict(int)
        for flight_id, slot_id in assignment.items():
            slot_demand[slot_id] += 1
        
        for slot_id, demand in slot_demand.items():
            slot_node = graph.get_slot_node(slot_id)
            if slot_node and demand > slot_node.weather_adjusted_capacity:
                violations.append(
                    f"Slot {slot_id} overloaded: {demand} > {slot_node.weather_adjusted_capacity}"
                )
        
        # Check turnaround constraints
        violations.extend(self._check_turnaround_constraints(assignment, graph))
        
        # Check wake separation constraints
        violations.extend(self._check_wake_separation_constraints(assignment, graph))
        
        # Check curfew constraints
        violations.extend(self._check_curfew_constraints(assignment, graph))
        
        return violations
    
    def autonomous_optimize(self, trigger_conditions: Dict[str, Any], 
                           flights: List[Flight]) -> OptimizationResult:
        """
        Perform autonomous optimization based on trigger conditions.
        
        Args:
            trigger_conditions: Conditions that triggered optimization
            flights: Flights to optimize
            
        Returns:
            OptimizationResult with autonomous decision
        """
        # Build graph
        graph = self.build_feasibility_graph(flights)
        
        # Determine weights based on trigger conditions
        weights = self._determine_autonomous_weights(trigger_conditions)
        
        # Solve with appropriate urgency (use heuristic to avoid OR-Tools issues)
        urgency = trigger_conditions.get("urgency", "medium")
        solver = "heuristic"  # Always use heuristic for now to avoid OR-Tools bus errors
        
        result = self.solve_assignment(graph, weights, solver)
        
        # Add autonomous decision metadata
        result.cost_breakdown["autonomous_trigger"] = str(trigger_conditions)
        
        return result
    
    def what_if_analysis(self, base_schedule: List[Flight], 
                        changes: List[Tuple[str, datetime]]) -> Dict[str, Any]:
        """
        Analyze impact of proposed changes using graph optimization.
        
        Args:
            base_schedule: Current flight schedule
            changes: List of (flight_id, new_time) changes
            
        Returns:
            Impact analysis results
        """
        # Create modified schedule
        modified_flights = base_schedule.copy()
        
        for flight_id, new_time in changes:
            for flight in modified_flights:
                if flight.flight_id == flight_id:
                    # Update flight time
                    flight.departure.scheduled = new_time.time()
                    break
        
        # Build graphs for before and after
        base_graph = self.build_feasibility_graph(base_schedule)
        modified_graph = self.build_feasibility_graph(modified_flights)
        
        # Solve both scenarios using heuristic solver to avoid OR-Tools issues
        base_result = self.solve_assignment(base_graph, solver="heuristic")
        modified_result = self.solve_assignment(modified_graph, solver="heuristic")
        
        # Calculate impact metrics
        impact = {
            "cost_change": modified_result.total_cost - base_result.total_cost,
            "cost_change_percent": ((modified_result.total_cost - base_result.total_cost) / 
                                  base_result.total_cost * 100) if base_result.total_cost > 0 else 0,
            "constraint_violations_change": (len(modified_result.constraint_violations) - 
                                           len(base_result.constraint_violations)),
            "affected_flights": len(changes),
            "recommendation": "approve" if modified_result.total_cost < base_result.total_cost else "reject"
        }
        
        return impact
    
    def _create_time_slots(self, flights: List[Flight], 
                          time_window_hours: int,
                          slot_interval_minutes: int) -> List[SlotNode]:
        """Create time slots for the optimization window."""
        if not flights:
            return []
        
        # Determine time range
        flight_times = []
        for flight in flights:
            flight_time = self._get_flight_datetime(flight)
            if flight_time:
                flight_times.append(flight_time)
        
        if not flight_times:
            return []
        
        start_time = min(flight_times) - timedelta(hours=time_window_hours)
        end_time = max(flight_times) + timedelta(hours=time_window_hours)
        
        # Create slots
        slots = []
        current_time = start_time.replace(
            minute=(start_time.minute // slot_interval_minutes) * slot_interval_minutes,
            second=0, microsecond=0
        )
        
        # Available runways (simplified - could be made configurable)
        runways = ["09L", "09R", "27L", "27R"]
        
        while current_time <= end_time:
            for runway in runways:
                slot_id = f"{runway}_{current_time.strftime('%H%M')}"
                
                # Base capacity (operations per hour, scaled to slot interval)
                base_capacity = 30  # operations per hour
                slot_capacity = max(1, int(base_capacity * (slot_interval_minutes / 60)))
                
                # Weather adjustment (simplified)
                weather_regime = WeatherRegime.CALM  # Could be determined from weather data
                weather_factor = self.weather_capacity_factors.get(weather_regime, 1.0)
                weather_adjusted_capacity = max(1, int(slot_capacity * weather_factor))
                
                # Check if curfew
                is_curfew = current_time.hour in self.curfew_hours
                if is_curfew:
                    weather_adjusted_capacity = max(1, weather_adjusted_capacity // 4)
                
                slot_node = SlotNode(
                    slot_id=slot_id,
                    runway=runway,
                    timestamp=current_time,
                    capacity=slot_capacity,
                    weather_adjusted_capacity=weather_adjusted_capacity,
                    is_curfew=is_curfew,
                    weather_regime=weather_regime
                )
                
                slots.append(slot_node)
            
            current_time += timedelta(minutes=slot_interval_minutes)
        
        return slots
    
    def _create_edge(self, flight_node: FlightNode, slot_node: SlotNode, 
                    flights: List[Flight]) -> Optional[WeightedEdge]:
        """Create edge between flight and slot with feasibility checking."""
        # Basic feasibility check - don't allow assignments too far from original
        time_delta_hours = abs((slot_node.timestamp - flight_node.original_slot).total_seconds() / 3600)
        if time_delta_hours > 6:  # Max 6 hours deviation
            return None
        
        # Calculate costs
        weights = ObjectiveWeights().normalize()
        total_cost, cost_breakdown = self.calculate_edge_costs(flight_node, slot_node, weights)
        
        # Create edge
        edge = WeightedEdge(
            flight_id=flight_node.flight_id,
            slot_id=slot_node.slot_id,
            cost=total_cost,
            cost_breakdown=cost_breakdown,
            feasible=True
        )
        
        # Check constraints
        violations = self._check_edge_constraints(flight_node, slot_node, flights)
        for violation in violations:
            edge.add_violation(violation)
        
        return edge
    
    def _check_edge_constraints(self, flight_node: FlightNode, slot_node: SlotNode,
                               flights: List[Flight]) -> List[str]:
        """Check constraints for a specific flight-slot assignment."""
        violations = []
        
        # Curfew constraint
        if slot_node.is_curfew and "heavy_aircraft" in flight_node.constraints:
            violations.append("Heavy aircraft not allowed during curfew")
        
        # Runway suitability (simplified)
        if flight_node.aircraft_type in ["B777", "B787"] and slot_node.runway.endswith("R"):
            # Prefer left runways for heavy aircraft (simplified rule)
            pass  # This is just a preference, not a hard constraint
        
        return violations
    
    def _calculate_taxi_cost(self, flight_node: FlightNode, slot_node: SlotNode) -> float:
        """Calculate taxi cost based on runway assignment."""
        # Simplified taxi cost model
        base_taxi_minutes = 8.0  # Base taxi time
        
        # Runway-specific adjustments
        runway_taxi_factors = {
            "09L": 1.0,
            "09R": 1.3,  # Longer taxi
            "27L": 0.8,  # Shorter taxi
            "27R": 1.1
        }
        
        taxi_factor = runway_taxi_factors.get(slot_node.runway, 1.0)
        taxi_minutes = base_taxi_minutes * taxi_factor
        
        # Cost per minute of taxi time
        return taxi_minutes * 0.5
    
    def _calculate_fairness_cost(self, flight_node: FlightNode, slot_node: SlotNode) -> float:
        """Calculate fairness cost to ensure balanced treatment."""
        # Prefer to minimize large deviations from original schedule
        time_delta_minutes = abs((slot_node.timestamp - flight_node.original_slot).total_seconds() / 60)
        
        # Exponential penalty for large deviations
        if time_delta_minutes > 60:
            return (time_delta_minutes - 60) * 0.1
        
        return 0.0
    
    def _calculate_co2_cost(self, flight_node: FlightNode, slot_node: SlotNode) -> float:
        """Calculate CO₂ impact cost."""
        # Simplified CO₂ model based on taxi time and delay
        taxi_cost = self._calculate_taxi_cost(flight_node, slot_node)
        
        # Additional CO₂ for delays (engines running longer)
        time_delta_minutes = abs((slot_node.timestamp - flight_node.original_slot).total_seconds() / 60)
        delay_co2 = time_delta_minutes * 0.2 if time_delta_minutes > 0 else 0
        
        return taxi_cost * 0.3 + delay_co2  # Convert to CO₂ cost units
    
    def _get_flight_datetime(self, flight: Flight) -> Optional[datetime]:
        """Get flight datetime from Flight object."""
        if not flight.departure.scheduled:
            return None
        
        flight_date = flight.flight_date or date.today()
        return datetime.combine(flight_date, flight.departure.scheduled)
    
    def _generate_graph_constraints(self, flights: List[Flight]) -> List[GraphConstraint]:
        """Generate graph-level constraints."""
        constraints = []
        
        # Turnaround constraints
        turnaround_flights = self._find_turnaround_pairs(flights)
        for i, (arrival_flight, departure_flight) in enumerate(turnaround_flights):
            constraint = GraphConstraint(
                constraint_id=f"turnaround_{i}",
                constraint_type="turnaround",
                affected_flights=[arrival_flight.flight_id, departure_flight.flight_id],
                description=f"Minimum turnaround time between {arrival_flight.flight_id} and {departure_flight.flight_id}",
                severity="high"
            )
            constraints.append(constraint)
        
        return constraints
    
    def _find_turnaround_pairs(self, flights: List[Flight]) -> List[Tuple[Flight, Flight]]:
        """Find flight pairs that represent turnarounds (same aircraft)."""
        # Simplified - group by aircraft registration if available
        aircraft_flights = defaultdict(list)
        
        for flight in flights:
            if flight.aircraft_registration:
                aircraft_flights[flight.aircraft_registration].append(flight)
        
        turnaround_pairs = []
        for aircraft, aircraft_flight_list in aircraft_flights.items():
            # Sort by time and find consecutive arrival-departure pairs
            sorted_flights = sorted(aircraft_flight_list, 
                                  key=lambda f: self._get_flight_datetime(f) or datetime.min)
            
            for i in range(len(sorted_flights) - 1):
                current = sorted_flights[i]
                next_flight = sorted_flights[i + 1]
                
                # Check if this could be a turnaround (arrival followed by departure)
                if (current.destination and next_flight.origin and 
                    current.destination.code == next_flight.origin.code):
                    turnaround_pairs.append((current, next_flight))
        
        return turnaround_pairs
    
    def _determine_autonomous_weights(self, trigger_conditions: Dict[str, Any]) -> ObjectiveWeights:
        """Determine optimization weights based on autonomous trigger conditions."""
        weights = ObjectiveWeights()
        
        # Adjust weights based on trigger type
        trigger_type = trigger_conditions.get("type", "general")
        
        if trigger_type == "capacity_overload":
            # Prioritize delay reduction and fairness
            weights.delay_weight = 2.0
            weights.fairness_weight = 1.5
            weights.taxi_weight = 0.2
            weights.co2_weight = 0.1
            
        elif trigger_type == "weather_degradation":
            # Prioritize safety and capacity
            weights.delay_weight = 1.5
            weights.curfew_weight = 3.0
            weights.taxi_weight = 0.5
            
        elif trigger_type == "curfew_violation":
            # Heavily penalize curfew violations
            weights.curfew_weight = 5.0
            weights.delay_weight = 1.0
            
        return weights.normalize()
    
    def _solve_with_cp_sat(self, graph: BipartiteGraph, 
                          weights: ObjectiveWeights) -> OptimizationResult:
        """Solve using CP-SAT solver."""
        if not ORTOOLS_AVAILABLE:
            raise Exception("OR-Tools not available")
        
        # Import OR-Tools only when needed
        try:
            from ortools.sat.python import cp_model
        except ImportError:
            raise Exception("OR-Tools import failed")
        
        model = cp_model.CpModel()
        
        # Decision variables: flight_id -> slot_id assignment
        assignment_vars = {}
        
        for flight_node in graph.flight_nodes:
            assignment_vars[flight_node.flight_id] = {}
            flight_edges = graph.get_flight_edges(flight_node.flight_id)
            
            for edge in flight_edges:
                if edge.feasible:
                    var_name = f"assign_{flight_node.flight_id}_{edge.slot_id}"
                    assignment_vars[flight_node.flight_id][edge.slot_id] = model.NewBoolVar(var_name)
        
        # Constraint: Each flight assigned to exactly one slot
        for flight_node in graph.flight_nodes:
            if flight_node.flight_id in assignment_vars:
                model.Add(sum(assignment_vars[flight_node.flight_id].values()) == 1)
        
        # Constraint: Slot capacity limits
        for slot_node in graph.slot_nodes:
            slot_assignments = []
            for flight_id, slot_vars in assignment_vars.items():
                if slot_node.slot_id in slot_vars:
                    slot_assignments.append(slot_vars[slot_node.slot_id])
            
            if slot_assignments:
                model.Add(sum(slot_assignments) <= slot_node.weather_adjusted_capacity)
        
        # Objective: Minimize total cost
        objective_terms = []
        
        for flight_id, slot_vars in assignment_vars.items():
            for slot_id, var in slot_vars.items():
                edge = graph.get_edge(flight_id, slot_id)
                if edge and edge.feasible:
                    # Scale cost for integer optimization
                    scaled_cost = int(edge.cost * 100)
                    objective_terms.append(scaled_cost * var)
        
        if objective_terms:
            model.Minimize(sum(objective_terms))
        
        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.solution_time_limit_seconds
        status = solver.Solve(model)
        
        # Extract solution
        assignment = {}
        total_cost = 0.0
        cost_breakdown = defaultdict(float)
        
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            for flight_id, slot_vars in assignment_vars.items():
                for slot_id, var in slot_vars.items():
                    if solver.Value(var) == 1:
                        assignment[flight_id] = slot_id
                        edge = graph.get_edge(flight_id, slot_id)
                        if edge:
                            total_cost += edge.cost
                            for cost_type, cost_value in edge.cost_breakdown.items():
                                cost_breakdown[cost_type] += cost_value
        
        solver_status = "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE" if status == cp_model.FEASIBLE else "FAILED"
        
        return OptimizationResult(
            assignment=assignment,
            total_cost=total_cost,
            cost_breakdown=dict(cost_breakdown),
            solver_status=solver_status,
            optimization_time_seconds=0  # Will be set by caller
        )
    
    def _solve_with_min_cost_flow(self, graph: BipartiteGraph, 
                                 weights: ObjectiveWeights) -> OptimizationResult:
        """Solve using min-cost flow algorithm (simplified implementation)."""
        # This is a simplified greedy approach since we don't have networkx
        assignment = {}
        total_cost = 0.0
        cost_breakdown = defaultdict(float)
        
        # Sort flights by priority (could be based on delay, airline, etc.)
        sorted_flights = sorted(graph.flight_nodes, 
                              key=lambda f: f.priority_score, reverse=True)
        
        # Track slot utilization
        slot_utilization = {slot.slot_id: 0 for slot in graph.slot_nodes}
        
        # Assign flights to best available slots
        for flight_node in sorted_flights:
            best_edge = None
            best_cost = float('inf')
            
            flight_edges = graph.get_flight_edges(flight_node.flight_id)
            feasible_edges = [e for e in flight_edges if e.feasible]
            
            for edge in feasible_edges:
                slot_node = graph.get_slot_node(edge.slot_id)
                if (slot_node and 
                    slot_utilization[edge.slot_id] < slot_node.weather_adjusted_capacity and
                    edge.cost < best_cost):
                    best_edge = edge
                    best_cost = edge.cost
            
            if best_edge:
                assignment[flight_node.flight_id] = best_edge.slot_id
                slot_utilization[best_edge.slot_id] += 1
                total_cost += best_edge.cost
                
                for cost_type, cost_value in best_edge.cost_breakdown.items():
                    cost_breakdown[cost_type] += cost_value
        
        return OptimizationResult(
            assignment=assignment,
            total_cost=total_cost,
            cost_breakdown=dict(cost_breakdown),
            solver_status="FEASIBLE",
            optimization_time_seconds=0  # Will be set by caller
        )
    
    def _solve_with_heuristic(self, graph: BipartiteGraph, 
                             weights: ObjectiveWeights) -> OptimizationResult:
        """Solve using heuristic approach for fast solutions."""
        assignment = {}
        total_cost = 0.0
        cost_breakdown = defaultdict(float)
        
        # Simple greedy assignment with local optimization
        slot_utilization = {slot.slot_id: 0 for slot in graph.slot_nodes}
        
        # Sort flights by urgency (time deviation from original)
        def flight_urgency(flight_node):
            min_deviation = float('inf')
            for edge in graph.get_flight_edges(flight_node.flight_id):
                if edge.feasible:
                    slot_node = graph.get_slot_node(edge.slot_id)
                    if slot_node:
                        deviation = abs((slot_node.timestamp - flight_node.original_slot).total_seconds())
                        min_deviation = min(min_deviation, deviation)
            return min_deviation
        
        sorted_flights = sorted(graph.flight_nodes, key=flight_urgency)
        
        # Assign each flight to its best available slot
        for flight_node in sorted_flights:
            best_edge = None
            best_score = float('inf')
            
            for edge in graph.get_flight_edges(flight_node.flight_id):
                if not edge.feasible:
                    continue
                
                slot_node = graph.get_slot_node(edge.slot_id)
                if not slot_node or slot_utilization[edge.slot_id] >= slot_node.weather_adjusted_capacity:
                    continue
                
                # Score combines cost and utilization preference
                utilization_penalty = slot_utilization[edge.slot_id] * 10
                score = edge.cost + utilization_penalty
                
                if score < best_score:
                    best_edge = edge
                    best_score = score
            
            if best_edge:
                assignment[flight_node.flight_id] = best_edge.slot_id
                slot_utilization[best_edge.slot_id] += 1
                total_cost += best_edge.cost
                
                for cost_type, cost_value in best_edge.cost_breakdown.items():
                    cost_breakdown[cost_type] += cost_value
        
        return OptimizationResult(
            assignment=assignment,
            total_cost=total_cost,
            cost_breakdown=dict(cost_breakdown),
            solver_status="HEURISTIC",
            optimization_time_seconds=0  # Will be set by caller
        )
    
    def _check_turnaround_constraints(self, assignment: Dict[str, str], 
                                     graph: BipartiteGraph) -> List[str]:
        """Check turnaround time constraints."""
        violations = []
        
        # Find turnaround constraints
        turnaround_constraints = [c for c in graph.constraints if c.constraint_type == "turnaround"]
        
        for constraint in turnaround_constraints:
            if len(constraint.affected_flights) >= 2:
                arrival_flight_id = constraint.affected_flights[0]
                departure_flight_id = constraint.affected_flights[1]
                
                arrival_slot_id = assignment.get(arrival_flight_id)
                departure_slot_id = assignment.get(departure_flight_id)
                
                if arrival_slot_id and departure_slot_id:
                    arrival_slot = graph.get_slot_node(arrival_slot_id)
                    departure_slot = graph.get_slot_node(departure_slot_id)
                    
                    if arrival_slot and departure_slot:
                        turnaround_minutes = (departure_slot.timestamp - arrival_slot.timestamp).total_seconds() / 60
                        
                        # Get flight node to determine aircraft type
                        departure_flight_node = graph.get_flight_node(departure_flight_id)
                        min_turnaround = self.min_turnaround_minutes.get(
                            departure_flight_node.aircraft_type if departure_flight_node else "DEFAULT",
                            self.min_turnaround_minutes["DEFAULT"]
                        )
                        
                        if turnaround_minutes < min_turnaround:
                            violations.append(
                                f"Turnaround violation: {turnaround_minutes:.1f}min < {min_turnaround}min for {departure_flight_id}"
                            )
        
        return violations
    
    def _check_wake_separation_constraints(self, assignment: Dict[str, str], 
                                          graph: BipartiteGraph) -> List[str]:
        """Check wake turbulence separation constraints."""
        violations = []
        
        # Group assignments by runway and sort by time
        runway_assignments = defaultdict(list)
        
        for flight_id, slot_id in assignment.items():
            slot_node = graph.get_slot_node(slot_id)
            flight_node = graph.get_flight_node(flight_id)
            
            if slot_node and flight_node:
                runway_assignments[slot_node.runway].append((slot_node.timestamp, flight_node))
        
        # Check separation for each runway
        for runway, assignments in runway_assignments.items():
            sorted_assignments = sorted(assignments, key=lambda x: x[0])
            
            for i in range(len(sorted_assignments) - 1):
                current_time, current_flight = sorted_assignments[i]
                next_time, next_flight = sorted_assignments[i + 1]
                
                separation_minutes = (next_time - current_time).total_seconds() / 60
                
                # Get wake categories
                current_wake = self._get_wake_category(current_flight.aircraft_type)
                next_wake = self._get_wake_category(next_flight.aircraft_type)
                
                required_separation = self.wake_separation_minutes.get(
                    (current_wake, next_wake),
                    self.wake_separation_minutes[("DEFAULT", "DEFAULT")]
                )
                
                if separation_minutes < required_separation:
                    violations.append(
                        f"Wake separation violation: {separation_minutes:.1f}min < {required_separation}min between {current_flight.flight_id} and {next_flight.flight_id}"
                    )
        
        return violations
    
    def _check_curfew_constraints(self, assignment: Dict[str, str], 
                                 graph: BipartiteGraph) -> List[str]:
        """Check curfew constraints."""
        violations = []
        
        for flight_id, slot_id in assignment.items():
            slot_node = graph.get_slot_node(slot_id)
            flight_node = graph.get_flight_node(flight_id)
            
            if slot_node and flight_node and slot_node.is_curfew:
                # Check if this is a restricted operation during curfew
                if "heavy_aircraft" in flight_node.constraints:
                    violations.append(
                        f"Curfew violation: Heavy aircraft {flight_id} scheduled during curfew at {slot_node.timestamp}"
                    )
        
        return violations
    
    def _get_wake_category(self, aircraft_type: str) -> str:
        """Get wake turbulence category for aircraft type."""
        if not aircraft_type:
            return "DEFAULT"
        
        aircraft_type = aircraft_type.upper()
        
        if any(heavy in aircraft_type for heavy in ["B777", "B787", "A330", "A340", "B747"]):
            return "HEAVY"
        elif any(light in aircraft_type for light in ["ATR", "DHC", "CRJ"]):
            return "LIGHT"
        else:
            return "MEDIUM"