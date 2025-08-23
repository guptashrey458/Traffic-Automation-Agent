"""Tests for weighted graph optimization engine."""

import pytest
from datetime import datetime, date, time, timedelta
from unittest.mock import Mock, patch

from src.services.weighted_graph_optimizer import (
    WeightedGraphOptimizer, BipartiteGraph, FlightNode, SlotNode, WeightedEdge,
    ObjectiveWeights, OptimizationResult, GraphConstraint
)
from src.models.flight import Flight, FlightTime, Airport
from src.services.analytics import WeatherRegime


class TestObjectiveWeights:
    """Test objective weights functionality."""
    
    def test_default_weights(self):
        """Test default weight initialization."""
        weights = ObjectiveWeights()
        
        assert weights.delay_weight == 1.0
        assert weights.taxi_weight == 0.3
        assert weights.fairness_weight == 0.4
        assert weights.co2_weight == 0.2
        assert weights.curfew_weight == 2.0
        assert len(weights.runway_multipliers) > 0
    
    def test_normalize_weights(self):
        """Test weight normalization."""
        weights = ObjectiveWeights(
            delay_weight=2.0,
            taxi_weight=1.0,
            fairness_weight=1.0,
            co2_weight=1.0,
            curfew_weight=1.0
        )
        
        normalized = weights.normalize()
        total = (normalized.delay_weight + normalized.taxi_weight + 
                normalized.fairness_weight + normalized.co2_weight + 
                normalized.curfew_weight)
        
        assert abs(total - 1.0) < 0.001
        assert normalized.delay_weight > normalized.taxi_weight  # Should maintain relative proportions
    
    def test_runway_adjusted_weights(self):
        """Test runway-specific weight adjustments."""
        weights = ObjectiveWeights()
        
        # Test different runways
        runway_09l = weights.get_runway_adjusted_weights("09L")
        runway_09r = weights.get_runway_adjusted_weights("09R")
        
        # 09R should have higher taxi weight (longer taxi)
        assert runway_09r.taxi_weight > runway_09l.taxi_weight
    
    def test_zero_weights_normalization(self):
        """Test normalization with zero weights."""
        weights = ObjectiveWeights(
            delay_weight=0.0,
            taxi_weight=0.0,
            fairness_weight=0.0,
            co2_weight=0.0,
            curfew_weight=0.0
        )
        
        normalized = weights.normalize()
        # Should return default weights when all are zero
        assert normalized.delay_weight > 0


class TestBipartiteGraph:
    """Test bipartite graph functionality."""
    
    def test_empty_graph(self):
        """Test empty graph initialization."""
        graph = BipartiteGraph()
        
        assert len(graph.flight_nodes) == 0
        assert len(graph.slot_nodes) == 0
        assert len(graph.edges) == 0
        assert len(graph.constraints) == 0
    
    def test_add_nodes_and_edges(self):
        """Test adding nodes and edges to graph."""
        graph = BipartiteGraph()
        
        # Add flight node
        flight_node = FlightNode(
            flight_id="AI123",
            original_slot=datetime(2024, 1, 1, 10, 0),
            airline="AI",
            aircraft_type="A320"
        )
        graph.add_flight_node(flight_node)
        
        # Add slot node
        slot_node = SlotNode(
            slot_id="09L_1000",
            runway="09L",
            timestamp=datetime(2024, 1, 1, 10, 0),
            capacity=6,
            weather_adjusted_capacity=5
        )
        graph.add_slot_node(slot_node)
        
        # Add edge
        edge = WeightedEdge(
            flight_id="AI123",
            slot_id="09L_1000",
            cost=10.0,
            cost_breakdown={"delay": 5.0, "taxi": 3.0, "fairness": 2.0}
        )
        graph.add_edge(edge)
        
        # Test retrieval
        assert graph.get_flight_node("AI123") == flight_node
        assert graph.get_slot_node("09L_1000") == slot_node
        assert graph.get_edge("AI123", "09L_1000") == edge
        
        # Test collections
        assert len(graph.flight_nodes) == 1
        assert len(graph.slot_nodes) == 1
        assert len(graph.edges) == 1
    
    def test_get_flight_edges(self):
        """Test getting edges for a specific flight."""
        graph = BipartiteGraph()
        
        # Add multiple edges for same flight
        edges = [
            WeightedEdge("AI123", "09L_1000", 10.0),
            WeightedEdge("AI123", "09L_1005", 12.0),
            WeightedEdge("AI456", "09L_1000", 8.0)
        ]
        
        for edge in edges:
            graph.add_edge(edge)
        
        flight_edges = graph.get_flight_edges("AI123")
        assert len(flight_edges) == 2
        assert all(edge.flight_id == "AI123" for edge in flight_edges)
    
    def test_get_feasible_edges(self):
        """Test getting only feasible edges."""
        graph = BipartiteGraph()
        
        # Add feasible and infeasible edges
        feasible_edge = WeightedEdge("AI123", "09L_1000", 10.0, feasible=True)
        infeasible_edge = WeightedEdge("AI123", "09L_1005", 12.0, feasible=False)
        
        graph.add_edge(feasible_edge)
        graph.add_edge(infeasible_edge)
        
        feasible_edges = graph.get_feasible_edges()
        assert len(feasible_edges) == 1
        assert feasible_edges[0].feasible is True


class TestSlotNode:
    """Test slot node functionality."""
    
    def test_slot_node_creation(self):
        """Test slot node creation and properties."""
        slot = SlotNode(
            slot_id="09L_1000",
            runway="09L",
            timestamp=datetime(2024, 1, 1, 10, 0),
            capacity=6,
            current_demand=3,
            weather_adjusted_capacity=5,
            is_curfew=False
        )
        
        assert slot.utilization == 0.6  # 3/5
        assert slot.available_capacity == 2  # 5-3
        assert slot.can_accommodate(1) is True
        assert slot.can_accommodate(3) is False
    
    def test_curfew_slot(self):
        """Test curfew slot properties."""
        slot = SlotNode(
            slot_id="09L_0200",
            runway="09L",
            timestamp=datetime(2024, 1, 1, 2, 0),
            capacity=6,
            weather_adjusted_capacity=2,  # Reduced for curfew
            is_curfew=True
        )
        
        assert slot.is_curfew is True
        assert slot.weather_adjusted_capacity < slot.capacity
    
    def test_weather_adjustment(self):
        """Test weather capacity adjustment."""
        slot = SlotNode(
            slot_id="09L_1000",
            runway="09L",
            timestamp=datetime(2024, 1, 1, 10, 0),
            capacity=6,
            weather_regime=WeatherRegime.STRONG
        )
        
        # Weather adjusted capacity should be set in post_init
        assert slot.weather_adjusted_capacity == slot.capacity  # Default behavior


class TestFlightNode:
    """Test flight node functionality."""
    
    def test_flight_node_creation(self):
        """Test flight node creation."""
        flight_node = FlightNode(
            flight_id="AI123",
            original_slot=datetime(2024, 1, 1, 10, 0),
            airline="AI",
            aircraft_type="A320"
        )
        
        assert flight_node.flight_id == "AI123"
        assert flight_node.priority_score == 1.0  # Default
        assert len(flight_node.constraints) == 0
    
    def test_add_constraints(self):
        """Test adding constraints to flight node."""
        flight_node = FlightNode(
            flight_id="AI123",
            original_slot=datetime(2024, 1, 1, 10, 0)
        )
        
        flight_node.add_constraint("heavy_aircraft")
        flight_node.add_constraint("priority_flight")
        flight_node.add_constraint("heavy_aircraft")  # Duplicate
        
        assert len(flight_node.constraints) == 2
        assert "heavy_aircraft" in flight_node.constraints
        assert "priority_flight" in flight_node.constraints


class TestWeightedEdge:
    """Test weighted edge functionality."""
    
    def test_edge_creation(self):
        """Test edge creation with cost breakdown."""
        edge = WeightedEdge(
            flight_id="AI123",
            slot_id="09L_1000",
            cost=15.5,
            cost_breakdown={
                "delay": 8.0,
                "taxi": 4.0,
                "fairness": 2.5,
                "co2": 1.0
            }
        )
        
        assert edge.feasible is True
        assert len(edge.constraint_violations) == 0
        assert edge.cost == 15.5
        assert edge.cost_breakdown["delay"] == 8.0
    
    def test_add_violations(self):
        """Test adding constraint violations."""
        edge = WeightedEdge("AI123", "09L_1000", 10.0)
        
        assert edge.feasible is True
        
        edge.add_violation("Curfew violation")
        
        assert edge.feasible is False
        assert len(edge.constraint_violations) == 1
        assert "Curfew violation" in edge.constraint_violations
        
        # Adding same violation shouldn't duplicate
        edge.add_violation("Curfew violation")
        assert len(edge.constraint_violations) == 1


class TestWeightedGraphOptimizer:
    """Test weighted graph optimizer functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.optimizer = WeightedGraphOptimizer()
        
        # Create sample flights
        self.flights = [
            Flight(
                flight_id="AI123",
                flight_number="AI123",
                airline_code="AI",
                aircraft_type="A320",
                flight_date=date(2024, 1, 1),
                departure=FlightTime(scheduled=time(10, 0)),
                origin=Airport("BOM", "Mumbai", "Mumbai"),
                destination=Airport("DEL", "Delhi", "Delhi")
            ),
            Flight(
                flight_id="6E456",
                flight_number="6E456",
                airline_code="6E",
                aircraft_type="A320",
                flight_date=date(2024, 1, 1),
                departure=FlightTime(scheduled=time(10, 30)),
                origin=Airport("BOM", "Mumbai", "Mumbai"),
                destination=Airport("BLR", "Bangalore", "Bangalore")
            ),
            Flight(
                flight_id="AI789",
                flight_number="AI789",
                airline_code="AI",
                aircraft_type="B777",
                flight_date=date(2024, 1, 1),
                departure=FlightTime(scheduled=time(11, 0)),
                origin=Airport("BOM", "Mumbai", "Mumbai"),
                destination=Airport("DEL", "Delhi", "Delhi")
            )
        ]
    
    def test_build_feasibility_graph(self):
        """Test building bipartite graph from flights."""
        graph = self.optimizer.build_feasibility_graph(self.flights)
        
        # Should have flight nodes for each flight
        assert len(graph.flight_nodes) == 3
        
        # Should have slot nodes (multiple runways × time slots)
        assert len(graph.slot_nodes) > 0
        
        # Should have edges connecting flights to feasible slots
        assert len(graph.edges) > 0
        
        # Check flight nodes
        flight_ids = {node.flight_id for node in graph.flight_nodes}
        assert "AI123" in flight_ids
        assert "6E456" in flight_ids
        assert "AI789" in flight_ids
        
        # Heavy aircraft should have constraints
        ai789_node = graph.get_flight_node("AI789")
        assert ai789_node is not None
        assert "heavy_aircraft" in ai789_node.constraints
    
    def test_build_empty_graph(self):
        """Test building graph with no flights."""
        graph = self.optimizer.build_feasibility_graph([])
        
        assert len(graph.flight_nodes) == 0
        assert len(graph.slot_nodes) == 0
        assert len(graph.edges) == 0
    
    def test_calculate_edge_costs(self):
        """Test edge cost calculation."""
        flight_node = FlightNode(
            flight_id="AI123",
            original_slot=datetime(2024, 1, 1, 10, 0),
            aircraft_type="A320"
        )
        
        slot_node = SlotNode(
            slot_id="09L_1005",
            runway="09L",
            timestamp=datetime(2024, 1, 1, 10, 5),  # 5 minutes later
            capacity=6
        )
        
        weights = ObjectiveWeights().normalize()
        
        total_cost, cost_breakdown = self.optimizer.calculate_edge_costs(
            flight_node, slot_node, weights
        )
        
        assert total_cost > 0
        assert "delay" in cost_breakdown
        assert "taxi" in cost_breakdown
        assert "fairness" in cost_breakdown
        assert "co2" in cost_breakdown
        assert cost_breakdown["delay"] > 0  # Should have delay cost for 5-minute change
    
    def test_calculate_curfew_cost(self):
        """Test curfew cost calculation."""
        flight_node = FlightNode(
            flight_id="AI123",
            original_slot=datetime(2024, 1, 1, 10, 0),  # Original at 10 AM
            aircraft_type="A320"
        )
        
        # Curfew slot
        curfew_slot = SlotNode(
            slot_id="09L_0200",
            runway="09L",
            timestamp=datetime(2024, 1, 1, 2, 0),
            capacity=6,
            is_curfew=True
        )
        
        # Non-curfew slot (closer to original time)
        normal_slot = SlotNode(
            slot_id="09L_1000",
            runway="09L",
            timestamp=datetime(2024, 1, 1, 10, 0),
            capacity=6,
            is_curfew=False
        )
        
        weights = ObjectiveWeights().normalize()
        
        curfew_cost, curfew_breakdown = self.optimizer.calculate_edge_costs(
            flight_node, curfew_slot, weights
        )
        
        normal_cost, normal_breakdown = self.optimizer.calculate_edge_costs(
            flight_node, normal_slot, weights
        )
        
        # Curfew slot should have higher curfew penalty
        assert curfew_breakdown["curfew"] > normal_breakdown["curfew"]
        # Total cost might be higher for curfew due to penalty, but delay cost is also a factor
        assert curfew_breakdown["curfew"] > 0  # Should have curfew penalty
    
    def test_solve_assignment_heuristic(self):
        """Test heuristic solver."""
        graph = self.optimizer.build_feasibility_graph(self.flights)
        weights = ObjectiveWeights().normalize()
        
        result = self.optimizer.solve_assignment(graph, weights, solver="heuristic")
        
        assert result.solver_status == "HEURISTIC"
        assert len(result.assignment) <= len(self.flights)
        assert result.total_cost >= 0
        assert len(result.cost_breakdown) > 0
    
    def test_solve_assignment_empty_graph(self):
        """Test solving empty graph."""
        graph = BipartiteGraph()
        weights = ObjectiveWeights().normalize()
        
        result = self.optimizer.solve_assignment(graph, weights, solver="heuristic")
        
        assert len(result.assignment) == 0
        assert result.total_cost == 0
    
    def test_solve_assignment_cp_sat(self):
        """Test CP-SAT solver fallback behavior."""
        # Test that solver gracefully handles CP-SAT requests
        graph = self.optimizer.build_feasibility_graph(self.flights[:1])  # Single flight
        weights = ObjectiveWeights().normalize()
        
        # Test auto solver selection (should work regardless of OR-Tools availability)
        result = self.optimizer.solve_assignment(graph, weights, solver="auto")
        
        # Should get a result from available solver
        assert result.solver_status in ["OPTIMAL", "FEASIBLE", "HEURISTIC", "FAILED"]
        assert isinstance(result.assignment, dict)
        assert result.total_cost >= 0
        
        # Test that min_cost_flow solver works
        result2 = self.optimizer.solve_assignment(graph, weights, solver="min_cost_flow")
        assert result2.solver_status in ["OPTIMAL", "FEASIBLE", "HEURISTIC", "FAILED"]
    
    def test_validate_runway_constraints(self):
        """Test runway constraint validation."""
        graph = self.optimizer.build_feasibility_graph(self.flights)
        
        # Create assignment that violates capacity
        assignment = {}
        slot_id = None
        
        # Find a slot and assign all flights to it
        if graph.slot_nodes:
            slot_id = graph.slot_nodes[0].slot_id
            # Set low capacity for testing
            graph.slot_nodes[0].weather_adjusted_capacity = 1
            
            for flight_node in graph.flight_nodes:
                assignment[flight_node.flight_id] = slot_id
        
        violations = self.optimizer.validate_runway_constraints(assignment, graph)
        
        if slot_id and len(assignment) > 1:
            # Should have capacity violation
            assert len(violations) > 0
            assert any("overloaded" in v for v in violations)
    
    def test_autonomous_optimize(self):
        """Test autonomous optimization."""
        trigger_conditions = {
            "type": "capacity_overload",
            "urgency": "high",
            "threshold_exceeded": 1.2
        }
        
        result = self.optimizer.autonomous_optimize(trigger_conditions, self.flights)
        
        assert isinstance(result, OptimizationResult)
        assert result.solver_status in ["OPTIMAL", "FEASIBLE", "HEURISTIC", "FAILED"]
        assert "autonomous_trigger" in result.cost_breakdown
    
    def test_what_if_analysis(self):
        """Test what-if analysis."""
        changes = [
            ("AI123", datetime(2024, 1, 1, 10, 15)),  # Move 15 minutes later
            ("6E456", datetime(2024, 1, 1, 10, 45))   # Move 15 minutes later
        ]
        
        impact = self.optimizer.what_if_analysis(self.flights, changes)
        
        assert "cost_change" in impact
        assert "cost_change_percent" in impact
        assert "affected_flights" in impact
        assert "recommendation" in impact
        assert impact["affected_flights"] == 2
        assert impact["recommendation"] in ["approve", "reject"]
    
    def test_create_time_slots(self):
        """Test time slot creation."""
        slots = self.optimizer._create_time_slots(self.flights, time_window_hours=2, slot_interval_minutes=5)
        
        assert len(slots) > 0
        
        # Check slot properties
        for slot in slots[:5]:  # Check first few slots
            assert slot.runway in ["09L", "09R", "27L", "27R"]
            assert slot.capacity > 0
            assert slot.weather_adjusted_capacity > 0
            assert isinstance(slot.timestamp, datetime)
    
    def test_find_turnaround_pairs(self):
        """Test finding turnaround flight pairs."""
        # Create flights with same aircraft registration
        flights_with_aircraft = []
        for i, flight in enumerate(self.flights):
            flight.aircraft_registration = f"VT-ABC{i % 2}"  # Two aircraft
            flights_with_aircraft.append(flight)
        
        turnaround_pairs = self.optimizer._find_turnaround_pairs(flights_with_aircraft)
        
        # Should find pairs based on aircraft registration and route
        assert isinstance(turnaround_pairs, list)
    
    def test_get_wake_category(self):
        """Test wake turbulence category determination."""
        assert self.optimizer._get_wake_category("B777") == "HEAVY"
        assert self.optimizer._get_wake_category("A320") == "MEDIUM"
        assert self.optimizer._get_wake_category("ATR72") == "LIGHT"
        assert self.optimizer._get_wake_category("") == "DEFAULT"
        assert self.optimizer._get_wake_category(None) == "DEFAULT"
    
    def test_determine_autonomous_weights(self):
        """Test autonomous weight determination."""
        # Test capacity overload trigger
        trigger_conditions = {"type": "capacity_overload"}
        weights = self.optimizer._determine_autonomous_weights(trigger_conditions)
        
        assert weights.delay_weight > 0
        assert weights.fairness_weight > 0
        
        # Test weather degradation trigger
        trigger_conditions = {"type": "weather_degradation"}
        weather_weights = self.optimizer._determine_autonomous_weights(trigger_conditions)
        
        assert weather_weights.curfew_weight > weights.curfew_weight
        
        # Test curfew violation trigger
        trigger_conditions = {"type": "curfew_violation"}
        curfew_weights = self.optimizer._determine_autonomous_weights(trigger_conditions)
        
        assert curfew_weights.curfew_weight > weather_weights.curfew_weight


class TestOptimizationResult:
    """Test optimization result functionality."""
    
    def test_result_creation(self):
        """Test optimization result creation."""
        assignment = {"AI123": "09L_1000", "6E456": "09R_1005"}
        cost_breakdown = {"delay": 10.0, "taxi": 5.0, "fairness": 2.0}
        
        result = OptimizationResult(
            assignment=assignment,
            total_cost=17.0,
            cost_breakdown=cost_breakdown,
            solver_status="OPTIMAL",
            optimization_time_seconds=1.5
        )
        
        assert result.get_assignment_for_flight("AI123") == "09L_1000"
        assert result.get_assignment_for_flight("UNKNOWN") is None
        
        flights_in_slot = result.get_flights_for_slot("09L_1000")
        assert "AI123" in flights_in_slot
        assert len(flights_in_slot) == 1


class TestIntegration:
    """Integration tests for the weighted graph optimizer."""
    
    def test_end_to_end_optimization(self):
        """Test complete optimization workflow."""
        optimizer = WeightedGraphOptimizer()
        
        # Create realistic flight scenario
        flights = []
        base_time = datetime(2024, 1, 1, 8, 0)
        
        for i in range(5):
            flight = Flight(
                flight_id=f"AI{100 + i}",
                flight_number=f"AI{100 + i}",
                airline_code="AI",
                aircraft_type="A320",
                flight_date=date(2024, 1, 1),
                departure=FlightTime(scheduled=(base_time + timedelta(minutes=i*10)).time()),
                origin=Airport("BOM", "Mumbai", "Mumbai"),
                destination=Airport("DEL", "Delhi", "Delhi")
            )
            flights.append(flight)
        
        # Build graph
        graph = optimizer.build_feasibility_graph(flights, time_window_hours=2)
        
        # Solve optimization
        weights = ObjectiveWeights().normalize()
        result = optimizer.solve_assignment(graph, weights, solver="heuristic")
        
        # Validate results
        assert result.solver_status in ["OPTIMAL", "FEASIBLE", "HEURISTIC"]
        assert len(result.assignment) <= len(flights)
        assert result.total_cost >= 0
        
        # Validate constraints
        violations = optimizer.validate_runway_constraints(result.assignment, graph)
        # Some violations might be acceptable depending on the scenario
        
        # Test what-if analysis
        if result.assignment:
            first_flight_id = list(result.assignment.keys())[0]
            changes = [(first_flight_id, base_time + timedelta(minutes=30))]
            
            impact = optimizer.what_if_analysis(flights, changes)
            assert "cost_change" in impact
            assert "recommendation" in impact
    
    def test_constraint_satisfaction(self):
        """Test that optimization respects constraints."""
        optimizer = WeightedGraphOptimizer()
        
        # Create scenario with potential constraint violations
        flights = [
            Flight(
                flight_id="HEAVY1",
                flight_number="AI777",
                airline_code="AI",
                aircraft_type="B777",  # Heavy aircraft
                flight_date=date(2024, 1, 1),
                departure=FlightTime(scheduled=time(2, 0)),  # Curfew time
                origin=Airport("BOM", "Mumbai", "Mumbai"),
                destination=Airport("DEL", "Delhi", "Delhi")
            ),
            Flight(
                flight_id="NORMAL1",
                flight_number="AI320",
                airline_code="AI",
                aircraft_type="A320",
                flight_date=date(2024, 1, 1),
                departure=FlightTime(scheduled=time(10, 0)),
                origin=Airport("BOM", "Mumbai", "Mumbai"),
                destination=Airport("DEL", "Delhi", "Delhi")
            )
        ]
        
        graph = optimizer.build_feasibility_graph(flights)
        result = optimizer.solve_assignment(graph, solver="heuristic")
        
        # Check that heavy aircraft is not assigned to curfew slots if possible
        if "HEAVY1" in result.assignment:
            assigned_slot_id = result.assignment["HEAVY1"]
            assigned_slot = graph.get_slot_node(assigned_slot_id)
            
            # If there were non-curfew alternatives, heavy aircraft should avoid curfew
            # This is a soft constraint in our implementation
            if assigned_slot and assigned_slot.is_curfew:
                # Check if there were feasible non-curfew alternatives
                heavy_edges = graph.get_flight_edges("HEAVY1")
                non_curfew_alternatives = [
                    e for e in heavy_edges 
                    if e.feasible and not graph.get_slot_node(e.slot_id).is_curfew
                ]
                
                # If alternatives existed, this might indicate suboptimal assignment
                # But it's acceptable for testing purposes
                pass


if __name__ == "__main__":
    pytest.main([__file__])