#!/usr/bin/env python3
"""
Demo script for the Weighted Graph Optimization Engine.

This script demonstrates the key features of the weighted graph optimizer:
- Building bipartite graphs from flight data
- Multi-objective cost calculation with runway-dependent weights
- Constraint satisfaction and feasibility checking
- Heuristic and min-cost flow solvers
- What-if analysis for schedule changes
- Autonomous optimization triggers
"""

import sys
import os
from datetime import datetime, date, time, timedelta
from typing import List

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.flight import Flight, FlightTime, Airport
from src.services.weighted_graph_optimizer import (
    WeightedGraphOptimizer, ObjectiveWeights, BipartiteGraph
)
from src.services.analytics import WeatherRegime


def create_sample_flights() -> List[Flight]:
    """Create sample flights for demonstration."""
    flights = []
    base_date = date(2024, 1, 15)
    
    # Morning rush flights
    flight_data = [
        ("AI123", "Air India", "A320", time(8, 0), "BOM", "DEL"),
        ("6E456", "IndiGo", "A320", time(8, 15), "BOM", "BLR"),
        ("SG789", "SpiceJet", "B737", time(8, 30), "BOM", "CCU"),
        ("AI234", "Air India", "B777", time(8, 45), "BOM", "DEL"),  # Heavy aircraft
        ("UK567", "Vistara", "A321", time(9, 0), "BOM", "HYD"),
        ("6E890", "IndiGo", "A320", time(9, 15), "BOM", "BLR"),
        ("AI345", "Air India", "A320", time(9, 30), "BOM", "MAA"),
        ("SG012", "SpiceJet", "B737", time(9, 45), "BOM", "GOI"),
    ]
    
    for i, (flight_no, airline, aircraft, dep_time, origin, dest) in enumerate(flight_data):
        flight = Flight(
            flight_id=f"{flight_no}_{i}",
            flight_number=flight_no,
            airline_code=flight_no[:2],
            aircraft_type=aircraft,
            aircraft_registration=f"VT-{airline[:3]}{i:02d}",
            flight_date=base_date,
            departure=FlightTime(scheduled=dep_time),
            origin=Airport.from_string(f"{origin} ({origin})"),
            destination=Airport.from_string(f"{dest} ({dest})")
        )
        flights.append(flight)
    
    return flights


def demonstrate_graph_construction():
    """Demonstrate bipartite graph construction."""
    print("=" * 60)
    print("WEIGHTED GRAPH OPTIMIZATION ENGINE DEMO")
    print("=" * 60)
    
    # Create sample flights
    flights = create_sample_flights()
    print(f"\n1. Created {len(flights)} sample flights for morning rush period")
    
    for flight in flights[:3]:  # Show first 3
        print(f"   - {flight.flight_number}: {flight.aircraft_type} "
              f"{flight.departure.scheduled} {flight.origin.code}→{flight.destination.code}")
    print(f"   ... and {len(flights)-3} more flights")
    
    # Initialize optimizer
    optimizer = WeightedGraphOptimizer()
    
    # Build bipartite graph
    print(f"\n2. Building bipartite graph (flights ↔ runway-time slots)...")
    graph = optimizer.build_feasibility_graph(
        flights, 
        time_window_hours=2,  # ±2 hours from original schedule
        slot_interval_minutes=5  # 5-minute time slots
    )
    
    print(f"   - Flight nodes: {len(graph.flight_nodes)}")
    print(f"   - Slot nodes: {len(graph.slot_nodes)}")
    print(f"   - Feasible edges: {len(graph.get_feasible_edges())}")
    print(f"   - Total edges: {len(graph.edges)}")
    print(f"   - Graph constraints: {len(graph.constraints)}")
    
    # Show sample slot nodes
    print(f"\n3. Sample runway-time slots:")
    for slot in graph.slot_nodes[:5]:
        print(f"   - {slot.slot_id}: {slot.runway} at {slot.timestamp.strftime('%H:%M')} "
              f"(capacity: {slot.weather_adjusted_capacity}, curfew: {slot.is_curfew})")
    
    return optimizer, graph, flights


def demonstrate_cost_calculation():
    """Demonstrate multi-objective cost calculation."""
    print(f"\n4. Multi-objective cost calculation with runway-dependent weights:")
    
    optimizer = WeightedGraphOptimizer()
    
    # Create sample flight and slots
    sample_graph = optimizer.build_feasibility_graph(create_sample_flights()[:1])
    flight_node = sample_graph.flight_nodes[0]
    
    # Different runway slots at same time
    from src.services.weighted_graph_optimizer import SlotNode
    
    slots = [
        SlotNode("09L_0800", "09L", datetime(2024, 1, 15, 8, 0), 6),  # Preferred runway
        SlotNode("09R_0800", "09R", datetime(2024, 1, 15, 8, 0), 6),  # Longer taxi
        SlotNode("09L_0805", "09L", datetime(2024, 1, 15, 8, 5), 6),  # 5min delay
        SlotNode("09L_0200", "09L", datetime(2024, 1, 15, 2, 0), 6, is_curfew=True),  # Curfew
    ]
    
    weights = ObjectiveWeights().normalize()
    
    print(f"   Flight: {flight_node.flight_id} originally at {flight_node.original_slot.strftime('%H:%M')}")
    print(f"   Objective weights: delay={weights.delay_weight:.2f}, taxi={weights.taxi_weight:.2f}, "
          f"fairness={weights.fairness_weight:.2f}, CO₂={weights.co2_weight:.2f}, curfew={weights.curfew_weight:.2f}")
    
    for slot in slots:
        total_cost, breakdown = optimizer.calculate_edge_costs(flight_node, slot, weights)
        print(f"\n   Slot {slot.slot_id} ({'CURFEW' if slot.is_curfew else 'NORMAL'}):")
        print(f"     Total cost: {total_cost:.2f}")
        for cost_type, cost_value in breakdown.items():
            if cost_value > 0:
                print(f"     - {cost_type}: {cost_value:.2f}")


def demonstrate_optimization():
    """Demonstrate schedule optimization."""
    print(f"\n5. Schedule optimization using heuristic solver:")
    
    optimizer, graph, flights = demonstrate_graph_construction()
    
    # Set up optimization weights
    weights = ObjectiveWeights(
        delay_weight=1.0,
        taxi_weight=0.3,
        fairness_weight=0.5,
        co2_weight=0.2,
        curfew_weight=2.0
    ).normalize()
    
    # Solve optimization
    result = optimizer.solve_assignment(graph, weights, solver="heuristic")
    
    print(f"   Solver status: {result.solver_status}")
    print(f"   Total cost: {result.total_cost:.2f}")
    print(f"   Flights assigned: {len(result.assignment)}")
    print(f"   Cost breakdown:")
    for cost_type, cost_value in result.cost_breakdown.items():
        if cost_value > 0:
            print(f"     - {cost_type}: {cost_value:.2f}")
    
    # Show some assignments
    print(f"\n   Sample assignments:")
    for i, (flight_id, slot_id) in enumerate(list(result.assignment.items())[:4]):
        slot_node = graph.get_slot_node(slot_id)
        flight_node = graph.get_flight_node(flight_id)
        if slot_node and flight_node:
            original_time = flight_node.original_slot.strftime('%H:%M')
            assigned_time = slot_node.timestamp.strftime('%H:%M')
            runway = slot_node.runway
            print(f"     - {flight_id}: {original_time} → {assigned_time} on {runway}")
    
    return result


def demonstrate_constraint_validation():
    """Demonstrate constraint validation."""
    print(f"\n6. Constraint validation:")
    
    optimizer, graph, flights = demonstrate_graph_construction()
    result = demonstrate_optimization()
    
    # Validate constraints
    violations = optimizer.validate_runway_constraints(result.assignment, graph)
    
    print(f"   Constraint violations found: {len(violations)}")
    for violation in violations[:3]:  # Show first 3
        print(f"     - {violation}")
    
    if len(violations) > 3:
        print(f"     ... and {len(violations)-3} more violations")


def demonstrate_what_if_analysis():
    """Demonstrate what-if analysis."""
    print(f"\n7. What-if analysis:")
    
    flights = create_sample_flights()
    optimizer = WeightedGraphOptimizer()
    
    # Propose some changes
    changes = [
        (flights[0].flight_id, datetime(2024, 1, 15, 8, 30)),  # Move first flight 30min later
        (flights[1].flight_id, datetime(2024, 1, 15, 8, 45)),  # Move second flight 30min later
    ]
    
    print(f"   Proposed changes:")
    for flight_id, new_time in changes:
        original_flight = next(f for f in flights if f.flight_id == flight_id)
        original_time = datetime.combine(original_flight.flight_date, original_flight.departure.scheduled)
        delta_minutes = (new_time - original_time).total_seconds() / 60
        print(f"     - {flight_id}: {original_time.strftime('%H:%M')} → {new_time.strftime('%H:%M')} "
              f"({delta_minutes:+.0f} min)")
    
    # Analyze impact
    impact = optimizer.what_if_analysis(flights, changes)
    
    print(f"\n   Impact analysis:")
    print(f"     - Cost change: {impact['cost_change']:+.2f} ({impact['cost_change_percent']:+.1f}%)")
    print(f"     - Constraint violations change: {impact['constraint_violations_change']:+d}")
    print(f"     - Affected flights: {impact['affected_flights']}")
    print(f"     - Recommendation: {impact['recommendation'].upper()}")


def demonstrate_autonomous_optimization():
    """Demonstrate autonomous optimization."""
    print(f"\n8. Autonomous optimization:")
    
    flights = create_sample_flights()
    optimizer = WeightedGraphOptimizer()
    
    # Simulate different trigger conditions
    trigger_scenarios = [
        {
            "type": "capacity_overload",
            "urgency": "high",
            "threshold_exceeded": 1.3,
            "affected_runway": "09L"
        },
        {
            "type": "weather_degradation", 
            "urgency": "medium",
            "weather_regime": "strong",
            "capacity_reduction": 0.7
        },
        {
            "type": "curfew_violation",
            "urgency": "critical",
            "flights_affected": 2
        }
    ]
    
    for i, trigger in enumerate(trigger_scenarios, 1):
        print(f"\n   Scenario {i}: {trigger['type'].replace('_', ' ').title()}")
        print(f"     Trigger: {trigger}")
        
        result = optimizer.autonomous_optimize(trigger, flights)
        
        print(f"     Result: {result.solver_status} (cost: {result.total_cost:.2f})")
        print(f"     Flights optimized: {len(result.assignment)}")
        
        if result.constraint_violations:
            print(f"     Violations: {len(result.constraint_violations)}")


def demonstrate_runway_dependent_weights():
    """Demonstrate runway-dependent weight adjustments."""
    print(f"\n9. Runway-dependent weight optimization:")
    
    base_weights = ObjectiveWeights()
    
    runways = ["09L", "09R", "27L", "27R"]
    
    print(f"   Base weights: taxi={base_weights.taxi_weight:.2f}, CO₂={base_weights.co2_weight:.2f}")
    print(f"   Runway adjustments:")
    
    for runway in runways:
        adjusted = base_weights.get_runway_adjusted_weights(runway)
        taxi_factor = adjusted.taxi_weight / base_weights.taxi_weight
        co2_factor = adjusted.co2_weight / base_weights.co2_weight
        
        print(f"     - {runway}: taxi×{taxi_factor:.2f}, CO₂×{co2_factor:.2f} "
              f"({'longer taxi' if taxi_factor > 1 else 'shorter taxi' if taxi_factor < 1 else 'standard taxi'})")


def main():
    """Run the complete demonstration."""
    try:
        # Core functionality
        demonstrate_graph_construction()
        demonstrate_cost_calculation()
        demonstrate_optimization()
        demonstrate_constraint_validation()
        
        # Advanced features
        demonstrate_what_if_analysis()
        demonstrate_autonomous_optimization()
        demonstrate_runway_dependent_weights()
        
        print(f"\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"\nKey features demonstrated:")
        print(f"✓ Bipartite graph construction (flights ↔ runway-time slots)")
        print(f"✓ Multi-objective cost calculation with runway-dependent weights")
        print(f"✓ Constraint satisfaction (turnaround, wake separation, curfew)")
        print(f"✓ Heuristic and min-cost flow solvers with CP-SAT fallback")
        print(f"✓ What-if analysis for schedule change impact assessment")
        print(f"✓ Autonomous optimization with policy-based triggers")
        print(f"✓ Runway-specific weight adjustments for taxi and CO₂ costs")
        
        print(f"\nThe weighted graph optimization engine is ready for integration!")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())