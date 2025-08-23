#!/usr/bin/env python3
"""
Demo script for cascade impact analysis system.

This script demonstrates the cascade analysis functionality including:
- Building dependency graphs for flight cascades
- Calculating centrality metrics and impact scores
- Identifying high-impact flights
- Tracing downstream impacts
"""

import os
import sys
from datetime import datetime, date, time, timedelta
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.services.cascade_analysis import CascadeAnalysisService, CascadeType, ImpactSeverity
from src.models.flight import Flight, Airport, FlightTime


def create_sample_flights():
    """Create sample flights for cascade analysis demonstration."""
    flights = []
    base_date = date(2024, 1, 15)
    
    # Create airports
    bom = Airport(code="BOM", name="Mumbai (BOM)", city="Mumbai")
    del_airport = Airport(code="DEL", name="Delhi (DEL)", city="Delhi")
    blr = Airport(code="BLR", name="Bangalore (BLR)", city="Bangalore")
    
    print("Creating sample flight data for cascade analysis...")
    
    # Scenario: Multiple aircraft with turnaround operations at BOM
    # This will create cascade dependencies when delays occur
    
    # Aircraft VT-ABC: DEL->BOM->DEL (with delays)
    flight1 = Flight(
        flight_id="AI101_001",
        flight_number="AI101",
        aircraft_registration="VT-ABC",
        origin=del_airport,
        destination=bom,
        flight_date=base_date
    )
    flight1.arrival.scheduled = time(8, 0)
    flight1.arrival.actual = datetime.combine(base_date, time(8, 25))  # 25 min delay
    flight1.arr_delay_min = 25
    flights.append(flight1)
    
    flight2 = Flight(
        flight_id="AI102_001",
        flight_number="AI102",
        aircraft_registration="VT-ABC",  # Same aircraft
        origin=bom,
        destination=del_airport,
        flight_date=base_date
    )
    flight2.departure.scheduled = time(10, 0)
    flight2.departure.actual = datetime.combine(base_date, time(10, 30))  # 30 min delay (cascade)
    flight2.dep_delay_min = 30
    flights.append(flight2)
    
    # Aircraft VT-XYZ: BLR->BOM->BLR (on time initially)
    flight3 = Flight(
        flight_id="6E201_001",
        flight_number="6E201",
        aircraft_registration="VT-XYZ",
        origin=blr,
        destination=bom,
        flight_date=base_date
    )
    flight3.arrival.scheduled = time(9, 0)
    flight3.arrival.actual = datetime.combine(base_date, time(9, 0))  # On time
    flight3.arr_delay_min = 0
    flights.append(flight3)
    
    flight4 = Flight(
        flight_id="6E202_001",
        flight_number="6E202",
        aircraft_registration="VT-XYZ",  # Same aircraft
        origin=bom,
        destination=blr,
        flight_date=base_date
    )
    flight4.departure.scheduled = time(11, 0)
    flight4.departure.actual = datetime.combine(base_date, time(11, 15))  # 15 min delay
    flight4.dep_delay_min = 15
    flights.append(flight4)
    
    # Aircraft VT-PQR: Multiple legs creating longer cascade
    flight5 = Flight(
        flight_id="UK501_001",
        flight_number="UK501",
        aircraft_registration="VT-PQR",
        origin=del_airport,
        destination=bom,
        flight_date=base_date
    )
    flight5.arrival.scheduled = time(7, 30)
    flight5.arrival.actual = datetime.combine(base_date, time(8, 0))  # 30 min delay
    flight5.arr_delay_min = 30
    flights.append(flight5)
    
    flight6 = Flight(
        flight_id="UK502_001",
        flight_number="UK502",
        aircraft_registration="VT-PQR",  # Same aircraft
        origin=bom,
        destination=blr,
        flight_date=base_date
    )
    flight6.departure.scheduled = time(9, 30)
    flight6.departure.actual = datetime.combine(base_date, time(10, 15))  # 45 min delay (cascade)
    flight6.dep_delay_min = 45
    flights.append(flight6)
    
    flight7 = Flight(
        flight_id="UK503_001",
        flight_number="UK503",
        aircraft_registration="VT-PQR",  # Same aircraft
        origin=blr,
        destination=del_airport,
        flight_date=base_date
    )
    flight7.departure.scheduled = time(12, 0)
    flight7.departure.actual = datetime.combine(base_date, time(13, 0))  # 60 min delay (cascade)
    flight7.dep_delay_min = 60
    flights.append(flight7)
    
    # Additional flights for runway bank dependencies
    flight8 = Flight(
        flight_id="SG301_001",
        flight_number="SG301",
        aircraft_registration="VT-STU",
        origin=bom,
        destination=del_airport,
        flight_date=base_date
    )
    flight8.departure.scheduled = time(10, 5)
    flight8.departure.actual = datetime.combine(base_date, time(10, 35))  # 30 min delay
    flight8.dep_delay_min = 30
    flights.append(flight8)
    
    flight9 = Flight(
        flight_id="AI201_001",
        flight_number="AI201",
        aircraft_registration="VT-VWX",
        origin=bom,
        destination=blr,
        flight_date=base_date
    )
    flight9.departure.scheduled = time(10, 10)
    flight9.departure.actual = datetime.combine(base_date, time(10, 40))  # 30 min delay (runway impact)
    flight9.dep_delay_min = 30
    flights.append(flight9)
    
    print(f"Created {len(flights)} sample flights with various delay scenarios")
    return flights


def demonstrate_cascade_analysis():
    """Demonstrate cascade analysis functionality."""
    print("=" * 60)
    print("FLIGHT DELAY CASCADE IMPACT ANALYSIS DEMO")
    print("=" * 60)
    
    # Create sample data
    flights = create_sample_flights()
    
    # Initialize cascade analysis service
    cascade_service = CascadeAnalysisService()
    
    print("\n1. BUILDING CASCADE DEPENDENCY GRAPH")
    print("-" * 40)
    
    # Build cascade graph
    graph = cascade_service.build_cascade_graph(
        flights=flights,
        airport_code="BOM",
        analysis_date=date(2024, 1, 15)
    )
    
    print(f"✓ Built cascade graph with {len(graph.nodes)} flights and {len(graph.edges)} dependencies")
    print(f"✓ Average cascade depth: {graph.avg_cascade_depth:.1f}")
    print(f"✓ Maximum cascade depth: {graph.max_cascade_depth}")
    
    # Show dependency types
    dependency_counts = {}
    for edge in graph.edges:
        dep_type = edge.cascade_type.value
        dependency_counts[dep_type] = dependency_counts.get(dep_type, 0) + 1
    
    print("\nDependency breakdown:")
    for dep_type, count in dependency_counts.items():
        print(f"  - {dep_type.replace('_', ' ').title()}: {count} dependencies")
    
    print("\n2. CENTRALITY METRICS ANALYSIS")
    print("-" * 40)
    
    # Show top flights by different centrality measures
    nodes_by_pagerank = sorted(graph.nodes.values(), key=lambda n: n.pagerank_score, reverse=True)[:5]
    nodes_by_betweenness = sorted(graph.nodes.values(), key=lambda n: n.betweenness_centrality, reverse=True)[:5]
    
    print("Top flights by PageRank centrality:")
    for i, node in enumerate(nodes_by_pagerank, 1):
        print(f"  {i}. {node.flight_number} (Score: {node.pagerank_score:.4f})")
    
    print("\nTop flights by Betweenness centrality:")
    for i, node in enumerate(nodes_by_betweenness, 1):
        print(f"  {i}. {node.flight_number} (Score: {node.betweenness_centrality:.4f})")
    
    print("\n3. HIGH-IMPACT FLIGHT IDENTIFICATION")
    print("-" * 40)
    
    # Identify high-impact flights
    high_impact_flights = cascade_service.identify_high_impact_flights(graph, top_n=5)
    
    print("TOP 5 HIGH-IMPACT FLIGHTS:")
    for flight in high_impact_flights:
        print(f"\n🔴 Rank #{flight.impact_rank}: {flight.flight_number}")
        print(f"   Aircraft: {flight.aircraft_registration}")
        print(f"   Impact Score: {flight.impact_score:.3f}")
        print(f"   Severity: {flight.severity.value.upper()}")
        print(f"   Downstream Flights: {flight.downstream_flights}")
        print(f"   Total Downstream Delay: {flight.total_downstream_delay:.0f} minutes")
        print(f"   Cascade Depth: {flight.cascade_depth}")
        
        if flight.impact_justification:
            print("   Justification:")
            for reason in flight.impact_justification:
                print(f"     • {reason}")
        
        # Show impact breakdown
        if flight.same_tail_impacts:
            print(f"   Same-tail impacts: {', '.join(flight.same_tail_impacts)}")
        if flight.runway_impacts:
            print(f"   Runway impacts: {', '.join(flight.runway_impacts)}")
    
    print("\n4. DOWNSTREAM IMPACT TRACING")
    print("-" * 40)
    
    # Trace downstream impact for the highest impact flight
    if high_impact_flights:
        top_flight = high_impact_flights[0]
        print(f"Tracing downstream impact for {top_flight.flight_number}...")
        
        impact_trace = cascade_service.trace_downstream_impact(
            flight_id=top_flight.flight_id,
            graph=graph,
            max_depth=5
        )
        
        print(f"✓ Affects {impact_trace['downstream_flights']} downstream flights")
        print(f"✓ Maximum cascade depth: {impact_trace['max_depth_reached']}")
        print(f"✓ Total downstream delay: {impact_trace['total_downstream_delay']:.0f} minutes")
        
        if impact_trace['impacts']:
            print("\nDownstream impact chain:")
            for impact in impact_trace['impacts'][:5]:  # Show first 5
                print(f"  Depth {impact['depth']}: {impact['flight_number']} "
                      f"({impact['delay_minutes']:.0f} min delay, {impact['cascade_type']})")
    
    print("\n5. DEPENDENCY ANALYSIS")
    print("-" * 40)
    
    # Show strongest dependencies
    strongest_edges = sorted(graph.edges, key=lambda e: e.dependency_strength, reverse=True)[:5]
    
    print("STRONGEST CASCADE DEPENDENCIES:")
    for i, edge in enumerate(strongest_edges, 1):
        source_node = graph.nodes[edge.source_flight_id]
        target_node = graph.nodes[edge.target_flight_id]
        
        print(f"\n{i}. {source_node.flight_number} → {target_node.flight_number}")
        print(f"   Type: {edge.cascade_type.value.replace('_', ' ').title()}")
        print(f"   Strength: {edge.dependency_strength:.3f}")
        print(f"   Expected delay propagation: {edge.expected_delay_propagation:.1f} minutes")
        print(f"   Confidence: {edge.confidence:.1f}")
        
        if edge.aircraft_registration:
            print(f"   Aircraft: {edge.aircraft_registration}")
        if edge.time_gap_minutes:
            print(f"   Time gap: {edge.time_gap_minutes:.0f} minutes")
    
    print("\n6. OPERATIONAL INSIGHTS")
    print("-" * 40)
    
    # Generate operational insights
    total_delay = sum(node.delay_minutes or 0 for node in graph.nodes.values())
    delayed_flights = sum(1 for node in graph.nodes.values() if (node.delay_minutes or 0) > 15)
    
    print(f"📊 SUMMARY STATISTICS:")
    print(f"   • Total flights analyzed: {len(graph.nodes)}")
    print(f"   • Flights with significant delays (>15 min): {delayed_flights}")
    print(f"   • Total delay minutes: {total_delay:.0f}")
    print(f"   • Average delay per flight: {total_delay/len(graph.nodes):.1f} minutes")
    print(f"   • Cascade dependencies identified: {len(graph.edges)}")
    
    # Recommendations
    print(f"\n💡 OPERATIONAL RECOMMENDATIONS:")
    
    if high_impact_flights:
        critical_flights = [f for f in high_impact_flights if f.severity in [ImpactSeverity.HIGH, ImpactSeverity.CRITICAL]]
        if critical_flights:
            print(f"   • Focus delay mitigation on {len(critical_flights)} high-impact flights")
            print(f"   • Priority flights: {', '.join(f.flight_number for f in critical_flights[:3])}")
    
    same_tail_deps = len([e for e in graph.edges if e.cascade_type == CascadeType.SAME_TAIL])
    if same_tail_deps > 0:
        print(f"   • Monitor {same_tail_deps} same-tail turnarounds for delay propagation")
    
    runway_deps = len([e for e in graph.edges if e.cascade_type == CascadeType.RUNWAY_BANK])
    if runway_deps > 0:
        print(f"   • Consider runway optimization for {runway_deps} sequential operations")
    
    if graph.max_cascade_depth > 3:
        print(f"   • Long cascade chains detected (depth {graph.max_cascade_depth}) - implement early intervention")
    
    print("\n" + "=" * 60)
    print("CASCADE ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    try:
        demonstrate_cascade_analysis()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()