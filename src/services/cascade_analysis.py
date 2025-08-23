"""Cascade impact analysis service for flight delay propagation and high-impact flight identification."""

import os
from datetime import datetime, date, time, timedelta
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from collections import defaultdict, deque
import networkx as nx

from ..models.flight import Flight
from .database import FlightDatabaseService, QueryResult


class CascadeType(Enum):
    """Types of cascade dependencies."""
    SAME_TAIL = "same_tail"           # Same aircraft turnaround
    STAND_TURNOVER = "stand_turnover" # Stand/gate dependency
    RUNWAY_BANK = "runway_bank"       # Runway slot dependency
    CREW_ROTATION = "crew_rotation"   # Crew scheduling dependency


class ImpactSeverity(Enum):
    """Severity levels for cascade impacts."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CascadeEdge:
    """Edge in the cascade dependency graph."""
    source_flight_id: str
    target_flight_id: str
    cascade_type: CascadeType
    dependency_strength: float  # 0.0 to 1.0
    expected_delay_propagation: float  # minutes
    confidence: float = 0.8  # confidence in the dependency
    
    # Additional context
    aircraft_registration: Optional[str] = None
    stand_number: Optional[str] = None
    runway: Optional[str] = None
    time_gap_minutes: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "source_flight_id": self.source_flight_id,
            "target_flight_id": self.target_flight_id,
            "cascade_type": self.cascade_type.value,
            "dependency_strength": round(self.dependency_strength, 3),
            "expected_delay_propagation": round(self.expected_delay_propagation, 1),
            "confidence": round(self.confidence, 3),
            "context": {
                "aircraft_registration": self.aircraft_registration,
                "stand_number": self.stand_number,
                "runway": self.runway,
                "time_gap_minutes": round(self.time_gap_minutes, 1) if self.time_gap_minutes else None
            }
        }


@dataclass
class CascadeNode:
    """Node in the cascade dependency graph representing a flight."""
    flight_id: str
    flight_number: str
    aircraft_registration: Optional[str]
    airport_code: str
    scheduled_time: datetime
    actual_time: Optional[datetime] = None
    delay_minutes: Optional[float] = None
    
    # Centrality metrics
    in_degree: int = 0
    out_degree: int = 0
    betweenness_centrality: float = 0.0
    closeness_centrality: float = 0.0
    eigenvector_centrality: float = 0.0
    pagerank_score: float = 0.0
    
    # Impact metrics
    downstream_flights: int = 0
    total_downstream_delay: float = 0.0
    cascade_depth: int = 0
    impact_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "flight_id": self.flight_id,
            "flight_number": self.flight_number,
            "aircraft_registration": self.aircraft_registration,
            "airport_code": self.airport_code,
            "scheduled_time": self.scheduled_time.isoformat(),
            "actual_time": self.actual_time.isoformat() if self.actual_time else None,
            "delay_minutes": round(self.delay_minutes, 1) if self.delay_minutes else None,
            "centrality_metrics": {
                "in_degree": self.in_degree,
                "out_degree": self.out_degree,
                "betweenness_centrality": round(self.betweenness_centrality, 4),
                "closeness_centrality": round(self.closeness_centrality, 4),
                "eigenvector_centrality": round(self.eigenvector_centrality, 4),
                "pagerank_score": round(self.pagerank_score, 4)
            },
            "impact_metrics": {
                "downstream_flights": self.downstream_flights,
                "total_downstream_delay": round(self.total_downstream_delay, 1),
                "cascade_depth": self.cascade_depth,
                "impact_score": round(self.impact_score, 3)
            }
        }


@dataclass
class CascadeGraph:
    """Complete cascade dependency graph."""
    nodes: Dict[str, CascadeNode] = field(default_factory=dict)
    edges: List[CascadeEdge] = field(default_factory=list)
    airport_code: str = ""
    analysis_date: date = field(default_factory=date.today)
    
    # Graph statistics
    total_flights: int = 0
    total_dependencies: int = 0
    avg_cascade_depth: float = 0.0
    max_cascade_depth: int = 0
    
    # NetworkX graph for analysis
    _nx_graph: Optional[nx.DiGraph] = field(default=None, init=False)
    
    def build_networkx_graph(self) -> nx.DiGraph:
        """Build NetworkX graph for centrality calculations."""
        if self._nx_graph is None:
            self._nx_graph = nx.DiGraph()
            
            # Add nodes
            for flight_id, node in self.nodes.items():
                self._nx_graph.add_node(flight_id, **node.to_dict())
            
            # Add edges
            for edge in self.edges:
                self._nx_graph.add_edge(
                    edge.source_flight_id,
                    edge.target_flight_id,
                    weight=edge.dependency_strength,
                    **edge.to_dict()
                )
        
        return self._nx_graph
    
    def get_high_impact_flights(self, top_n: int = 10) -> List[CascadeNode]:
        """Get top N high-impact flights ranked by impact score."""
        sorted_nodes = sorted(
            self.nodes.values(),
            key=lambda n: n.impact_score,
            reverse=True
        )
        return sorted_nodes[:top_n]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "airport_code": self.airport_code,
            "analysis_date": self.analysis_date.isoformat(),
            "statistics": {
                "total_flights": self.total_flights,
                "total_dependencies": self.total_dependencies,
                "avg_cascade_depth": round(self.avg_cascade_depth, 2),
                "max_cascade_depth": self.max_cascade_depth
            },
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges]
        }


@dataclass
class HighImpactFlight:
    """High-impact flight with detailed impact analysis."""
    flight_id: str
    flight_number: str
    aircraft_registration: Optional[str]
    airport_code: str
    scheduled_time: datetime
    impact_score: float
    impact_rank: int
    downstream_flights: int
    total_downstream_delay: float
    cascade_depth: int
    actual_time: Optional[datetime] = None
    delay_minutes: Optional[float] = None
    
    # Centrality scores
    centrality_scores: Dict[str, float] = field(default_factory=dict)
    
    # Impact breakdown
    same_tail_impacts: List[str] = field(default_factory=list)
    stand_impacts: List[str] = field(default_factory=list)
    runway_impacts: List[str] = field(default_factory=list)
    
    # Justification
    impact_justification: List[str] = field(default_factory=list)
    severity: ImpactSeverity = ImpactSeverity.MEDIUM
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "flight_id": self.flight_id,
            "flight_number": self.flight_number,
            "aircraft_registration": self.aircraft_registration,
            "airport_code": self.airport_code,
            "scheduled_time": self.scheduled_time.isoformat(),
            "actual_time": self.actual_time.isoformat() if self.actual_time else None,
            "delay_minutes": round(self.delay_minutes, 1) if self.delay_minutes else None,
            "impact_metrics": {
                "impact_score": round(self.impact_score, 3),
                "impact_rank": self.impact_rank,
                "downstream_flights": self.downstream_flights,
                "total_downstream_delay": round(self.total_downstream_delay, 1),
                "cascade_depth": self.cascade_depth,
                "severity": self.severity.value
            },
            "centrality_scores": {k: round(v, 4) for k, v in self.centrality_scores.items()},
            "impact_breakdown": {
                "same_tail_impacts": self.same_tail_impacts,
                "stand_impacts": self.stand_impacts,
                "runway_impacts": self.runway_impacts
            },
            "impact_justification": self.impact_justification
        }


class CascadeAnalysisService:
    """Service for analyzing flight delay cascades and identifying high-impact flights."""
    
    def __init__(self, db_service: Optional[FlightDatabaseService] = None):
        """
        Initialize the cascade analysis service.
        
        Args:
            db_service: Database service for flight data
        """
        self.db_service = db_service or FlightDatabaseService()
        
        # Configuration parameters
        self.max_turnaround_gap_hours = 24  # Maximum gap for same-tail dependencies
        self.min_turnaround_minutes = 30    # Minimum turnaround time
        self.max_turnaround_minutes = 720   # Maximum turnaround time (12 hours)
        self.stand_turnover_minutes = 60    # Time window for stand dependencies
        self.runway_bank_minutes = 30       # Time window for runway dependencies
    
    def build_cascade_graph(self, flights: List[Flight], 
                          airport_code: str,
                          analysis_date: date) -> CascadeGraph:
        """
        Build cascade dependency graph for flights.
        
        Args:
            flights: List of flights to analyze
            airport_code: Airport code for analysis
            analysis_date: Date of analysis
            
        Returns:
            CascadeGraph with nodes and edges
        """
        graph = CascadeGraph(
            airport_code=airport_code,
            analysis_date=analysis_date,
            total_flights=len(flights)
        )
        
        # Create nodes from flights
        self._create_cascade_nodes(flights, graph)
        
        # Build same-tail dependencies
        self._build_same_tail_dependencies(flights, graph)
        
        # Build stand turnover dependencies
        self._build_stand_dependencies(flights, graph)
        
        # Build runway bank dependencies
        self._build_runway_dependencies(flights, graph)
        
        # Calculate centrality metrics
        self._calculate_centrality_metrics(graph)
        
        # Calculate impact scores
        self._calculate_impact_scores(graph)
        
        # Update graph statistics
        self._update_graph_statistics(graph)
        
        return graph
    
    def identify_high_impact_flights(self, graph: CascadeGraph, 
                                   top_n: int = 10) -> List[HighImpactFlight]:
        """
        Identify high-impact flights from cascade graph.
        
        Args:
            graph: Cascade dependency graph
            top_n: Number of top flights to return
            
        Returns:
            List of HighImpactFlight objects ranked by impact
        """
        high_impact_flights = []
        top_nodes = graph.get_high_impact_flights(top_n)
        
        for rank, node in enumerate(top_nodes, 1):
            # Get impact breakdown
            same_tail_impacts = self._get_same_tail_impacts(node.flight_id, graph)
            stand_impacts = self._get_stand_impacts(node.flight_id, graph)
            runway_impacts = self._get_runway_impacts(node.flight_id, graph)
            
            # Generate justification
            justification = self._generate_impact_justification(node, graph)
            
            # Determine severity
            severity = self._determine_impact_severity(node)
            
            high_impact_flight = HighImpactFlight(
                flight_id=node.flight_id,
                flight_number=node.flight_number,
                aircraft_registration=node.aircraft_registration,
                airport_code=node.airport_code,
                scheduled_time=node.scheduled_time,
                actual_time=node.actual_time,
                delay_minutes=node.delay_minutes,
                impact_score=node.impact_score,
                impact_rank=rank,
                downstream_flights=node.downstream_flights,
                total_downstream_delay=node.total_downstream_delay,
                cascade_depth=node.cascade_depth,
                centrality_scores={
                    "betweenness": node.betweenness_centrality,
                    "closeness": node.closeness_centrality,
                    "eigenvector": node.eigenvector_centrality,
                    "pagerank": node.pagerank_score
                },
                same_tail_impacts=same_tail_impacts,
                stand_impacts=stand_impacts,
                runway_impacts=runway_impacts,
                impact_justification=justification,
                severity=severity
            )
            
            high_impact_flights.append(high_impact_flight)
        
        return high_impact_flights
    
    def trace_downstream_impact(self, flight_id: str, 
                              graph: CascadeGraph,
                              max_depth: int = 5) -> Dict[str, Any]:
        """
        Trace downstream impact of a specific flight.
        
        Args:
            flight_id: Flight to trace impact for
            graph: Cascade dependency graph
            max_depth: Maximum depth to trace
            
        Returns:
            Dictionary with downstream impact details
        """
        if flight_id not in graph.nodes:
            return {"error": "Flight not found in graph"}
        
        nx_graph = graph.build_networkx_graph()
        
        # BFS to trace downstream impacts
        visited = set()
        queue = deque([(flight_id, 0)])  # (flight_id, depth)
        downstream_impacts = []
        total_delay = 0.0
        
        while queue and len(visited) < 100:  # Limit to prevent infinite loops
            current_flight, depth = queue.popleft()
            
            if current_flight in visited or depth > max_depth:
                continue
            
            visited.add(current_flight)
            
            if depth > 0:  # Don't include the source flight
                node = graph.nodes[current_flight]
                downstream_impacts.append({
                    "flight_id": current_flight,
                    "flight_number": node.flight_number,
                    "depth": depth,
                    "delay_minutes": node.delay_minutes or 0,
                    "cascade_type": self._get_cascade_type_to_flight(flight_id, current_flight, graph)
                })
                total_delay += node.delay_minutes or 0
            
            # Add successors to queue
            for successor in nx_graph.successors(current_flight):
                if successor not in visited:
                    queue.append((successor, depth + 1))
        
        return {
            "source_flight_id": flight_id,
            "downstream_flights": len(downstream_impacts),
            "max_depth_reached": max(impact["depth"] for impact in downstream_impacts) if downstream_impacts else 0,
            "total_downstream_delay": round(total_delay, 1),
            "impacts": downstream_impacts
        }
    
    def _create_cascade_nodes(self, flights: List[Flight], graph: CascadeGraph) -> None:
        """Create cascade nodes from flight data."""
        for flight in flights:
            # Determine if this is a departure or arrival at the airport
            scheduled_time = None
            actual_time = None
            delay_minutes = None
            
            if flight.origin and flight.origin.code == graph.airport_code:
                # Departure from this airport
                scheduled_time = flight.departure.scheduled
                actual_time = flight.departure.actual
                delay_minutes = flight.dep_delay_min
            elif flight.destination and flight.destination.code == graph.airport_code:
                # Arrival at this airport
                scheduled_time = flight.arrival.scheduled
                actual_time = flight.arrival.actual
                delay_minutes = flight.arr_delay_min
            
            if scheduled_time:
                # Convert time to datetime if needed
                if isinstance(scheduled_time, time):
                    scheduled_time = datetime.combine(graph.analysis_date, scheduled_time)
                
                node = CascadeNode(
                    flight_id=flight.flight_id,
                    flight_number=flight.flight_number,
                    aircraft_registration=flight.aircraft_registration,
                    airport_code=graph.airport_code,
                    scheduled_time=scheduled_time,
                    actual_time=actual_time,
                    delay_minutes=delay_minutes
                )
                
                graph.nodes[flight.flight_id] = node
    
    def _build_same_tail_dependencies(self, flights: List[Flight], graph: CascadeGraph) -> None:
        """Build same-tail turnaround dependencies."""
        # Group flights by aircraft registration
        aircraft_flights = defaultdict(list)
        
        for flight in flights:
            if flight.aircraft_registration and flight.aircraft_registration != "UNKNOWN":
                aircraft_flights[flight.aircraft_registration].append(flight)
        
        # For each aircraft, find turnaround dependencies
        for aircraft_reg, aircraft_flight_list in aircraft_flights.items():
            # Sort flights by time
            sorted_flights = sorted(aircraft_flight_list, key=lambda f: self._get_flight_time(f))
            
            # Find consecutive flights at the same airport (turnarounds)
            for i in range(len(sorted_flights) - 1):
                arrival_flight = sorted_flights[i]
                departure_flight = sorted_flights[i + 1]
                
                # Check if this is a turnaround at the analysis airport
                if (arrival_flight.destination and arrival_flight.destination.code == graph.airport_code and
                    departure_flight.origin and departure_flight.origin.code == graph.airport_code):
                    
                    # Calculate time gap
                    arrival_time = self._get_flight_time(arrival_flight)
                    departure_time = self._get_flight_time(departure_flight)
                    time_gap_minutes = (departure_time - arrival_time).total_seconds() / 60
                    
                    # Check if it's a reasonable turnaround
                    if self.min_turnaround_minutes <= time_gap_minutes <= self.max_turnaround_minutes:
                        # Calculate dependency strength based on time gap
                        dependency_strength = self._calculate_turnaround_dependency_strength(time_gap_minutes)
                        
                        # Estimate delay propagation
                        delay_propagation = self._estimate_delay_propagation(
                            arrival_flight, departure_flight, CascadeType.SAME_TAIL
                        )
                        
                        edge = CascadeEdge(
                            source_flight_id=arrival_flight.flight_id,
                            target_flight_id=departure_flight.flight_id,
                            cascade_type=CascadeType.SAME_TAIL,
                            dependency_strength=dependency_strength,
                            expected_delay_propagation=delay_propagation,
                            aircraft_registration=aircraft_reg,
                            time_gap_minutes=time_gap_minutes,
                            confidence=0.9  # High confidence for same-tail dependencies
                        )
                        
                        graph.edges.append(edge)
    
    def _build_stand_dependencies(self, flights: List[Flight], graph: CascadeGraph) -> None:
        """Build stand/gate turnover dependencies."""
        # Group flights by stand/gate (if available)
        # For now, we'll use a simplified approach based on time proximity
        # In a real implementation, this would use actual stand assignment data
        
        departures = []
        arrivals = []
        
        for flight in flights:
            if flight.origin and flight.origin.code == graph.airport_code:
                departures.append(flight)
            elif flight.destination and flight.destination.code == graph.airport_code:
                arrivals.append(flight)
        
        # Find potential stand turnovers (arrival followed by departure within time window)
        for arrival in arrivals:
            arrival_time = self._get_flight_time(arrival)
            
            for departure in departures:
                departure_time = self._get_flight_time(departure)
                time_gap_minutes = (departure_time - arrival_time).total_seconds() / 60
                
                # Check if it's within stand turnover window and not the same aircraft
                if (0 < time_gap_minutes <= self.stand_turnover_minutes and
                    arrival.aircraft_registration != departure.aircraft_registration):
                    
                    # Calculate dependency strength (weaker than same-tail)
                    dependency_strength = 0.3 * (1 - time_gap_minutes / self.stand_turnover_minutes)
                    
                    # Estimate delay propagation
                    delay_propagation = self._estimate_delay_propagation(
                        arrival, departure, CascadeType.STAND_TURNOVER
                    )
                    
                    edge = CascadeEdge(
                        source_flight_id=arrival.flight_id,
                        target_flight_id=departure.flight_id,
                        cascade_type=CascadeType.STAND_TURNOVER,
                        dependency_strength=dependency_strength,
                        expected_delay_propagation=delay_propagation,
                        time_gap_minutes=time_gap_minutes,
                        confidence=0.5  # Medium confidence for stand dependencies
                    )
                    
                    graph.edges.append(edge)
    
    def _build_runway_dependencies(self, flights: List[Flight], graph: CascadeGraph) -> None:
        """Build runway bank dependencies."""
        # Group flights by runway and time proximity
        # This creates dependencies between flights using the same runway in close succession
        
        departures = []
        arrivals = []
        
        for flight in flights:
            if flight.origin and flight.origin.code == graph.airport_code:
                departures.append(flight)
            elif flight.destination and flight.destination.code == graph.airport_code:
                arrivals.append(flight)
        
        # Sort by time
        departures.sort(key=self._get_flight_time)
        arrivals.sort(key=self._get_flight_time)
        
        # Build dependencies within runway banks
        self._build_runway_bank_dependencies(departures, graph, "departure")
        self._build_runway_bank_dependencies(arrivals, graph, "arrival")
    
    def _build_runway_bank_dependencies(self, flights: List[Flight], 
                                      graph: CascadeGraph, 
                                      operation_type: str) -> None:
        """Build dependencies within a runway bank (consecutive operations)."""
        for i in range(len(flights) - 1):
            current_flight = flights[i]
            next_flight = flights[i + 1]
            
            current_time = self._get_flight_time(current_flight)
            next_time = self._get_flight_time(next_flight)
            time_gap_minutes = (next_time - current_time).total_seconds() / 60
            
            # Check if flights are in the same runway bank
            if 0 < time_gap_minutes <= self.runway_bank_minutes:
                # Calculate dependency strength (weaker than turnaround dependencies)
                dependency_strength = 0.2 * (1 - time_gap_minutes / self.runway_bank_minutes)
                
                # Estimate delay propagation
                delay_propagation = self._estimate_delay_propagation(
                    current_flight, next_flight, CascadeType.RUNWAY_BANK
                )
                
                edge = CascadeEdge(
                    source_flight_id=current_flight.flight_id,
                    target_flight_id=next_flight.flight_id,
                    cascade_type=CascadeType.RUNWAY_BANK,
                    dependency_strength=dependency_strength,
                    expected_delay_propagation=delay_propagation,
                    time_gap_minutes=time_gap_minutes,
                    confidence=0.4  # Lower confidence for runway dependencies
                )
                
                graph.edges.append(edge)
    
    def _calculate_centrality_metrics(self, graph: CascadeGraph) -> None:
        """Calculate centrality metrics for all nodes."""
        if not graph.edges:
            return
        
        nx_graph = graph.build_networkx_graph()
        
        try:
            # Calculate various centrality measures
            betweenness = nx.betweenness_centrality(nx_graph)
            closeness = nx.closeness_centrality(nx_graph)
            pagerank = nx.pagerank(nx_graph, weight='weight')
            
            # Eigenvector centrality (may fail for disconnected graphs)
            try:
                eigenvector = nx.eigenvector_centrality(nx_graph, weight='weight')
            except:
                eigenvector = {node: 0.0 for node in nx_graph.nodes()}
            
            # Update node metrics
            for flight_id, node in graph.nodes.items():
                node.in_degree = nx_graph.in_degree(flight_id)
                node.out_degree = nx_graph.out_degree(flight_id)
                node.betweenness_centrality = betweenness.get(flight_id, 0.0)
                node.closeness_centrality = closeness.get(flight_id, 0.0)
                node.eigenvector_centrality = eigenvector.get(flight_id, 0.0)
                node.pagerank_score = pagerank.get(flight_id, 0.0)
                
        except Exception as e:
            print(f"Error calculating centrality metrics: {e}")
    
    def _calculate_impact_scores(self, graph: CascadeGraph) -> None:
        """Calculate impact scores for all nodes."""
        nx_graph = graph.build_networkx_graph()
        
        for flight_id, node in graph.nodes.items():
            # Calculate downstream metrics
            downstream_flights = set()
            total_downstream_delay = 0.0
            max_depth = 0
            
            # BFS to find all downstream flights
            visited = set()
            queue = deque([(flight_id, 0)])
            
            while queue:
                current_flight, depth = queue.popleft()
                
                if current_flight in visited:
                    continue
                
                visited.add(current_flight)
                
                if depth > 0:  # Don't count the source flight
                    downstream_flights.add(current_flight)
                    downstream_node = graph.nodes.get(current_flight)
                    if downstream_node and downstream_node.delay_minutes:
                        total_downstream_delay += downstream_node.delay_minutes
                    max_depth = max(max_depth, depth)
                
                # Add successors
                for successor in nx_graph.successors(current_flight):
                    if successor not in visited and depth < 10:  # Limit depth
                        queue.append((successor, depth + 1))
            
            # Update node metrics
            node.downstream_flights = len(downstream_flights)
            node.total_downstream_delay = total_downstream_delay
            node.cascade_depth = max_depth
            
            # Calculate composite impact score
            node.impact_score = self._calculate_composite_impact_score(node)
    
    def _calculate_composite_impact_score(self, node: CascadeNode) -> float:
        """Calculate composite impact score for a node."""
        # Weighted combination of different factors
        weights = {
            "centrality": 0.3,
            "downstream_count": 0.25,
            "downstream_delay": 0.25,
            "cascade_depth": 0.1,
            "own_delay": 0.1
        }
        
        # Normalize centrality (use PageRank as primary centrality measure)
        centrality_score = node.pagerank_score
        
        # Normalize downstream count (log scale to handle large numbers)
        downstream_score = np.log1p(node.downstream_flights) / 10.0
        
        # Normalize downstream delay (log scale)
        delay_score = np.log1p(node.total_downstream_delay) / 100.0
        
        # Normalize cascade depth
        depth_score = min(node.cascade_depth / 5.0, 1.0)
        
        # Own delay contribution
        own_delay_score = np.log1p(node.delay_minutes or 0) / 50.0
        
        # Calculate weighted score
        impact_score = (
            weights["centrality"] * centrality_score +
            weights["downstream_count"] * downstream_score +
            weights["downstream_delay"] * delay_score +
            weights["cascade_depth"] * depth_score +
            weights["own_delay"] * own_delay_score
        )
        
        return min(impact_score, 1.0)  # Cap at 1.0
    
    def _update_graph_statistics(self, graph: CascadeGraph) -> None:
        """Update graph-level statistics."""
        graph.total_dependencies = len(graph.edges)
        
        if graph.nodes:
            cascade_depths = [node.cascade_depth for node in graph.nodes.values()]
            graph.avg_cascade_depth = sum(cascade_depths) / len(cascade_depths)
            graph.max_cascade_depth = max(cascade_depths)
    
    def _get_flight_time(self, flight: Flight) -> datetime:
        """Get the primary time for a flight (actual if available, otherwise scheduled)."""
        # For departures, use departure time
        if flight.departure.actual:
            return flight.departure.actual
        elif flight.departure.scheduled:
            if isinstance(flight.departure.scheduled, time):
                return datetime.combine(date.today(), flight.departure.scheduled)
            return flight.departure.scheduled
        
        # For arrivals, use arrival time
        if flight.arrival.actual:
            return flight.arrival.actual
        elif flight.arrival.scheduled:
            if isinstance(flight.arrival.scheduled, time):
                return datetime.combine(date.today(), flight.arrival.scheduled)
            return flight.arrival.scheduled
        
        # Fallback
        return datetime.now()
    
    def _calculate_turnaround_dependency_strength(self, time_gap_minutes: float) -> float:
        """Calculate dependency strength for turnaround operations."""
        # Stronger dependency for shorter turnarounds
        if time_gap_minutes <= 60:
            return 0.9  # Very strong dependency
        elif time_gap_minutes <= 120:
            return 0.7  # Strong dependency
        elif time_gap_minutes <= 240:
            return 0.5  # Medium dependency
        else:
            return 0.3  # Weak dependency
    
    def _estimate_delay_propagation(self, source_flight: Flight, 
                                  target_flight: Flight, 
                                  cascade_type: CascadeType) -> float:
        """Estimate expected delay propagation between flights."""
        source_delay = source_flight.dep_delay_min or source_flight.arr_delay_min or 0
        
        if source_delay <= 0:
            return 0.0
        
        # Propagation factors by cascade type
        propagation_factors = {
            CascadeType.SAME_TAIL: 0.8,        # High propagation for same aircraft
            CascadeType.STAND_TURNOVER: 0.3,   # Medium propagation for stand conflicts
            CascadeType.RUNWAY_BANK: 0.2,      # Low propagation for runway delays
            CascadeType.CREW_ROTATION: 0.6     # High propagation for crew issues
        }
        
        factor = propagation_factors.get(cascade_type, 0.2)
        return source_delay * factor
    
    def _get_same_tail_impacts(self, flight_id: str, graph: CascadeGraph) -> List[str]:
        """Get same-tail impact flights for a given flight."""
        impacts = []
        for edge in graph.edges:
            if (edge.source_flight_id == flight_id and 
                edge.cascade_type == CascadeType.SAME_TAIL):
                target_node = graph.nodes.get(edge.target_flight_id)
                if target_node:
                    impacts.append(f"{target_node.flight_number} (tail: {edge.aircraft_registration})")
        return impacts
    
    def _get_stand_impacts(self, flight_id: str, graph: CascadeGraph) -> List[str]:
        """Get stand turnover impact flights for a given flight."""
        impacts = []
        for edge in graph.edges:
            if (edge.source_flight_id == flight_id and 
                edge.cascade_type == CascadeType.STAND_TURNOVER):
                target_node = graph.nodes.get(edge.target_flight_id)
                if target_node:
                    impacts.append(f"{target_node.flight_number} (stand turnover)")
        return impacts
    
    def _get_runway_impacts(self, flight_id: str, graph: CascadeGraph) -> List[str]:
        """Get runway bank impact flights for a given flight."""
        impacts = []
        for edge in graph.edges:
            if (edge.source_flight_id == flight_id and 
                edge.cascade_type == CascadeType.RUNWAY_BANK):
                target_node = graph.nodes.get(edge.target_flight_id)
                if target_node:
                    impacts.append(f"{target_node.flight_number} (runway bank)")
        return impacts
    
    def _get_cascade_type_to_flight(self, source_flight_id: str, 
                                  target_flight_id: str, 
                                  graph: CascadeGraph) -> str:
        """Get cascade type between two flights."""
        for edge in graph.edges:
            if (edge.source_flight_id == source_flight_id and 
                edge.target_flight_id == target_flight_id):
                return edge.cascade_type.value
        return "unknown"
    
    def _generate_impact_justification(self, node: CascadeNode, 
                                     graph: CascadeGraph) -> List[str]:
        """Generate justification for high impact classification."""
        justification = []
        
        if node.downstream_flights > 10:
            justification.append(f"Affects {node.downstream_flights} downstream flights")
        
        if node.total_downstream_delay > 100:
            justification.append(f"Causes {node.total_downstream_delay:.0f} minutes of downstream delays")
        
        if node.cascade_depth > 3:
            justification.append(f"Creates cascade chains up to {node.cascade_depth} levels deep")
        
        if node.pagerank_score > 0.01:
            justification.append("High centrality in the dependency network")
        
        if node.betweenness_centrality > 0.1:
            justification.append("Critical bridge between different parts of the network")
        
        if not justification:
            justification.append("Significant impact on overall network stability")
        
        return justification
    
    def _determine_impact_severity(self, node: CascadeNode) -> ImpactSeverity:
        """Determine impact severity based on node metrics."""
        if node.impact_score > 0.8 or node.downstream_flights > 20:
            return ImpactSeverity.CRITICAL
        elif node.impact_score > 0.6 or node.downstream_flights > 10:
            return ImpactSeverity.HIGH
        elif node.impact_score > 0.3 or node.downstream_flights > 5:
            return ImpactSeverity.MEDIUM
        else:
            return ImpactSeverity.LOW