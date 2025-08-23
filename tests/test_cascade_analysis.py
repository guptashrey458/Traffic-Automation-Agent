"""Tests for cascade impact analysis service."""

import pytest
from datetime import datetime, date, time, timedelta
from unittest.mock import Mock, patch
import networkx as nx

from src.services.cascade_analysis import (
    CascadeAnalysisService, CascadeGraph, CascadeNode, CascadeEdge,
    HighImpactFlight, CascadeType, ImpactSeverity
)
from src.models.flight import Flight, Airport, FlightTime


@pytest.fixture
def mock_db_service():
    """Mock database service."""
    return Mock()


@pytest.fixture
def cascade_service(mock_db_service):
    """Create cascade analysis service with mocked database."""
    return CascadeAnalysisService(db_service=mock_db_service)


@pytest.fixture
def sample_flights():
    """Create sample flights for testing."""
    flights = []
    base_date = date(2024, 1, 15)
    
    # Create Mumbai airport
    bom = Airport(code="BOM", name="Mumbai", city="Mumbai")
    del_airport = Airport(code="DEL", name="Delhi", city="Delhi")
    
    # Flight 1: AI101 arrival at BOM at 08:00
    flight1 = Flight(
        flight_id="flight_001",
        flight_number="AI101",
        aircraft_registration="VT-ABC",
        origin=del_airport,
        destination=bom,
        flight_date=base_date
    )
    flight1.arrival.scheduled = time(8, 0)
    flight1.arrival.actual = datetime.combine(base_date, time(8, 15))  # 15 min delay
    flight1.arr_delay_min = 15
    flights.append(flight1)
    
    # Flight 2: AI102 departure from BOM at 10:00 (same aircraft as flight1)
    flight2 = Flight(
        flight_id="flight_002",
        flight_number="AI102",
        aircraft_registration="VT-ABC",
        origin=bom,
        destination=del_airport,
        flight_date=base_date
    )
    flight2.departure.scheduled = time(10, 0)
    flight2.departure.actual = datetime.combine(base_date, time(10, 20))  # 20 min delay
    flight2.dep_delay_min = 20
    flights.append(flight2)
    
    # Flight 3: 6E201 arrival at BOM at 08:30
    flight3 = Flight(
        flight_id="flight_003",
        flight_number="6E201",
        aircraft_registration="VT-XYZ",
        origin=del_airport,
        destination=bom,
        flight_date=base_date
    )
    flight3.arrival.scheduled = time(8, 30)
    flight3.arrival.actual = datetime.combine(base_date, time(8, 30))  # On time
    flight3.arr_delay_min = 0
    flights.append(flight3)
    
    # Flight 4: 6E202 departure from BOM at 10:30 (same aircraft as flight3)
    flight4 = Flight(
        flight_id="flight_004",
        flight_number="6E202",
        aircraft_registration="VT-XYZ",
        origin=bom,
        destination=del_airport,
        flight_date=base_date
    )
    flight4.departure.scheduled = time(10, 30)
    flight4.departure.actual = datetime.combine(base_date, time(10, 30))  # On time
    flight4.dep_delay_min = 0
    flights.append(flight4)
    
    # Flight 5: UK501 departure from BOM at 10:05 (different aircraft, close time)
    flight5 = Flight(
        flight_id="flight_005",
        flight_number="UK501",
        aircraft_registration="VT-PQR",
        origin=bom,
        destination=del_airport,
        flight_date=base_date
    )
    flight5.departure.scheduled = time(10, 5)
    flight5.departure.actual = datetime.combine(base_date, time(10, 25))  # 20 min delay
    flight5.dep_delay_min = 20
    flights.append(flight5)
    
    return flights


class TestCascadeAnalysisService:
    """Test cases for CascadeAnalysisService."""
    
    def test_initialization(self, cascade_service):
        """Test service initialization."""
        assert cascade_service.db_service is not None
        assert cascade_service.max_turnaround_gap_hours == 24
        assert cascade_service.min_turnaround_minutes == 30
        assert cascade_service.max_turnaround_minutes == 720
    
    def test_build_cascade_graph_basic(self, cascade_service, sample_flights):
        """Test basic cascade graph construction."""
        graph = cascade_service.build_cascade_graph(
            flights=sample_flights,
            airport_code="BOM",
            analysis_date=date(2024, 1, 15)
        )
        
        assert isinstance(graph, CascadeGraph)
        assert graph.airport_code == "BOM"
        assert graph.analysis_date == date(2024, 1, 15)
        assert graph.total_flights == 5
        assert len(graph.nodes) > 0
        assert len(graph.edges) > 0
    
    def test_create_cascade_nodes(self, cascade_service, sample_flights):
        """Test cascade node creation."""
        graph = CascadeGraph(airport_code="BOM", analysis_date=date(2024, 1, 15))
        cascade_service._create_cascade_nodes(sample_flights, graph)
        
        # Should create nodes for flights involving BOM
        assert len(graph.nodes) == 5  # All flights involve BOM
        
        # Check specific nodes
        flight1_node = graph.nodes.get("flight_001")
        assert flight1_node is not None
        assert flight1_node.flight_number == "AI101"
        assert flight1_node.aircraft_registration == "VT-ABC"
        assert flight1_node.delay_minutes == 15
        
        flight2_node = graph.nodes.get("flight_002")
        assert flight2_node is not None
        assert flight2_node.flight_number == "AI102"
        assert flight2_node.delay_minutes == 20
    
    def test_build_same_tail_dependencies(self, cascade_service, sample_flights):
        """Test same-tail dependency construction."""
        graph = CascadeGraph(airport_code="BOM", analysis_date=date(2024, 1, 15))
        cascade_service._create_cascade_nodes(sample_flights, graph)
        cascade_service._build_same_tail_dependencies(sample_flights, graph)
        
        # Should find same-tail dependencies
        same_tail_edges = [e for e in graph.edges if e.cascade_type == CascadeType.SAME_TAIL]
        assert len(same_tail_edges) >= 2  # At least AI101->AI102 and 6E201->6E202
        
        # Check specific dependency
        ai_dependency = next((e for e in same_tail_edges 
                            if e.source_flight_id == "flight_001" and e.target_flight_id == "flight_002"), None)
        assert ai_dependency is not None
        assert ai_dependency.aircraft_registration == "VT-ABC"
        assert ai_dependency.dependency_strength >= 0.5  # Should be strong dependency
        assert ai_dependency.confidence == 0.9
    
    def test_build_runway_dependencies(self, cascade_service, sample_flights):
        """Test runway bank dependency construction."""
        graph = CascadeGraph(airport_code="BOM", analysis_date=date(2024, 1, 15))
        cascade_service._create_cascade_nodes(sample_flights, graph)
        cascade_service._build_runway_dependencies(sample_flights, graph)
        
        # Should find runway dependencies between close departures
        runway_edges = [e for e in graph.edges if e.cascade_type == CascadeType.RUNWAY_BANK]
        assert len(runway_edges) > 0
        
        # Check dependency properties
        for edge in runway_edges:
            assert edge.dependency_strength <= 0.2  # Runway dependencies are weaker
            assert edge.confidence == 0.4
            assert edge.time_gap_minutes <= 30  # Within runway bank window
    
    def test_calculate_centrality_metrics(self, cascade_service, sample_flights):
        """Test centrality metrics calculation."""
        graph = cascade_service.build_cascade_graph(
            flights=sample_flights,
            airport_code="BOM",
            analysis_date=date(2024, 1, 15)
        )
        
        # Check that centrality metrics are calculated
        for node in graph.nodes.values():
            assert hasattr(node, 'betweenness_centrality')
            assert hasattr(node, 'closeness_centrality')
            assert hasattr(node, 'pagerank_score')
            assert node.pagerank_score >= 0
    
    def test_calculate_impact_scores(self, cascade_service, sample_flights):
        """Test impact score calculation."""
        graph = cascade_service.build_cascade_graph(
            flights=sample_flights,
            airport_code="BOM",
            analysis_date=date(2024, 1, 15)
        )
        
        # Check that impact scores are calculated
        for node in graph.nodes.values():
            assert hasattr(node, 'impact_score')
            assert 0 <= node.impact_score <= 1
            assert hasattr(node, 'downstream_flights')
            assert hasattr(node, 'cascade_depth')
    
    def test_identify_high_impact_flights(self, cascade_service, sample_flights):
        """Test high-impact flight identification."""
        graph = cascade_service.build_cascade_graph(
            flights=sample_flights,
            airport_code="BOM",
            analysis_date=date(2024, 1, 15)
        )
        
        high_impact_flights = cascade_service.identify_high_impact_flights(graph, top_n=3)
        
        assert len(high_impact_flights) <= 3
        assert all(isinstance(f, HighImpactFlight) for f in high_impact_flights)
        
        # Check ranking (should be sorted by impact score)
        for i, flight in enumerate(high_impact_flights):
            assert flight.impact_rank == i + 1
            if i > 0:
                assert flight.impact_score <= high_impact_flights[i-1].impact_score
    
    def test_trace_downstream_impact(self, cascade_service, sample_flights):
        """Test downstream impact tracing."""
        graph = cascade_service.build_cascade_graph(
            flights=sample_flights,
            airport_code="BOM",
            analysis_date=date(2024, 1, 15)
        )
        
        # Trace impact from first flight
        impact_trace = cascade_service.trace_downstream_impact("flight_001", graph)
        
        assert "source_flight_id" in impact_trace
        assert impact_trace["source_flight_id"] == "flight_001"
        assert "downstream_flights" in impact_trace
        assert "total_downstream_delay" in impact_trace
        assert "impacts" in impact_trace
        
        # Should find at least the same-tail connection
        assert impact_trace["downstream_flights"] >= 1
    
    def test_get_flight_time(self, cascade_service, sample_flights):
        """Test flight time extraction."""
        flight = sample_flights[0]  # AI101 with actual arrival time
        flight_time = cascade_service._get_flight_time(flight)
        
        assert isinstance(flight_time, datetime)
        assert flight_time == flight.arrival.actual
    
    def test_calculate_turnaround_dependency_strength(self, cascade_service):
        """Test turnaround dependency strength calculation."""
        # Very short turnaround
        strength_short = cascade_service._calculate_turnaround_dependency_strength(45)
        assert strength_short == 0.9
        
        # Medium turnaround
        strength_medium = cascade_service._calculate_turnaround_dependency_strength(90)
        assert strength_medium == 0.7
        
        # Long turnaround
        strength_long = cascade_service._calculate_turnaround_dependency_strength(300)
        assert strength_long == 0.3
    
    def test_estimate_delay_propagation(self, cascade_service, sample_flights):
        """Test delay propagation estimation."""
        source_flight = sample_flights[0]  # AI101 with 15 min delay
        target_flight = sample_flights[1]  # AI102
        
        propagation = cascade_service._estimate_delay_propagation(
            source_flight, target_flight, CascadeType.SAME_TAIL
        )
        
        # Should propagate 80% of delay for same-tail
        expected = 15 * 0.8
        assert abs(propagation - expected) < 0.1
    
    def test_composite_impact_score_calculation(self, cascade_service):
        """Test composite impact score calculation."""
        node = CascadeNode(
            flight_id="test_001",
            flight_number="TEST001",
            aircraft_registration="VT-TEST",
            airport_code="BOM",
            scheduled_time=datetime.now(),
            delay_minutes=30,
            downstream_flights=5,
            total_downstream_delay=100,
            cascade_depth=3,
            pagerank_score=0.05
        )
        
        impact_score = cascade_service._calculate_composite_impact_score(node)
        
        assert 0 <= impact_score <= 1
        assert impact_score > 0  # Should have some impact with these metrics
    
    def test_determine_impact_severity(self, cascade_service):
        """Test impact severity determination."""
        # Critical impact
        critical_node = CascadeNode(
            flight_id="critical",
            flight_number="CRIT001",
            aircraft_registration="VT-CRIT",
            airport_code="BOM",
            scheduled_time=datetime.now(),
            impact_score=0.9,
            downstream_flights=25
        )
        severity = cascade_service._determine_impact_severity(critical_node)
        assert severity == ImpactSeverity.CRITICAL
        
        # Low impact
        low_node = CascadeNode(
            flight_id="low",
            flight_number="LOW001",
            aircraft_registration="VT-LOW",
            airport_code="BOM",
            scheduled_time=datetime.now(),
            impact_score=0.1,
            downstream_flights=2
        )
        severity = cascade_service._determine_impact_severity(low_node)
        assert severity == ImpactSeverity.LOW


class TestCascadeGraph:
    """Test cases for CascadeGraph."""
    
    def test_cascade_graph_initialization(self):
        """Test cascade graph initialization."""
        graph = CascadeGraph(
            airport_code="BOM",
            analysis_date=date(2024, 1, 15)
        )
        
        assert graph.airport_code == "BOM"
        assert graph.analysis_date == date(2024, 1, 15)
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0
        assert graph.total_flights == 0
    
    def test_build_networkx_graph(self):
        """Test NetworkX graph construction."""
        graph = CascadeGraph()
        
        # Add sample nodes and edges
        node1 = CascadeNode(
            flight_id="flight_001",
            flight_number="AI101",
            aircraft_registration="VT-ABC",
            airport_code="BOM",
            scheduled_time=datetime.now()
        )
        node2 = CascadeNode(
            flight_id="flight_002",
            flight_number="AI102",
            aircraft_registration="VT-ABC",
            airport_code="BOM",
            scheduled_time=datetime.now()
        )
        
        graph.nodes["flight_001"] = node1
        graph.nodes["flight_002"] = node2
        
        edge = CascadeEdge(
            source_flight_id="flight_001",
            target_flight_id="flight_002",
            cascade_type=CascadeType.SAME_TAIL,
            dependency_strength=0.8,
            expected_delay_propagation=10
        )
        graph.edges.append(edge)
        
        nx_graph = graph.build_networkx_graph()
        
        assert isinstance(nx_graph, nx.DiGraph)
        assert len(nx_graph.nodes()) == 2
        assert len(nx_graph.edges()) == 1
        assert nx_graph.has_edge("flight_001", "flight_002")
    
    def test_get_high_impact_flights(self):
        """Test high-impact flight retrieval."""
        graph = CascadeGraph()
        
        # Add nodes with different impact scores
        for i in range(5):
            node = CascadeNode(
                flight_id=f"flight_{i:03d}",
                flight_number=f"TEST{i:03d}",
                aircraft_registration=f"VT-{i:03d}",
                airport_code="BOM",
                scheduled_time=datetime.now(),
                impact_score=i * 0.2  # 0.0, 0.2, 0.4, 0.6, 0.8
            )
            graph.nodes[node.flight_id] = node
        
        high_impact = graph.get_high_impact_flights(top_n=3)
        
        assert len(high_impact) == 3
        # Should be sorted by impact score (highest first)
        assert abs(high_impact[0].impact_score - 0.8) < 0.001
        assert abs(high_impact[1].impact_score - 0.6) < 0.001
        assert abs(high_impact[2].impact_score - 0.4) < 0.001
    
    def test_to_dict(self):
        """Test graph serialization to dictionary."""
        graph = CascadeGraph(
            airport_code="BOM",
            analysis_date=date(2024, 1, 15),
            total_flights=5
        )
        
        graph_dict = graph.to_dict()
        
        assert graph_dict["airport_code"] == "BOM"
        assert graph_dict["analysis_date"] == "2024-01-15"
        assert graph_dict["statistics"]["total_flights"] == 5
        assert "nodes" in graph_dict
        assert "edges" in graph_dict


class TestCascadeEdge:
    """Test cases for CascadeEdge."""
    
    def test_cascade_edge_creation(self):
        """Test cascade edge creation."""
        edge = CascadeEdge(
            source_flight_id="flight_001",
            target_flight_id="flight_002",
            cascade_type=CascadeType.SAME_TAIL,
            dependency_strength=0.8,
            expected_delay_propagation=12.5,
            aircraft_registration="VT-ABC",
            time_gap_minutes=90
        )
        
        assert edge.source_flight_id == "flight_001"
        assert edge.target_flight_id == "flight_002"
        assert edge.cascade_type == CascadeType.SAME_TAIL
        assert edge.dependency_strength == 0.8
        assert edge.expected_delay_propagation == 12.5
        assert edge.aircraft_registration == "VT-ABC"
        assert edge.time_gap_minutes == 90
    
    def test_edge_to_dict(self):
        """Test edge serialization to dictionary."""
        edge = CascadeEdge(
            source_flight_id="flight_001",
            target_flight_id="flight_002",
            cascade_type=CascadeType.RUNWAY_BANK,
            dependency_strength=0.3,
            expected_delay_propagation=5.2,
            runway="09L"
        )
        
        edge_dict = edge.to_dict()
        
        assert edge_dict["source_flight_id"] == "flight_001"
        assert edge_dict["target_flight_id"] == "flight_002"
        assert edge_dict["cascade_type"] == "runway_bank"
        assert edge_dict["dependency_strength"] == 0.3
        assert edge_dict["expected_delay_propagation"] == 5.2
        assert edge_dict["context"]["runway"] == "09L"


class TestHighImpactFlight:
    """Test cases for HighImpactFlight."""
    
    def test_high_impact_flight_creation(self):
        """Test high-impact flight creation."""
        flight = HighImpactFlight(
            flight_id="flight_001",
            flight_number="AI101",
            aircraft_registration="VT-ABC",
            airport_code="BOM",
            scheduled_time=datetime(2024, 1, 15, 10, 0),
            actual_time=datetime(2024, 1, 15, 10, 20),
            delay_minutes=20,
            impact_score=0.85,
            impact_rank=1,
            downstream_flights=15,
            total_downstream_delay=180,
            cascade_depth=4,
            severity=ImpactSeverity.HIGH
        )
        
        assert flight.flight_id == "flight_001"
        assert flight.impact_score == 0.85
        assert flight.impact_rank == 1
        assert flight.downstream_flights == 15
        assert flight.severity == ImpactSeverity.HIGH
    
    def test_high_impact_flight_to_dict(self):
        """Test high-impact flight serialization."""
        flight = HighImpactFlight(
            flight_id="flight_001",
            flight_number="AI101",
            aircraft_registration="VT-ABC",
            airport_code="BOM",
            scheduled_time=datetime(2024, 1, 15, 10, 0),
            impact_score=0.75,
            impact_rank=2,
            downstream_flights=8,
            total_downstream_delay=95.5,
            cascade_depth=3,
            centrality_scores={"pagerank": 0.05, "betweenness": 0.12},
            impact_justification=["High centrality", "Multiple downstream impacts"],
            severity=ImpactSeverity.MEDIUM
        )
        
        flight_dict = flight.to_dict()
        
        assert flight_dict["flight_id"] == "flight_001"
        assert flight_dict["impact_metrics"]["impact_score"] == 0.75
        assert flight_dict["impact_metrics"]["impact_rank"] == 2
        assert flight_dict["impact_metrics"]["severity"] == "medium"
        assert flight_dict["centrality_scores"]["pagerank"] == 0.05
        assert len(flight_dict["impact_justification"]) == 2


if __name__ == "__main__":
    pytest.main([__file__])