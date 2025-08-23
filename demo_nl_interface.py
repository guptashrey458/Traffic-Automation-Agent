#!/usr/bin/env python3
"""
Demo script for the Natural Language Interface.

This script demonstrates how to use the NL interface to process
natural language queries about flight scheduling.
"""

import os
import sys
from datetime import datetime, date
from unittest.mock import Mock

# Add src to path for imports
sys.path.append('src')

from src.services.nl_interface import (
    NLInterface, ConversationContext, QueryIntent
)
from src.services.analytics import AnalyticsEngine, PeakAnalysis, TimeBucket, TrafficLevel
from src.services.delay_prediction import DelayRiskPredictor, DelayPrediction, DelayRiskLevel
from src.services.schedule_optimizer import ScheduleOptimizer, OptimizationResult
from src.services.whatif_simulator import WhatIfSimulator, ImpactCard
from src.services.database import FlightDatabaseService


def create_mock_services():
    """Create mock services for demonstration."""
    
    # Mock Analytics Engine
    analytics_engine = Mock(spec=AnalyticsEngine)
    mock_peak_analysis = PeakAnalysis(
        airport="BOM",
        analysis_date=date.today(),
        bucket_minutes=10,
        time_buckets=[
            TimeBucket(
                start_time=datetime(2024, 1, 1, 8, 0),
                end_time=datetime(2024, 1, 1, 8, 10),
                bucket_minutes=10,
                scheduled_departures=8,
                actual_departures=6,
                total_demand=8,
                capacity=6,
                utilization=1.33,
                overload=2,
                traffic_level=TrafficLevel.HIGH
            ),
            TimeBucket(
                start_time=datetime(2024, 1, 1, 8, 10),
                end_time=datetime(2024, 1, 1, 8, 20),
                bucket_minutes=10,
                scheduled_departures=5,
                actual_departures=5,
                total_demand=5,
                capacity=6,
                utilization=0.83,
                overload=0,
                traffic_level=TrafficLevel.MEDIUM
            )
        ],
        overload_windows=[],
        avg_utilization=0.85,
        recommendations=["Consider moving 2 flights from 08:00-08:10 slot to reduce overload"]
    )
    analytics_engine.analyze_peaks.return_value = mock_peak_analysis
    
    # Mock Delay Risk Predictor
    delay_predictor = Mock(spec=DelayRiskPredictor)
    from src.services.delay_prediction import PredictionConfidence
    mock_prediction = DelayPrediction(
        flight_id="AI2739_20240101",
        prediction_type="departure",
        delay_probability=0.65,
        risk_level=DelayRiskLevel.HIGH,
        is_likely_delayed=True,
        expected_delay_minutes=22.5,
        delay_range_min=15.0,
        delay_range_max=30.0,
        confidence=PredictionConfidence.HIGH,
        confidence_score=0.85,
        key_risk_factors=["Peak slot demand", "Aircraft turnaround time", "Weather conditions"],
        prediction_timestamp=datetime.now()
    )
    delay_predictor.predict_departure_delay.return_value = mock_prediction
    
    # Mock Schedule Optimizer
    schedule_optimizer = Mock(spec=ScheduleOptimizer)
    mock_optimization = Mock()
    mock_optimization.cost_reduction = 125.5
    mock_optimization.affected_flights = ["AI2739", "6E1234", "UK955"]
    schedule_optimizer.optimize_schedule.return_value = mock_optimization
    
    # Mock What-If Simulator
    whatif_simulator = Mock(spec=WhatIfSimulator)
    mock_impact = Mock()
    mock_impact.delay_delta = -8.5  # 8.5 minutes improvement
    mock_impact.affected_flights = ["AI2739", "6E1234"]
    whatif_simulator.simulate_scenario.return_value = mock_impact
    
    # Mock Database Service
    database_service = Mock(spec=FlightDatabaseService)
    
    return {
        'analytics_engine': analytics_engine,
        'delay_predictor': delay_predictor,
        'schedule_optimizer': schedule_optimizer,
        'whatif_simulator': whatif_simulator,
        'database_service': database_service
    }


def demo_nl_interface():
    """Demonstrate the natural language interface capabilities."""
    
    print("🛩️  Flight Scheduling Natural Language Interface Demo")
    print("=" * 60)
    
    # Create mock services
    services = create_mock_services()
    
    # Initialize NL interface
    nl_interface = NLInterface(**services)
    
    # Create conversation context
    context = ConversationContext(
        user_id="demo_user",
        session_id="demo_session",
        preferred_airport="BOM"
    )
    
    # Demo queries
    demo_queries = [
        "What are the busiest 10-minute slots at BOM?",
        "What's the delay risk for AI 2739?",
        "Move AI 2739 by +15 minutes—what's the impact?",
        "Show me operational constraints for BOM",
        "Optimize schedule for Mumbai airport"
    ]
    
    print("\n🔍 Processing Natural Language Queries:")
    print("-" * 40)
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n{i}. Query: \"{query}\"")
        
        # Process the query
        response = nl_interface.process_query(query, context)
        
        # Display results
        print(f"   Intent: {response.intent.value}")
        print(f"   Confidence: {response.confidence_score:.1%}")
        
        if response.parameters.airport_code:
            print(f"   Airport: {response.parameters.airport_code}")
        if response.parameters.flight_number:
            print(f"   Flight: {response.parameters.flight_number}")
        if response.parameters.time_shift_minutes:
            print(f"   Time Shift: {response.parameters.time_shift_minutes:+d} minutes")
        if response.parameters.time_bucket_minutes:
            print(f"   Time Bucket: {response.parameters.time_bucket_minutes} minutes")
        
        print(f"   Response: {response.natural_language_response}")
        
        if response.suggestions:
            print(f"   Suggestions: {', '.join(response.suggestions[:2])}")
    
    print("\n" + "=" * 60)
    print("✅ Demo completed successfully!")
    
    # Show intent classification accuracy
    print("\n📊 Intent Classification Examples:")
    print("-" * 30)
    
    test_queries = [
        ("Which time slots have overload at BOM?", QueryIntent.ASK_PEAKS),
        ("Risk assessment for flight AI 2739", QueryIntent.ASK_RISK),
        ("What if I move 6E1234 by -10 minutes?", QueryIntent.ASK_WHATIF),
        ("Minimize delays at Delhi airport", QueryIntent.ASK_OPTIMIZE),
        ("What are the turnaround time constraints?", QueryIntent.ASK_CONSTRAINTS)
    ]
    
    for query, expected_intent in test_queries:
        classified_intent = nl_interface.classify_intent(query)
        status = "✅" if classified_intent == expected_intent else "❌"
        print(f"{status} \"{query}\" → {classified_intent.value}")
    
    print("\n🎯 Parameter Extraction Examples:")
    print("-" * 30)
    
    param_queries = [
        "What are the busiest slots at DEL?",
        "Delay risk for AI 2739",
        "Move flight by +20 minutes",
        "Show 5-minute bucket analysis"
    ]
    
    for query in param_queries:
        intent = nl_interface.classify_intent(query)
        params = nl_interface.extract_parameters(query, intent, context)
        
        extracted = []
        if params.airport_code:
            extracted.append(f"Airport: {params.airport_code}")
        if params.flight_number:
            extracted.append(f"Flight: {params.flight_number}")
        if params.time_shift_minutes is not None:
            extracted.append(f"Shift: {params.time_shift_minutes:+d}min")
        if params.time_bucket_minutes:
            extracted.append(f"Bucket: {params.time_bucket_minutes}min")
        
        print(f"📝 \"{query}\"")
        print(f"   → {', '.join(extracted) if extracted else 'No parameters'}")


if __name__ == "__main__":
    try:
        demo_nl_interface()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()