#!/usr/bin/env python3
"""
Simple demo of the Natural Language Interface core functionality.

This demo focuses on intent classification and parameter extraction
without requiring complex mock data structures.
"""

import sys
sys.path.append('src')

from src.services.nl_interface import (
    NLInterface, ConversationContext, QueryIntent, QueryParameters
)
from unittest.mock import Mock


def demo_intent_classification():
    """Demo intent classification capabilities."""
    
    print("🧠 Intent Classification Demo")
    print("=" * 40)
    
    # Create minimal NL interface for classification only
    nl_interface = NLInterface(
        analytics_engine=Mock(),
        delay_predictor=Mock(),
        schedule_optimizer=Mock(),
        whatif_simulator=Mock(),
        database_service=Mock()
    )
    
    # Test queries with expected intents
    test_cases = [
        # Peak traffic queries
        ("What are the busiest slots at BOM?", QueryIntent.ASK_PEAKS),
        ("Show me peak traffic analysis", QueryIntent.ASK_PEAKS),
        ("Which time slots have overload?", QueryIntent.ASK_PEAKS),
        ("Demand heatmap for Delhi airport", QueryIntent.ASK_PEAKS),
        
        # Delay risk queries
        ("What's the delay risk for AI 2739?", QueryIntent.ASK_RISK),
        ("Probability of delay for flight 6E1234", QueryIntent.ASK_RISK),
        ("Risk assessment for AI 2739", QueryIntent.ASK_RISK),
        ("Is AI 2739 likely to be delayed?", QueryIntent.ASK_RISK),
        
        # What-if queries
        ("Move AI 2739 by +10 minutes—impact?", QueryIntent.ASK_WHATIF),
        ("What if I shift flight 6E1234 by -15 minutes?", QueryIntent.ASK_WHATIF),
        ("Impact of moving AI 2739 earlier", QueryIntent.ASK_WHATIF),
        
        # Optimization queries
        ("Optimize schedule for BOM", QueryIntent.ASK_OPTIMIZE),
        ("Minimize delays at Delhi", QueryIntent.ASK_OPTIMIZE),
        ("Improve the schedule", QueryIntent.ASK_OPTIMIZE),
        
        # Constraints queries
        ("What are the operational rules?", QueryIntent.ASK_CONSTRAINTS),
        ("Show me constraints", QueryIntent.ASK_CONSTRAINTS),
        ("Turnaround time restrictions", QueryIntent.ASK_CONSTRAINTS),
        
        # Status queries
        ("Flight status for AI 2739", QueryIntent.ASK_STATUS),
        ("Show flight AI 2739", QueryIntent.ASK_STATUS),
    ]
    
    correct = 0
    total = len(test_cases)
    
    for query, expected_intent in test_cases:
        classified_intent = nl_interface.classify_intent(query)
        is_correct = classified_intent == expected_intent
        
        if is_correct:
            correct += 1
            status = "✅"
        else:
            status = "❌"
        
        print(f"{status} \"{query}\"")
        print(f"   Expected: {expected_intent.value}")
        print(f"   Got: {classified_intent.value}")
        print()
    
    accuracy = correct / total * 100
    print(f"🎯 Intent Classification Accuracy: {accuracy:.1f}% ({correct}/{total})")
    return accuracy


def demo_parameter_extraction():
    """Demo parameter extraction capabilities."""
    
    print("\n🔍 Parameter Extraction Demo")
    print("=" * 40)
    
    nl_interface = NLInterface(
        analytics_engine=Mock(),
        delay_predictor=Mock(),
        schedule_optimizer=Mock(),
        whatif_simulator=Mock(),
        database_service=Mock()
    )
    
    context = ConversationContext(
        user_id="demo",
        session_id="demo",
        preferred_airport="BOM"
    )
    
    test_cases = [
        # Airport code extraction
        ("What are the busiest slots at DEL?", {"airport_code": "DEL"}),
        ("Peak traffic at BOM airport", {"airport_code": "BOM"}),
        
        # Flight number extraction
        ("Delay risk for AI 2739", {"flight_number": "AI2739"}),
        ("Status of flight 6E 1234", {"flight_number": "6E1234"}),
        
        # Time shift extraction
        ("Move AI 2739 by +10 minutes", {"time_shift_minutes": 10}),
        ("Shift flight by -15min", {"time_shift_minutes": -15}),
        ("Move by +30m", {"time_shift_minutes": 30}),
        
        # Time bucket extraction
        ("Show 5min bucket analysis", {"time_bucket_minutes": 5}),
        ("Peak traffic in 10 minute buckets", {"time_bucket_minutes": 10}),
        
        # Multiple parameters
        ("Move AI 2739 at BOM by +20 minutes", {
            "airport_code": "BOM", 
            "flight_number": "AI2739", 
            "time_shift_minutes": 20
        }),
    ]
    
    for query, expected_params in test_cases:
        intent = nl_interface.classify_intent(query)
        params = nl_interface.extract_parameters(query, intent, context)
        
        print(f"📝 \"{query}\"")
        
        # Check each expected parameter
        all_correct = True
        for param_name, expected_value in expected_params.items():
            actual_value = getattr(params, param_name)
            if actual_value == expected_value:
                print(f"   ✅ {param_name}: {actual_value}")
            else:
                print(f"   ❌ {param_name}: expected {expected_value}, got {actual_value}")
                all_correct = False
        
        if all_correct:
            print("   🎯 All parameters extracted correctly!")
        print()


def demo_conversation_context():
    """Demo conversation context and memory."""
    
    print("\n💭 Conversation Context Demo")
    print("=" * 40)
    
    nl_interface = NLInterface(
        analytics_engine=Mock(),
        delay_predictor=Mock(),
        schedule_optimizer=Mock(),
        whatif_simulator=Mock(),
        database_service=Mock()
    )
    
    context = ConversationContext(
        user_id="demo",
        session_id="demo"
    )
    
    # Simulate a conversation
    conversation = [
        "What are the busiest slots at BOM?",
        "Show me 5-minute buckets",  # Should remember BOM
        "What about DEL airport?",
        "Risk for AI 2739",  # Should remember DEL
        "Move that flight by +10 minutes"  # Should remember AI 2739
    ]
    
    print("🗣️  Simulated Conversation:")
    print("-" * 25)
    
    for i, query in enumerate(conversation, 1):
        intent = nl_interface.classify_intent(query)
        params = nl_interface.extract_parameters(query, intent, context)
        
        # Update context (simplified)
        nl_interface.update_context(context, query, intent, params)
        
        print(f"{i}. User: \"{query}\"")
        print(f"   Intent: {intent.value}")
        
        extracted = []
        if params.airport_code:
            extracted.append(f"Airport: {params.airport_code}")
        if params.flight_number:
            extracted.append(f"Flight: {params.flight_number}")
        if params.time_shift_minutes is not None:
            extracted.append(f"Shift: {params.time_shift_minutes:+d}min")
        if params.time_bucket_minutes:
            extracted.append(f"Bucket: {params.time_bucket_minutes}min")
        
        if extracted:
            print(f"   Parameters: {', '.join(extracted)}")
        
        print(f"   Context: Airport={context.preferred_airport}, Bucket={context.preferred_time_bucket}min")
        print()


def main():
    """Run all demos."""
    
    print("🛩️  Natural Language Interface - Core Functionality Demo")
    print("=" * 65)
    
    try:
        # Run intent classification demo
        accuracy = demo_intent_classification()
        
        # Run parameter extraction demo
        demo_parameter_extraction()
        
        # Run conversation context demo
        demo_conversation_context()
        
        print("\n" + "=" * 65)
        print("✅ All demos completed successfully!")
        print(f"🎯 Overall Intent Classification Accuracy: {accuracy:.1f}%")
        
        if accuracy >= 90:
            print("🏆 Excellent performance!")
        elif accuracy >= 80:
            print("👍 Good performance!")
        else:
            print("⚠️  Consider improving intent patterns")
            
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()