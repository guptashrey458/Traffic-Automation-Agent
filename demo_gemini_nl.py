#!/usr/bin/env python3
"""
Demo of Gemini-powered Natural Language Interface.
Shows actual Gemini usage with rate limiting and intelligent routing.
"""

import os
import sys
import time
from datetime import datetime

# Add src to path
sys.path.insert(0, '.')

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.services.nl_router import process_query, get_nl_stats, health_check


def demo_gemini_nl():
    """Demo Gemini-powered NL processing."""
    
    print("🤖 Gemini-Powered Natural Language Interface Demo")
    print("=" * 55)
    
    # Health check first
    health = health_check()
    print(f"🏥 System Health:")
    print(f"   Local processor: {health['local_processor']}")
    print(f"   Gemini available: {'✅' if health['gemini_available'] else '❌'}")
    print(f"   Gemini healthy: {'✅' if health['gemini_healthy'] else '❌'}")
    
    if not health['gemini_available']:
        print("\n⚠️ Gemini not available - will use local processing only")
        print("   Set GOOGLE_API_KEY or GEMINI_API_KEY to enable Gemini")
    
    print(f"\n🧪 Processing Test Queries:")
    print("-" * 30)
    
    # Test queries designed to showcase different scenarios
    test_scenarios = [
        {
            "name": "Clear Intent (Local Should Handle)",
            "queries": [
                "What are the busiest slots at BOM?",
                "Delay risk for AI 2739"
            ]
        },
        {
            "name": "Ambiguous Intent (Gemini Should Help)",
            "queries": [
                "Can you help me understand the situation at Mumbai?",
                "I need to know about that flight we discussed",
                "Move it a bit earlier please"
            ]
        },
        {
            "name": "Complex Parameters (Gemini Should Extract)",
            "queries": [
                "What happens if I move flight AI two seven three nine by plus fifteen minutes?",
                "Show me peak traffic analysis for Mumbai airport in five minute intervals",
                "I want to reschedule the Air India flight to Delhi by moving it twenty minutes later"
            ]
        }
    ]
    
    total_queries = 0
    gemini_used = 0
    
    for scenario in test_scenarios:
        print(f"\n📋 {scenario['name']}")
        print("-" * len(scenario['name']) + "---")
        
        for query in scenario['queries']:
            print(f"\n💬 Query: \"{query}\"")
            
            # Process query
            start_time = time.time()
            result = process_query(query)
            processing_time = time.time() - start_time
            
            # Display results
            provider_emoji = "🤖" if result.provider.value == "gemini" else "🔧"
            cache_emoji = "💾" if result.cache_hit else "🆕"
            
            print(f"   {provider_emoji}{cache_emoji} Provider: {result.provider.value}")
            print(f"   🎯 Intent: {result.intent}")
            print(f"   📊 Confidence: {result.confidence:.2f}")
            print(f"   ⏱️ Latency: {result.latency_ms}ms")
            
            # Show extracted parameters
            params = []
            if result.airport_code:
                params.append(f"Airport: {result.airport_code}")
            if result.flight_number:
                params.append(f"Flight: {result.flight_number}")
            if result.time_shift_minutes is not None:
                params.append(f"Time shift: {result.time_shift_minutes:+d}min")
            if result.time_bucket_minutes:
                params.append(f"Bucket: {result.time_bucket_minutes}min")
            
            if params:
                print(f"   📝 Parameters: {', '.join(params)}")
            
            if result.error:
                print(f"   ⚠️ Error: {result.error}")
            
            total_queries += 1
            if result.provider.value == "gemini":
                gemini_used += 1
            
            # Small delay to respect rate limits
            time.sleep(0.5)
    
    # Show final statistics
    print(f"\n📊 Processing Statistics")
    print("=" * 25)
    
    stats = get_nl_stats()
    
    print(f"Total queries processed: {stats['total_queries']}")
    print(f"Gemini usage: {stats['gemini_usage_pct']:.1f}%")
    print(f"Cache hit rate: {stats['cache_hit_pct']:.1f}%")
    print(f"Average latency: {stats['avg_latency_ms']}ms")
    print(f"Average confidence: {stats['avg_confidence']}")
    print(f"Gemini healthy: {'✅' if stats['gemini_healthy'] else '❌'}")
    
    # Recommendations
    print(f"\n💡 Recommendations")
    print("-" * 17)
    
    if stats['gemini_usage_pct'] > 50:
        print("🎯 Great! Gemini is handling complex queries effectively")
    elif stats['gemini_usage_pct'] > 20:
        print("👍 Good balance between local and Gemini processing")
    elif stats['gemini_usage_pct'] > 0:
        print("🔧 Mostly using local processing - consider more complex queries")
    else:
        print("⚠️ Gemini not used - check API key and rate limits")
    
    if stats['cache_hit_pct'] > 30:
        print("💾 Good cache utilization - reducing API costs")
    
    if stats['avg_confidence'] > 0.8:
        print("🎯 High confidence in results - system is working well")
    elif stats['avg_confidence'] > 0.6:
        print("👍 Reasonable confidence - consider tuning thresholds")
    else:
        print("⚠️ Low confidence - may need better training data")


def demo_conversation_flow():
    """Demo conversational flow with context."""
    
    print(f"\n💭 Conversational Flow Demo")
    print("=" * 30)
    
    # Simulate a conversation
    conversation = [
        ("What are the busiest slots at Mumbai?", {"preferred_airport": None}),
        ("Show me 5-minute buckets", {"preferred_airport": "BOM"}),
        ("What about the delay risk?", {"preferred_airport": "BOM", "last_intent": "ask_peaks"}),
        ("For flight AI 2739", {"preferred_airport": "BOM", "last_intent": "ask_risk"}),
        ("Move it by plus ten minutes", {"preferred_airport": "BOM", "flight_context": "AI2739"})
    ]
    
    print("🗣️ Simulated Conversation:")
    print("-" * 25)
    
    for i, (query, context) in enumerate(conversation, 1):
        print(f"\n{i}. User: \"{query}\"")
        
        result = process_query(query, context)
        
        provider_emoji = "🤖" if result.provider.value == "gemini" else "🔧"
        print(f"   {provider_emoji} Intent: {result.intent} (confidence: {result.confidence:.2f})")
        
        # Show context understanding
        if result.airport_code and not any("BOM" in query or "Mumbai" in query for query in [query]):
            print(f"   🧠 Inferred airport from context: {result.airport_code}")
        
        if result.flight_number and "AI" not in query:
            print(f"   🧠 Inferred flight from context: {result.flight_number}")
        
        time.sleep(0.5)


def main():
    """Run the demo."""
    
    try:
        # Main demo
        demo_gemini_nl()
        
        # Conversation demo
        demo_conversation_flow()
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"   Timestamp: {datetime.now().isoformat()}")
        
        # Final health check
        final_health = health_check()
        if final_health['gemini_healthy']:
            print("   ✅ Gemini Pro is working correctly")
        else:
            print("   ⚠️ Gemini Pro may have issues (check rate limits)")
        
    except KeyboardInterrupt:
        print(f"\n\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()