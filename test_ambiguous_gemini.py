#!/usr/bin/env python3
"""
Test Gemini with ambiguous queries that should trigger Gemini usage.
"""

import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, '.')

from src.services.nl_router import process_query, get_nl_stats

def test_ambiguous():
    """Test queries that should use Gemini."""
    
    print("🤖 Testing Gemini with Ambiguous Queries")
    print("=" * 40)
    
    # Wait for rate limit
    print("⏳ Waiting 15 seconds for rate limit reset...")
    time.sleep(15)
    
    # Test ambiguous queries that local system should struggle with
    ambiguous_queries = [
        "Can you help me understand the situation at the airport?",
        "I need to know about that flight we discussed earlier",
        "Move it a bit earlier please",
        "What happens if I reschedule the morning departure?",
        "Show me the analysis for peak hours"
    ]
    
    print("\n🧪 Testing ambiguous queries:")
    print("-" * 30)
    
    for query in ambiguous_queries:
        print(f"\n💬 \"{query}\"")
        
        result = process_query(query)
        
        provider_emoji = "🤖" if result.provider.value == "gemini" else "🔧"
        print(f"   {provider_emoji} Provider: {result.provider.value}")
        print(f"   🎯 Intent: {result.intent}")
        print(f"   📊 Confidence: {result.confidence:.2f}")
        print(f"   ⏱️ Latency: {result.latency_ms}ms")
        
        if result.error:
            print(f"   ⚠️ Error: {result.error}")
        
        # Wait between queries to respect rate limits
        time.sleep(3)
    
    # Show stats
    stats = get_nl_stats()
    print(f"\n📊 Final Stats:")
    print(f"   Total queries: {stats['total_queries']}")
    print(f"   Gemini usage: {stats['gemini_usage_pct']:.1f}%")
    print(f"   Average confidence: {stats['avg_confidence']:.2f}")
    
    if stats['gemini_usage_pct'] > 0:
        print("✅ Gemini is being used for ambiguous queries!")
    else:
        print("⚠️ Gemini not used - queries may not be ambiguous enough")

if __name__ == "__main__":
    test_ambiguous()