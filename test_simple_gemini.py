#!/usr/bin/env python3
"""
Simple Gemini test to verify it's working.
"""

import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, '.')

from src.services.nl_router import process_query, health_check

def test_simple():
    """Test a single query."""
    
    print("🧪 Simple Gemini Test")
    print("=" * 20)
    
    # Wait a bit for rate limit to reset
    print("⏳ Waiting 10 seconds for rate limit reset...")
    time.sleep(10)
    
    # Health check
    health = health_check()
    print(f"Gemini available: {health['gemini_available']}")
    print(f"Gemini healthy: {health['gemini_healthy']}")
    
    if not health['gemini_available']:
        print("❌ Gemini not available")
        return
    
    # Test one query
    print("\n🧪 Testing query...")
    query = "What are the busiest slots at Mumbai airport?"
    
    result = process_query(query)
    
    print(f"Query: \"{query}\"")
    print(f"Provider: {result.provider.value}")
    print(f"Intent: {result.intent}")
    print(f"Confidence: {result.confidence}")
    print(f"Airport: {result.airport_code}")
    print(f"Latency: {result.latency_ms}ms")
    
    if result.provider.value == "gemini":
        print("✅ Gemini is working!")
    else:
        print("⚠️ Using fallback, not Gemini")

if __name__ == "__main__":
    test_simple()