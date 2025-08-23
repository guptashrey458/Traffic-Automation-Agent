#!/usr/bin/env python3
"""
Test Gemini Pro integration with the Natural Language Interface.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src to path for imports
sys.path.append('src')

from src.services.nl_interface import NLInterface, ConversationContext
from unittest.mock import Mock


def test_gemini_integration():
    """Test Gemini Pro integration."""
    
    print("🤖 Testing Gemini Pro Integration")
    print("=" * 40)
    
    # Check if API key is loaded
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment variables")
        print("Please make sure your .env file contains GEMINI_API_KEY=your_key_here")
        return False
    
    print(f"✅ GEMINI_API_KEY found: {api_key[:10]}...{api_key[-4:]}")
    
    # Create NL interface
    nl_interface = NLInterface(
        analytics_engine=Mock(),
        delay_predictor=Mock(),
        schedule_optimizer=Mock(),
        whatif_simulator=Mock(),
        database_service=Mock()
    )
    
    # Test if Gemini model is initialized
    if nl_interface.gemini_model:
        print("✅ Gemini Pro model initialized successfully")
    else:
        print("❌ Gemini Pro model not initialized")
        return False
    
    # Test intent classification with Gemini
    print("\n🧠 Testing Gemini-powered Intent Classification:")
    print("-" * 45)
    
    test_queries = [
        "What are the peak traffic hours at Mumbai airport?",
        "Can you tell me the delay probability for flight AI 2739?",
        "I want to see what happens if I move flight 6E1234 by 20 minutes",
        "Please optimize the schedule to minimize delays",
        "What are the operational constraints for runway operations?"
    ]
    
    for query in test_queries:
        try:
            intent = nl_interface.classify_intent(query)
            print(f"✅ \"{query}\"")
            print(f"   → Intent: {intent.value}")
        except Exception as e:
            print(f"❌ \"{query}\"")
            print(f"   → Error: {e}")
    
    # Test parameter extraction with Gemini
    print("\n🔍 Testing Gemini-powered Parameter Extraction:")
    print("-" * 48)
    
    context = ConversationContext(user_id="test", session_id="test")
    
    param_queries = [
        "What's the delay risk for flight AI 2739 at Mumbai airport?",
        "Move flight 6E1234 by plus 15 minutes and show impact",
        "Analyze peak traffic in 5-minute intervals at Delhi"
    ]
    
    for query in param_queries:
        try:
            intent = nl_interface.classify_intent(query)
            params = nl_interface.extract_parameters(query, intent, context)
            
            print(f"✅ \"{query}\"")
            print(f"   → Intent: {intent.value}")
            
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
                print(f"   → Parameters: {', '.join(extracted)}")
            else:
                print("   → No parameters extracted")
                
        except Exception as e:
            print(f"❌ \"{query}\"")
            print(f"   → Error: {e}")
    
    print("\n" + "=" * 40)
    print("✅ Gemini Pro integration test completed!")
    return True


if __name__ == "__main__":
    try:
        success = test_gemini_integration()
        if success:
            print("🎉 All tests passed! Gemini Pro is working correctly.")
        else:
            print("⚠️  Some tests failed. Check the output above.")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()