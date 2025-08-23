#!/usr/bin/env python3
"""
Gemini health check script.
Verifies that Gemini Pro is working correctly.
"""

import os
import sys
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


def check_gemini():
    """Check Gemini Pro health."""
    
    print("🔍 Gemini Pro Health Check")
    print("=" * 30)
    
    # Check if library is available
    if not GEMINI_AVAILABLE:
        print("❌ google-generativeai library not installed")
        print("   Run: pip install google-generativeai")
        return False
    
    print("✅ google-generativeai library available")
    
    # Check API key
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ No API key found")
        print("   Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable")
        return False
    
    print(f"✅ API key found: {api_key[:10]}...{api_key[-4:]}")
    
    # Check model configuration
    model_name = os.getenv("NL_GEMINI_MODEL", "gemini-1.5-pro")
    rpm = os.getenv("NL_GEMINI_RPM", "24")
    use_gemini = os.getenv("NL_USE_GEMINI", "true")
    
    print(f"✅ Model: {model_name}")
    print(f"✅ Rate limit: {rpm} RPM")
    print(f"✅ Enabled: {use_gemini}")
    
    # Test API connection
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        
        print("\n🧪 Testing API connection...")
        
        # Simple test
        test_prompt = f'Reply with JSON only: {{"status": "ok", "timestamp": "{datetime.now().isoformat()}", "model": "{model_name}"}}'
        
        response = model.generate_content(
            test_prompt,
            generation_config={
                "response_mime_type": "application/json"
            }
        )
        
        result = json.loads(response.text)
        
        if result.get("status") == "ok":
            print("✅ API connection successful")
            print(f"   Response: {json.dumps(result, indent=2)}")
            
            # Test NL processing
            print("\n🧠 Testing NL processing...")
            
            nl_prompt = """
            Extract intent and parameters for this flight scheduling query.
            
            Query: "What are the busiest slots at BOM airport?"
            
            Return JSON with:
            - intent: one of [ask_peaks, ask_risk, ask_whatif, ask_optimize, ask_constraints, ask_status, unknown]
            - airport_code: 3-letter code if found
            - confidence: 0.0 to 1.0
            """
            
            nl_response = model.generate_content(
                nl_prompt,
                generation_config={
                    "response_mime_type": "application/json"
                }
            )
            
            nl_result = json.loads(nl_response.text)
            print(f"✅ NL processing successful")
            print(f"   Intent: {nl_result.get('intent')}")
            print(f"   Airport: {nl_result.get('airport_code')}")
            print(f"   Confidence: {nl_result.get('confidence')}")
            
            return True
            
        else:
            print("❌ Unexpected response format")
            print(f"   Got: {result}")
            return False
            
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing failed: {e}")
        print(f"   Raw response: {response.text}")
        return False
        
    except Exception as e:
        if "429" in str(e):
            print("⚠️ Rate limit exceeded (429)")
            print("   This is expected if you've been testing frequently")
            print("   Wait a minute and try again")
            return False
        else:
            print(f"❌ API call failed: {e}")
            return False


def check_nl_router():
    """Check the NL Router integration."""
    
    print("\n🔄 NL Router Integration Check")
    print("=" * 35)
    
    try:
        from src.services.nl_router import nl_router, process_query
        
        # Health check
        health = nl_router.health_check()
        print(f"✅ NL Router initialized")
        print(f"   Local processor: {health['local_processor']}")
        print(f"   Gemini available: {health['gemini_available']}")
        print(f"   Gemini healthy: {health['gemini_healthy']}")
        
        # Test query
        print("\n🧪 Testing query processing...")
        
        test_queries = [
            "What are the busiest slots at BOM?",
            "Delay risk for AI 2739",
            "Move flight by +10 minutes"
        ]
        
        for query in test_queries:
            result = process_query(query)
            provider_emoji = "🤖" if result.provider.value == "gemini" else "🔧"
            print(f"{provider_emoji} \"{query}\"")
            print(f"   → Intent: {result.intent}")
            print(f"   → Provider: {result.provider.value}")
            print(f"   → Confidence: {result.confidence:.2f}")
            print(f"   → Latency: {result.latency_ms}ms")
            
            if result.airport_code:
                print(f"   → Airport: {result.airport_code}")
            if result.flight_number:
                print(f"   → Flight: {result.flight_number}")
            if result.time_shift_minutes is not None:
                print(f"   → Time shift: {result.time_shift_minutes:+d}min")
        
        # Show stats
        stats = nl_router.get_stats()
        if stats['total_queries'] > 0:
            print(f"\n📊 Processing Stats:")
            print(f"   Total queries: {stats['total_queries']}")
            print(f"   Gemini usage: {stats['gemini_usage_pct']:.1f}%")
            print(f"   Cache hits: {stats['cache_hit_pct']:.1f}%")
            print(f"   Avg latency: {stats['avg_latency_ms']}ms")
            print(f"   Avg confidence: {stats['avg_confidence']}")
        
        return True
        
    except Exception as e:
        print(f"❌ NL Router test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all health checks."""
    
    print("🛩️ Flight Scheduling NL System Health Check")
    print("=" * 50)
    
    gemini_ok = check_gemini()
    router_ok = check_nl_router()
    
    print("\n" + "=" * 50)
    
    if gemini_ok and router_ok:
        print("🎉 All systems healthy! Gemini Pro is working correctly.")
        print("\n💡 Tips:")
        print("   - Set NL_USE_GEMINI=true to enable Gemini")
        print("   - Set NL_GEMINI_RPM=24 for safe rate limiting")
        print("   - Use gemini-1.5-pro for best accuracy")
        return True
    else:
        print("⚠️ Some issues found. Check the output above.")
        if not gemini_ok:
            print("   - Gemini Pro is not working")
        if not router_ok:
            print("   - NL Router has issues")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)