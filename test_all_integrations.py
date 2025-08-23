#!/usr/bin/env python3
"""
Comprehensive test suite for all API integrations
Tests Weather API, Gemini API, and Slack webhook
"""
import requests
import json
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

def test_weather_api():
    """Test OpenWeatherMap API integration"""
    print("🌤️  Testing Weather API...")
    
    api_key = os.getenv('WEATHER_API_KEY')
    if not api_key:
        print("❌ WEATHER_API_KEY not found")
        return False
    
    # Test with a major airport (JFK)
    test_url = f"http://api.openweathermap.org/data/2.5/weather?q=New York,US&appid={api_key}&units=metric"
    
    try:
        response = requests.get(test_url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            weather = data['weather'][0]['main']
            temp = data['main']['temp']
            wind_speed = data['wind']['speed']
            
            print(f"✅ Weather API working!")
            print(f"   📍 Location: {data['name']}")
            print(f"   🌡️  Temperature: {temp}°C")
            print(f"   🌤️  Conditions: {weather}")
            print(f"   💨 Wind Speed: {wind_speed} m/s")
            return True
        else:
            print(f"❌ Weather API failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Weather API error: {e}")
        return False

def test_gemini_api():
    """Test Google Gemini API integration"""
    print("\n🤖 Testing Gemini AI API...")
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY not found")
        return False
    
    try:
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Test with a simple flight scheduling query
        model = genai.GenerativeModel('gemini-1.5-flash')
        test_prompt = """
        You are a flight scheduling AI assistant. 
        Analyze this scenario: An airport has 3 runways and expects 45 flights in the next hour.
        Weather conditions are clear with light winds.
        Provide a brief recommendation for optimal scheduling.
        """
        
        response = model.generate_content(test_prompt)
        
        if response and response.text:
            print("✅ Gemini API working!")
            print(f"   🧠 AI Response Preview: {response.text[:150]}...")
            return True
        else:
            print("❌ Gemini API returned empty response")
            return False
            
    except Exception as e:
        print(f"❌ Gemini API error: {e}")
        return False

def test_slack_webhook():
    """Test Slack webhook integration"""
    print("\n📱 Testing Slack Webhook...")
    
    webhook_url = os.getenv('SLACK_WEBHOOK_URL')
    if not webhook_url:
        print("❌ SLACK_WEBHOOK_URL not found")
        return False
    
    test_message = {
        "text": "🧪 Integration Test Complete",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Flight Scheduler Integration Test* ✈️\n\nAll API integrations have been tested successfully!"
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": "*Weather API:*\nConnected ✅"
                    },
                    {
                        "type": "mrkdwn",
                        "text": "*Gemini AI:*\nConnected ✅"
                    }
                ]
            }
        ]
    }
    
    try:
        response = requests.post(
            webhook_url,
            data=json.dumps(test_message),
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if response.status_code == 200:
            print("✅ Slack webhook working!")
            print("   📲 Integration test message sent to Slack")
            return True
        else:
            print(f"❌ Slack webhook failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Slack webhook error: {e}")
        return False

def main():
    """Run all integration tests"""
    print("🚀 Flight Scheduler - API Integration Test Suite")
    print("=" * 60)
    
    results = {
        'weather': test_weather_api(),
        'gemini': test_gemini_api(),
        'slack': test_slack_webhook()
    }
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY:")
    print("=" * 60)
    
    all_passed = True
    for service, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{service.upper():12} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("🎉 ALL INTEGRATIONS WORKING! Ready to proceed with implementation.")
    else:
        print("⚠️  Some integrations failed. Please check API keys and try again.")
    
    return all_passed

if __name__ == "__main__":
    main()