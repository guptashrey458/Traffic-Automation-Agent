#!/usr/bin/env python3
"""
Quick test script to verify Slack webhook functionality
"""
import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_slack_webhook():
    """Test the Slack webhook with a simple message"""
    webhook_url = os.getenv('SLACK_WEBHOOK_URL')
    
    if not webhook_url:
        print("❌ SLACK_WEBHOOK_URL not found in environment variables")
        return False
    
    print(f"🔗 Testing webhook: {webhook_url[:50]}...")
    
    # Test message payload
    test_message = {
        "text": "🚀 Flight Scheduler Alert System Test",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Flight Scheduler System Test* ✈️\n\nThis is a test message from your Traffic Automation Agent!"
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": "*Status:*\nWebhook Connected ✅"
                    },
                    {
                        "type": "mrkdwn",
                        "text": "*System:*\nAlert System Ready 🔔"
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
            print("✅ Slack webhook test successful!")
            print("📱 Check your Slack channel for the test message")
            return True
        else:
            print(f"❌ Webhook failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Error sending to Slack: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Slack Webhook Integration...")
    print("=" * 50)
    
    success = test_slack_webhook()
    
    print("=" * 50)
    if success:
        print("🎉 Slack integration is working correctly!")
    else:
        print("🔧 Please check your webhook URL and try again")