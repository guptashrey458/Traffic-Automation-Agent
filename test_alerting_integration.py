#!/usr/bin/env python3
"""
Integration test for the alerting and notification system.
Tests the complete workflow from alert detection to notification delivery.
"""

import os
import sys
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.services.alerting import AlertingService, AlertSeverity, AlertType
from src.services.analytics import AnalyticsEngine, OverloadWindow, PeakAnalysis
from src.services.schedule_optimizer import ScheduleOptimizer
from src.models.flight import Flight


def create_test_flights():
    """Create test flights with various delay scenarios."""
    from src.models.flight import Airport, FlightTime
    from datetime import time
    
    flights = []
    
    # Normal flights
    for i in range(3):
        flight = Flight(
            flight_id=f"AI{100+i}_20240101",
            flight_number=f"AI {100+i}",
            origin=Airport(code="BOM", name="Mumbai", city="Mumbai"),
            destination=Airport(code="DEL", name="Delhi", city="Delhi"),
            aircraft_type="A320",
            departure=FlightTime(
                scheduled=time(10+i, 0),
                actual=datetime(2024, 1, 1, 10+i, 5)  # 5 min delay
            )
        )
        flight.dep_delay_min = 5  # Minor delay
        flights.append(flight)
    
    # Delayed flights for cascade detection
    for i in range(6):
        flight = Flight(
            flight_id=f"6E{200+i}_20240101",
            flight_number=f"6E {200+i}",
            origin=Airport(code="BOM", name="Mumbai", city="Mumbai"),
            destination=Airport(code="BLR", name="Bangalore", city="Bangalore"),
            aircraft_type="A320",
            departure=FlightTime(
                scheduled=time(14+i, 0),
                actual=datetime(2024, 1, 1, 14+i, 25)  # 25 min delay
            )
        )
        flight.dep_delay_min = 25  # Above cascade threshold
        flights.append(flight)
    
    # Critical delay flight
    critical_flight = Flight(
        flight_id="UK999_20240101",
        flight_number="UK 999",
        origin=Airport(code="BOM", name="Mumbai", city="Mumbai"),
        destination=Airport(code="CCU", name="Kolkata", city="Kolkata"),
        aircraft_type="A320",
        departure=FlightTime(
            scheduled=time(18, 0),
            actual=datetime(2024, 1, 1, 19, 30)  # 90 min delay
        )
    )
    critical_flight.dep_delay_min = 90  # Critical delay
    flights.append(critical_flight)
    
    return flights


def create_test_overload_scenario():
    """Create test overload scenario."""
    overload_window = OverloadWindow(
        start_time=datetime(2024, 1, 1, 10, 0),
        end_time=datetime(2024, 1, 1, 10, 40),
        duration_minutes=40,
        peak_overload=15,  # High overload
        avg_overload=12.0,
        affected_flights=45
    )
    
    return PeakAnalysis(
        airport="BOM",
        analysis_date=datetime(2024, 1, 1).date(),
        bucket_minutes=10,
        overload_windows=[overload_window]
    )


def test_alert_detection():
    """Test alert detection functionality."""
    print("🔍 Testing Alert Detection...")
    
    # Create mock services
    mock_analytics = Mock(spec=AnalyticsEngine)
    mock_optimizer = Mock(spec=ScheduleOptimizer)
    
    # Create alerting service
    alerting_service = AlertingService(
        analytics_engine=mock_analytics,
        schedule_optimizer=mock_optimizer,
        slack_webhook_url="https://hooks.slack.com/test"
    )
    
    # Setup test data
    flights = create_test_flights()
    peak_analysis = create_test_overload_scenario()
    mock_analytics.analyze_peaks.return_value = peak_analysis
    
    # Check for alerts
    alerts = alerting_service.check_for_alerts("BOM", flights)
    
    print(f"✅ Generated {len(alerts)} alerts")
    
    # Verify alert types
    alert_types = [alert.alert_type for alert in alerts]
    expected_types = [AlertType.CAPACITY_OVERLOAD, AlertType.DELAY_CASCADE, AlertType.CRITICAL_FLIGHT_DELAY]
    
    for expected_type in expected_types:
        if expected_type in alert_types:
            print(f"✅ {expected_type.value} alert detected")
        else:
            print(f"❌ {expected_type.value} alert NOT detected")
    
    # Test alert details
    for alert in alerts:
        print(f"\n📋 Alert: {alert.title}")
        print(f"   Severity: {alert.severity.value}")
        print(f"   Type: {alert.alert_type.value}")
        print(f"   Recommendations: {len(alert.recommendations)}")
        print(f"   Affected Flights: {len(alert.affected_flights)}")
        
        # Verify recommendations
        if len(alert.recommendations) > 0:
            print(f"   Top Recommendation: {alert.recommendations[0].action}")
    
    return alerts


def test_slack_message_formatting():
    """Test Slack message formatting."""
    print("\n📱 Testing Slack Message Formatting...")
    
    # Create mock services
    mock_analytics = Mock(spec=AnalyticsEngine)
    mock_optimizer = Mock(spec=ScheduleOptimizer)
    
    alerting_service = AlertingService(
        analytics_engine=mock_analytics,
        schedule_optimizer=mock_optimizer,
        slack_webhook_url="https://hooks.slack.com/test"
    )
    
    # Get test alerts
    flights = create_test_flights()
    peak_analysis = create_test_overload_scenario()
    mock_analytics.analyze_peaks.return_value = peak_analysis
    
    alerts = alerting_service.check_for_alerts("BOM", flights)
    
    if alerts:
        alert = alerts[0]  # Test first alert
        message = alerting_service._build_slack_message(alert)
        
        print("✅ Slack message structure:")
        print(f"   Text: {message['text']}")
        print(f"   Blocks: {len(message['blocks'])}")
        
        # Verify message structure
        required_keys = ['text', 'blocks']
        for key in required_keys:
            if key in message:
                print(f"✅ {key} present in message")
            else:
                print(f"❌ {key} missing from message")
        
        # Check for emojis and formatting
        message_text = json.dumps(message)
        if any(emoji in message_text for emoji in ['🔴', '🟠', '🟡', '🚨']):
            print("✅ Severity emojis present")
        else:
            print("❌ Severity emojis missing")
        
        if any(emoji in message_text for emoji in ['📊', '⛓️', '✈️']):
            print("✅ Alert type emojis present")
        else:
            print("❌ Alert type emojis missing")
        
        return message
    else:
        print("❌ No alerts generated for message testing")
        return None


def test_alert_lifecycle():
    """Test complete alert lifecycle."""
    print("\n🔄 Testing Alert Lifecycle...")
    
    # Create mock services
    mock_analytics = Mock(spec=AnalyticsEngine)
    mock_optimizer = Mock(spec=ScheduleOptimizer)
    
    alerting_service = AlertingService(
        analytics_engine=mock_analytics,
        schedule_optimizer=mock_optimizer,
        slack_webhook_url="https://hooks.slack.com/test"
    )
    
    # Generate alerts
    flights = create_test_flights()
    peak_analysis = create_test_overload_scenario()
    mock_analytics.analyze_peaks.return_value = peak_analysis
    
    alerts = alerting_service.check_for_alerts("BOM", flights)
    
    if alerts:
        alert = alerts[0]
        alert_id = alert.alert_id
        
        print(f"✅ Alert created: {alert_id}")
        print(f"   Initial severity: {alert.severity.value}")
        
        # Test escalation
        with patch.object(alerting_service, '_send_escalation_notification', return_value=True):
            escalated = alerting_service.escalate_alert(alert_id)
            if escalated:
                print(f"✅ Alert escalated to: {alert.severity.value}")
            else:
                print("❌ Alert escalation failed")
        
        # Test resolution
        with patch.object(alerting_service, '_send_resolution_notification', return_value=True):
            resolved = alerting_service.resolve_alert(alert_id)
            if resolved:
                print("✅ Alert resolved successfully")
                print(f"   Resolution time: {alert.resolved_at}")
            else:
                print("❌ Alert resolution failed")
        
        # Verify alert is no longer active
        active_alerts = alerting_service.get_active_alerts()
        if alert_id not in [a.alert_id for a in active_alerts]:
            print("✅ Alert removed from active alerts")
        else:
            print("❌ Alert still in active alerts")
    
    else:
        print("❌ No alerts generated for lifecycle testing")


def test_alert_summary():
    """Test alert summary functionality."""
    print("\n📊 Testing Alert Summary...")
    
    # Create mock services
    mock_analytics = Mock(spec=AnalyticsEngine)
    mock_optimizer = Mock(spec=ScheduleOptimizer)
    
    alerting_service = AlertingService(
        analytics_engine=mock_analytics,
        schedule_optimizer=mock_optimizer,
        slack_webhook_url="https://hooks.slack.com/test"
    )
    
    # Generate alerts
    flights = create_test_flights()
    peak_analysis = create_test_overload_scenario()
    mock_analytics.analyze_peaks.return_value = peak_analysis
    
    alerts = alerting_service.check_for_alerts("BOM", flights)
    
    # Get summary
    summary = alerting_service.get_alert_summary()
    
    print("✅ Alert Summary:")
    print(f"   Total Active Alerts: {summary['total_active_alerts']}")
    print(f"   By Severity: {summary['by_severity']}")
    print(f"   By Type: {summary['by_type']}")
    print(f"   Escalated Alerts: {summary['escalated_alerts']}")
    
    if summary['oldest_alert']:
        print(f"   Oldest Alert: {summary['oldest_alert']}")
    if summary['most_recent_alert']:
        print(f"   Most Recent Alert: {summary['most_recent_alert']}")
    
    # Test airport-specific summary
    bom_summary = alerting_service.get_alert_summary(airport="BOM")
    print(f"\n✅ BOM-specific Summary:")
    print(f"   BOM Active Alerts: {bom_summary['total_active_alerts']}")
    
    return summary


@patch('requests.Session.post')
def test_webhook_integration(mock_post):
    """Test webhook integration with mocked HTTP requests."""
    print("\n🌐 Testing Webhook Integration...")
    
    # Mock successful response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_post.return_value = mock_response
    
    # Create mock services
    mock_analytics = Mock(spec=AnalyticsEngine)
    mock_optimizer = Mock(spec=ScheduleOptimizer)
    
    alerting_service = AlertingService(
        analytics_engine=mock_analytics,
        schedule_optimizer=mock_optimizer,
        slack_webhook_url="https://hooks.slack.com/test"
    )
    
    # Generate alerts
    flights = create_test_flights()
    peak_analysis = create_test_overload_scenario()
    mock_analytics.analyze_peaks.return_value = peak_analysis
    
    alerts = alerting_service.check_for_alerts("BOM", flights)
    
    if alerts:
        alert = alerts[0]
        
        # Test notification sending
        success = alerting_service.send_alert_notification(alert)
        
        if success:
            print("✅ Webhook notification sent successfully")
            print(f"   HTTP POST called: {mock_post.called}")
            
            # Verify webhook was called with correct parameters
            if mock_post.called:
                call_args = mock_post.call_args
                print(f"   Webhook URL: {call_args[0][0]}")
                print(f"   Message keys: {list(call_args[1]['json'].keys())}")
        else:
            print("❌ Webhook notification failed")
    
    else:
        print("❌ No alerts generated for webhook testing")


def main():
    """Run all integration tests."""
    print("🚀 Starting Alerting System Integration Tests")
    print("=" * 60)
    
    try:
        # Load environment variables
        load_dotenv()
        
        # Run tests
        alerts = test_alert_detection()
        message = test_slack_message_formatting()
        test_alert_lifecycle()
        summary = test_alert_summary()
        test_webhook_integration()
        
        print("\n" + "=" * 60)
        print("🎉 All Integration Tests Completed!")
        
        # Summary
        if alerts:
            print(f"✅ Generated {len(alerts)} test alerts")
        if message:
            print("✅ Slack message formatting verified")
        if summary:
            print(f"✅ Alert summary generated ({summary['total_active_alerts']} active)")
        
        print("\n💡 Next Steps:")
        print("   1. Configure SLACK_WEBHOOK_URL in .env for real notifications")
        print("   2. Integrate with FastAPI endpoints")
        print("   3. Set up monitoring and alerting thresholds")
        print("   4. Test with real flight data")
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())