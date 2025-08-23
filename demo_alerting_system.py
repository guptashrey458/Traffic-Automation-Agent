#!/usr/bin/env python3
"""
Demo script for the Alerting and Notification System.
Shows the complete workflow from alert detection to notification delivery.
"""

import os
import sys
import json
from datetime import datetime, timedelta, time, date
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.services.alerting import AlertingService, AlertSeverity, AlertType
from src.services.analytics import AnalyticsEngine, OverloadWindow, PeakAnalysis
from src.services.schedule_optimizer import ScheduleOptimizer
from src.services.database import FlightDatabaseService
from src.models.flight import Flight, Airport, FlightTime


def create_demo_scenario():
    """Create a realistic demo scenario with various alert conditions."""
    print("🎬 Creating Demo Scenario...")
    
    # Create flights representing a busy morning at Mumbai (BOM)
    flights = []
    
    # Peak hour flights (8-10 AM) - will trigger capacity overload
    peak_flights = [
        ("AI 101", "BOM", "DEL", time(8, 0), 5),    # Minor delay
        ("6E 202", "BOM", "BLR", time(8, 10), 15),  # Moderate delay
        ("UK 303", "BOM", "CCU", time(8, 15), 8),   # Minor delay
        ("SG 404", "BOM", "MAA", time(8, 20), 22),  # Cascade delay
        ("AI 505", "BOM", "HYD", time(8, 25), 18),  # Cascade delay
        ("6E 606", "BOM", "GOI", time(8, 30), 25),  # Cascade delay
        ("UK 707", "BOM", "AMD", time(8, 35), 30),  # Cascade delay
        ("SG 808", "BOM", "PNQ", time(8, 40), 20),  # Cascade delay
        ("AI 909", "BOM", "JAI", time(8, 45), 16),  # Cascade delay
        ("6E 010", "BOM", "IXC", time(8, 50), 12),  # Minor delay
    ]
    
    for i, (flight_no, origin, dest, sched_time, delay_min) in enumerate(peak_flights):
        actual_time = datetime.combine(date.today(), sched_time) + timedelta(minutes=delay_min)
        
        flight = Flight(
            flight_id=f"{flight_no.replace(' ', '')}_{date.today().strftime('%Y%m%d')}",
            flight_number=flight_no,
            origin=Airport(code=origin, name=f"{origin} Airport", city=origin),
            destination=Airport(code=dest, name=f"{dest} Airport", city=dest),
            aircraft_type="A320",
            departure=FlightTime(
                scheduled=sched_time,
                actual=actual_time
            )
        )
        flight.dep_delay_min = delay_min
        flights.append(flight)
    
    # Add a critical delay flight
    critical_flight = Flight(
        flight_id=f"AI999_{date.today().strftime('%Y%m%d')}",
        flight_number="AI 999",
        origin=Airport(code="BOM", name="Mumbai Airport", city="Mumbai"),
        destination=Airport(code="DEL", name="Delhi Airport", city="Delhi"),
        aircraft_type="B777",
        departure=FlightTime(
            scheduled=time(9, 0),
            actual=datetime.combine(date.today(), time(10, 45))  # 105 min delay
        )
    )
    critical_flight.dep_delay_min = 105  # Critical delay
    flights.append(critical_flight)
    
    print(f"✅ Created {len(flights)} demo flights")
    print(f"   - Peak hour flights: {len(peak_flights)}")
    print(f"   - Critical delay flights: 1")
    print(f"   - Total delays > 15 min: {len([f for f in flights if f.dep_delay_min > 15])}")
    
    return flights


def create_demo_overload_scenario():
    """Create overload scenario for capacity alerts."""
    # Simulate high demand period
    overload_window = OverloadWindow(
        start_time=datetime.combine(date.today(), time(8, 0)),
        end_time=datetime.combine(date.today(), time(9, 0)),
        duration_minutes=60,
        peak_overload=18,  # 18 flights over capacity
        avg_overload=12.5,
        affected_flights=55
    )
    
    return PeakAnalysis(
        airport="BOM",
        analysis_date=date.today(),
        bucket_minutes=10,
        overload_windows=[overload_window],
        peak_hour=8,
        peak_demand=45,
        total_capacity=30,
        avg_utilization=1.2,  # 120% utilization
        recommendations=[
            "Implement ground delay program for arriving flights",
            "Coordinate with adjacent airports for traffic redistribution",
            "Activate additional runway configurations"
        ]
    )


def demo_alert_detection():
    """Demonstrate alert detection capabilities."""
    print("\n🔍 Alert Detection Demo")
    print("=" * 50)
    
    # Create mock services (in real scenario, these would be actual services)
    from unittest.mock import Mock
    mock_analytics = Mock(spec=AnalyticsEngine)
    mock_optimizer = Mock(spec=ScheduleOptimizer)
    
    # Create alerting service
    alerting_service = AlertingService(
        analytics_engine=mock_analytics,
        schedule_optimizer=mock_optimizer
    )
    
    # Setup demo data
    flights = create_demo_scenario()
    peak_analysis = create_demo_overload_scenario()
    mock_analytics.analyze_peaks.return_value = peak_analysis
    
    print(f"📊 Analyzing {len(flights)} flights at BOM...")
    print(f"   Peak demand: {peak_analysis.peak_demand} flights/hour")
    print(f"   Capacity: {peak_analysis.total_capacity} flights/hour")
    print(f"   Utilization: {peak_analysis.avg_utilization:.1%}")
    
    # Check for alerts
    alerts = alerting_service.check_for_alerts("BOM", flights)
    
    print(f"\n🚨 Generated {len(alerts)} alerts:")
    
    for i, alert in enumerate(alerts, 1):
        print(f"\n{i}. {alert.title}")
        print(f"   🔥 Severity: {alert.severity.value.upper()}")
        print(f"   📋 Type: {alert.alert_type.value.replace('_', ' ').title()}")
        print(f"   ✈️  Affected Flights: {len(alert.affected_flights)}")
        print(f"   💡 Recommendations: {len(alert.recommendations)}")
        
        if alert.recommendations:
            print(f"   🎯 Top Recommendation: {alert.recommendations[0].action}")
        
        if alert.metrics:
            key_metrics = []
            for key, value in list(alert.metrics.items())[:3]:  # Show top 3 metrics
                if isinstance(value, (int, float)):
                    key_metrics.append(f"{key.replace('_', ' ').title()}: {value}")
                else:
                    key_metrics.append(f"{key.replace('_', ' ').title()}: {value}")
            if key_metrics:
                print(f"   📈 Key Metrics: {', '.join(key_metrics)}")
    
    return alerts, alerting_service


def demo_slack_notifications(alerts, alerting_service):
    """Demonstrate Slack notification functionality."""
    print("\n📱 Slack Notifications Demo")
    print("=" * 50)
    
    if not alerting_service.slack_webhook_url:
        print("⚠️  No Slack webhook configured - showing message format only")
        
        if alerts:
            alert = alerts[0]  # Demo with first alert
            message = alerting_service._build_slack_message(alert)
            
            print(f"\n📝 Sample Slack Message for: {alert.title}")
            print(f"   Text: {message['text']}")
            print(f"   Blocks: {len(message['blocks'])} sections")
            
            # Show formatted message structure
            print("\n📋 Message Structure:")
            for i, block in enumerate(message['blocks'], 1):
                if block['type'] == 'header':
                    print(f"   {i}. Header: {block['text']['text']}")
                elif block['type'] == 'section':
                    content = block['text']['text'][:100] + "..." if len(block['text']['text']) > 100 else block['text']['text']
                    print(f"   {i}. Section: {content}")
        
        return False
    
    else:
        print("📤 Sending notifications to Slack...")
        
        success_count = 0
        for alert in alerts:
            try:
                success = alerting_service.send_alert_notification(alert)
                if success:
                    print(f"   ✅ Sent: {alert.title}")
                    success_count += 1
                else:
                    print(f"   ❌ Failed: {alert.title}")
            except Exception as e:
                print(f"   ❌ Error sending {alert.title}: {e}")
        
        print(f"\n📊 Notification Results: {success_count}/{len(alerts)} sent successfully")
        return success_count > 0


def demo_alert_management(alerting_service):
    """Demonstrate alert management capabilities."""
    print("\n🔧 Alert Management Demo")
    print("=" * 50)
    
    # Get active alerts
    active_alerts = alerting_service.get_active_alerts()
    print(f"📋 Active Alerts: {len(active_alerts)}")
    
    if active_alerts:
        # Show alert summary
        summary = alerting_service.get_alert_summary()
        print(f"\n📊 Alert Summary:")
        print(f"   Total Active: {summary['total_active_alerts']}")
        print(f"   By Severity: {summary['by_severity']}")
        print(f"   By Type: {summary['by_type']}")
        print(f"   Escalated: {summary['escalated_alerts']}")
        
        # Demo escalation
        first_alert = active_alerts[0]
        if first_alert.severity != AlertSeverity.CRITICAL:
            print(f"\n⬆️  Escalating alert: {first_alert.title}")
            original_severity = first_alert.severity
            success = alerting_service.escalate_alert(first_alert.alert_id)
            if success:
                print(f"   ✅ Escalated from {original_severity.value} to {first_alert.severity.value}")
            else:
                print(f"   ❌ Escalation failed")
        
        # Demo resolution
        if len(active_alerts) > 1:
            alert_to_resolve = active_alerts[1]
            print(f"\n✅ Resolving alert: {alert_to_resolve.title}")
            success = alerting_service.resolve_alert(alert_to_resolve.alert_id)
            if success:
                print(f"   ✅ Alert resolved successfully")
                print(f"   ⏱️  Resolution time: {alert_to_resolve.resolved_at}")
            else:
                print(f"   ❌ Resolution failed")
        
        # Show updated summary
        updated_summary = alerting_service.get_alert_summary()
        print(f"\n📊 Updated Summary:")
        print(f"   Active Alerts: {updated_summary['total_active_alerts']}")
        print(f"   Escalated: {updated_summary['escalated_alerts']}")


def demo_api_integration():
    """Demonstrate API integration capabilities."""
    print("\n🌐 API Integration Demo")
    print("=" * 50)
    
    print("📡 Available API Endpoints:")
    endpoints = [
        "POST /alerts/check - Check for new alerts",
        "GET /alerts/active - Get active alerts",
        "GET /alerts/summary - Get alert summary",
        "POST /alerts/{id}/resolve - Resolve an alert",
        "POST /alerts/{id}/escalate - Escalate an alert",
        "POST /alerts/test-notification - Test webhook"
    ]
    
    for endpoint in endpoints:
        print(f"   • {endpoint}")
    
    print("\n📝 Example API Usage:")
    print("""
    # Check for alerts
    curl -X POST "http://localhost:8000/alerts/check" \\
         -H "Content-Type: application/json" \\
         -d '{"airport": "BOM", "force_check": true}'
    
    # Get active alerts
    curl "http://localhost:8000/alerts/active?airport=BOM&severity=high"
    
    # Test notification
    curl -X POST "http://localhost:8000/alerts/test-notification"
    """)


def demo_real_time_monitoring():
    """Demonstrate real-time monitoring capabilities."""
    print("\n⏱️  Real-Time Monitoring Demo")
    print("=" * 50)
    
    print("🔄 Continuous Monitoring Features:")
    features = [
        "✅ Automatic alert detection every 5 minutes",
        "✅ Threshold-based capacity overload detection",
        "✅ Delay cascade pattern recognition",
        "✅ Critical flight delay identification",
        "✅ Instant Slack notifications",
        "✅ Alert escalation for unresolved issues",
        "✅ Resolution confirmation notifications"
    ]
    
    for feature in features:
        print(f"   {feature}")
    
    print("\n⚙️  Configuration Options:")
    config_options = [
        "ALERT_THRESHOLDS__CAPACITY_OVERLOAD_THRESHOLD=0.9",
        "ALERT_THRESHOLDS__DELAY_THRESHOLD_MINUTES=15",
        "ALERT_THRESHOLDS__CASCADE_IMPACT_THRESHOLD=5",
        "SLACK_WEBHOOK_URL=https://hooks.slack.com/...",
    ]
    
    for option in config_options:
        print(f"   • {option}")


def main():
    """Run the complete alerting system demo."""
    print("🚀 Agentic Flight Scheduler - Alerting System Demo")
    print("=" * 60)
    
    # Load environment variables
    load_dotenv()
    
    try:
        # Demo alert detection
        alerts, alerting_service = demo_alert_detection()
        
        # Demo Slack notifications
        demo_slack_notifications(alerts, alerting_service)
        
        # Demo alert management
        demo_alert_management(alerting_service)
        
        # Demo API integration
        demo_api_integration()
        
        # Demo real-time monitoring
        demo_real_time_monitoring()
        
        print("\n" + "=" * 60)
        print("🎉 Alerting System Demo Complete!")
        
        print("\n📋 Summary:")
        print(f"   ✅ Alert Detection: {len(alerts)} alerts generated")
        print(f"   ✅ Notification System: {'Configured' if alerting_service.slack_webhook_url else 'Not configured'}")
        print(f"   ✅ Alert Management: Escalation & Resolution")
        print(f"   ✅ API Integration: 6 endpoints available")
        print(f"   ✅ Real-time Monitoring: Continuous operation")
        
        print("\n🚀 Next Steps:")
        print("   1. Configure SLACK_WEBHOOK_URL in .env for real notifications")
        print("   2. Start the FastAPI server: uvicorn src.api.main:app --reload")
        print("   3. Test API endpoints with curl or Postman")
        print("   4. Set up monitoring dashboard integration")
        print("   5. Configure alert thresholds for your environment")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())