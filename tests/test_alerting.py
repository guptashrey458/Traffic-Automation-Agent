"""Tests for the alerting and notification system."""

import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import List

from src.services.alerting import (
    AlertingService, Alert, AlertSeverity, AlertType, AlertRecommendation
)
from src.services.analytics import AnalyticsEngine, OverloadWindow, PeakAnalysis
from src.services.schedule_optimizer import ScheduleOptimizer
from src.models.flight import Flight


@pytest.fixture
def mock_analytics_engine():
    """Mock analytics engine."""
    engine = Mock(spec=AnalyticsEngine)
    return engine


@pytest.fixture
def mock_schedule_optimizer():
    """Mock schedule optimizer."""
    optimizer = Mock(spec=ScheduleOptimizer)
    return optimizer


@pytest.fixture
def alerting_service(mock_analytics_engine, mock_schedule_optimizer):
    """Create alerting service for testing."""
    return AlertingService(
        analytics_engine=mock_analytics_engine,
        schedule_optimizer=mock_schedule_optimizer,
        slack_webhook_url="https://hooks.slack.com/test"
    )


@pytest.fixture
def sample_flights():
    """Create sample flights for testing."""
    from src.models.flight import Airport, FlightTime
    from datetime import time
    
    # Create flights with delays
    flights = []
    
    # Normal flight with minor delay
    flight1 = Flight(
        flight_id="AI123_20240101",
        flight_number="AI 123",
        origin=Airport(code="BOM", name="Mumbai", city="Mumbai"),
        destination=Airport(code="DEL", name="Delhi", city="Delhi"),
        aircraft_type="A320",
        departure=FlightTime(
            scheduled=time(10, 0),
            actual=datetime(2024, 1, 1, 10, 20)  # 20 min delay
        )
    )
    flight1.dep_delay_min = 20  # Set delay manually for testing
    flights.append(flight1)
    
    # Critical delay flight
    flight2 = Flight(
        flight_id="6E456_20240101",
        flight_number="6E 456",
        origin=Airport(code="BOM", name="Mumbai", city="Mumbai"),
        destination=Airport(code="BLR", name="Bangalore", city="Bangalore"),
        aircraft_type="A320",
        departure=FlightTime(
            scheduled=time(11, 0),
            actual=datetime(2024, 1, 1, 12, 15)  # 75 min delay
        )
    )
    flight2.dep_delay_min = 75  # Critical delay
    flights.append(flight2)
    
    # Normal flight with minor delay
    flight3 = Flight(
        flight_id="UK789_20240101",
        flight_number="UK 789",
        origin=Airport(code="BOM", name="Mumbai", city="Mumbai"),
        destination=Airport(code="CCU", name="Kolkata", city="Kolkata"),
        aircraft_type="A320",
        departure=FlightTime(
            scheduled=time(12, 0),
            actual=datetime(2024, 1, 1, 12, 5)  # 5 min delay
        )
    )
    flight3.dep_delay_min = 5  # Minor delay
    flights.append(flight3)
    
    return flights


@pytest.fixture
def sample_overload_window():
    """Create sample overload window."""
    return OverloadWindow(
        start_time=datetime(2024, 1, 1, 10, 0),
        end_time=datetime(2024, 1, 1, 10, 30),
        duration_minutes=30,
        peak_overload=12,
        avg_overload=8.5,
        affected_flights=25
    )


@pytest.fixture
def sample_peak_analysis(sample_overload_window):
    """Create sample peak analysis."""
    return PeakAnalysis(
        airport="BOM",
        analysis_date=datetime(2024, 1, 1).date(),
        bucket_minutes=10,
        overload_windows=[sample_overload_window]
    )


class TestAlertingService:
    """Test the AlertingService class."""
    
    def test_initialization(self, mock_analytics_engine, mock_schedule_optimizer):
        """Test alerting service initialization."""
        service = AlertingService(
            analytics_engine=mock_analytics_engine,
            schedule_optimizer=mock_schedule_optimizer,
            slack_webhook_url="https://test.webhook.url"
        )
        
        assert service.analytics_engine == mock_analytics_engine
        assert service.schedule_optimizer == mock_schedule_optimizer
        assert service.slack_webhook_url == "https://test.webhook.url"
        assert service.capacity_overload_threshold == 0.9
        assert service.delay_threshold_minutes == 15
        assert service.cascade_impact_threshold == 5
        assert len(service.active_alerts) == 0
        assert len(service.alert_history) == 0
    
    def test_check_for_alerts_capacity_overload(self, alerting_service, sample_flights, 
                                               sample_peak_analysis, mock_analytics_engine):
        """Test capacity overload alert detection."""
        mock_analytics_engine.analyze_peaks.return_value = sample_peak_analysis
        
        alerts = alerting_service.check_for_alerts("BOM", sample_flights)
        
        assert len(alerts) >= 1
        overload_alert = next((a for a in alerts if a.alert_type == AlertType.CAPACITY_OVERLOAD), None)
        assert overload_alert is not None
        assert overload_alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]
        assert overload_alert.airport == "BOM"
        assert "Capacity Overload" in overload_alert.title
        assert len(overload_alert.recommendations) <= 3
        assert overload_alert.metrics["peak_overload"] == 12
    
    def test_check_for_alerts_delay_cascade(self, alerting_service, mock_analytics_engine):
        """Test delay cascade alert detection."""
        from src.models.flight import Airport, FlightTime
        from datetime import time
        
        # Create flights with significant delays
        delayed_flights = []
        for i in range(6):  # Above cascade threshold
            flight = Flight(
                flight_id=f"AI{100+i}_20240101",
                flight_number=f"AI {100+i}",
                origin=Airport(code="BOM", name="Mumbai", city="Mumbai"),
                destination=Airport(code="DEL", name="Delhi", city="Delhi"),
                aircraft_type="A320",
                departure=FlightTime(
                    scheduled=time(10+i, 0),
                    actual=datetime(2024, 1, 1, 10+i, 20)  # 20 min delay
                )
            )
            flight.dep_delay_min = 20  # Above threshold
            delayed_flights.append(flight)
        
        # Mock empty peak analysis
        mock_analytics_engine.analyze_peaks.return_value = PeakAnalysis(
            airport="BOM", 
            analysis_date=datetime(2024, 1, 1).date(),
            bucket_minutes=10, 
            overload_windows=[]
        )
        
        alerts = alerting_service.check_for_alerts("BOM", delayed_flights)
        
        cascade_alert = next((a for a in alerts if a.alert_type == AlertType.DELAY_CASCADE), None)
        assert cascade_alert is not None
        assert cascade_alert.severity in [AlertSeverity.MEDIUM, AlertSeverity.HIGH]
        assert len(cascade_alert.affected_flights) == 6
        assert cascade_alert.metrics["delayed_flights_count"] == 6
    
    def test_check_for_alerts_critical_delay(self, alerting_service, sample_flights, mock_analytics_engine):
        """Test critical flight delay alert detection."""
        # Mock empty peak analysis
        mock_analytics_engine.analyze_peaks.return_value = PeakAnalysis(
            airport="BOM", 
            analysis_date=datetime(2024, 1, 1).date(),
            bucket_minutes=10, 
            overload_windows=[]
        )
        
        alerts = alerting_service.check_for_alerts("BOM", sample_flights)
        
        critical_alert = next((a for a in alerts if a.alert_type == AlertType.CRITICAL_FLIGHT_DELAY), None)
        assert critical_alert is not None
        assert critical_alert.severity == AlertSeverity.CRITICAL
        assert "6E 456" in critical_alert.title  # The flight with 75min delay
        assert critical_alert.metrics["delay_minutes"] == 75.0
    
    def test_calculate_overload_severity(self, alerting_service):
        """Test overload severity calculation."""
        # Critical severity
        critical_overload = OverloadWindow(
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(minutes=60),
            duration_minutes=60,
            peak_overload=15,
            avg_overload=12.0,
            affected_flights=50
        )
        assert alerting_service._calculate_overload_severity(critical_overload) == AlertSeverity.CRITICAL
        
        # High severity
        high_overload = OverloadWindow(
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(minutes=30),
            duration_minutes=30,
            peak_overload=10,
            avg_overload=8.0,
            affected_flights=30
        )
        assert alerting_service._calculate_overload_severity(high_overload) == AlertSeverity.HIGH
        
        # Medium severity
        medium_overload = OverloadWindow(
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(minutes=15),
            duration_minutes=15,
            peak_overload=5,
            avg_overload=4.0,
            affected_flights=15
        )
        assert alerting_service._calculate_overload_severity(medium_overload) == AlertSeverity.MEDIUM
        
        # Low severity
        low_overload = OverloadWindow(
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(minutes=10),
            duration_minutes=10,
            peak_overload=3,
            avg_overload=2.0,
            affected_flights=8
        )
        assert alerting_service._calculate_overload_severity(low_overload) == AlertSeverity.LOW
    
    def test_generate_overload_recommendations(self, alerting_service):
        """Test overload recommendation generation."""
        severe_overload = OverloadWindow(
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(minutes=30),
            duration_minutes=30,
            peak_overload=12,
            avg_overload=10.0,
            affected_flights=40
        )
        
        recommendations = alerting_service._generate_overload_recommendations(severe_overload)
        
        assert len(recommendations) == 3
        assert all(isinstance(rec, AlertRecommendation) for rec in recommendations)
        assert recommendations[0].priority == 1
        assert "ground delay" in recommendations[0].action.lower()
        assert all(rec.estimated_improvement is not None for rec in recommendations)
    
    @patch('requests.Session.post')
    def test_send_alert_notification_success(self, mock_post, alerting_service):
        """Test successful alert notification sending."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        alert = Alert(
            alert_id="test_alert",
            alert_type=AlertType.CAPACITY_OVERLOAD,
            severity=AlertSeverity.HIGH,
            title="Test Alert",
            description="Test description",
            airport="BOM",
            timestamp=datetime.now(),
            recommendations=[
                AlertRecommendation(
                    action="Test action",
                    impact="Test impact",
                    priority=1,
                    estimated_improvement="10% improvement"
                )
            ],
            metrics={"test_metric": 42}
        )
        
        result = alerting_service.send_alert_notification(alert)
        
        assert result is True
        mock_post.assert_called_once()
        
        # Verify message structure
        call_args = mock_post.call_args
        message = call_args[1]['json']
        assert 'text' in message
        assert 'blocks' in message
        assert len(message['blocks']) >= 3  # Header, description, recommendations
    
    @patch('requests.Session.post')
    def test_send_alert_notification_failure(self, mock_post, alerting_service):
        """Test failed alert notification sending."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response
        
        alert = Alert(
            alert_id="test_alert",
            alert_type=AlertType.CAPACITY_OVERLOAD,
            severity=AlertSeverity.HIGH,
            title="Test Alert",
            description="Test description",
            airport="BOM",
            timestamp=datetime.now()
        )
        
        result = alerting_service.send_alert_notification(alert)
        
        assert result is False
    
    def test_send_alert_notification_no_webhook(self, mock_analytics_engine, mock_schedule_optimizer):
        """Test alert notification when no webhook URL is configured."""
        service = AlertingService(
            analytics_engine=mock_analytics_engine,
            schedule_optimizer=mock_schedule_optimizer,
            slack_webhook_url=None
        )
        
        alert = Alert(
            alert_id="test_alert",
            alert_type=AlertType.CAPACITY_OVERLOAD,
            severity=AlertSeverity.HIGH,
            title="Test Alert",
            description="Test description",
            airport="BOM",
            timestamp=datetime.now()
        )
        
        result = service.send_alert_notification(alert)
        
        assert result is False
    
    def test_build_slack_message(self, alerting_service):
        """Test Slack message building."""
        alert = Alert(
            alert_id="test_alert",
            alert_type=AlertType.CAPACITY_OVERLOAD,
            severity=AlertSeverity.HIGH,
            title="Capacity Overload at BOM",
            description="Peak overload detected",
            airport="BOM",
            timestamp=datetime(2024, 1, 1, 10, 30),
            affected_flights=["AI123", "6E456"],
            recommendations=[
                AlertRecommendation(
                    action="Implement ground delay program",
                    impact="Reduce overload by 8 flights",
                    priority=1,
                    estimated_improvement="30-50% delay reduction"
                )
            ],
            metrics={
                "peak_overload": 12,
                "duration_minutes": 30,
                "affected_flights_count": 25
            }
        )
        
        message = alerting_service._build_slack_message(alert)
        
        assert 'text' in message
        assert 'blocks' in message
        assert len(message['blocks']) >= 4  # Header, description, metrics, recommendations
        
        # Check header
        header_block = message['blocks'][0]
        assert header_block['type'] == 'header'
        assert '🔴' in header_block['text']['text']  # High severity emoji
        
        # Check description block
        desc_block = message['blocks'][1]
        assert 'BOM' in desc_block['text']['text']
        assert 'HIGH' in desc_block['text']['text']
        
        # Check metrics block
        metrics_block = next(b for b in message['blocks'] if 'Key Metrics' in b['text']['text'])
        assert 'Peak Overload:* 12' in metrics_block['text']['text']
        
        # Check recommendations block
        rec_block = next(b for b in message['blocks'] if 'Top Recommendations' in b['text']['text'])
        assert 'ground delay' in rec_block['text']['text'].lower()
        assert '30-50% delay reduction' in rec_block['text']['text']
    
    def test_resolve_alert(self, alerting_service):
        """Test alert resolution."""
        alert = Alert(
            alert_id="test_alert",
            alert_type=AlertType.CAPACITY_OVERLOAD,
            severity=AlertSeverity.HIGH,
            title="Test Alert",
            description="Test description",
            airport="BOM",
            timestamp=datetime.now()
        )
        
        # Add to active alerts
        alerting_service.active_alerts[alert.alert_id] = alert
        
        with patch.object(alerting_service, '_send_resolution_notification', return_value=True):
            result = alerting_service.resolve_alert("test_alert")
        
        assert result is True
        assert alert.resolved is True
        assert alert.resolved_at is not None
        assert "test_alert" not in alerting_service.active_alerts
    
    def test_resolve_nonexistent_alert(self, alerting_service):
        """Test resolving non-existent alert."""
        result = alerting_service.resolve_alert("nonexistent_alert")
        assert result is False
    
    def test_escalate_alert(self, alerting_service):
        """Test alert escalation."""
        alert = Alert(
            alert_id="test_alert",
            alert_type=AlertType.CAPACITY_OVERLOAD,
            severity=AlertSeverity.MEDIUM,
            title="Test Alert",
            description="Test description",
            airport="BOM",
            timestamp=datetime.now()
        )
        
        # Add to active alerts
        alerting_service.active_alerts[alert.alert_id] = alert
        
        with patch.object(alerting_service, '_send_escalation_notification', return_value=True):
            result = alerting_service.escalate_alert("test_alert")
        
        assert result is True
        assert alert.severity == AlertSeverity.HIGH
        assert alert.escalated is True
        assert alert.escalated_at is not None
    
    def test_escalate_critical_alert(self, alerting_service):
        """Test escalating already critical alert."""
        alert = Alert(
            alert_id="test_alert",
            alert_type=AlertType.CAPACITY_OVERLOAD,
            severity=AlertSeverity.CRITICAL,
            title="Test Alert",
            description="Test description",
            airport="BOM",
            timestamp=datetime.now()
        )
        
        # Add to active alerts
        alerting_service.active_alerts[alert.alert_id] = alert
        
        result = alerting_service.escalate_alert("test_alert")
        
        assert result is False  # Cannot escalate critical alert
        assert alert.escalated is False
    
    def test_get_active_alerts_no_filter(self, alerting_service):
        """Test getting all active alerts."""
        alerts = [
            Alert(
                alert_id="alert1",
                alert_type=AlertType.CAPACITY_OVERLOAD,
                severity=AlertSeverity.HIGH,
                title="Alert 1",
                description="Description 1",
                airport="BOM",
                timestamp=datetime.now()
            ),
            Alert(
                alert_id="alert2",
                alert_type=AlertType.DELAY_CASCADE,
                severity=AlertSeverity.CRITICAL,
                title="Alert 2",
                description="Description 2",
                airport="DEL",
                timestamp=datetime.now() + timedelta(minutes=5)
            )
        ]
        
        for alert in alerts:
            alerting_service.active_alerts[alert.alert_id] = alert
        
        active_alerts = alerting_service.get_active_alerts()
        
        assert len(active_alerts) == 2
        # Should be sorted by severity (critical first)
        assert active_alerts[0].severity == AlertSeverity.CRITICAL
        assert active_alerts[1].severity == AlertSeverity.HIGH
    
    def test_get_active_alerts_with_filters(self, alerting_service):
        """Test getting active alerts with filters."""
        alerts = [
            Alert(
                alert_id="alert1",
                alert_type=AlertType.CAPACITY_OVERLOAD,
                severity=AlertSeverity.HIGH,
                title="Alert 1",
                description="Description 1",
                airport="BOM",
                timestamp=datetime.now()
            ),
            Alert(
                alert_id="alert2",
                alert_type=AlertType.DELAY_CASCADE,
                severity=AlertSeverity.HIGH,
                title="Alert 2",
                description="Description 2",
                airport="DEL",
                timestamp=datetime.now()
            )
        ]
        
        for alert in alerts:
            alerting_service.active_alerts[alert.alert_id] = alert
        
        # Filter by airport
        bom_alerts = alerting_service.get_active_alerts(airport="BOM")
        assert len(bom_alerts) == 1
        assert bom_alerts[0].airport == "BOM"
        
        # Filter by severity
        high_alerts = alerting_service.get_active_alerts(severity=AlertSeverity.HIGH)
        assert len(high_alerts) == 2
        assert all(a.severity == AlertSeverity.HIGH for a in high_alerts)
        
        # Filter by both
        bom_high_alerts = alerting_service.get_active_alerts(airport="BOM", severity=AlertSeverity.HIGH)
        assert len(bom_high_alerts) == 1
        assert bom_high_alerts[0].airport == "BOM"
        assert bom_high_alerts[0].severity == AlertSeverity.HIGH
    
    def test_get_alert_summary(self, alerting_service):
        """Test getting alert summary."""
        alerts = [
            Alert(
                alert_id="alert1",
                alert_type=AlertType.CAPACITY_OVERLOAD,
                severity=AlertSeverity.HIGH,
                title="Alert 1",
                description="Description 1",
                airport="BOM",
                timestamp=datetime.now(),
                escalated=True
            ),
            Alert(
                alert_id="alert2",
                alert_type=AlertType.DELAY_CASCADE,
                severity=AlertSeverity.CRITICAL,
                title="Alert 2",
                description="Description 2",
                airport="BOM",
                timestamp=datetime.now() + timedelta(minutes=5)
            ),
            Alert(
                alert_id="alert3",
                alert_type=AlertType.CAPACITY_OVERLOAD,
                severity=AlertSeverity.MEDIUM,
                title="Alert 3",
                description="Description 3",
                airport="DEL",
                timestamp=datetime.now() - timedelta(minutes=10)
            )
        ]
        
        for alert in alerts:
            alerting_service.active_alerts[alert.alert_id] = alert
        
        summary = alerting_service.get_alert_summary()
        
        assert summary["total_active_alerts"] == 3
        assert summary["by_severity"]["critical"] == 1
        assert summary["by_severity"]["high"] == 1
        assert summary["by_severity"]["medium"] == 1
        assert summary["by_type"]["capacity_overload"] == 2
        assert summary["by_type"]["delay_cascade"] == 1
        assert summary["escalated_alerts"] == 1
        assert summary["oldest_alert"] is not None
        assert summary["most_recent_alert"] is not None
        
        # Test with airport filter
        bom_summary = alerting_service.get_alert_summary(airport="BOM")
        assert bom_summary["total_active_alerts"] == 2
    
    def test_system_error_alert_creation(self, alerting_service, sample_flights, mock_analytics_engine):
        """Test system error alert creation when analytics fails."""
        mock_analytics_engine.analyze_peaks.side_effect = Exception("Database connection failed")
        
        alerts = alerting_service.check_for_alerts("BOM", sample_flights)
        
        assert len(alerts) == 1
        error_alert = alerts[0]
        assert error_alert.alert_type == AlertType.SYSTEM_ERROR
        assert error_alert.severity == AlertSeverity.HIGH
        assert "System Error" in error_alert.title
        assert "Database connection failed" in error_alert.description
        assert len(error_alert.recommendations) == 3


class TestAlertRecommendation:
    """Test the AlertRecommendation class."""
    
    def test_alert_recommendation_creation(self):
        """Test creating alert recommendations."""
        rec = AlertRecommendation(
            action="Test action",
            impact="Test impact",
            priority=1,
            estimated_improvement="10% improvement"
        )
        
        assert rec.action == "Test action"
        assert rec.impact == "Test impact"
        assert rec.priority == 1
        assert rec.estimated_improvement == "10% improvement"


class TestAlert:
    """Test the Alert class."""
    
    def test_alert_creation(self):
        """Test creating alerts."""
        alert = Alert(
            alert_id="test_alert",
            alert_type=AlertType.CAPACITY_OVERLOAD,
            severity=AlertSeverity.HIGH,
            title="Test Alert",
            description="Test description",
            airport="BOM",
            timestamp=datetime.now(),
            affected_flights=["AI123", "6E456"],
            recommendations=[
                AlertRecommendation(
                    action="Test action",
                    impact="Test impact",
                    priority=1
                )
            ],
            metrics={"test_metric": 42}
        )
        
        assert alert.alert_id == "test_alert"
        assert alert.alert_type == AlertType.CAPACITY_OVERLOAD
        assert alert.severity == AlertSeverity.HIGH
        assert alert.title == "Test Alert"
        assert alert.description == "Test description"
        assert alert.airport == "BOM"
        assert len(alert.affected_flights) == 2
        assert len(alert.recommendations) == 1
        assert alert.metrics["test_metric"] == 42
        assert alert.resolved is False
        assert alert.escalated is False


if __name__ == "__main__":
    pytest.main([__file__])