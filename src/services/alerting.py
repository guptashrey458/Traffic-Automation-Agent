"""Alerting and notification system for flight scheduling operations."""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .analytics import AnalyticsEngine, OverloadWindow, PeakAnalysis
from .schedule_optimizer import ScheduleOptimizer, OptimizationResult
from ..models.flight import Flight


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts that can be triggered."""
    CAPACITY_OVERLOAD = "capacity_overload"
    DELAY_CASCADE = "delay_cascade"
    CRITICAL_FLIGHT_DELAY = "critical_flight_delay"
    SYSTEM_ERROR = "system_error"


@dataclass
class AlertRecommendation:
    """A single recommendation to resolve an alert."""
    action: str
    impact: str
    priority: int
    estimated_improvement: Optional[str] = None


@dataclass
class Alert:
    """An alert with details and recommendations."""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    description: str
    airport: str
    timestamp: datetime
    affected_flights: List[str] = field(default_factory=list)
    recommendations: List[AlertRecommendation] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    escalated: bool = False
    escalated_at: Optional[datetime] = None


class AlertingService:
    """Service for detecting alerts and sending notifications."""
    
    def __init__(self, 
                 analytics_engine: AnalyticsEngine,
                 schedule_optimizer: ScheduleOptimizer,
                 slack_webhook_url: Optional[str] = None):
        """Initialize the alerting service.
        
        Args:
            analytics_engine: Analytics engine for capacity analysis
            schedule_optimizer: Optimizer for generating recommendations
            slack_webhook_url: Slack webhook URL for notifications
        """
        self.analytics_engine = analytics_engine
        self.schedule_optimizer = schedule_optimizer
        self.slack_webhook_url = slack_webhook_url or os.getenv('SLACK_WEBHOOK_URL')
        
        # Alert thresholds from environment
        self.capacity_overload_threshold = float(os.getenv('ALERT_THRESHOLDS__CAPACITY_OVERLOAD_THRESHOLD', '0.9'))
        self.delay_threshold_minutes = int(os.getenv('ALERT_THRESHOLDS__DELAY_THRESHOLD_MINUTES', '15'))
        self.cascade_impact_threshold = int(os.getenv('ALERT_THRESHOLDS__CASCADE_IMPACT_THRESHOLD', '5'))
        
        # Active alerts tracking
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # HTTP session for webhook requests
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        self.logger = logging.getLogger(__name__)
    
    def check_for_alerts(self, airport: str, flights: List[Flight]) -> List[Alert]:
        """Check for alert conditions and generate alerts.
        
        Args:
            airport: Airport code to analyze
            flights: List of flights to analyze
            
        Returns:
            List of new alerts generated
        """
        new_alerts = []
        
        try:
            # Analyze current traffic patterns
            peak_analysis = self.analytics_engine.analyze_peaks(
                airport=airport,
                bucket_minutes=10
            )
            
            # Check for capacity overload alerts
            overload_alerts = self._check_capacity_overload(airport, peak_analysis)
            new_alerts.extend(overload_alerts)
            
            # Check for delay cascade alerts
            cascade_alerts = self._check_delay_cascades(airport, flights)
            new_alerts.extend(cascade_alerts)
            
            # Check for critical flight delays
            delay_alerts = self._check_critical_delays(airport, flights)
            new_alerts.extend(delay_alerts)
            
            # Store new alerts
            for alert in new_alerts:
                self.active_alerts[alert.alert_id] = alert
                self.alert_history.append(alert)
            
            self.logger.info(f"Generated {len(new_alerts)} new alerts for {airport}")
            
        except Exception as e:
            self.logger.error(f"Error checking for alerts: {e}")
            # Generate system error alert
            error_alert = self._create_system_error_alert(airport, str(e))
            new_alerts.append(error_alert)
            self.active_alerts[error_alert.alert_id] = error_alert
        
        return new_alerts
    
    def _check_capacity_overload(self, airport: str, peak_analysis: PeakAnalysis) -> List[Alert]:
        """Check for capacity overload conditions."""
        alerts = []
        
        for overload_window in peak_analysis.overload_windows:
            # Calculate severity based on overload magnitude and duration
            severity = self._calculate_overload_severity(overload_window)
            
            if severity == AlertSeverity.LOW:
                continue  # Don't alert for low severity overloads
            
            alert_id = f"overload_{airport}_{overload_window.start_time.strftime('%Y%m%d_%H%M')}"
            
            # Skip if we already have an active alert for this window
            if alert_id in self.active_alerts:
                continue
            
            # Generate recommendations
            recommendations = self._generate_overload_recommendations(overload_window)
            
            alert = Alert(
                alert_id=alert_id,
                alert_type=AlertType.CAPACITY_OVERLOAD,
                severity=severity,
                title=f"Capacity Overload at {airport}",
                description=f"Peak overload of {overload_window.peak_overload} flights detected from "
                           f"{overload_window.start_time.strftime('%H:%M')} to "
                           f"{overload_window.end_time.strftime('%H:%M')}",
                airport=airport,
                timestamp=datetime.now(),
                affected_flights=[],  # Would need flight IDs from overload window
                recommendations=recommendations,
                metrics={
                    "peak_overload": overload_window.peak_overload,
                    "avg_overload": overload_window.avg_overload,
                    "duration_minutes": overload_window.duration_minutes,
                    "affected_flights_count": overload_window.affected_flights
                }
            )
            
            alerts.append(alert)
        
        return alerts
    
    def _check_delay_cascades(self, airport: str, flights: List[Flight]) -> List[Alert]:
        """Check for delay cascade conditions."""
        alerts = []
        
        # Count flights with significant delays
        delayed_flights = [f for f in flights if f.dep_delay_min and f.dep_delay_min > self.delay_threshold_minutes]
        
        if len(delayed_flights) >= self.cascade_impact_threshold:
            alert_id = f"cascade_{airport}_{datetime.now().strftime('%Y%m%d_%H%M')}"
            
            # Skip if we already have a recent cascade alert
            if any(alert.alert_type == AlertType.DELAY_CASCADE and 
                   alert.timestamp > datetime.now() - timedelta(hours=1)
                   for alert in self.active_alerts.values()):
                return alerts
            
            severity = AlertSeverity.HIGH if len(delayed_flights) > 10 else AlertSeverity.MEDIUM
            
            recommendations = self._generate_cascade_recommendations(delayed_flights)
            
            alert = Alert(
                alert_id=alert_id,
                alert_type=AlertType.DELAY_CASCADE,
                severity=severity,
                title=f"Delay Cascade Detected at {airport}",
                description=f"{len(delayed_flights)} flights experiencing delays > {self.delay_threshold_minutes} minutes",
                airport=airport,
                timestamp=datetime.now(),
                affected_flights=[f.flight_id for f in delayed_flights],
                recommendations=recommendations,
                metrics={
                    "delayed_flights_count": len(delayed_flights),
                    "avg_delay": sum(f.dep_delay_min for f in delayed_flights) / len(delayed_flights),
                    "max_delay": max(f.dep_delay_min for f in delayed_flights)
                }
            )
            
            alerts.append(alert)
        
        return alerts
    
    def _check_critical_delays(self, airport: str, flights: List[Flight]) -> List[Alert]:
        """Check for critical individual flight delays."""
        alerts = []
        
        critical_delay_threshold = 60  # 1 hour
        
        for flight in flights:
            if (flight.dep_delay_min and flight.dep_delay_min > critical_delay_threshold):
                alert_id = f"critical_delay_{flight.flight_id}"
                
                # Skip if we already have an alert for this flight
                if alert_id in self.active_alerts:
                    continue
                
                recommendations = self._generate_critical_delay_recommendations(flight)
                
                alert = Alert(
                    alert_id=alert_id,
                    alert_type=AlertType.CRITICAL_FLIGHT_DELAY,
                    severity=AlertSeverity.CRITICAL,
                    title=f"Critical Delay: {flight.flight_number}",
                    description=f"Flight {flight.flight_number} delayed by {flight.dep_delay_min:.0f} minutes",
                    airport=airport,
                    timestamp=datetime.now(),
                    affected_flights=[flight.flight_id],
                    recommendations=recommendations,
                    metrics={
                        "delay_minutes": flight.dep_delay_min,
                        "flight_no": flight.flight_number,
                        "origin": flight.origin.code if flight.origin else "UNK",
                        "destination": flight.destination.code if flight.destination else "UNK"
                    }
                )
                
                alerts.append(alert)
        
        return alerts
    
    def _calculate_overload_severity(self, overload_window: OverloadWindow) -> AlertSeverity:
        """Calculate severity level for capacity overload."""
        if overload_window.peak_overload >= 15 or overload_window.duration_minutes >= 60:
            return AlertSeverity.CRITICAL
        elif overload_window.peak_overload >= 10 or overload_window.duration_minutes >= 30:
            return AlertSeverity.HIGH
        elif overload_window.peak_overload >= 5 or overload_window.duration_minutes >= 15:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW
    
    def _generate_overload_recommendations(self, overload_window: OverloadWindow) -> List[AlertRecommendation]:
        """Generate recommendations for capacity overload."""
        recommendations = []
        
        if overload_window.peak_overload >= 10:
            recommendations.extend([
                AlertRecommendation(
                    action="Implement ground delay program for arriving flights",
                    impact=f"Could reduce overload by ~{min(overload_window.peak_overload, 8)} flights",
                    priority=1,
                    estimated_improvement="30-50% delay reduction"
                ),
                AlertRecommendation(
                    action="Coordinate with adjacent airports for traffic redistribution",
                    impact="Distribute excess demand across regional airports",
                    priority=2,
                    estimated_improvement="20-30% capacity relief"
                ),
                AlertRecommendation(
                    action="Activate additional runway configurations if available",
                    impact="Increase capacity by 15-25%",
                    priority=3,
                    estimated_improvement="15-25% capacity increase"
                )
            ])
        elif overload_window.peak_overload >= 5:
            recommendations.extend([
                AlertRecommendation(
                    action="Optimize departure sequencing to reduce taxi delays",
                    impact=f"Could improve throughput by 2-3 flights per hour",
                    priority=1,
                    estimated_improvement="10-15% efficiency gain"
                ),
                AlertRecommendation(
                    action="Implement tactical flow management for peak period",
                    impact="Smooth traffic distribution over time",
                    priority=2,
                    estimated_improvement="20% delay reduction"
                ),
                AlertRecommendation(
                    action="Consider voluntary schedule adjustments with airlines",
                    impact="Reduce peak demand through collaboration",
                    priority=3,
                    estimated_improvement="Variable based on airline cooperation"
                )
            ])
        
        return recommendations[:3]  # Return top 3 recommendations
    
    def _generate_cascade_recommendations(self, delayed_flights: List[Flight]) -> List[AlertRecommendation]:
        """Generate recommendations for delay cascades."""
        return [
            AlertRecommendation(
                action="Prioritize departure clearances for connecting flights",
                impact=f"Reduce cascade impact for {len(delayed_flights)} delayed flights",
                priority=1,
                estimated_improvement="25-40% cascade reduction"
            ),
            AlertRecommendation(
                action="Coordinate with airlines for passenger rebooking options",
                impact="Minimize passenger disruption from delays",
                priority=2,
                estimated_improvement="Improved passenger experience"
            ),
            AlertRecommendation(
                action="Implement tactical ground stops if delays worsen",
                impact="Prevent further cascade propagation",
                priority=3,
                estimated_improvement="Stop cascade growth"
            )
        ]
    
    def _generate_critical_delay_recommendations(self, flight: Flight) -> List[AlertRecommendation]:
        """Generate recommendations for critical flight delays."""
        return [
            AlertRecommendation(
                action=f"Expedite departure clearance for {flight.flight_number}",
                impact="Minimize further delay accumulation",
                priority=1,
                estimated_improvement="5-10 minute delay reduction"
            ),
            AlertRecommendation(
                action="Coordinate with destination airport for priority arrival slot",
                impact="Reduce arrival delays and passenger impact",
                priority=2,
                estimated_improvement="Faster turnaround at destination"
            ),
            AlertRecommendation(
                action="Alert airline operations for passenger service recovery",
                impact="Proactive passenger communication and rebooking",
                priority=3,
                estimated_improvement="Improved customer satisfaction"
            )
        ]
    
    def _create_system_error_alert(self, airport: str, error_message: str) -> Alert:
        """Create a system error alert."""
        return Alert(
            alert_id=f"system_error_{airport}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            alert_type=AlertType.SYSTEM_ERROR,
            severity=AlertSeverity.HIGH,
            title=f"System Error at {airport}",
            description=f"Error in alerting system: {error_message}",
            airport=airport,
            timestamp=datetime.now(),
            recommendations=[
                AlertRecommendation(
                    action="Check system logs for detailed error information",
                    impact="Identify root cause of system issue",
                    priority=1
                ),
                AlertRecommendation(
                    action="Verify data connectivity and service health",
                    impact="Ensure all components are operational",
                    priority=2
                ),
                AlertRecommendation(
                    action="Contact system administrator if issue persists",
                    impact="Escalate for technical resolution",
                    priority=3
                )
            ],
            metrics={"error_message": error_message}
        )
    
    def send_alert_notification(self, alert: Alert) -> bool:
        """Send alert notification via Slack webhook.
        
        Args:
            alert: Alert to send notification for
            
        Returns:
            True if notification sent successfully, False otherwise
        """
        if not self.slack_webhook_url:
            self.logger.warning("No Slack webhook URL configured, skipping notification")
            return False
        
        try:
            # Build Slack message
            message = self._build_slack_message(alert)
            
            # Send webhook request
            response = self.session.post(
                self.slack_webhook_url,
                json=message,
                timeout=10
            )
            
            if response.status_code == 200:
                self.logger.info(f"Alert notification sent successfully: {alert.alert_id}")
                return True
            else:
                self.logger.error(f"Failed to send alert notification: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error sending alert notification: {e}")
            return False
    
    def _build_slack_message(self, alert: Alert) -> Dict[str, Any]:
        """Build Slack message payload for alert."""
        # Severity emoji mapping
        severity_emoji = {
            AlertSeverity.LOW: "🟡",
            AlertSeverity.MEDIUM: "🟠", 
            AlertSeverity.HIGH: "🔴",
            AlertSeverity.CRITICAL: "🚨"
        }
        
        # Alert type emoji mapping
        type_emoji = {
            AlertType.CAPACITY_OVERLOAD: "📊",
            AlertType.DELAY_CASCADE: "⛓️",
            AlertType.CRITICAL_FLIGHT_DELAY: "✈️",
            AlertType.SYSTEM_ERROR: "⚠️"
        }
        
        emoji = severity_emoji.get(alert.severity, "⚠️")
        type_icon = type_emoji.get(alert.alert_type, "📋")
        
        # Build main message
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} {alert.title}"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"{type_icon} *{alert.alert_type.value.replace('_', ' ').title()}*\n"
                           f"📍 *Airport:* {alert.airport}\n"
                           f"⏰ *Time:* {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
                           f"🔥 *Severity:* {alert.severity.value.upper()}\n\n"
                           f"*Description:*\n{alert.description}"
                }
            }
        ]
        
        # Add metrics if available
        if alert.metrics:
            metrics_text = ""
            for key, value in alert.metrics.items():
                formatted_key = key.replace('_', ' ').title()
                if isinstance(value, float):
                    metrics_text += f"• *{formatted_key}:* {value:.1f}\n"
                else:
                    metrics_text += f"• *{formatted_key}:* {value}\n"
            
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Key Metrics:*\n{metrics_text}"
                }
            })
        
        # Add recommendations
        if alert.recommendations:
            rec_text = ""
            for i, rec in enumerate(alert.recommendations[:3], 1):  # Top 3 recommendations
                rec_text += f"{i}. *{rec.action}*\n   _{rec.impact}_\n"
                if rec.estimated_improvement:
                    rec_text += f"   📈 {rec.estimated_improvement}\n"
                rec_text += "\n"
            
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*🎯 Top Recommendations:*\n{rec_text}"
                }
            })
        
        # Add affected flights if any
        if alert.affected_flights:
            flights_text = ", ".join(alert.affected_flights[:10])  # Show first 10
            if len(alert.affected_flights) > 10:
                flights_text += f" and {len(alert.affected_flights) - 10} more"
            
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*✈️ Affected Flights:*\n{flights_text}"
                }
            })
        
        return {
            "text": f"{emoji} {alert.title}",
            "blocks": blocks
        }
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved and send confirmation notification.
        
        Args:
            alert_id: ID of alert to resolve
            
        Returns:
            True if alert was resolved successfully, False otherwise
        """
        if alert_id not in self.active_alerts:
            self.logger.warning(f"Alert {alert_id} not found in active alerts")
            return False
        
        alert = self.active_alerts[alert_id]
        alert.resolved = True
        alert.resolved_at = datetime.now()
        
        # Remove from active alerts
        del self.active_alerts[alert_id]
        
        # Send resolution notification
        self._send_resolution_notification(alert)
        
        self.logger.info(f"Alert {alert_id} resolved")
        return True
    
    def _send_resolution_notification(self, alert: Alert) -> bool:
        """Send alert resolution notification."""
        if not self.slack_webhook_url:
            return False
        
        try:
            message = {
                "text": f"✅ Alert Resolved: {alert.title}",
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"✅ *Alert Resolved*\n\n"
                                   f"*Alert:* {alert.title}\n"
                                   f"*Airport:* {alert.airport}\n"
                                   f"*Resolved At:* {alert.resolved_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
                                   f"*Duration:* {(alert.resolved_at - alert.timestamp).total_seconds() / 60:.0f} minutes"
                        }
                    }
                ]
            }
            
            response = self.session.post(
                self.slack_webhook_url,
                json=message,
                timeout=10
            )
            
            return response.status_code == 200
            
        except Exception as e:
            self.logger.error(f"Error sending resolution notification: {e}")
            return False
    
    def escalate_alert(self, alert_id: str) -> bool:
        """Escalate an alert to higher severity level.
        
        Args:
            alert_id: ID of alert to escalate
            
        Returns:
            True if alert was escalated successfully, False otherwise
        """
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        
        # Don't escalate if already critical or already escalated
        if alert.severity == AlertSeverity.CRITICAL or alert.escalated:
            return False
        
        # Escalate severity
        severity_order = [AlertSeverity.LOW, AlertSeverity.MEDIUM, AlertSeverity.HIGH, AlertSeverity.CRITICAL]
        current_index = severity_order.index(alert.severity)
        if current_index < len(severity_order) - 1:
            alert.severity = severity_order[current_index + 1]
        
        alert.escalated = True
        alert.escalated_at = datetime.now()
        
        # Send escalation notification
        self._send_escalation_notification(alert)
        
        self.logger.warning(f"Alert {alert_id} escalated to {alert.severity.value}")
        return True
    
    def _send_escalation_notification(self, alert: Alert) -> bool:
        """Send alert escalation notification."""
        if not self.slack_webhook_url:
            return False
        
        try:
            message = {
                "text": f"🚨 ESCALATED: {alert.title}",
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"🚨 *ALERT ESCALATED*\n\n"
                                   f"*Alert:* {alert.title}\n"
                                   f"*Airport:* {alert.airport}\n"
                                   f"*New Severity:* {alert.severity.value.upper()}\n"
                                   f"*Escalated At:* {alert.escalated_at.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                                   f"*Immediate action required!*"
                        }
                    }
                ]
            }
            
            response = self.session.post(
                self.slack_webhook_url,
                json=message,
                timeout=10
            )
            
            return response.status_code == 200
            
        except Exception as e:
            self.logger.error(f"Error sending escalation notification: {e}")
            return False
    
    def get_active_alerts(self, airport: Optional[str] = None, 
                         severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get active alerts with optional filtering.
        
        Args:
            airport: Filter by airport code
            severity: Filter by severity level
            
        Returns:
            List of active alerts matching filters
        """
        alerts = list(self.active_alerts.values())
        
        if airport:
            alerts = [a for a in alerts if a.airport == airport]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        # Sort by severity (critical first) then by timestamp
        severity_order = {
            AlertSeverity.CRITICAL: 0,
            AlertSeverity.HIGH: 1,
            AlertSeverity.MEDIUM: 2,
            AlertSeverity.LOW: 3
        }
        
        alerts.sort(key=lambda a: (severity_order[a.severity], a.timestamp))
        
        return alerts
    
    def get_alert_summary(self, airport: Optional[str] = None) -> Dict[str, Any]:
        """Get summary of alert status.
        
        Args:
            airport: Filter by airport code
            
        Returns:
            Dictionary with alert summary statistics
        """
        alerts = self.get_active_alerts(airport)
        
        summary = {
            "total_active_alerts": len(alerts),
            "by_severity": {
                "critical": len([a for a in alerts if a.severity == AlertSeverity.CRITICAL]),
                "high": len([a for a in alerts if a.severity == AlertSeverity.HIGH]),
                "medium": len([a for a in alerts if a.severity == AlertSeverity.MEDIUM]),
                "low": len([a for a in alerts if a.severity == AlertSeverity.LOW])
            },
            "by_type": {
                "capacity_overload": len([a for a in alerts if a.alert_type == AlertType.CAPACITY_OVERLOAD]),
                "delay_cascade": len([a for a in alerts if a.alert_type == AlertType.DELAY_CASCADE]),
                "critical_flight_delay": len([a for a in alerts if a.alert_type == AlertType.CRITICAL_FLIGHT_DELAY]),
                "system_error": len([a for a in alerts if a.alert_type == AlertType.SYSTEM_ERROR])
            },
            "escalated_alerts": len([a for a in alerts if a.escalated]),
            "oldest_alert": min([a.timestamp for a in alerts]) if alerts else None,
            "most_recent_alert": max([a.timestamp for a in alerts]) if alerts else None
        }
        
        return summary