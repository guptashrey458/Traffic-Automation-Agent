"""Console-based alerting system that mimics Slack notifications."""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts."""
    CAPACITY_OVERLOAD = "capacity_overload"
    WEATHER_IMPACT = "weather_impact"
    DELAY_CASCADE = "delay_cascade"
    RUNWAY_CLOSURE = "runway_closure"
    SYSTEM_OPTIMIZATION = "system_optimization"
    AUTONOMOUS_ACTION = "autonomous_action"


@dataclass
class ConsoleAlert:
    """Console alert data structure."""
    alert_id: str
    timestamp: datetime
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    airport: Optional[str] = None
    affected_flights: int = 0
    recommendations: List[str] = None
    metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []
        if self.metrics is None:
            self.metrics = {}


class ConsoleAlertingService:
    """Service for console-based alerting that mimics Slack notifications."""
    
    def __init__(self):
        self.alert_history: List[ConsoleAlert] = []
        self.active_alerts: Dict[str, ConsoleAlert] = {}
        
        # Emoji mappings for different alert types and severities
        self.severity_emojis = {
            AlertSeverity.LOW: "🟢",
            AlertSeverity.MEDIUM: "🟡",
            AlertSeverity.HIGH: "🔴",
            AlertSeverity.CRITICAL: "🚨"
        }
        
        self.type_emojis = {
            AlertType.CAPACITY_OVERLOAD: "📈",
            AlertType.WEATHER_IMPACT: "🌦️",
            AlertType.DELAY_CASCADE: "⛓️",
            AlertType.RUNWAY_CLOSURE: "🚧",
            AlertType.SYSTEM_OPTIMIZATION: "🤖",
            AlertType.AUTONOMOUS_ACTION: "⚡"
        }
    
    def send_alert(self, alert: ConsoleAlert) -> None:
        """Send a console alert in Slack-style format."""
        try:
            # Store alert
            self.alert_history.append(alert)
            self.active_alerts[alert.alert_id] = alert
            
            # Print Slack-style alert
            self._print_slack_style_alert(alert)
            
            logger.info(f"Console alert sent: {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Error sending console alert: {e}")
    
    def _print_slack_style_alert(self, alert: ConsoleAlert) -> None:
        """Print alert in Slack-style format."""
        severity_emoji = self.severity_emojis.get(alert.severity, "ℹ️")
        type_emoji = self.type_emojis.get(alert.alert_type, "📢")
        timestamp_str = alert.timestamp.strftime("%H:%M:%S")
        
        # Header
        print(f"\n{severity_emoji} {type_emoji} **FLIGHT OPERATIONS ALERT** [{timestamp_str}]")
        print("─" * 70)
        
        # Title and severity
        print(f"🎯 **{alert.title}**")
        print(f"📊 Severity: {alert.severity.value.upper()}")
        
        # Airport if specified
        if alert.airport:
            print(f"🛫 Airport: {alert.airport}")
        
        # Affected flights
        if alert.affected_flights > 0:
            print(f"✈️  Affected Flights: {alert.affected_flights}")
        
        # Main message
        print(f"💬 {alert.message}")
        
        # Metrics if available
        if alert.metrics:
            print("📈 **Metrics:**")
            for key, value in alert.metrics.items():
                if isinstance(value, float):
                    print(f"   • {key}: {value:.2f}")
                else:
                    print(f"   • {key}: {value}")
        
        # Recommendations
        if alert.recommendations:
            print("💡 **Recommended Actions:**")
            for i, rec in enumerate(alert.recommendations, 1):
                print(f"   {i}. {rec}")
        
        # Footer
        print("─" * 70)
        print(f"🤖 Autonomous Flight Scheduler | Alert ID: {alert.alert_id}")
        print()
    
    def send_capacity_overload_alert(self, airport: str, overload_percentage: float, 
                                   affected_flights: int, time_window: str) -> str:
        """Send capacity overload alert."""
        severity = AlertSeverity.HIGH if overload_percentage > 1.2 else AlertSeverity.MEDIUM
        
        alert = ConsoleAlert(
            alert_id=f"CAP_{airport}_{datetime.now().strftime('%H%M%S')}",
            timestamp=datetime.now(),
            alert_type=AlertType.CAPACITY_OVERLOAD,
            severity=severity,
            title=f"Capacity Overload Detected at {airport}",
            message=f"Airport capacity exceeded by {(overload_percentage-1)*100:.1f}% in {time_window} window",
            airport=airport,
            affected_flights=affected_flights,
            metrics={
                "Overload Percentage": f"{overload_percentage*100:.1f}%",
                "Time Window": time_window,
                "Current Demand": affected_flights,
                "Capacity Threshold": int(affected_flights / overload_percentage)
            },
            recommendations=[
                f"Optimize {min(10, affected_flights//2)} high-impact flights",
                "Consider runway reallocation",
                "Implement ground delay program if needed"
            ]
        )
        
        self.send_alert(alert)
        return alert.alert_id
    
    def send_weather_impact_alert(self, airport: str, weather_regime: str, 
                                capacity_reduction: float, affected_flights: int) -> str:
        """Send weather impact alert."""
        severity_map = {
            "calm": AlertSeverity.LOW,
            "medium": AlertSeverity.MEDIUM,
            "strong": AlertSeverity.HIGH,
            "severe": AlertSeverity.CRITICAL
        }
        
        severity = severity_map.get(weather_regime.lower(), AlertSeverity.MEDIUM)
        
        alert = ConsoleAlert(
            alert_id=f"WX_{airport}_{datetime.now().strftime('%H%M%S')}",
            timestamp=datetime.now(),
            alert_type=AlertType.WEATHER_IMPACT,
            severity=severity,
            title=f"Weather Impact at {airport}",
            message=f"{weather_regime.title()} weather reducing capacity by {capacity_reduction*100:.0f}%",
            airport=airport,
            affected_flights=affected_flights,
            metrics={
                "Weather Regime": weather_regime.title(),
                "Capacity Reduction": f"{capacity_reduction*100:.0f}%",
                "Affected Operations": affected_flights,
                "Estimated Delay Impact": f"{capacity_reduction * 15:.1f} minutes"
            },
            recommendations=[
                "Activate weather contingency procedures",
                "Proactively reschedule weather-sensitive flights",
                "Coordinate with ATC for flow management"
            ]
        )
        
        self.send_alert(alert)
        return alert.alert_id
    
    def send_autonomous_action_alert(self, action_type: str, confidence: float, 
                                   affected_flights: int, expected_benefit: str) -> str:
        """Send autonomous action notification."""
        alert = ConsoleAlert(
            alert_id=f"AUTO_{datetime.now().strftime('%H%M%S')}",
            timestamp=datetime.now(),
            alert_type=AlertType.AUTONOMOUS_ACTION,
            severity=AlertSeverity.MEDIUM,
            title="Autonomous Action Executed",
            message=f"System executed {action_type} with {confidence*100:.0f}% confidence",
            affected_flights=affected_flights,
            metrics={
                "Action Type": action_type,
                "Confidence Level": f"{confidence*100:.0f}%",
                "Expected Benefit": expected_benefit,
                "Execution Time": "< 30 seconds"
            },
            recommendations=[
                "Monitor system performance",
                "Review optimization results",
                "Validate autonomous decisions"
            ]
        )
        
        self.send_alert(alert)
        return alert.alert_id
    
    def send_optimization_complete_alert(self, flights_optimized: int, delay_reduction: float,
                                       runway_changes: int, success_rate: float) -> str:
        """Send optimization completion notification."""
        alert = ConsoleAlert(
            alert_id=f"OPT_{datetime.now().strftime('%H%M%S')}",
            timestamp=datetime.now(),
            alert_type=AlertType.SYSTEM_OPTIMIZATION,
            severity=AlertSeverity.LOW,
            title="Schedule Optimization Completed",
            message=f"Successfully optimized {flights_optimized} flights with {delay_reduction:.1f}min delay reduction",
            affected_flights=flights_optimized,
            metrics={
                "Flights Optimized": flights_optimized,
                "Delay Reduction": f"{delay_reduction:.1f} minutes",
                "Runway Changes": runway_changes,
                "Success Rate": f"{success_rate*100:.0f}%",
                "Optimization Time": "< 5 minutes"
            },
            recommendations=[
                "Implement optimized schedule",
                "Monitor for cascade effects",
                "Update operational procedures"
            ]
        )
        
        self.send_alert(alert)
        return alert.alert_id
    
    def send_delay_cascade_alert(self, trigger_flight: str, cascade_depth: int,
                               downstream_flights: int, estimated_impact: float) -> str:
        """Send delay cascade alert."""
        severity = AlertSeverity.HIGH if cascade_depth > 3 else AlertSeverity.MEDIUM
        
        alert = ConsoleAlert(
            alert_id=f"CASCADE_{datetime.now().strftime('%H%M%S')}",
            timestamp=datetime.now(),
            alert_type=AlertType.DELAY_CASCADE,
            severity=severity,
            title="Delay Cascade Detected",
            message=f"Flight {trigger_flight} triggering cascade affecting {downstream_flights} flights",
            affected_flights=downstream_flights,
            metrics={
                "Trigger Flight": trigger_flight,
                "Cascade Depth": cascade_depth,
                "Downstream Impact": downstream_flights,
                "Estimated Total Delay": f"{estimated_impact:.1f} minutes"
            },
            recommendations=[
                f"Prioritize {trigger_flight} for immediate attention",
                "Consider alternative routing for affected flights",
                "Activate cascade mitigation procedures"
            ]
        )
        
        self.send_alert(alert)
        return alert.alert_id
    
    def resolve_alert(self, alert_id: str, resolution_message: str = None) -> bool:
        """Resolve an active alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            del self.active_alerts[alert_id]
            
            # Print resolution notification
            print(f"\n✅ **ALERT RESOLVED** [{datetime.now().strftime('%H:%M:%S')}]")
            print("─" * 50)
            print(f"🎯 Alert: {alert.title}")
            print(f"🆔 ID: {alert_id}")
            if resolution_message:
                print(f"💬 Resolution: {resolution_message}")
            print("─" * 50)
            print()
            
            logger.info(f"Alert resolved: {alert_id}")
            return True
        
        return False
    
    def get_active_alerts(self) -> List[ConsoleAlert]:
        """Get list of active alerts."""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 50) -> List[ConsoleAlert]:
        """Get alert history."""
        return self.alert_history[-limit:]
    
    def print_alert_summary(self) -> None:
        """Print summary of alert activity."""
        total_alerts = len(self.alert_history)
        active_count = len(self.active_alerts)
        
        if total_alerts == 0:
            print("📊 No alerts generated")
            return
        
        # Count by severity
        severity_counts = {}
        for alert in self.alert_history:
            severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1
        
        # Count by type
        type_counts = {}
        for alert in self.alert_history:
            type_counts[alert.alert_type] = type_counts.get(alert.alert_type, 0) + 1
        
        print(f"\n📊 **ALERT ACTIVITY SUMMARY**")
        print("─" * 40)
        print(f"Total Alerts: {total_alerts}")
        print(f"Active Alerts: {active_count}")
        print(f"Resolved Alerts: {total_alerts - active_count}")
        print()
        
        print("By Severity:")
        for severity, count in severity_counts.items():
            emoji = self.severity_emojis.get(severity, "ℹ️")
            print(f"   {emoji} {severity.value.title()}: {count}")
        
        print("\nBy Type:")
        for alert_type, count in type_counts.items():
            emoji = self.type_emojis.get(alert_type, "📢")
            print(f"   {emoji} {alert_type.value.replace('_', ' ').title()}: {count}")
        
        print("─" * 40)
        print()


# Global instance
console_alerting_service = ConsoleAlertingService()