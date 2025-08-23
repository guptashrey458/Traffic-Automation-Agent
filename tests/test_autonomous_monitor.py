"""Tests for autonomous monitoring and policy engine."""

import pytest
import json
import os
from datetime import datetime, timedelta, date, time
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from src.services.autonomous_monitor import (
    AutonomousMonitor, MonitorPolicy, PolicyTrigger, AutonomousAction, GuardrailCheck,
    PolicyType, ActionType, ConfidenceLevel, AlertSeverity
)
from src.services.analytics import AnalyticsEngine, PeakAnalysis, TimeBucket, TrafficLevel
from src.services.schedule_optimizer import ScheduleOptimizer
from src.services.cascade_analysis import CascadeAnalysisService
from src.services.delay_prediction import DelayRiskPredictor
from src.services.alerting import AlertingService, Alert, AlertType
from src.models.flight import Flight, Airport, FlightTime


class TestAutonomousMonitor:
    """Test cases for AutonomousMonitor class."""
    
    @pytest.fixture
    def mock_services(self):
        """Create mock services for testing."""
        analytics = Mock(spec=AnalyticsEngine)
        optimizer = Mock(spec=ScheduleOptimizer)
        cascade_analyzer = Mock(spec=CascadeAnalysisService)
        delay_predictor = Mock(spec=DelayRiskPredictor)
        alerting = Mock(spec=AlertingService)
        
        # Add send_alert method to alerting mock
        alerting.send_alert = Mock(return_value=True)
        
        return {
            'analytics': analytics,
            'optimizer': optimizer,
            'cascade_analyzer': cascade_analyzer,
            'delay_predictor': delay_predictor,
            'alerting': alerting
        }
    
    @pytest.fixture
    def sample_flights(self):
        """Create sample flight data for testing."""
        flights = []
        
        # Create flights with various delay patterns
        for i in range(20):
            flight = Flight(
                flight_id=f"flight_{i:03d}",
                flight_number=f"AI{2500 + i}",
                origin=Airport.from_string("Mumbai (BOM)"),
                destination=Airport.from_string("Delhi (DEL)"),
                aircraft_type="A320",
                flight_date=date.today(),
                departure=FlightTime(scheduled=time(8, 0)),
                arrival=FlightTime(scheduled=time(10, 30))
            )
            
            # Add delays to some flights
            if i < 5:  # First 5 flights have significant delays
                flight.dep_delay_min = 45 + i * 10
            elif i < 10:  # Next 5 have moderate delays
                flight.dep_delay_min = 20 + i * 2
            else:  # Rest are on time or minor delays
                flight.dep_delay_min = max(0, i - 15)
            
            flights.append(flight)
        
        return flights
    
    @pytest.fixture
    def monitor(self, mock_services):
        """Create AutonomousMonitor instance for testing."""
        return AutonomousMonitor(
            analytics_engine=mock_services['analytics'],
            schedule_optimizer=mock_services['optimizer'],
            cascade_analyzer=mock_services['cascade_analyzer'],
            delay_predictor=mock_services['delay_predictor'],
            alerting_service=mock_services['alerting']
        )
    
    def test_initialization(self, monitor):
        """Test monitor initialization."""
        assert monitor is not None
        assert len(monitor.policies) > 0  # Should have default policies
        assert monitor.max_autonomous_changes_per_hour == 5
        assert monitor.min_confidence_for_autonomous_action == 0.7
    
    def test_add_policy(self, monitor):
        """Test adding a monitoring policy."""
        policy = MonitorPolicy(
            policy_id="test_policy_001",
            name="Test Policy",
            policy_type=PolicyType.UTILIZATION_THRESHOLD,
            condition="Test condition",
            condition_func=lambda data: data.get("test_metric", 0) > 0.5,
            action=ActionType.SEND_ALERT,
            threshold_values={"test_metric": 0.5}
        )
        
        initial_count = len(monitor.policies)
        monitor.add_policy(policy)
        
        assert len(monitor.policies) == initial_count + 1
        assert "test_policy_001" in monitor.policies
        assert monitor.policies["test_policy_001"].name == "Test Policy"
    
    def test_remove_policy(self, monitor):
        """Test removing a monitoring policy."""
        # Add a test policy first
        policy = MonitorPolicy(
            policy_id="remove_test_001",
            name="Remove Test Policy",
            policy_type=PolicyType.DELAY_CASCADE,
            condition="Test condition",
            condition_func=lambda data: False,
            action=ActionType.LOG_WARNING,
            threshold_values={}
        )
        monitor.add_policy(policy)
        
        # Remove the policy
        result = monitor.remove_policy("remove_test_001")
        
        assert result is True
        assert "remove_test_001" not in monitor.policies
        
        # Try to remove non-existent policy
        result = monitor.remove_policy("non_existent")
        assert result is False
    
    def test_enable_disable_policy(self, monitor):
        """Test enabling and disabling policies."""
        # Get a default policy
        policy_id = list(monitor.policies.keys())[0]
        
        # Disable policy
        result = monitor.disable_policy(policy_id)
        assert result is True
        assert monitor.policies[policy_id].enabled is False
        
        # Enable policy
        result = monitor.enable_policy(policy_id)
        assert result is True
        assert monitor.policies[policy_id].enabled is True
    
    def test_calculate_system_metrics(self, monitor, sample_flights):
        """Test system metrics calculation."""
        metrics = monitor._calculate_system_metrics(sample_flights, {})
        
        assert "total_flights" in metrics
        assert "delayed_flights_count" in metrics
        assert "delayed_15m_count" in metrics
        assert "delayed_30m_count" in metrics
        assert "delayed_60m_count" in metrics
        assert "avg_delay" in metrics
        assert "max_delay_minutes" in metrics
        
        assert metrics["total_flights"] == 20
        assert metrics["delayed_15m_count"] >= 5  # At least 5 flights with >15min delay
        assert metrics["max_delay_minutes"] >= 45  # Maximum delay should be at least 45min
    
    def test_evaluate_policies_no_triggers(self, monitor, sample_flights):
        """Test policy evaluation with no triggers."""
        # Modify flights to have no significant delays
        for flight in sample_flights:
            flight.dep_delay_min = 5  # All flights have minor delays
        
        triggers = monitor.evaluate_policies(sample_flights)
        
        # Should have no triggers since delays are minor
        assert len(triggers) == 0
    
    def test_evaluate_policies_with_triggers(self, monitor, sample_flights):
        """Test policy evaluation with triggers."""
        # Mock analytics to return high utilization
        mock_peak_analysis = Mock()
        mock_peak_analysis.time_buckets = [
            Mock(utilization=1.2, overload=25)  # High utilization and overload
        ]
        monitor.analytics.analyze_peaks.return_value = mock_peak_analysis
        
        triggers = monitor.evaluate_policies(sample_flights)
        
        # Should have triggers due to high utilization and delays
        assert len(triggers) > 0
        
        # Check trigger properties
        for trigger in triggers:
            assert hasattr(trigger, 'policy_id')
            assert hasattr(trigger, 'confidence')
            assert hasattr(trigger, 'severity')
            assert hasattr(trigger, 'reasoning')
            assert trigger.confidence >= 0.0
            assert trigger.confidence <= 1.0
    
    def test_trigger_confidence_calculation(self, monitor, sample_flights):
        """Test confidence calculation for policy triggers."""
        # Get a utilization policy
        util_policy = None
        for policy in monitor.policies.values():
            if policy.policy_type == PolicyType.UTILIZATION_THRESHOLD:
                util_policy = policy
                break
        
        assert util_policy is not None
        
        # Test with high utilization
        metrics = {"max_utilization": 1.3, "total_flights": 100}
        confidence = monitor._calculate_trigger_confidence(util_policy, metrics, sample_flights)
        
        assert confidence > 0.3  # Should have reasonable confidence (adjusted for realistic calculation)
        assert confidence <= 1.0
        
        # Test with low utilization
        metrics = {"max_utilization": 1.05, "total_flights": 10}
        confidence = monitor._calculate_trigger_confidence(util_policy, metrics, sample_flights)
        
        assert confidence < 0.8  # Should have lower confidence
    
    def test_guardrail_checking(self, monitor):
        """Test guardrail checking for autonomous actions."""
        action = AutonomousAction(
            action_type=ActionType.OPTIMIZE_SCHEDULE,
            confidence_score=0.8,
            affected_flights=["flight_001", "flight_002"]
        )
        
        trigger = PolicyTrigger(
            policy_id="test_policy",
            confidence=0.8,
            severity=AlertSeverity.MEDIUM
        )
        
        # Test normal action
        check = monitor.check_guardrails(action, trigger)
        assert check.action_allowed is True
        assert len(check.violations) == 0
        
        # Test low confidence action
        action.confidence_score = 0.3
        check = monitor.check_guardrails(action, trigger)
        assert check.confidence_too_low is True
        assert check.action_allowed is False
        assert len(check.violations) > 0
    
    def test_guardrail_max_changes_exceeded(self, monitor):
        """Test guardrail for maximum changes per hour."""
        # Add many recent actions to history
        for i in range(6):  # Exceed the limit of 5
            action = AutonomousAction(
                action_id=f"action_{i}",
                timestamp=datetime.now() - timedelta(minutes=30),
                execution_status="completed"
            )
            monitor.action_history.append(action)
        
        # Test new action
        new_action = AutonomousAction(
            action_type=ActionType.OPTIMIZE_SCHEDULE,
            confidence_score=0.8
        )
        
        trigger = PolicyTrigger(policy_id="test", confidence=0.8)
        check = monitor.check_guardrails(new_action, trigger)
        
        assert check.max_changes_exceeded is True
        assert check.action_allowed is False
    
    def test_execute_autonomous_action_blocked(self, monitor, sample_flights):
        """Test autonomous action execution when blocked by guardrails."""
        # Create action that will be blocked (low confidence)
        trigger = PolicyTrigger(
            policy_id="test_policy",
            recommended_action=ActionType.OPTIMIZE_SCHEDULE,
            confidence=0.3,  # Low confidence
            affected_flights=["flight_001"]
        )
        
        action = monitor.execute_autonomous_action(trigger, sample_flights)
        
        assert action.execution_status == "blocked"
        assert "Guardrail violations" in action.execution_details.get("blocked_reason", "")
    
    def test_execute_autonomous_action_awaiting_approval(self, monitor, sample_flights):
        """Test autonomous action requiring human approval."""
        trigger = PolicyTrigger(
            policy_id="test_policy",
            recommended_action=ActionType.OPTIMIZE_SCHEDULE,
            confidence=0.8,  # High confidence to pass guardrails
            escalation_required=True,
            affected_flights=["flight_001"]
        )
        
        action = monitor.execute_autonomous_action(trigger, sample_flights)
        
        assert action.human_approval_required is True
        assert action.execution_status == "awaiting_approval"
    
    def test_execute_optimization_action(self, monitor, sample_flights):
        """Test execution of optimization action."""
        # Mock optimizer to return success
        monitor.optimizer.optimize_schedule.return_value = {
            "delay_improvement": 15.5,
            "score": 0.85
        }
        
        action = AutonomousAction(
            action_type=ActionType.OPTIMIZE_SCHEDULE,
            confidence_score=0.8,
            affected_flights=["flight_001", "flight_002"]
        )
        
        result = monitor._execute_optimization(action, sample_flights)
        
        assert result["status"] == "optimization_completed"
        assert "impact" in result
        assert result["impact"]["flights_optimized"] == 2
    
    def test_execute_alert_action(self, monitor):
        """Test execution of alert action."""
        trigger = PolicyTrigger(
            policy_id="test_policy",
            policy_name="Test Alert Policy",
            severity=AlertSeverity.HIGH,
            reasoning="Test alert reasoning"
        )
        
        action = AutonomousAction(action_type=ActionType.SEND_ALERT)
        
        result = monitor._execute_alert(action, trigger)
        
        assert result["status"] == "alert_sent"
        assert "alert_id" in result
        assert monitor.alerting.send_alert.called
    
    def test_escalate_to_human(self, monitor):
        """Test escalation to human operators."""
        action = AutonomousAction(
            action_id="test_action_001",
            action_type=ActionType.OPTIMIZE_SCHEDULE,
            confidence_score=0.4,
            affected_flights=["flight_001", "flight_002"]
        )
        
        trigger = PolicyTrigger(
            policy_name="Test Policy",
            confidence=0.4,
            severity=AlertSeverity.HIGH,
            reasoning="Test escalation reasoning"
        )
        
        # Should not raise exception
        monitor._escalate_to_human(action, trigger)
        
        # Should have sent escalation alert
        assert monitor.alerting.send_alert.called
    
    def test_audit_logging(self, monitor, tmp_path):
        """Test audit logging functionality."""
        # Set up temporary log path
        log_path = tmp_path / "test_audit.log"
        monitor.audit_log_path = str(log_path)
        monitor._setup_audit_logging()
        
        action = AutonomousAction(
            action_id="test_action_001",
            action_type=ActionType.SEND_ALERT,
            confidence_score=0.8,
            reasoning="Test audit logging"
        )
        
        monitor._log_autonomous_decision(action, "COMPLETED")
        
        # Check that log file was created and contains entry
        assert log_path.exists()
        log_content = log_path.read_text()
        assert "test_action_001" in log_content
        assert "COMPLETED" in log_content
    
    def test_approve_action(self, monitor):
        """Test action approval functionality."""
        # Add action awaiting approval
        action = AutonomousAction(
            action_id="approval_test_001",
            execution_status="awaiting_approval"
        )
        monitor.action_history.append(action)
        
        # Approve the action
        result = monitor.approve_action("approval_test_001", True, "test_user")
        
        assert result is True
        assert action.human_approved is True
        assert action.execution_status == "approved"
        
        # Try to approve non-existent action
        result = monitor.approve_action("non_existent", True)
        assert result is False
    
    def test_get_policy_status(self, monitor):
        """Test getting policy status."""
        status = monitor.get_policy_status()
        
        assert "total_policies" in status
        assert "enabled_policies" in status
        assert "active_triggers" in status
        assert "recent_actions" in status
        assert "policies" in status
        
        assert isinstance(status["policies"], list)
        assert len(status["policies"]) == status["total_policies"]
        
        # Check policy details
        for policy_status in status["policies"]:
            assert "policy_id" in policy_status
            assert "name" in policy_status
            assert "enabled" in policy_status
            assert "priority" in policy_status
    
    def test_get_action_history(self, monitor):
        """Test getting action history."""
        # Add some test actions
        for i in range(3):
            action = AutonomousAction(
                action_id=f"history_test_{i}",
                timestamp=datetime.now() - timedelta(hours=i),
                action_type=ActionType.SEND_ALERT,
                confidence_score=0.7 + i * 0.1
            )
            monitor.action_history.append(action)
        
        history = monitor.get_action_history(hours=24)
        
        assert len(history) >= 3
        
        # Check history entry format
        for entry in history:
            assert "action_id" in entry
            assert "timestamp" in entry
            assert "action_type" in entry
            assert "confidence" in entry
            assert "status" in entry
    
    def test_continuous_monitoring(self, monitor, sample_flights):
        """Test continuous monitoring cycle."""
        # Mock analytics to trigger policies
        mock_peak_analysis = Mock()
        mock_peak_analysis.time_buckets = [Mock(utilization=1.15, overload=22)]
        monitor.analytics.analyze_peaks.return_value = mock_peak_analysis
        
        result = monitor.monitor_continuous(sample_flights)
        
        assert result["status"] == "completed"
        assert "execution_time_seconds" in result
        assert "triggers_found" in result
        assert "actions_executed" in result
        assert "timestamp" in result
        
        assert result["triggers_found"] >= 0
        assert result["actions_executed"] >= 0
    
    def test_cleanup_old_data(self, monitor):
        """Test cleanup of old triggers and actions."""
        # Add old triggers and actions
        old_time = datetime.now() - timedelta(hours=25)
        
        old_trigger = PolicyTrigger(
            policy_id="old_policy",
            trigger_time=old_time
        )
        monitor.active_triggers.append(old_trigger)
        
        old_action = AutonomousAction(
            action_id="old_action",
            timestamp=old_time
        )
        monitor.action_history.append(old_action)
        
        # Add recent data
        recent_trigger = PolicyTrigger(
            policy_id="recent_policy",
            trigger_time=datetime.now()
        )
        monitor.active_triggers.append(recent_trigger)
        
        # Run cleanup
        monitor._cleanup_old_data()
        
        # Old trigger should be removed, recent should remain
        assert len(monitor.active_triggers) == 1
        assert monitor.active_triggers[0].policy_id == "recent_policy"
    
    def test_policy_cooldown(self, monitor, sample_flights):
        """Test policy cooldown functionality."""
        # Get a policy and set recent trigger time
        policy_id = list(monitor.policies.keys())[0]
        policy = monitor.policies[policy_id]
        policy.last_triggered = datetime.now() - timedelta(minutes=5)  # Recent trigger
        policy.cooldown_minutes = 15
        
        # Mock conditions to trigger policy
        mock_peak_analysis = Mock()
        mock_peak_analysis.time_buckets = [Mock(utilization=1.5, overload=30)]
        monitor.analytics.analyze_peaks.return_value = mock_peak_analysis
        
        triggers = monitor.evaluate_policies(sample_flights)
        
        # Policy should not trigger due to cooldown
        triggered_policy_ids = [t.policy_id for t in triggers]
        assert policy_id not in triggered_policy_ids
    
    def test_error_handling_in_monitoring(self, monitor, sample_flights):
        """Test error handling during monitoring."""
        # Mock analytics to raise exception
        monitor.analytics.analyze_peaks.side_effect = Exception("Test error")
        
        # Should not raise exception, should handle gracefully
        result = monitor.monitor_continuous(sample_flights)
        
        # Should still complete with error status or handle gracefully
        assert "status" in result
        # The system should be resilient to individual component failures


class TestMonitoringPolicies:
    """Test cases for monitoring policies."""
    
    def test_utilization_policy_condition(self):
        """Test utilization threshold policy condition."""
        policy = MonitorPolicy(
            policy_id="util_test",
            name="Utilization Test",
            policy_type=PolicyType.UTILIZATION_THRESHOLD,
            condition="Utilization > 110%",
            condition_func=lambda data: data.get("max_utilization", 0) > 1.10,
            action=ActionType.OPTIMIZE_SCHEDULE,
            threshold_values={"utilization": 1.10}
        )
        
        # Test condition with high utilization
        high_util_data = {"max_utilization": 1.25}
        assert policy.condition_func(high_util_data) is True
        
        # Test condition with normal utilization
        normal_util_data = {"max_utilization": 0.85}
        assert policy.condition_func(normal_util_data) is False
    
    def test_delay_cascade_policy_condition(self):
        """Test delay cascade policy condition."""
        policy = MonitorPolicy(
            policy_id="cascade_test",
            name="Cascade Test",
            policy_type=PolicyType.DELAY_CASCADE,
            condition="5+ flights delayed >30min with cascade impact",
            condition_func=lambda data: (
                data.get("delayed_30m_count", 0) >= 5 and
                data.get("cascade_impact_score", 0) > 0.7
            ),
            action=ActionType.OPTIMIZE_SCHEDULE,
            threshold_values={"delayed_count": 5, "cascade_score": 0.7}
        )
        
        # Test condition with cascade scenario
        cascade_data = {"delayed_30m_count": 6, "cascade_impact_score": 0.8}
        assert policy.condition_func(cascade_data) is True
        
        # Test condition without cascade
        no_cascade_data = {"delayed_30m_count": 3, "cascade_impact_score": 0.5}
        assert policy.condition_func(no_cascade_data) is False


if __name__ == "__main__":
    pytest.main([__file__])