"""Autonomous monitoring and policy engine for flight scheduling operations."""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid
from collections import defaultdict
import numpy as np

from .analytics import AnalyticsEngine, PeakAnalysis, OverloadWindow, TrafficLevel
from .schedule_optimizer import ScheduleOptimizer, OptimizationResult
from .cascade_analysis import CascadeAnalysisService, CascadeNode
from .delay_prediction import DelayRiskPredictor
from .alerting import AlertingService, Alert, AlertSeverity, AlertType
from .console_alerting import console_alerting_service
from ..models.flight import Flight


class PolicyType(Enum):
    """Types of monitoring policies."""
    UTILIZATION_THRESHOLD = "utilization_threshold"
    DELAY_CASCADE = "delay_cascade"
    CAPACITY_OVERLOAD = "capacity_overload"
    CRITICAL_DELAY = "critical_delay"
    SYSTEM_HEALTH = "system_health"


class ActionType(Enum):
    """Types of autonomous actions."""
    OPTIMIZE_SCHEDULE = "optimize_schedule"
    SEND_ALERT = "send_alert"
    ADJUST_CAPACITY = "adjust_capacity"
    ESCALATE_HUMAN = "escalate_human"
    LOG_WARNING = "log_warning"


class ConfidenceLevel(Enum):
    """Confidence levels for autonomous decisions."""
    LOW = "low"          # 0.0 - 0.4
    MEDIUM = "medium"    # 0.4 - 0.7
    HIGH = "high"        # 0.7 - 0.9
    VERY_HIGH = "very_high"  # 0.9 - 1.0


@dataclass
class MonitorPolicy:
    """A monitoring policy that defines conditions and actions."""
    policy_id: str
    name: str
    policy_type: PolicyType
    condition: str  # Human-readable condition description
    condition_func: Callable[[Dict[str, Any]], bool]  # Function to evaluate condition
    action: ActionType
    threshold_values: Dict[str, float]
    cooldown_minutes: int = 15
    enabled: bool = True
    priority: int = 1  # Higher number = higher priority
    max_changes_per_hour: int = 10
    min_confidence_required: float = 0.6
    created_at: datetime = field(default_factory=datetime.now)
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0


@dataclass
class PolicyTrigger:
    """A triggered policy with evaluation results."""
    trigger_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    policy_id: str = ""
    policy_name: str = ""
    trigger_time: datetime = field(default_factory=datetime.now)
    condition_values: Dict[str, float] = field(default_factory=dict)
    severity: AlertSeverity = AlertSeverity.MEDIUM
    recommended_action: ActionType = ActionType.LOG_WARNING
    confidence: float = 0.0
    affected_flights: List[str] = field(default_factory=list)
    reasoning: str = ""
    escalation_required: bool = False


@dataclass
class GuardrailCheck:
    """Result of guardrail validation for an autonomous action."""
    action_allowed: bool = True
    violations: List[str] = field(default_factory=list)
    max_changes_exceeded: bool = False
    min_notice_violated: bool = False
    fairness_constraints_ok: bool = True
    confidence_too_low: bool = False
    alternative_suggestions: List[str] = field(default_factory=list)
    risk_assessment: str = "low"


@dataclass
class AutonomousAction:
    """An autonomous action taken by the system."""
    action_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trigger_id: str = ""
    action_type: ActionType = ActionType.LOG_WARNING
    timestamp: datetime = field(default_factory=datetime.now)
    confidence_score: float = 0.0
    affected_flights: List[str] = field(default_factory=list)
    expected_impact: Dict[str, float] = field(default_factory=dict)
    actual_impact: Optional[Dict[str, float]] = None
    human_approval_required: bool = False
    human_approved: Optional[bool] = None
    reasoning: str = ""
    execution_status: str = "pending"  # pending, executing, completed, failed, cancelled
    execution_details: Dict[str, Any] = field(default_factory=dict)
    rollback_available: bool = False


class AutonomousMonitor:
    """Autonomous monitoring and policy engine."""
    
    def __init__(self, 
                 analytics_engine: AnalyticsEngine,
                 schedule_optimizer: ScheduleOptimizer,
                 cascade_analyzer: CascadeAnalysisService,
                 delay_predictor: DelayRiskPredictor,
                 alerting_service: AlertingService):
        """Initialize the autonomous monitor."""
        self.analytics = analytics_engine
        self.optimizer = schedule_optimizer
        self.cascade_analyzer = cascade_analyzer
        self.delay_predictor = delay_predictor
        self.alerting = alerting_service
        
        # Policy management
        self.policies: Dict[str, MonitorPolicy] = {}
        self.active_triggers: List[PolicyTrigger] = []
        self.action_history: List[AutonomousAction] = []
        
        # Configuration
        self.max_autonomous_changes_per_hour = 5
        self.min_confidence_for_autonomous_action = 0.7
        self.escalation_threshold = 0.5
        self.audit_log_path = "logs/autonomous_decisions.log"
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self._setup_audit_logging()
        
        # Initialize default policies
        self._initialize_default_policies()
    
    def _setup_audit_logging(self):
        """Setup audit logging for autonomous decisions."""
        os.makedirs("logs", exist_ok=True)
        
        # Create audit logger
        audit_logger = logging.getLogger("autonomous_audit")
        audit_logger.setLevel(logging.INFO)
        
        # Create file handler
        handler = logging.FileHandler(self.audit_log_path)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        audit_logger.addHandler(handler)
        
        self.audit_logger = audit_logger
    
    def _initialize_default_policies(self):
        """Initialize default monitoring policies."""
        
        # Utilization overload policy
        utilization_policy = MonitorPolicy(
            policy_id="util_overload_001",
            name="Runway Utilization Overload",
            policy_type=PolicyType.UTILIZATION_THRESHOLD,
            condition="Runway utilization > 110% for 15+ minutes",
            condition_func=lambda data: data.get("max_utilization", 0) > 1.10,
            action=ActionType.OPTIMIZE_SCHEDULE,
            threshold_values={"utilization": 1.10, "duration_minutes": 15},
            cooldown_minutes=30,
            priority=3,
            min_confidence_required=0.8
        )
        
        # Delay cascade policy
        cascade_policy = MonitorPolicy(
            policy_id="cascade_001",
            name="Delay Cascade Detection",
            policy_type=PolicyType.DELAY_CASCADE,
            condition="5+ flights delayed >30min with cascade impact",
            condition_func=lambda data: (
                data.get("delayed_30m_count", 0) >= 5 and
                data.get("cascade_impact_score", 0) > 0.7
            ),
            action=ActionType.OPTIMIZE_SCHEDULE,
            threshold_values={"delayed_count": 5, "delay_threshold": 30, "cascade_score": 0.7},
            cooldown_minutes=20,
            priority=4,
            min_confidence_required=0.75
        )
        
        # Critical delay policy
        critical_delay_policy = MonitorPolicy(
            policy_id="critical_delay_001",
            name="Critical Flight Delays",
            policy_type=PolicyType.CRITICAL_DELAY,
            condition="High-impact flight delayed >60min",
            condition_func=lambda data: (
                data.get("max_delay_minutes", 0) > 60 and
                data.get("high_impact_flight_delayed", False)
            ),
            action=ActionType.SEND_ALERT,
            threshold_values={"delay_threshold": 60, "impact_threshold": 0.8},
            cooldown_minutes=10,
            priority=5,
            min_confidence_required=0.6
        )
        
        # Capacity overload policy
        capacity_policy = MonitorPolicy(
            policy_id="capacity_001",
            name="Runway Capacity Overload",
            policy_type=PolicyType.CAPACITY_OVERLOAD,
            condition="Demand exceeds capacity by 20+ flights",
            condition_func=lambda data: data.get("overload_count", 0) >= 20,
            action=ActionType.ESCALATE_HUMAN,
            threshold_values={"overload_threshold": 20},
            cooldown_minutes=45,
            priority=2,
            min_confidence_required=0.5
        )
        
        # Add policies
        for policy in [utilization_policy, cascade_policy, critical_delay_policy, capacity_policy]:
            self.add_policy(policy)
    
    def add_policy(self, policy: MonitorPolicy) -> None:
        """Add a monitoring policy."""
        self.policies[policy.policy_id] = policy
        self.logger.info(f"Added monitoring policy: {policy.name} ({policy.policy_id})")
    
    def remove_policy(self, policy_id: str) -> bool:
        """Remove a monitoring policy."""
        if policy_id in self.policies:
            policy = self.policies.pop(policy_id)
            self.logger.info(f"Removed monitoring policy: {policy.name} ({policy_id})")
            return True
        return False
    
    def enable_policy(self, policy_id: str) -> bool:
        """Enable a monitoring policy."""
        if policy_id in self.policies:
            self.policies[policy_id].enabled = True
            return True
        return False
    
    def disable_policy(self, policy_id: str) -> bool:
        """Disable a monitoring policy."""
        if policy_id in self.policies:
            self.policies[policy_id].enabled = False
            return True
        return False
    
    def evaluate_policies(self, flights: List[Flight], context: Dict[str, Any] = None) -> List[PolicyTrigger]:
        """Evaluate all enabled policies against current conditions."""
        if context is None:
            context = {}
        
        triggers = []
        current_time = datetime.now()
        
        # Get current system metrics
        metrics = self._calculate_system_metrics(flights, context)
        
        # Evaluate each enabled policy
        for policy in self.policies.values():
            if not policy.enabled:
                continue
            
            # Check cooldown period
            if (policy.last_triggered and 
                current_time - policy.last_triggered < timedelta(minutes=policy.cooldown_minutes)):
                continue
            
            # Evaluate policy condition
            try:
                condition_met = policy.condition_func(metrics)
                if condition_met:
                    trigger = self._create_policy_trigger(policy, metrics, flights)
                    triggers.append(trigger)
                    
                    # Update policy state
                    policy.last_triggered = current_time
                    policy.trigger_count += 1
                    
                    self.logger.info(f"Policy triggered: {policy.name} (confidence: {trigger.confidence:.2f})")
                    
            except Exception as e:
                self.logger.error(f"Error evaluating policy {policy.name}: {str(e)}")
        
        # Sort triggers by priority and confidence
        triggers.sort(key=lambda t: (
            self.policies[t.policy_id].priority,
            t.confidence
        ), reverse=True)
        
        self.active_triggers.extend(triggers)
        return triggers
    
    def _calculate_system_metrics(self, flights: List[Flight], context: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate current system metrics for policy evaluation."""
        metrics = {}
        
        try:
            # Basic flight metrics
            total_flights = len(flights)
            delayed_flights = [f for f in flights if f.dep_delay_min and f.dep_delay_min > 15]
            delayed_30m = [f for f in flights if f.dep_delay_min and f.dep_delay_min > 30]
            delayed_60m = [f for f in flights if f.dep_delay_min and f.dep_delay_min > 60]
            
            metrics.update({
                "total_flights": total_flights,
                "delayed_flights_count": len(delayed_flights),
                "delayed_15m_count": len(delayed_flights),
                "delayed_30m_count": len(delayed_30m),
                "delayed_60m_count": len(delayed_60m),
                "avg_delay": np.mean([f.dep_delay_min for f in delayed_flights]) if delayed_flights else 0,
                "max_delay_minutes": max([f.dep_delay_min for f in delayed_flights]) if delayed_flights else 0
            })
            
            # Get peak analysis if available
            if hasattr(self.analytics, 'analyze_peaks'):
                try:
                    peak_analysis = self.analytics.analyze_peaks("BOM", 10)  # Default to BOM, 10-min buckets
                    if peak_analysis:
                        max_util = max([bucket.utilization for bucket in peak_analysis.time_buckets]) if peak_analysis.time_buckets else 0
                        overload_count = sum([bucket.overload for bucket in peak_analysis.time_buckets]) if peak_analysis.time_buckets else 0
                        
                        metrics.update({
                            "max_utilization": max_util,
                            "overload_count": overload_count,
                            "peak_buckets": len(peak_analysis.time_buckets)
                        })
                except Exception as e:
                    self.logger.warning(f"Could not get peak analysis: {str(e)}")
            
            # Get cascade analysis if available
            if hasattr(self.cascade_analyzer, 'analyze_cascades'):
                try:
                    cascade_result = self.cascade_analyzer.analyze_cascades(flights)
                    if cascade_result and hasattr(cascade_result, 'high_impact_flights'):
                        high_impact_delayed = [
                            f for f in cascade_result.high_impact_flights 
                            if any(flight.flight_id == f.flight_id and flight.dep_delay_min and flight.dep_delay_min > 30 
                                  for flight in flights)
                        ]
                        
                        metrics.update({
                            "cascade_impact_score": len(high_impact_delayed) / max(len(delayed_30m), 1),
                            "high_impact_flight_delayed": len(high_impact_delayed) > 0,
                            "high_impact_flights_count": len(cascade_result.high_impact_flights) if hasattr(cascade_result, 'high_impact_flights') else 0
                        })
                except Exception as e:
                    self.logger.warning(f"Could not get cascade analysis: {str(e)}")
            
            # Add context metrics
            metrics.update(context)
            
        except Exception as e:
            self.logger.error(f"Error calculating system metrics: {str(e)}")
            # Return basic metrics to avoid complete failure
            metrics = {
                "total_flights": len(flights),
                "delayed_flights_count": 0,
                "max_utilization": 0,
                "overload_count": 0,
                "cascade_impact_score": 0
            }
        
        return metrics
    
    def _create_policy_trigger(self, policy: MonitorPolicy, metrics: Dict[str, Any], flights: List[Flight]) -> PolicyTrigger:
        """Create a policy trigger with confidence scoring."""
        
        # Calculate confidence based on data quality and threshold exceedance
        confidence = self._calculate_trigger_confidence(policy, metrics, flights)
        
        # Determine severity based on policy type and metrics
        severity = self._determine_trigger_severity(policy, metrics, confidence)
        
        # Identify affected flights
        affected_flights = self._identify_affected_flights(policy, metrics, flights)
        
        # Generate reasoning
        reasoning = self._generate_trigger_reasoning(policy, metrics, confidence)
        
        # Determine if escalation is required
        escalation_required = (
            confidence < self.escalation_threshold or
            policy.action == ActionType.ESCALATE_HUMAN or
            severity == AlertSeverity.CRITICAL
        )
        
        trigger = PolicyTrigger(
            policy_id=policy.policy_id,
            policy_name=policy.name,
            condition_values=metrics,
            severity=severity,
            recommended_action=policy.action,
            confidence=confidence,
            affected_flights=affected_flights,
            reasoning=reasoning,
            escalation_required=escalation_required
        )
        
        return trigger
    
    def _calculate_trigger_confidence(self, policy: MonitorPolicy, metrics: Dict[str, Any], flights: List[Flight]) -> float:
        """Calculate confidence score for a policy trigger."""
        confidence_factors = []
        
        # Data quality factor
        data_quality = min(1.0, len(flights) / 100)  # Assume 100+ flights is good data
        confidence_factors.append(data_quality * 0.3)
        
        # Threshold exceedance factor
        if policy.policy_type == PolicyType.UTILIZATION_THRESHOLD:
            util = metrics.get("max_utilization", 0)
            threshold = policy.threshold_values.get("utilization", 1.0)
            exceedance = min(1.0, (util - threshold) / threshold) if threshold > 0 else 0
            confidence_factors.append(exceedance * 0.4)
        
        elif policy.policy_type == PolicyType.DELAY_CASCADE:
            delayed_count = metrics.get("delayed_30m_count", 0)
            threshold = policy.threshold_values.get("delayed_count", 5)
            exceedance = min(1.0, delayed_count / threshold) if threshold > 0 else 0
            confidence_factors.append(exceedance * 0.4)
        
        elif policy.policy_type == PolicyType.CAPACITY_OVERLOAD:
            overload = metrics.get("overload_count", 0)
            threshold = policy.threshold_values.get("overload_threshold", 20)
            exceedance = min(1.0, overload / threshold) if threshold > 0 else 0
            confidence_factors.append(exceedance * 0.4)
        
        else:
            confidence_factors.append(0.3)  # Default moderate confidence
        
        # Historical accuracy factor (simplified)
        historical_accuracy = 0.8  # Would be calculated from past trigger outcomes
        confidence_factors.append(historical_accuracy * 0.3)
        
        return min(1.0, sum(confidence_factors))
    
    def _determine_trigger_severity(self, policy: MonitorPolicy, metrics: Dict[str, Any], confidence: float) -> AlertSeverity:
        """Determine the severity of a policy trigger."""
        
        # Base severity on policy priority and confidence
        if policy.priority >= 4 and confidence >= 0.8:
            return AlertSeverity.CRITICAL
        elif policy.priority >= 3 and confidence >= 0.6:
            return AlertSeverity.HIGH
        elif policy.priority >= 2 and confidence >= 0.4:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW
    
    def _identify_affected_flights(self, policy: MonitorPolicy, metrics: Dict[str, Any], flights: List[Flight]) -> List[str]:
        """Identify flights affected by a policy trigger."""
        affected = []
        
        if policy.policy_type in [PolicyType.DELAY_CASCADE, PolicyType.CRITICAL_DELAY]:
            # Find delayed flights
            for flight in flights:
                if flight.dep_delay_min and flight.dep_delay_min > 15:
                    affected.append(flight.flight_id)
        
        elif policy.policy_type == PolicyType.UTILIZATION_THRESHOLD:
            # Find flights in peak periods (simplified)
            affected = [f.flight_id for f in flights[:10]]  # Top 10 as example
        
        return affected[:20]  # Limit to 20 flights for manageability
    
    def _generate_trigger_reasoning(self, policy: MonitorPolicy, metrics: Dict[str, Any], confidence: float) -> str:
        """Generate human-readable reasoning for a policy trigger."""
        
        reasoning_parts = [f"Policy '{policy.name}' triggered with {confidence:.1%} confidence."]
        
        if policy.policy_type == PolicyType.UTILIZATION_THRESHOLD:
            util = metrics.get("max_utilization", 0)
            reasoning_parts.append(f"Peak utilization reached {util:.1%}, exceeding threshold of {policy.threshold_values.get('utilization', 1.0):.1%}.")
        
        elif policy.policy_type == PolicyType.DELAY_CASCADE:
            delayed = metrics.get("delayed_30m_count", 0)
            cascade_score = metrics.get("cascade_impact_score", 0)
            reasoning_parts.append(f"{delayed} flights delayed >30min with cascade impact score of {cascade_score:.2f}.")
        
        elif policy.policy_type == PolicyType.CAPACITY_OVERLOAD:
            overload = metrics.get("overload_count", 0)
            reasoning_parts.append(f"Capacity overload detected with {overload} excess flights.")
        
        elif policy.policy_type == PolicyType.CRITICAL_DELAY:
            max_delay = metrics.get("max_delay_minutes", 0)
            reasoning_parts.append(f"Critical delay detected: {max_delay} minutes maximum delay.")
        
        return " ".join(reasoning_parts)
    
    def execute_autonomous_action(self, trigger: PolicyTrigger, flights: List[Flight]) -> AutonomousAction:
        """Execute an autonomous action based on a policy trigger."""
        
        # Create action record
        action = AutonomousAction(
            trigger_id=trigger.trigger_id,
            action_type=trigger.recommended_action,
            confidence_score=trigger.confidence,
            affected_flights=trigger.affected_flights,
            reasoning=trigger.reasoning
        )
        
        # Check guardrails before execution
        guardrail_check = self.check_guardrails(action, trigger)
        
        if not guardrail_check.action_allowed:
            action.execution_status = "blocked"
            action.execution_details = {
                "blocked_reason": "Guardrail violations",
                "violations": guardrail_check.violations
            }
            self._log_autonomous_decision(action, "BLOCKED")
            return action
        
        # Determine if human approval is required
        action.human_approval_required = (
            trigger.escalation_required or
            trigger.confidence < self.min_confidence_for_autonomous_action or
            guardrail_check.risk_assessment == "high"
        )
        
        if action.human_approval_required:
            action.execution_status = "awaiting_approval"
            self._log_autonomous_decision(action, "AWAITING_APPROVAL")
            self._escalate_to_human(action, trigger)
            return action
        
        # Execute the action
        try:
            action.execution_status = "executing"
            execution_result = self._execute_action(action, trigger, flights)
            
            action.execution_details = execution_result
            action.actual_impact = execution_result.get("impact", {})
            action.execution_status = "completed"
            
            self._log_autonomous_decision(action, "COMPLETED")
            
        except Exception as e:
            action.execution_status = "failed"
            action.execution_details = {"error": str(e)}
            self._log_autonomous_decision(action, "FAILED")
            self.logger.error(f"Failed to execute autonomous action {action.action_id}: {str(e)}")
        
        # Store action in history
        self.action_history.append(action)
        
        return action
    
    def check_guardrails(self, action: AutonomousAction, trigger: PolicyTrigger) -> GuardrailCheck:
        """Check guardrails for an autonomous action."""
        
        check = GuardrailCheck()
        current_time = datetime.now()
        
        # Check maximum changes per hour
        recent_actions = [
            a for a in self.action_history 
            if (current_time - a.timestamp).total_seconds() < 3600 and
               a.execution_status == "completed"
        ]
        
        if len(recent_actions) >= self.max_autonomous_changes_per_hour:
            check.max_changes_exceeded = True
            check.violations.append(f"Maximum changes per hour exceeded ({len(recent_actions)}/{self.max_autonomous_changes_per_hour})")
        
        # Check minimum confidence threshold
        if action.confidence_score < self.min_confidence_for_autonomous_action:
            check.confidence_too_low = True
            check.violations.append(f"Confidence too low ({action.confidence_score:.2f} < {self.min_confidence_for_autonomous_action:.2f})")
        
        # Check minimum notice period for schedule changes
        if action.action_type == ActionType.OPTIMIZE_SCHEDULE:
            # Simplified check - in reality would check flight departure times
            min_notice_hours = 2
            check.min_notice_violated = False  # Would implement proper check
        
        # Check fairness constraints
        affected_airlines = self._get_affected_airlines(action.affected_flights)
        if len(affected_airlines) == 1 and len(action.affected_flights) > 5:
            check.fairness_constraints_ok = False
            check.violations.append("Fairness constraint violated: too many flights from single airline affected")
        
        # Assess overall risk
        risk_factors = []
        if check.max_changes_exceeded:
            risk_factors.append("high_frequency")
        if check.confidence_too_low:
            risk_factors.append("low_confidence")
        if len(action.affected_flights) > 10:
            risk_factors.append("high_impact")
        
        if len(risk_factors) >= 2:
            check.risk_assessment = "high"
        elif len(risk_factors) == 1:
            check.risk_assessment = "medium"
        else:
            check.risk_assessment = "low"
        
        # Generate alternative suggestions if action is blocked
        if check.violations:
            check.action_allowed = False
            check.alternative_suggestions = self._generate_alternatives(action, trigger)
        
        return check
    
    def _get_affected_airlines(self, flight_ids: List[str]) -> Set[str]:
        """Get set of airlines affected by flight IDs (simplified implementation)."""
        # In a real implementation, would look up flights by ID
        # For now, return a dummy set
        return {"AI", "6E", "SG"}  # Example airlines
    
    def _generate_alternatives(self, action: AutonomousAction, trigger: PolicyTrigger) -> List[str]:
        """Generate alternative suggestions when an action is blocked."""
        alternatives = []
        
        if action.action_type == ActionType.OPTIMIZE_SCHEDULE:
            alternatives.extend([
                "Send alert to operations team instead of autonomous optimization",
                "Reduce scope to top 5 most critical flights only",
                "Schedule optimization for next available maintenance window"
            ])
        
        elif action.action_type == ActionType.SEND_ALERT:
            alternatives.extend([
                "Log warning instead of sending alert",
                "Escalate to human operator for review"
            ])
        
        return alternatives
    
    def _execute_action(self, action: AutonomousAction, trigger: PolicyTrigger, flights: List[Flight]) -> Dict[str, Any]:
        """Execute the actual autonomous action."""
        
        if action.action_type == ActionType.OPTIMIZE_SCHEDULE:
            return self._execute_optimization(action, flights)
        
        elif action.action_type == ActionType.SEND_ALERT:
            return self._execute_alert(action, trigger)
        
        elif action.action_type == ActionType.ADJUST_CAPACITY:
            return self._execute_capacity_adjustment(action, trigger)
        
        elif action.action_type == ActionType.LOG_WARNING:
            return self._execute_log_warning(action, trigger)
        
        else:
            raise ValueError(f"Unknown action type: {action.action_type}")
    
    def _execute_optimization(self, action: AutonomousAction, flights: List[Flight]) -> Dict[str, Any]:
        """Execute schedule optimization."""
        try:
            # Get affected flights
            affected_flights = [f for f in flights if f.flight_id in action.affected_flights]
            
            if not affected_flights:
                return {"status": "no_flights_to_optimize", "impact": {}}
            
            # Run optimization (simplified)
            optimization_result = self.optimizer.optimize_schedule(
                flights=affected_flights,
                constraints={"max_changes": 5, "min_improvement": 10}
            )
            
            impact = {
                "flights_optimized": len(affected_flights),
                "estimated_delay_reduction": optimization_result.get("delay_improvement", 0),
                "optimization_score": optimization_result.get("score", 0)
            }
            
            action.expected_impact = impact
            
            return {
                "status": "optimization_completed",
                "optimization_result": optimization_result,
                "impact": impact
            }
            
        except Exception as e:
            return {"status": "optimization_failed", "error": str(e), "impact": {}}
    
    def _execute_alert(self, action: AutonomousAction, trigger: PolicyTrigger) -> Dict[str, Any]:
        """Execute alert sending."""
        try:
            # Create alert
            alert = Alert(
                alert_id=str(uuid.uuid4()),
                alert_type=AlertType.CAPACITY_OVERLOAD,  # Map from trigger
                severity=trigger.severity,
                title=f"Autonomous Alert: {trigger.policy_name}",
                description=trigger.reasoning,
                airport="BOM",  # Default airport
                affected_flights=trigger.affected_flights,
                recommendations=[],
                timestamp=datetime.now()
            )
            
            # Send alert
            self.alerting.send_alert(alert)
            
            return {
                "status": "alert_sent",
                "alert_id": alert.alert_id,
                "impact": {"alerts_sent": 1}
            }
            
        except Exception as e:
            return {"status": "alert_failed", "error": str(e), "impact": {}}
    
    def _execute_capacity_adjustment(self, action: AutonomousAction, trigger: PolicyTrigger) -> Dict[str, Any]:
        """Execute capacity adjustment."""
        # Placeholder for capacity adjustment logic
        return {
            "status": "capacity_adjusted",
            "adjustment": "weather_based_reduction",
            "impact": {"capacity_change": -10}
        }
    
    def _execute_log_warning(self, action: AutonomousAction, trigger: PolicyTrigger) -> Dict[str, Any]:
        """Execute warning logging."""
        self.logger.warning(f"Autonomous warning: {trigger.reasoning}")
        return {
            "status": "warning_logged",
            "impact": {"warnings_logged": 1}
        }
    
    def _escalate_to_human(self, action: AutonomousAction, trigger: PolicyTrigger) -> None:
        """Escalate an action to human operators."""
        escalation_message = f"""
        AUTONOMOUS SYSTEM ESCALATION
        
        Action ID: {action.action_id}
        Trigger: {trigger.policy_name}
        Confidence: {trigger.confidence:.1%}
        Severity: {trigger.severity.value}
        
        Reasoning: {trigger.reasoning}
        
        Recommended Action: {action.action_type.value}
        Affected Flights: {len(action.affected_flights)}
        
        Please review and approve/reject this autonomous action.
        """
        
        # In a real system, would send to operations dashboard or notification system
        self.logger.info(f"ESCALATION: {escalation_message}")
        
        # Could also send via alerting system
        try:
            escalation_alert = Alert(
                alert_id=str(uuid.uuid4()),
                alert_type=AlertType.SYSTEM_ERROR,
                severity=AlertSeverity.HIGH,
                title="Autonomous System Escalation Required",
                description=escalation_message,
                airport="BOM",  # Default airport
                affected_flights=action.affected_flights,
                recommendations=[],
                timestamp=datetime.now()
            )
            self.alerting.send_alert(escalation_alert)
        except Exception as e:
            self.logger.error(f"Failed to send escalation alert: {str(e)}")
    
    def _log_autonomous_decision(self, action: AutonomousAction, status: str) -> None:
        """Log autonomous decision for audit trail."""
        log_entry = {
            "timestamp": action.timestamp.isoformat(),
            "action_id": action.action_id,
            "trigger_id": action.trigger_id,
            "action_type": action.action_type.value,
            "status": status,
            "confidence": action.confidence_score,
            "affected_flights_count": len(action.affected_flights),
            "reasoning": action.reasoning,
            "human_approval_required": action.human_approval_required,
            "execution_details": action.execution_details
        }
        
        self.audit_logger.info(json.dumps(log_entry))
    
    def get_policy_status(self) -> Dict[str, Any]:
        """Get status of all monitoring policies."""
        status = {
            "total_policies": len(self.policies),
            "enabled_policies": len([p for p in self.policies.values() if p.enabled]),
            "active_triggers": len(self.active_triggers),
            "recent_actions": len([a for a in self.action_history if (datetime.now() - a.timestamp).total_seconds() < 3600]),
            "policies": []
        }
        
        for policy in self.policies.values():
            policy_status = {
                "policy_id": policy.policy_id,
                "name": policy.name,
                "type": policy.policy_type.value,
                "enabled": policy.enabled,
                "priority": policy.priority,
                "trigger_count": policy.trigger_count,
                "last_triggered": policy.last_triggered.isoformat() if policy.last_triggered else None,
                "cooldown_minutes": policy.cooldown_minutes
            }
            status["policies"].append(policy_status)
        
        return status
    
    def get_action_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get autonomous action history."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_actions = [a for a in self.action_history if a.timestamp >= cutoff_time]
        
        return [
            {
                "action_id": action.action_id,
                "timestamp": action.timestamp.isoformat(),
                "action_type": action.action_type.value,
                "confidence": action.confidence_score,
                "status": action.execution_status,
                "affected_flights": len(action.affected_flights),
                "human_approval_required": action.human_approval_required,
                "reasoning": action.reasoning[:200] + "..." if len(action.reasoning) > 200 else action.reasoning
            }
            for action in recent_actions
        ]
    
    def approve_action(self, action_id: str, approved: bool, approver: str = "human") -> bool:
        """Approve or reject a pending autonomous action."""
        for action in self.action_history:
            if action.action_id == action_id and action.execution_status == "awaiting_approval":
                action.human_approved = approved
                
                if approved:
                    action.execution_status = "approved"
                    self._log_autonomous_decision(action, f"APPROVED_BY_{approver.upper()}")
                    # Could trigger actual execution here
                else:
                    action.execution_status = "rejected"
                    self._log_autonomous_decision(action, f"REJECTED_BY_{approver.upper()}")
                
                return True
        
        return False
    
    def monitor_continuous(self, flights: List[Flight], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run continuous monitoring cycle."""
        start_time = datetime.now()
        
        try:
            # Evaluate policies
            triggers = self.evaluate_policies(flights, context)
            
            # Execute autonomous actions for high-confidence triggers
            actions_executed = []
            for trigger in triggers:
                if trigger.confidence >= self.min_confidence_for_autonomous_action:
                    action = self.execute_autonomous_action(trigger, flights)
                    actions_executed.append(action)
            
            # Clean up old triggers and actions
            self._cleanup_old_data()
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "status": "completed",
                "execution_time_seconds": execution_time,
                "triggers_found": len(triggers),
                "actions_executed": len(actions_executed),
                "escalations_required": len([t for t in triggers if t.escalation_required]),
                "timestamp": start_time.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in continuous monitoring: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": start_time.isoformat()
            }
    
    def _cleanup_old_data(self) -> None:
        """Clean up old triggers and actions to prevent memory issues."""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        # Remove old triggers
        self.active_triggers = [t for t in self.active_triggers if t.trigger_time >= cutoff_time]
        
        # Keep only recent actions (but preserve audit log)
        if len(self.action_history) > 1000:
            self.action_history = self.action_history[-500:]  # Keep last 500 actions