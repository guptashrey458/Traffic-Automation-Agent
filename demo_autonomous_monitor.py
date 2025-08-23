#!/usr/bin/env python3
"""
Demo script for the Autonomous Monitoring and Policy Engine.

This script demonstrates the key features of the autonomous monitoring system:
- Policy-based condition evaluation
- Threshold detection for utilization overload and delay cascades
- Autonomous decision-making with confidence scoring
- Guardrail checking for maximum changes and fairness constraints
- Escalation logic for complex scenarios
- Audit logging for all autonomous decisions
"""

import os
import sys
from datetime import datetime, date, time
from typing import List
from unittest.mock import Mock

# Add src to path for imports
sys.path.append('src')

from src.services.autonomous_monitor import (
    AutonomousMonitor, MonitorPolicy, PolicyType, ActionType, AlertSeverity
)
from src.services.analytics import AnalyticsEngine
from src.services.schedule_optimizer import ScheduleOptimizer
from src.services.cascade_analysis import CascadeAnalysisService
from src.services.delay_prediction import DelayRiskPredictor
from src.services.alerting import AlertingService
from src.models.flight import Flight, Airport, FlightTime


def create_sample_flights() -> List[Flight]:
    """Create sample flight data with various delay patterns."""
    flights = []
    
    # Create flights with different delay scenarios
    delay_patterns = [
        # High-impact delayed flights
        (60, "AI2501", "Mumbai (BOM)", "Delhi (DEL)"),
        (45, "6E123", "Mumbai (BOM)", "Bangalore (BLR)"),
        (75, "SG456", "Mumbai (BOM)", "Chennai (MAA)"),
        (90, "AI2502", "Delhi (DEL)", "Mumbai (BOM)"),
        (35, "6E124", "Bangalore (BLR)", "Mumbai (BOM)"),
        
        # Moderate delays
        (25, "AI2503", "Mumbai (BOM)", "Kolkata (CCU)"),
        (20, "6E125", "Mumbai (BOM)", "Hyderabad (HYD)"),
        (30, "SG457", "Mumbai (BOM)", "Pune (PNQ)"),
        (18, "AI2504", "Delhi (DEL)", "Bangalore (BLR)"),
        (22, "6E126", "Chennai (MAA)", "Mumbai (BOM)"),
        
        # Minor delays and on-time flights
        (5, "AI2505", "Mumbai (BOM)", "Goa (GOI)"),
        (0, "6E127", "Mumbai (BOM)", "Ahmedabad (AMD)"),
        (8, "SG458", "Mumbai (BOM)", "Indore (IDR)"),
        (0, "AI2506", "Delhi (DEL)", "Chennai (MAA)"),
        (3, "6E128", "Bangalore (BLR)", "Delhi (DEL)"),
    ]
    
    for i, (delay, flight_no, origin_str, dest_str) in enumerate(delay_patterns):
        flight = Flight(
            flight_id=f"flight_{i:03d}",
            flight_number=flight_no,
            origin=Airport.from_string(origin_str),
            destination=Airport.from_string(dest_str),
            aircraft_type="A320",
            flight_date=date.today(),
            departure=FlightTime(scheduled=time(8 + i % 12, (i * 15) % 60)),
            arrival=FlightTime(scheduled=time(10 + i % 12, (i * 15) % 60))
        )
        
        # Set delay
        flight.dep_delay_min = delay
        
        flights.append(flight)
    
    return flights


def create_mock_services():
    """Create mock services for the demo."""
    # Create mock services
    analytics = Mock(spec=AnalyticsEngine)
    optimizer = Mock(spec=ScheduleOptimizer)
    cascade_analyzer = Mock(spec=CascadeAnalysisService)
    delay_predictor = Mock(spec=DelayRiskPredictor)
    alerting = Mock(spec=AlertingService)
    
    # Configure analytics mock to return high utilization
    mock_peak_analysis = Mock()
    mock_peak_analysis.time_buckets = [
        Mock(utilization=1.25, overload=30),  # High utilization
        Mock(utilization=1.15, overload=20),
        Mock(utilization=0.95, overload=0)
    ]
    analytics.analyze_peaks = Mock(return_value=mock_peak_analysis)
    
    # Configure cascade analyzer mock
    mock_cascade_result = Mock()
    mock_cascade_result.high_impact_flights = [Mock(flight_id="flight_000"), Mock(flight_id="flight_001")]
    cascade_analyzer.analyze_cascades = Mock(return_value=mock_cascade_result)
    
    # Configure optimizer mock
    optimizer.optimize_schedule = Mock(return_value={
        "delay_improvement": 25.5,
        "score": 0.87,
        "changes_made": 3
    })
    
    # Configure alerting mock
    alerting.send_alert = Mock(return_value=True)
    
    return analytics, optimizer, cascade_analyzer, delay_predictor, alerting


def demo_policy_evaluation():
    """Demonstrate policy evaluation and triggering."""
    print("=" * 60)
    print("AUTONOMOUS MONITORING SYSTEM DEMO")
    print("=" * 60)
    
    # Create sample data
    flights = create_sample_flights()
    analytics, optimizer, cascade_analyzer, delay_predictor, alerting = create_mock_services()
    
    # Create autonomous monitor
    monitor = AutonomousMonitor(
        analytics_engine=analytics,
        schedule_optimizer=optimizer,
        cascade_analyzer=cascade_analyzer,
        delay_predictor=delay_predictor,
        alerting_service=alerting
    )
    
    print(f"\n1. SYSTEM INITIALIZATION")
    print(f"   - Created autonomous monitor with {len(monitor.policies)} default policies")
    print(f"   - Loaded {len(flights)} sample flights")
    print(f"   - Flights with delays >30min: {len([f for f in flights if f.dep_delay_min and f.dep_delay_min > 30])}")
    print(f"   - Maximum delay: {max([f.dep_delay_min for f in flights if f.dep_delay_min])} minutes")
    
    # Show policy status
    policy_status = monitor.get_policy_status()
    print(f"\n2. POLICY STATUS")
    print(f"   - Total policies: {policy_status['total_policies']}")
    print(f"   - Enabled policies: {policy_status['enabled_policies']}")
    
    for policy in policy_status['policies'][:3]:  # Show first 3 policies
        print(f"   - {policy['name']}: Priority {policy['priority']}, Enabled: {policy['enabled']}")
    
    # Evaluate policies
    print(f"\n3. POLICY EVALUATION")
    triggers = monitor.evaluate_policies(flights)
    
    print(f"   - Policies evaluated: {len(monitor.policies)}")
    print(f"   - Triggers found: {len(triggers)}")
    
    for trigger in triggers:
        print(f"   - TRIGGER: {trigger.policy_name}")
        print(f"     * Confidence: {trigger.confidence:.1%}")
        print(f"     * Severity: {trigger.severity.value}")
        print(f"     * Action: {trigger.recommended_action.value}")
        print(f"     * Affected flights: {len(trigger.affected_flights)}")
        print(f"     * Reasoning: {trigger.reasoning[:100]}...")
    
    return monitor, flights, triggers


def demo_autonomous_actions(monitor, flights, triggers):
    """Demonstrate autonomous action execution."""
    print(f"\n4. AUTONOMOUS ACTION EXECUTION")
    
    if not triggers:
        print("   - No triggers to execute")
        return
    
    # Execute actions for triggers
    actions_executed = []
    for trigger in triggers[:2]:  # Execute first 2 triggers
        print(f"\n   Executing action for trigger: {trigger.policy_name}")
        
        action = monitor.execute_autonomous_action(trigger, flights)
        actions_executed.append(action)
        
        print(f"   - Action ID: {action.action_id}")
        print(f"   - Status: {action.execution_status}")
        print(f"   - Confidence: {action.confidence_score:.1%}")
        print(f"   - Human approval required: {action.human_approval_required}")
        
        if action.execution_status == "blocked":
            print(f"   - Blocked reason: {action.execution_details.get('blocked_reason', 'Unknown')}")
            violations = action.execution_details.get('violations', [])
            for violation in violations:
                print(f"     * {violation}")
        
        elif action.execution_status == "awaiting_approval":
            print(f"   - Escalated to human operators")
        
        elif action.execution_status == "completed":
            print(f"   - Action completed successfully")
            if action.actual_impact:
                for key, value in action.actual_impact.items():
                    print(f"     * {key}: {value}")
    
    return actions_executed


def demo_guardrails(monitor):
    """Demonstrate guardrail checking."""
    print(f"\n5. GUARDRAIL SYSTEM DEMONSTRATION")
    
    # Create a test action that should pass guardrails
    from src.services.autonomous_monitor import AutonomousAction, PolicyTrigger
    
    good_action = AutonomousAction(
        action_type=ActionType.SEND_ALERT,
        confidence_score=0.85,
        affected_flights=["flight_001", "flight_002"]
    )
    
    good_trigger = PolicyTrigger(
        policy_id="test_policy",
        confidence=0.85,
        severity=AlertSeverity.MEDIUM
    )
    
    print("   Testing GOOD action (high confidence, few flights):")
    check = monitor.check_guardrails(good_action, good_trigger)
    print(f"   - Action allowed: {check.action_allowed}")
    print(f"   - Risk assessment: {check.risk_assessment}")
    print(f"   - Violations: {len(check.violations)}")
    
    # Create a test action that should fail guardrails
    bad_action = AutonomousAction(
        action_type=ActionType.OPTIMIZE_SCHEDULE,
        confidence_score=0.3,  # Low confidence
        affected_flights=[f"flight_{i:03d}" for i in range(15)]  # Many flights
    )
    
    bad_trigger = PolicyTrigger(
        policy_id="test_policy",
        confidence=0.3,
        severity=AlertSeverity.HIGH
    )
    
    print("\n   Testing BAD action (low confidence, many flights):")
    check = monitor.check_guardrails(bad_action, bad_trigger)
    print(f"   - Action allowed: {check.action_allowed}")
    print(f"   - Risk assessment: {check.risk_assessment}")
    print(f"   - Violations: {len(check.violations)}")
    for violation in check.violations:
        print(f"     * {violation}")
    
    if check.alternative_suggestions:
        print("   - Alternative suggestions:")
        for suggestion in check.alternative_suggestions:
            print(f"     * {suggestion}")


def demo_continuous_monitoring(monitor, flights):
    """Demonstrate continuous monitoring cycle."""
    print(f"\n6. CONTINUOUS MONITORING CYCLE")
    
    # Run continuous monitoring
    result = monitor.monitor_continuous(flights)
    
    print(f"   - Status: {result['status']}")
    print(f"   - Execution time: {result['execution_time_seconds']:.2f} seconds")
    print(f"   - Triggers found: {result['triggers_found']}")
    print(f"   - Actions executed: {result['actions_executed']}")
    print(f"   - Escalations required: {result['escalations_required']}")
    
    # Show action history
    history = monitor.get_action_history(hours=1)
    print(f"\n   Recent action history ({len(history)} actions):")
    for action in history[-3:]:  # Show last 3 actions
        print(f"   - {action['timestamp'][:19]}: {action['action_type']} (confidence: {action['confidence']:.1%})")
        print(f"     Status: {action['status']}, Flights: {action['affected_flights']}")


def demo_audit_logging():
    """Demonstrate audit logging capabilities."""
    print(f"\n7. AUDIT LOGGING")
    
    # Check if audit log exists
    log_path = "logs/autonomous_decisions.log"
    if os.path.exists(log_path):
        print(f"   - Audit log location: {log_path}")
        
        # Read last few lines
        try:
            with open(log_path, 'r') as f:
                lines = f.readlines()
                print(f"   - Total log entries: {len(lines)}")
                
                if lines:
                    print("   - Recent entries:")
                    for line in lines[-2:]:  # Show last 2 entries
                        if line.strip():
                            # Parse JSON log entry
                            import json
                            try:
                                entry = json.loads(line.strip())
                                print(f"     * {entry['timestamp'][:19]}: {entry['action_type']} - {entry['status']}")
                            except:
                                print(f"     * {line.strip()[:100]}...")
        except Exception as e:
            print(f"   - Error reading audit log: {e}")
    else:
        print(f"   - Audit log not found (would be created at: {log_path})")


def main():
    """Run the autonomous monitoring demo."""
    try:
        # Run the demo
        monitor, flights, triggers = demo_policy_evaluation()
        actions = demo_autonomous_actions(monitor, flights, triggers)
        demo_guardrails(monitor)
        demo_continuous_monitoring(monitor, flights)
        demo_audit_logging()
        
        print(f"\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"The autonomous monitoring system demonstrated:")
        print(f"✓ Policy-based condition evaluation")
        print(f"✓ Threshold detection for utilization overload and delay cascades")
        print(f"✓ Autonomous decision-making with confidence scoring")
        print(f"✓ Guardrail checking for maximum changes and fairness constraints")
        print(f"✓ Escalation logic for complex scenarios")
        print(f"✓ Audit logging for all autonomous decisions")
        print(f"\nThe system is ready for integration with the flight scheduling platform.")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())