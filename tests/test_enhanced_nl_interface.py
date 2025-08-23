"""Tests for enhanced multi-provider NLP interface with autonomous agent orchestration."""

import pytest
import json
import time
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.services.enhanced_nl_interface import (
    EnhancedNLInterface, GeminiProvider, PerplexityProvider, OpenAIProvider, LocalProvider,
    ToolOrchestrator, AutonomousDecisionEngine, QueryIntent, ActionType, ConfidenceLevel,
    ProviderType, ProviderResponse, QueryParameters, ConversationContext, AutonomousAction,
    ToolResult, EnhancedNLResponse
)
from src.services.analytics import AnalyticsEngine
from src.services.delay_prediction import DelayRiskPredictor
from src.services.schedule_optimizer import ScheduleOptimizer
from src.services.whatif_simulator import WhatIfSimulator
from src.services.database import FlightDatabaseService


class TestProviders:
    """Test individual NLP providers."""
    
    def test_local_provider_intent_classification(self):
        """Test local provider intent classification."""
        provider = LocalProvider()
        
        # Test peak analysis query
        response = provider.classify_intent("What are the busiest slots at BOM?")
        assert response.success
        assert response.provider == ProviderType.LOCAL
        assert "ASK_PEAKS" in response.response
        assert response.confidence > 0.5
        
        # Test risk query
        response = provider.classify_intent("What's the delay risk for AI 2739?")
        assert response.success
        assert "ASK_RISK" in response.response
        
        # Test unknown query
        response = provider.classify_intent("What's the weather like?")
        assert response.success
        assert "UNKNOWN" in response.response
        assert response.confidence < 0.5
    
    def test_local_provider_parameter_extraction(self):
        """Test local provider parameter extraction."""
        provider = LocalProvider()
        
        query = "Move AI 2739 by +10 minutes at BOM"
        response = provider.extract_parameters(query, QueryIntent.ASK_WHATIF)
        
        assert response.success
        params = json.loads(response.response)
        assert params.get("airport_code") == "BOM"
        assert params.get("flight_number") == "AI2739"
        assert params.get("time_shift_minutes") == 10
    
    def test_local_provider_response_formatting(self):
        """Test local provider response formatting."""
        provider = LocalProvider()
        
        data = {
            "type": "peak_analysis",
            "airport": "BOM",
            "overload_windows": 3
        }
        
        response = provider.format_response(data, QueryIntent.ASK_PEAKS)
        assert response.success
        assert "BOM" in response.response
        assert "3" in response.response
    
    @patch('src.services.enhanced_nl_interface.genai')
    def test_gemini_provider_with_mock(self, mock_genai):
        """Test Gemini provider with mocked API."""
        # Mock Gemini response
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "ASK_PEAKS"
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'test_key'}):
            provider = GeminiProvider()
            provider.model = mock_model  # Override with mock
            
            response = provider.classify_intent("What are the busiest slots?")
            assert response.success
            assert response.provider == ProviderType.GEMINI
            assert response.response == "ASK_PEAKS"
    
    @patch('src.services.enhanced_nl_interface.requests')
    def test_perplexity_provider_with_mock(self, mock_requests):
        """Test Perplexity provider with mocked API."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "ASK_RISK"}}]
        }
        mock_requests.post.return_value = mock_response
        
        with patch.dict('os.environ', {'PERPLEXITY_API_KEY': 'test_key'}):
            provider = PerplexityProvider()
            
            response = provider.classify_intent("What's the delay risk?")
            assert response.success
            assert response.provider == ProviderType.PERPLEXITY
            assert response.response == "ASK_RISK"
    
    @patch('src.services.enhanced_nl_interface.requests')
    def test_perplexity_provider_rate_limit(self, mock_requests):
        """Test Perplexity provider rate limit handling."""
        # Mock rate limit response
        mock_response = Mock()
        mock_response.status_code = 429
        mock_requests.post.return_value = mock_response
        
        with patch.dict('os.environ', {'PERPLEXITY_API_KEY': 'test_key'}):
            provider = PerplexityProvider()
            
            response = provider.classify_intent("Test query")
            assert not response.success
            assert response.rate_limited
            assert response.provider == ProviderType.PERPLEXITY


class TestToolOrchestrator:
    """Test tool orchestration functionality."""
    
    @pytest.fixture
    def mock_services(self):
        """Create mock services for testing."""
        return {
            'analytics_engine': Mock(spec=AnalyticsEngine),
            'delay_predictor': Mock(spec=DelayRiskPredictor),
            'schedule_optimizer': Mock(spec=ScheduleOptimizer),
            'whatif_simulator': Mock(spec=WhatIfSimulator),
            'database_service': Mock(spec=FlightDatabaseService)
        }
    
    def test_tool_orchestrator_initialization(self, mock_services):
        """Test tool orchestrator initialization."""
        orchestrator = ToolOrchestrator(**mock_services)
        
        assert "analyze_peaks" in orchestrator.tools
        assert "predict_risk" in orchestrator.tools
        assert "simulate_whatif" in orchestrator.tools
        assert "optimize_schedule" in orchestrator.tools
    
    def test_execute_unknown_tool(self, mock_services):
        """Test execution of unknown tool."""
        orchestrator = ToolOrchestrator(**mock_services)
        
        result = orchestrator.execute_tool("unknown_tool", {})
        assert not result.success
        assert "not found" in result.error
        assert result.tool_name == "unknown_tool"
    
    def test_execute_peaks_analysis_tool(self, mock_services):
        """Test peaks analysis tool execution."""
        # Mock peak analysis result
        mock_peak_analysis = Mock()
        mock_peak_analysis.time_buckets = [Mock(overload=0), Mock(overload=1)]
        mock_services['analytics_engine'].analyze_peaks.return_value = mock_peak_analysis
        
        orchestrator = ToolOrchestrator(**mock_services)
        
        result = orchestrator.execute_tool("analyze_peaks", {
            "airport_code": "BOM",
            "time_bucket_minutes": 10
        })
        
        assert result.success
        assert result.data["type"] == "peak_analysis"
        assert result.data["airport"] == "BOM"
        assert result.data["overload_windows"] == 1
    
    def test_execute_risk_prediction_tool(self, mock_services):
        """Test risk prediction tool execution."""
        # Mock database query result
        mock_query_result = Mock()
        mock_query_result.data = [{
            'flight_id': 'test_id',
            'flight_number': 'AI2739',
            'date_local': date.today(),
            'origin_code': 'BOM',
            'destination_code': 'DEL',
            'aircraft_type': 'A320',
            'std_utc': datetime.now(),
            'sta_utc': datetime.now() + timedelta(hours=2),
            'atd_utc': None,
            'ata_utc': None,
            'dep_delay_minutes': None,
            'arr_delay_minutes': None
        }]
        mock_services['database_service'].query_flights_by_date_range.return_value = mock_query_result
        
        # Mock delay prediction
        mock_prediction = Mock()
        mock_prediction.risk_level.value = "medium"
        mock_prediction.delay_probability = 0.65
        mock_prediction.expected_delay_minutes = 12.5
        mock_services['delay_predictor'].predict_departure_delay.return_value = mock_prediction
        
        orchestrator = ToolOrchestrator(**mock_services)
        
        result = orchestrator.execute_tool("predict_risk", {
            "flight_number": "AI2739"
        })
        
        assert result.success
        assert result.data["type"] == "risk_prediction"
        assert result.data["flight_number"] == "AI2739"
        assert result.data["risk_level"] == "medium"
    
    def test_execute_tool_with_exception(self, mock_services):
        """Test tool execution with exception."""
        # Make analytics engine raise exception
        mock_services['analytics_engine'].analyze_peaks.side_effect = Exception("Test error")
        
        orchestrator = ToolOrchestrator(**mock_services)
        
        result = orchestrator.execute_tool("analyze_peaks", {})
        assert not result.success
        assert "Test error" in result.error


class TestAutonomousDecisionEngine:
    """Test autonomous decision-making functionality."""
    
    @pytest.fixture
    def mock_orchestrator(self):
        """Create mock tool orchestrator."""
        return Mock(spec=ToolOrchestrator)
    
    def test_decision_engine_initialization(self, mock_orchestrator):
        """Test decision engine initialization."""
        engine = AutonomousDecisionEngine(mock_orchestrator)
        
        assert ActionType.SEND_ALERT in engine.confidence_thresholds
        assert ActionType.OPTIMIZE_SCHEDULE in engine.confidence_thresholds
        assert engine.action_history == []
    
    def test_evaluate_autonomous_actions_for_optimization(self, mock_orchestrator):
        """Test autonomous action evaluation for optimization."""
        engine = AutonomousDecisionEngine(mock_orchestrator)
        
        context = ConversationContext(
            user_id="test_user",
            session_id="test_session",
            autonomous_permissions={"minor_optimizations": True}
        )
        
        parameters = QueryParameters(airport_code="BOM")
        
        actions = engine.evaluate_autonomous_actions(
            QueryIntent.ASK_OPTIMIZE, parameters, context
        )
        
        assert len(actions) == 1
        assert actions[0].action_type == ActionType.OPTIMIZE_SCHEDULE
        assert actions[0].confidence > 0.8
        assert not actions[0].requires_human_approval
    
    def test_evaluate_autonomous_actions_for_peaks(self, mock_orchestrator):
        """Test autonomous action evaluation for peaks analysis."""
        engine = AutonomousDecisionEngine(mock_orchestrator)
        
        context = ConversationContext(
            user_id="test_user",
            session_id="test_session"
        )
        
        parameters = QueryParameters(airport_code="DEL")
        
        actions = engine.evaluate_autonomous_actions(
            QueryIntent.ASK_PEAKS, parameters, context
        )
        
        assert len(actions) == 1
        assert actions[0].action_type == ActionType.SEND_ALERT
        assert actions[0].confidence > 0.6
    
    def test_should_execute_action_confidence_check(self, mock_orchestrator):
        """Test action execution decision based on confidence."""
        engine = AutonomousDecisionEngine(mock_orchestrator)
        
        context = ConversationContext(
            user_id="test_user",
            session_id="test_session",
            autonomous_permissions={"send_alerts": True}
        )
        
        # High confidence action should execute
        high_conf_action = AutonomousAction(
            action_id="test_1",
            action_type=ActionType.SEND_ALERT,
            confidence=0.8,
            reasoning="Test",
            parameters={},
            requires_human_approval=False,
            estimated_impact={}
        )
        
        assert engine.should_execute_action(high_conf_action, context)
        
        # Low confidence action should not execute
        low_conf_action = AutonomousAction(
            action_id="test_2",
            action_type=ActionType.SEND_ALERT,
            confidence=0.3,
            reasoning="Test",
            parameters={},
            requires_human_approval=False,
            estimated_impact={}
        )
        
        assert not engine.should_execute_action(low_conf_action, context)
    
    def test_should_execute_action_permission_check(self, mock_orchestrator):
        """Test action execution decision based on permissions."""
        engine = AutonomousDecisionEngine(mock_orchestrator)
        
        # Context without optimization permissions
        context = ConversationContext(
            user_id="test_user",
            session_id="test_session",
            autonomous_permissions={"minor_optimizations": False}
        )
        
        action = AutonomousAction(
            action_id="test_1",
            action_type=ActionType.OPTIMIZE_SCHEDULE,
            confidence=0.95,
            reasoning="Test",
            parameters={},
            requires_human_approval=False,
            estimated_impact={}
        )
        
        assert not engine.should_execute_action(action, context)
    
    def test_execute_alert_action(self, mock_orchestrator):
        """Test execution of alert action."""
        engine = AutonomousDecisionEngine(mock_orchestrator)
        
        action = AutonomousAction(
            action_id="test_alert",
            action_type=ActionType.SEND_ALERT,
            confidence=0.8,
            reasoning="Test alert",
            parameters={"airport_code": "BOM", "severity": "high"},
            requires_human_approval=False,
            estimated_impact={}
        )
        
        result = engine.execute_action(action)
        
        assert result["success"]
        assert result["data"]["alert_sent"]
        assert result["data"]["airport"] == "BOM"
        assert result["data"]["severity"] == "high"
        
        # Check action history
        assert len(engine.action_history) == 1
        assert engine.action_history[0]["action_id"] == "test_alert"
        assert engine.action_history[0]["success"]


class TestEnhancedNLInterface:
    """Test the main enhanced NL interface."""
    
    @pytest.fixture
    def mock_services(self):
        """Create mock services for testing."""
        return {
            'analytics_engine': Mock(spec=AnalyticsEngine),
            'delay_predictor': Mock(spec=DelayRiskPredictor),
            'schedule_optimizer': Mock(spec=ScheduleOptimizer),
            'whatif_simulator': Mock(spec=WhatIfSimulator),
            'database_service': Mock(spec=FlightDatabaseService)
        }
    
    def test_enhanced_nl_interface_initialization(self, mock_services):
        """Test enhanced NL interface initialization."""
        interface = EnhancedNLInterface(**mock_services)
        
        assert len(interface.providers) == 4
        assert isinstance(interface.providers[0], GeminiProvider)
        assert isinstance(interface.providers[1], PerplexityProvider)
        assert isinstance(interface.providers[2], OpenAIProvider)
        assert isinstance(interface.providers[3], LocalProvider)
        
        assert interface.tool_orchestrator is not None
        assert interface.decision_engine is not None
    
    def test_process_query_with_fallback(self, mock_services):
        """Test query processing with provider fallback."""
        interface = EnhancedNLInterface(**mock_services)
        
        # Mock all providers to fail except local
        for provider in interface.providers[:-1]:
            provider.classify_intent = Mock(return_value=ProviderResponse(
                provider=provider.__class__.__name__.replace('Provider', '').lower(),
                success=False,
                error="Mock failure"
            ))
        
        context = ConversationContext(
            user_id="test_user",
            session_id="test_session"
        )
        
        response = interface.process_query("What are the busiest slots at BOM?", context)
        
        assert isinstance(response, EnhancedNLResponse)
        assert response.provider_used == ProviderType.LOCAL
        assert response.confidence_score > 0.0
    
    def test_process_query_with_autonomous_actions(self, mock_services):
        """Test query processing with autonomous actions."""
        # Mock peak analysis result
        mock_peak_analysis = Mock()
        mock_peak_analysis.time_buckets = [Mock(overload=0), Mock(overload=1)]
        mock_services['analytics_engine'].analyze_peaks.return_value = mock_peak_analysis
        
        interface = EnhancedNLInterface(**mock_services)
        
        context = ConversationContext(
            user_id="test_user",
            session_id="test_session",
            autonomous_permissions={"send_alerts": True}
        )
        
        response = interface.process_query("What are the busiest slots at BOM?", context)
        
        assert response.intent == QueryIntent.ASK_PEAKS
        assert len(response.autonomous_actions) > 0
        assert response.autonomous_actions[0].action_type == ActionType.SEND_ALERT
    
    def test_provider_statistics(self, mock_services):
        """Test provider statistics tracking."""
        interface = EnhancedNLInterface(**mock_services)
        
        context = ConversationContext(
            user_id="test_user",
            session_id="test_session"
        )
        
        # Process a query to generate stats
        interface.process_query("Test query", context)
        
        stats = interface.get_provider_statistics()
        
        assert "provider_stats" in stats
        assert "total_calls" in stats
        assert "success_rate" in stats
        assert stats["total_calls"] > 0
    
    def test_autonomous_action_history(self, mock_services):
        """Test autonomous action history tracking."""
        interface = EnhancedNLInterface(**mock_services)
        
        context = ConversationContext(
            user_id="test_user",
            session_id="test_session",
            autonomous_permissions={"send_alerts": True}
        )
        
        # Process query that triggers autonomous action
        interface.process_query("What are the busiest slots?", context)
        
        history = interface.get_autonomous_action_history()
        assert isinstance(history, list)
    
    def test_confidence_calculation(self, mock_services):
        """Test confidence score calculation."""
        interface = EnhancedNLInterface(**mock_services)
        
        # Test with successful tool result
        tool_result = ToolResult(
            tool_name="test_tool",
            success=True,
            data={"type": "test"},
            confidence=0.8
        )
        
        confidence = interface._calculate_confidence(
            QueryIntent.ASK_PEAKS,
            QueryParameters(airport_code="BOM", time_bucket_minutes=10),
            tool_result,
            ProviderType.GEMINI
        )
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be reasonably high with good inputs
    
    def test_suggestion_generation(self, mock_services):
        """Test contextual suggestion generation."""
        interface = EnhancedNLInterface(**mock_services)
        
        suggestions = interface._generate_suggestions(
            QueryIntent.ASK_PEAKS,
            QueryParameters(airport_code="BOM"),
            {"type": "peak_analysis"}
        )
        
        assert isinstance(suggestions, list)
        assert len(suggestions) <= 3
        assert all(isinstance(s, str) for s in suggestions)
    
    def test_context_update(self, mock_services):
        """Test conversation context updates."""
        interface = EnhancedNLInterface(**mock_services)
        
        context = ConversationContext(
            user_id="test_user",
            session_id="test_session"
        )
        
        query = "What are the busiest slots at DEL?"
        intent = QueryIntent.ASK_PEAKS
        parameters = QueryParameters(airport_code="DEL", time_bucket_minutes=15)
        
        interface._update_context(context, query, intent, parameters)
        
        assert context.last_query == query
        assert context.last_intent == intent
        assert context.preferred_airport == "DEL"
        assert context.preferred_time_bucket == 15
        assert len(context.conversation_history) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])