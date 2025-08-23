"""Tests for natural language interface."""

import pytest
from datetime import datetime, date, time
from unittest.mock import Mock, patch
import os

from src.services.nl_interface import (
    NLInterface, QueryIntent, QueryParameters, ConversationContext, NLResponse
)
from src.services.analytics import AnalyticsEngine
from src.services.delay_prediction import DelayRiskPredictor
from src.services.schedule_optimizer import ScheduleOptimizer
from src.services.whatif_simulator import WhatIfSimulator
from src.services.database import FlightDatabaseService


class TestNLInterface:
    """Test cases for natural language interface."""
    
    @pytest.fixture
    def mock_services(self):
        """Create mock services for testing."""
        analytics_engine = Mock(spec=AnalyticsEngine)
        delay_predictor = Mock(spec=DelayRiskPredictor)
        schedule_optimizer = Mock(spec=ScheduleOptimizer)
        whatif_simulator = Mock(spec=WhatIfSimulator)
        database_service = Mock(spec=FlightDatabaseService)
        
        return {
            'analytics_engine': analytics_engine,
            'delay_predictor': delay_predictor,
            'schedule_optimizer': schedule_optimizer,
            'whatif_simulator': whatif_simulator,
            'database_service': database_service
        }
    
    @pytest.fixture
    def nl_interface(self, mock_services):
        """Create NL interface with mock services."""
        return NLInterface(**mock_services)
    
    @pytest.fixture
    def conversation_context(self):
        """Create test conversation context."""
        return ConversationContext(
            user_id="test_user",
            session_id="test_session",
            preferred_airport="BOM"
        )
    
    def test_intent_classification_peaks(self, nl_interface):
        """Test intent classification for peak traffic queries."""
        queries = [
            "What are the busiest slots at BOM?",
            "Show me peak traffic analysis",
            "Which time slots have overload?",
            "Demand heatmap for Delhi airport"
        ]
        
        for query in queries:
            intent = nl_interface.classify_intent(query)
            assert intent == QueryIntent.ASK_PEAKS, f"Failed for query: {query}"
    
    def test_intent_classification_risk(self, nl_interface):
        """Test intent classification for delay risk queries."""
        queries = [
            "What's the delay risk for AI 2739?",
            "Probability of delay for flight 6E1234",
            "Risk assessment for AI 2739",
            "Is AI 2739 likely to be delayed?"
        ]
        
        for query in queries:
            intent = nl_interface.classify_intent(query)
            assert intent == QueryIntent.ASK_RISK, f"Failed for query: {query}"
    
    def test_intent_classification_whatif(self, nl_interface):
        """Test intent classification for what-if queries."""
        queries = [
            "Move AI 2739 by +10 minutes—impact?",
            "What if I shift flight 6E1234 by -15 minutes?",
            "Impact of moving AI 2739 earlier",
            "Reschedule AI 2739 to 10:30"
        ]
        
        for query in queries:
            intent = nl_interface.classify_intent(query)
            assert intent == QueryIntent.ASK_WHATIF, f"Failed for query: {query}"
    
    def test_intent_classification_optimize(self, nl_interface):
        """Test intent classification for optimization queries."""
        queries = [
            "Optimize schedule for BOM",
            "Minimize delays at Delhi",
            "Improve the schedule",
            "Best schedule optimization"
        ]
        
        for query in queries:
            intent = nl_interface.classify_intent(query)
            assert intent == QueryIntent.ASK_OPTIMIZE, f"Failed for query: {query}"
    
    def test_parameter_extraction_airport_code(self, nl_interface, conversation_context):
        """Test extraction of airport codes."""
        query = "What are the busiest slots at BOM?"
        intent = QueryIntent.ASK_PEAKS
        
        params = nl_interface.extract_parameters(query, intent, conversation_context)
        assert params.airport_code == "BOM"
    
    def test_parameter_extraction_flight_number(self, nl_interface, conversation_context):
        """Test extraction of flight numbers."""
        queries_and_expected = [
            ("What's the delay risk for AI 2739?", "AI2739"),
            ("Risk for flight 6E 1234", "6E1234"),
            ("Check AI2739 status", "AI2739")
        ]
        
        for query, expected in queries_and_expected:
            params = nl_interface.extract_parameters(query, QueryIntent.ASK_RISK, conversation_context)
            assert params.flight_number == expected, f"Failed for query: {query}"
    
    def test_parameter_extraction_time_shift(self, nl_interface, conversation_context):
        """Test extraction of time shifts."""
        queries_and_expected = [
            ("Move AI 2739 by +10 minutes", 10),
            ("Shift flight by -15min", -15),
            ("Move by +30m", 30)
        ]
        
        for query, expected in queries_and_expected:
            params = nl_interface.extract_parameters(query, QueryIntent.ASK_WHATIF, conversation_context)
            assert params.time_shift_minutes == expected, f"Failed for query: {query}"
    
    def test_parameter_extraction_time_bucket(self, nl_interface, conversation_context):
        """Test extraction of time bucket sizes."""
        queries_and_expected = [
            ("Show 5min bucket analysis", 5),
            ("Peak traffic in 10 minute buckets", 10),
            ("30min bucket heatmap", 30)
        ]
        
        for query, expected in queries_and_expected:
            params = nl_interface.extract_parameters(query, QueryIntent.ASK_PEAKS, conversation_context)
            assert params.time_bucket_minutes == expected, f"Failed for query: {query}"
    
    def test_context_defaults(self, nl_interface):
        """Test that context provides default values."""
        context = ConversationContext(
            user_id="test",
            session_id="test",
            preferred_airport="DEL",
            preferred_time_bucket=15
        )
        
        query = "Show me peak traffic"
        params = nl_interface.extract_parameters(query, QueryIntent.ASK_PEAKS, context)
        
        assert params.airport_code == "DEL"
        assert params.time_bucket_minutes == 15
    
    def test_peaks_query_handling(self, nl_interface, mock_services, conversation_context):
        """Test handling of peak traffic queries."""
        # Mock the analytics engine response
        mock_peak_analysis = Mock()
        mock_peak_analysis.time_buckets = [Mock(overload=5, utilization=0.8)]
        mock_services['analytics_engine'].analyze_peaks.return_value = mock_peak_analysis
        
        params = QueryParameters(airport_code="BOM", time_bucket_minutes=10)
        result = nl_interface._handle_peaks_query(params)
        
        assert result["type"] == "peak_analysis"
        assert result["airport"] == "BOM"
        assert result["bucket_minutes"] == 10
        mock_services['analytics_engine'].analyze_peaks.assert_called_once_with(
            airport="BOM", bucket_minutes=10
        )
    
    def test_constraints_query_handling(self, nl_interface, conversation_context):
        """Test handling of constraints queries."""
        params = QueryParameters(airport_code="BOM")
        result = nl_interface._handle_constraints_query(params)
        
        assert result["type"] == "constraints"
        assert result["airport"] == "BOM"
        assert "runway_capacity" in result["constraints"]
        assert "turnaround_times" in result["constraints"]
    
    def test_confidence_calculation(self, nl_interface):
        """Test confidence score calculation."""
        # High confidence case
        intent = QueryIntent.ASK_PEAKS
        params = QueryParameters(airport_code="BOM", time_bucket_minutes=10)
        structured_data = {"type": "peak_analysis", "results": "success"}
        
        confidence = nl_interface.calculate_confidence(intent, params, structured_data)
        assert confidence > 0.7
        
        # Low confidence case
        intent = QueryIntent.UNKNOWN
        params = QueryParameters()
        structured_data = {"error": "Failed"}
        
        confidence = nl_interface.calculate_confidence(intent, params, structured_data)
        assert confidence < 0.6
    
    def test_context_update(self, nl_interface):
        """Test conversation context updates."""
        context = ConversationContext(user_id="test", session_id="test")
        query = "What are the busiest slots at BOM?"
        intent = QueryIntent.ASK_PEAKS
        params = QueryParameters(airport_code="BOM", time_bucket_minutes=10)
        
        nl_interface.update_context(context, query, intent, params)
        
        assert context.last_query == query
        assert context.last_intent == intent
        assert context.preferred_airport == "BOM"
        assert context.preferred_time_bucket == 10
        assert len(context.conversation_history) == 1
    
    def test_suggestion_generation(self, nl_interface):
        """Test suggestion generation for different intents."""
        # Test peaks suggestions
        suggestions = nl_interface.generate_suggestions(
            QueryIntent.ASK_PEAKS, 
            QueryParameters(airport_code="BOM")
        )
        assert len(suggestions) > 0
        assert any("delay risk" in s.lower() for s in suggestions)
        
        # Test risk suggestions
        suggestions = nl_interface.generate_suggestions(
            QueryIntent.ASK_RISK,
            QueryParameters(flight_number="AI2739")
        )
        assert len(suggestions) > 0
        assert any("what-if" in s.lower() for s in suggestions)
    
    def test_error_handling(self, nl_interface, mock_services, conversation_context):
        """Test error handling in query processing."""
        # Mock service to raise exception
        mock_services['analytics_engine'].analyze_peaks.side_effect = Exception("Test error")
        
        query = "What are the busiest slots?"
        response = nl_interface.process_query(query, conversation_context)
        
        # Intent should still be correctly identified
        assert response.intent == QueryIntent.ASK_PEAKS
        # But structured data should contain error
        assert "error" in response.structured_data
        assert "error" in response.natural_language_response.lower()
    
    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'})
    def test_gemini_integration_mock(self, nl_interface):
        """Test Gemini integration with mocked API."""
        # This test verifies the integration structure without actual API calls
        query = "What are the busiest slots at BOM?"
        
        # The classify_intent method should handle missing Gemini gracefully
        intent = nl_interface.classify_intent(query)
        assert intent in [QueryIntent.ASK_PEAKS, QueryIntent.UNKNOWN]
    
    def test_response_formatting_templates(self, nl_interface):
        """Test template-based response formatting."""
        # Test peak analysis formatting
        structured_data = {
            "type": "peak_analysis",
            "airport": "BOM",
            "overload_windows": 3,
            "max_utilization": 0.85
        }
        intent = QueryIntent.ASK_PEAKS
        params = QueryParameters(airport_code="BOM")
        
        response = nl_interface._format_response_with_templates(structured_data, intent, params)
        assert "BOM" in response
        assert "3" in response
        assert "85" in response or "0.85" in response
        
        # Test risk prediction formatting
        structured_data = {
            "type": "risk_prediction",
            "flight_number": "AI2739",
            "risk_level": "high",
            "probability": 0.65,
            "expected_delay": 25.5
        }
        intent = QueryIntent.ASK_RISK
        params = QueryParameters(flight_number="AI2739")
        
        response = nl_interface._format_response_with_templates(structured_data, intent, params)
        assert "AI2739" in response
        assert "HIGH" in response.upper()
        assert "65" in response or "0.65" in response
    
    def test_full_query_processing(self, nl_interface, mock_services, conversation_context):
        """Test complete query processing pipeline."""
        # Mock analytics engine
        mock_peak_analysis = Mock()
        mock_peak_analysis.time_buckets = [Mock(overload=2, utilization=0.75)]
        mock_services['analytics_engine'].analyze_peaks.return_value = mock_peak_analysis
        
        query = "What are the busiest 10-minute slots at BOM?"
        response = nl_interface.process_query(query, conversation_context)
        
        assert response.intent == QueryIntent.ASK_PEAKS
        assert response.parameters.airport_code == "BOM"
        assert response.parameters.time_bucket_minutes == 10
        assert response.structured_data["type"] == "peak_analysis"
        assert response.confidence_score > 0.5
        assert len(response.suggestions) > 0
        assert "BOM" in response.natural_language_response


if __name__ == "__main__":
    pytest.main([__file__, "-v"])