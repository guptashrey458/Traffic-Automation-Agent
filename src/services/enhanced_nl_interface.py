"""Enhanced multi-provider NLP interface with autonomous agent orchestration."""

import os
import json
import re
import time as time_module
import asyncio
import logging
from datetime import datetime, date, time, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
from functools import lru_cache
import hashlib
from abc import ABC, abstractmethod

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Multi-provider imports with fallback handling
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import requests
    PERPLEXITY_AVAILABLE = True
except ImportError:
    PERPLEXITY_AVAILABLE = False

from ..models.flight import Flight
from .analytics import AnalyticsEngine, PeakAnalysis, WeatherRegime
from .delay_prediction import DelayRiskPredictor, DelayPrediction
from .schedule_optimizer import ScheduleOptimizer, OptimizationResult
from .whatif_simulator import WhatIfSimulator, WhatIfScenario, ImpactAnalysis
from .database import FlightDatabaseService


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProviderType(Enum):
    """Available NLP providers in fallback chain."""
    GEMINI = "gemini"
    PERPLEXITY = "perplexity"
    OPENAI = "openai"
    LOCAL = "local"


class QueryIntent(Enum):
    """Types of natural language query intents."""
    ASK_PEAKS = "ask_peaks"
    ASK_RISK = "ask_risk"
    ASK_WHATIF = "ask_whatif"
    ASK_OPTIMIZE = "ask_optimize"
    ASK_CONSTRAINTS = "ask_constraints"
    ASK_STATUS = "ask_status"
    ASK_GENERAL = "ask_general"
    AUTONOMOUS_ACTION = "autonomous_action"
    UNKNOWN = "unknown"


class ActionType(Enum):
    """Types of autonomous actions."""
    OPTIMIZE_SCHEDULE = "optimize_schedule"
    SEND_ALERT = "send_alert"
    ADJUST_CAPACITY = "adjust_capacity"
    ESCALATE_TO_HUMAN = "escalate_to_human"
    MONITOR_CONDITION = "monitor_condition"


class ConfidenceLevel(Enum):
    """Confidence levels for autonomous actions."""
    VERY_LOW = "very_low"      # 0.0 - 0.3
    LOW = "low"                # 0.3 - 0.5
    MEDIUM = "medium"          # 0.5 - 0.7
    HIGH = "high"              # 0.7 - 0.9
    VERY_HIGH = "very_high"    # 0.9 - 1.0


@dataclass
class ProviderResponse:
    """Response from an NLP provider."""
    provider: ProviderType
    success: bool
    response: Optional[str] = None
    error: Optional[str] = None
    confidence: float = 0.0
    processing_time: float = 0.0
    rate_limited: bool = False


@dataclass
class QueryParameters:
    """Extracted parameters from natural language query."""
    airport_code: Optional[str] = None
    flight_number: Optional[str] = None
    time_shift_minutes: Optional[int] = None
    date_range: Optional[Tuple[date, date]] = None
    time_bucket_minutes: Optional[int] = None
    runway: Optional[str] = None
    aircraft_type: Optional[str] = None
    raw_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationContext:
    """Enhanced context for maintaining conversation state."""
    user_id: str
    session_id: str
    last_query: Optional[str] = None
    last_intent: Optional[QueryIntent] = None
    last_parameters: Optional[QueryParameters] = None
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    preferred_airport: Optional[str] = None
    preferred_time_bucket: int = 10
    user_expertise_level: str = "intermediate"  # beginner, intermediate, expert
    autonomous_permissions: Dict[str, bool] = field(default_factory=lambda: {
        "minor_optimizations": True,
        "send_alerts": True,
        "adjust_capacity": False,
        "major_changes": False
    })


@dataclass
class AutonomousAction:
    """Represents an autonomous action to be executed."""
    action_id: str
    action_type: ActionType
    confidence: float
    reasoning: str
    parameters: Dict[str, Any]
    requires_human_approval: bool
    estimated_impact: Dict[str, float]
    timeout_seconds: int = 300
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ToolResult:
    """Result from tool execution."""
    tool_name: str
    success: bool
    data: Dict[str, Any]
    error: Optional[str] = None
    execution_time: float = 0.0
    confidence: float = 0.0


@dataclass
class EnhancedNLResponse:
    """Enhanced response with autonomous capabilities."""
    intent: QueryIntent
    parameters: QueryParameters
    structured_data: Dict[str, Any]
    natural_language_response: str
    confidence_score: float
    provider_used: ProviderType
    reasoning: str
    autonomous_actions: List[AutonomousAction] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    processing_time: float = 0.0


class NLProvider(ABC):
    """Abstract base class for NLP providers."""
    
    @abstractmethod
    def classify_intent(self, query: str) -> ProviderResponse:
        """Classify query intent."""
        pass
    
    @abstractmethod
    def extract_parameters(self, query: str, intent: QueryIntent) -> ProviderResponse:
        """Extract parameters from query."""
        pass
    
    @abstractmethod
    def format_response(self, data: Dict[str, Any], intent: QueryIntent) -> ProviderResponse:
        """Format response in natural language."""
        pass
    
    @abstractmethod
    def generate_reasoning(self, action: AutonomousAction) -> ProviderResponse:
        """Generate reasoning for autonomous actions."""
        pass


class GeminiProvider(NLProvider):
    """Gemini Pro NLP provider."""
    
    def __init__(self):
        self.model = None
        self.last_call_time = 0
        self.call_interval = 4.5  # Rate limiting
        self.cache = {}
        
        if GEMINI_AVAILABLE:
            api_key = os.getenv('GEMINI_API_KEY')
            if api_key:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-1.5-flash-8b')
                logger.info("✅ Gemini provider initialized")
            else:
                logger.warning("GEMINI_API_KEY not found")
    
    def _rate_limit(self):
        """Apply rate limiting."""
        current_time = time_module.time()
        time_since_last = current_time - self.last_call_time
        if time_since_last < self.call_interval:
            sleep_time = self.call_interval - time_since_last
            time_module.sleep(sleep_time)
        self.last_call_time = time_module.time()
    
    def classify_intent(self, query: str) -> ProviderResponse:
        """Classify query intent using Gemini."""
        if not self.model:
            return ProviderResponse(
                provider=ProviderType.GEMINI,
                success=False,
                error="Gemini not available"
            )
        
        start_time = time_module.time()
        cache_key = f"intent_{hashlib.md5(query.encode()).hexdigest()}"
        
        if cache_key in self.cache:
            return ProviderResponse(
                provider=ProviderType.GEMINI,
                success=True,
                response=self.cache[cache_key],
                confidence=0.9,
                processing_time=time_module.time() - start_time
            )
        
        try:
            self._rate_limit()
            
            prompt = f"""
            Classify this flight scheduling query into ONE intent:
            
            Intents:
            - ASK_PEAKS: busiest slots, peak traffic, demand analysis
            - ASK_RISK: delay risk, probability of delays
            - ASK_WHATIF: impact of moving/changing flights
            - ASK_OPTIMIZE: optimize schedules, minimize delays
            - ASK_CONSTRAINTS: operational rules, restrictions
            - ASK_STATUS: flight status, information
            - ASK_GENERAL: general aviation questions
            - AUTONOMOUS_ACTION: system should take autonomous action
            - UNKNOWN: unclear queries
            
            Query: "{query}"
            
            Respond with ONLY the intent name.
            """
            
            response = self.model.generate_content(prompt)
            intent_str = response.text.strip().upper()
            
            self.cache[cache_key] = intent_str
            
            return ProviderResponse(
                provider=ProviderType.GEMINI,
                success=True,
                response=intent_str,
                confidence=0.85,
                processing_time=time_module.time() - start_time
            )
            
        except Exception as e:
            if "quota" in str(e).lower() or "rate" in str(e).lower():
                return ProviderResponse(
                    provider=ProviderType.GEMINI,
                    success=False,
                    error=str(e),
                    rate_limited=True,
                    processing_time=time_module.time() - start_time
                )
            else:
                return ProviderResponse(
                    provider=ProviderType.GEMINI,
                    success=False,
                    error=str(e),
                    processing_time=time_module.time() - start_time
                )
    
    def extract_parameters(self, query: str, intent: QueryIntent) -> ProviderResponse:
        """Extract parameters using Gemini."""
        if not self.model:
            return ProviderResponse(
                provider=ProviderType.GEMINI,
                success=False,
                error="Gemini not available"
            )
        
        start_time = time_module.time()
        cache_key = f"params_{hashlib.md5((query + intent.value).encode()).hexdigest()}"
        
        if cache_key in self.cache:
            return ProviderResponse(
                provider=ProviderType.GEMINI,
                success=True,
                response=self.cache[cache_key],
                confidence=0.8,
                processing_time=time_module.time() - start_time
            )
        
        try:
            self._rate_limit()
            
            prompt = f"""
            Extract parameters from this flight scheduling query:
            
            Query: "{query}"
            Intent: {intent.value}
            
            Extract if present (return null if not found):
            - airport_code: 3-letter IATA code (BOM, DEL, CCU, etc.)
            - flight_number: Flight number (AI2739, 6E1234, etc.)
            - time_shift_minutes: Time shift as integer (+10, -15, etc.)
            - time_bucket_minutes: Time bucket size (5, 10, 30)
            - runway: Runway identifier (09/27, 14/32, etc.)
            - aircraft_type: Aircraft type (A320, B737, etc.)
            
            Respond with valid JSON only:
            {{"airport_code": "BOM", "flight_number": "AI2739"}}
            """
            
            response = self.model.generate_content(prompt)
            result = response.text.strip()
            
            # Validate JSON
            json.loads(result)
            self.cache[cache_key] = result
            
            return ProviderResponse(
                provider=ProviderType.GEMINI,
                success=True,
                response=result,
                confidence=0.8,
                processing_time=time_module.time() - start_time
            )
            
        except Exception as e:
            return ProviderResponse(
                provider=ProviderType.GEMINI,
                success=False,
                error=str(e),
                rate_limited="quota" in str(e).lower(),
                processing_time=time_module.time() - start_time
            )
    
    def format_response(self, data: Dict[str, Any], intent: QueryIntent) -> ProviderResponse:
        """Format response using Gemini."""
        if not self.model:
            return ProviderResponse(
                provider=ProviderType.GEMINI,
                success=False,
                error="Gemini not available"
            )
        
        start_time = time_module.time()
        
        try:
            self._rate_limit()
            
            prompt = f"""
            Format this flight scheduling data into a clear response:
            
            Intent: {intent.value}
            Data: {json.dumps(data, default=str, indent=2)}
            
            Guidelines:
            - Use aviation terminology (A-CDM, CODA, slots, turnaround)
            - Be concise but informative
            - Include specific numbers and metrics
            - Suggest actionable next steps
            - Keep under 200 words
            """
            
            response = self.model.generate_content(prompt)
            
            return ProviderResponse(
                provider=ProviderType.GEMINI,
                success=True,
                response=response.text.strip(),
                confidence=0.85,
                processing_time=time_module.time() - start_time
            )
            
        except Exception as e:
            return ProviderResponse(
                provider=ProviderType.GEMINI,
                success=False,
                error=str(e),
                rate_limited="quota" in str(e).lower(),
                processing_time=time_module.time() - start_time
            )
    
    def generate_reasoning(self, action: AutonomousAction) -> ProviderResponse:
        """Generate reasoning for autonomous actions."""
        if not self.model:
            return ProviderResponse(
                provider=ProviderType.GEMINI,
                success=False,
                error="Gemini not available"
            )
        
        start_time = time_module.time()
        
        try:
            self._rate_limit()
            
            prompt = f"""
            Explain the reasoning for this autonomous action:
            
            Action Type: {action.action_type.value}
            Confidence: {action.confidence:.2f}
            Parameters: {json.dumps(action.parameters, default=str)}
            Estimated Impact: {json.dumps(action.estimated_impact, default=str)}
            
            Provide a clear, professional explanation of:
            1. Why this action was recommended
            2. What conditions triggered it
            3. Expected outcomes and benefits
            4. Any risks or considerations
            
            Keep under 150 words.
            """
            
            response = self.model.generate_content(prompt)
            
            return ProviderResponse(
                provider=ProviderType.GEMINI,
                success=True,
                response=response.text.strip(),
                confidence=0.8,
                processing_time=time_module.time() - start_time
            )
            
        except Exception as e:
            return ProviderResponse(
                provider=ProviderType.GEMINI,
                success=False,
                error=str(e),
                rate_limited="quota" in str(e).lower(),
                processing_time=time_module.time() - start_time
            )


class PerplexityProvider(NLProvider):
    """Perplexity API NLP provider."""
    
    def __init__(self):
        self.api_key = os.getenv('PERPLEXITY_API_KEY')
        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.last_call_time = 0
        self.call_interval = 1.0  # 1 second between calls
        
        if self.api_key:
            logger.info("✅ Perplexity provider initialized")
        else:
            logger.warning("PERPLEXITY_API_KEY not found")
    
    def _make_request(self, prompt: str) -> ProviderResponse:
        """Make request to Perplexity API."""
        if not self.api_key:
            return ProviderResponse(
                provider=ProviderType.PERPLEXITY,
                success=False,
                error="Perplexity API key not available"
            )
        
        start_time = time_module.time()
        
        # Rate limiting
        current_time = time_module.time()
        time_since_last = current_time - self.last_call_time
        if time_since_last < self.call_interval:
            time_module.sleep(self.call_interval - time_since_last)
        self.last_call_time = time_module.time()
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "llama-3.1-sonar-small-128k-online",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 500,
                "temperature": 0.1
            }
            
            response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 429:
                return ProviderResponse(
                    provider=ProviderType.PERPLEXITY,
                    success=False,
                    error="Rate limited",
                    rate_limited=True,
                    processing_time=time_module.time() - start_time
                )
            
            response.raise_for_status()
            result = response.json()
            
            return ProviderResponse(
                provider=ProviderType.PERPLEXITY,
                success=True,
                response=result["choices"][0]["message"]["content"],
                confidence=0.75,
                processing_time=time_module.time() - start_time
            )
            
        except Exception as e:
            return ProviderResponse(
                provider=ProviderType.PERPLEXITY,
                success=False,
                error=str(e),
                processing_time=time_module.time() - start_time
            )
    
    def classify_intent(self, query: str) -> ProviderResponse:
        """Classify intent using Perplexity."""
        prompt = f"""
        Classify this flight scheduling query into ONE intent:
        
        ASK_PEAKS, ASK_RISK, ASK_WHATIF, ASK_OPTIMIZE, ASK_CONSTRAINTS, 
        ASK_STATUS, ASK_GENERAL, AUTONOMOUS_ACTION, UNKNOWN
        
        Query: "{query}"
        
        Respond with ONLY the intent name.
        """
        return self._make_request(prompt)
    
    def extract_parameters(self, query: str, intent: QueryIntent) -> ProviderResponse:
        """Extract parameters using Perplexity."""
        prompt = f"""
        Extract parameters from: "{query}"
        Intent: {intent.value}
        
        Return JSON with: airport_code, flight_number, time_shift_minutes, 
        time_bucket_minutes, runway, aircraft_type (null if not found)
        """
        return self._make_request(prompt)
    
    def format_response(self, data: Dict[str, Any], intent: QueryIntent) -> ProviderResponse:
        """Format response using Perplexity."""
        prompt = f"""
        Format this flight data into a professional response:
        Intent: {intent.value}
        Data: {json.dumps(data, default=str)}
        
        Use aviation terminology, be concise, include metrics.
        """
        return self._make_request(prompt)
    
    def generate_reasoning(self, action: AutonomousAction) -> ProviderResponse:
        """Generate reasoning using Perplexity."""
        prompt = f"""
        Explain this autonomous action:
        Type: {action.action_type.value}
        Confidence: {action.confidence}
        Parameters: {action.parameters}
        
        Explain why, what conditions triggered it, expected outcomes.
        """
        return self._make_request(prompt)


class OpenAIProvider(NLProvider):
    """OpenAI GPT NLP provider."""
    
    def __init__(self):
        self.client = None
        self.last_call_time = 0
        self.call_interval = 0.5  # 0.5 seconds between calls
        
        if OPENAI_AVAILABLE:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.client = openai.OpenAI(api_key=api_key)
                logger.info("✅ OpenAI provider initialized")
            else:
                logger.warning("OPENAI_API_KEY not found")
    
    def _make_request(self, prompt: str) -> ProviderResponse:
        """Make request to OpenAI API."""
        if not self.client:
            return ProviderResponse(
                provider=ProviderType.OPENAI,
                success=False,
                error="OpenAI client not available"
            )
        
        start_time = time_module.time()
        
        # Rate limiting
        current_time = time_module.time()
        time_since_last = current_time - self.last_call_time
        if time_since_last < self.call_interval:
            time_module.sleep(self.call_interval - time_since_last)
        self.last_call_time = time_module.time()
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.1
            )
            
            return ProviderResponse(
                provider=ProviderType.OPENAI,
                success=True,
                response=response.choices[0].message.content,
                confidence=0.8,
                processing_time=time_module.time() - start_time
            )
            
        except Exception as e:
            rate_limited = "rate" in str(e).lower() or "quota" in str(e).lower()
            return ProviderResponse(
                provider=ProviderType.OPENAI,
                success=False,
                error=str(e),
                rate_limited=rate_limited,
                processing_time=time_module.time() - start_time
            )
    
    def classify_intent(self, query: str) -> ProviderResponse:
        """Classify intent using OpenAI."""
        prompt = f"""
        Classify this flight scheduling query into ONE intent:
        
        ASK_PEAKS, ASK_RISK, ASK_WHATIF, ASK_OPTIMIZE, ASK_CONSTRAINTS,
        ASK_STATUS, ASK_GENERAL, AUTONOMOUS_ACTION, UNKNOWN
        
        Query: "{query}"
        
        Respond with ONLY the intent name.
        """
        return self._make_request(prompt)
    
    def extract_parameters(self, query: str, intent: QueryIntent) -> ProviderResponse:
        """Extract parameters using OpenAI."""
        prompt = f"""
        Extract parameters from: "{query}"
        Intent: {intent.value}
        
        Return JSON with: airport_code, flight_number, time_shift_minutes,
        time_bucket_minutes, runway, aircraft_type (null if not found)
        """
        return self._make_request(prompt)
    
    def format_response(self, data: Dict[str, Any], intent: QueryIntent) -> ProviderResponse:
        """Format response using OpenAI."""
        prompt = f"""
        Format this flight data into a professional response:
        Intent: {intent.value}
        Data: {json.dumps(data, default=str)}
        
        Use aviation terminology, be concise, include metrics.
        """
        return self._make_request(prompt)
    
    def generate_reasoning(self, action: AutonomousAction) -> ProviderResponse:
        """Generate reasoning using OpenAI."""
        prompt = f"""
        Explain this autonomous action:
        Type: {action.action_type.value}
        Confidence: {action.confidence}
        Parameters: {action.parameters}
        
        Explain why, what conditions triggered it, expected outcomes.
        """
        return self._make_request(prompt)


class LocalProvider(NLProvider):
    """Local rule-based NLP provider (fallback)."""
    
    def __init__(self):
        self.intent_patterns = {
            QueryIntent.ASK_PEAKS: [
                r'busiest.*slots?', r'peak.*traffic', r'demand.*heatmap',
                r'overload.*windows?', r'capacity.*utilization'
            ],
            QueryIntent.ASK_RISK: [
                r'delay.*risk', r'probability.*delay', r'risk.*assessment',
                r'likely.*delayed?'
            ],
            QueryIntent.ASK_WHATIF: [
                r'move.*by', r'shift.*flight', r'what.*if.*move',
                r'impact.*of.*moving'
            ],
            QueryIntent.ASK_OPTIMIZE: [
                r'optimize.*schedule', r'improve.*schedule', r'minimize.*delays'
            ],
            QueryIntent.ASK_CONSTRAINTS: [
                r'operational.*rules', r'constraints', r'restrictions'
            ],
            QueryIntent.ASK_STATUS: [
                r'flight.*status', r'show.*flight', r'flight.*info'
            ]
        }
        
        self.param_patterns = {
            'airport_code': r'\b(BOM|DEL|CCU|MAA|HYD|BLR|GOI|PNQ|AMD)\b',
            'flight_number': r'\b([A-Z0-9]{2}\s*\d{3,4})\b',
            'time_shift_minutes': r'([+-]?\d+)\s*m(?:in(?:ute)?s?)?',
            'time_bucket_minutes': r'(\d+)\s*m(?:in(?:ute)?s?)?.*bucket'
        }
        
        logger.info("✅ Local provider initialized")
    
    def classify_intent(self, query: str) -> ProviderResponse:
        """Classify intent using pattern matching."""
        start_time = time_module.time()
        query_lower = query.lower()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return ProviderResponse(
                        provider=ProviderType.LOCAL,
                        success=True,
                        response=intent.value.upper(),
                        confidence=0.6,
                        processing_time=time_module.time() - start_time
                    )
        
        return ProviderResponse(
            provider=ProviderType.LOCAL,
            success=True,
            response="UNKNOWN",
            confidence=0.3,
            processing_time=time_module.time() - start_time
        )
    
    def extract_parameters(self, query: str, intent: QueryIntent) -> ProviderResponse:
        """Extract parameters using regex."""
        start_time = time_module.time()
        params = {}
        
        for param_name, pattern in self.param_patterns.items():
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                if param_name == 'airport_code':
                    params[param_name] = matches[0].upper()
                elif param_name == 'flight_number':
                    params[param_name] = matches[0].replace(' ', '')
                elif param_name in ['time_shift_minutes', 'time_bucket_minutes']:
                    params[param_name] = int(matches[0])
        
        return ProviderResponse(
            provider=ProviderType.LOCAL,
            success=True,
            response=json.dumps(params),
            confidence=0.5,
            processing_time=time_module.time() - start_time
        )
    
    def format_response(self, data: Dict[str, Any], intent: QueryIntent) -> ProviderResponse:
        """Format response using templates."""
        start_time = time_module.time()
        
        if "error" in data:
            response = f"I'm sorry, {data['error']}"
        elif data.get("type") == "peak_analysis":
            airport = data.get("airport", "airport")
            overload = data.get("overload_windows", 0)
            response = f"Peak analysis for {airport}: Found {overload} overload windows."
        elif data.get("type") == "risk_prediction":
            flight = data.get("flight_number", "flight")
            risk = data.get("risk_level", "unknown")
            response = f"Delay risk for {flight}: {risk} risk level."
        else:
            response = "Analysis completed. Please check the structured data."
        
        return ProviderResponse(
            provider=ProviderType.LOCAL,
            success=True,
            response=response,
            confidence=0.4,
            processing_time=time_module.time() - start_time
        )
    
    def generate_reasoning(self, action: AutonomousAction) -> ProviderResponse:
        """Generate basic reasoning."""
        start_time = time_module.time()
        
        reasoning = f"""
        Autonomous action recommended: {action.action_type.value}
        Confidence level: {action.confidence:.2f}
        
        This action was triggered based on system monitoring and predefined policies.
        Expected impact: {action.estimated_impact}
        """
        
        return ProviderResponse(
            provider=ProviderType.LOCAL,
            success=True,
            response=reasoning.strip(),
            confidence=0.3,
            processing_time=time_module.time() - start_time
        )


class ToolOrchestrator:
    """Orchestrates tool execution for autonomous agent decision-making."""
    
    def __init__(self, 
                 analytics_engine: AnalyticsEngine,
                 delay_predictor: DelayRiskPredictor,
                 schedule_optimizer: ScheduleOptimizer,
                 whatif_simulator: WhatIfSimulator,
                 database_service: FlightDatabaseService):
        self.analytics_engine = analytics_engine
        self.delay_predictor = delay_predictor
        self.schedule_optimizer = schedule_optimizer
        self.whatif_simulator = whatif_simulator
        self.database_service = database_service
        
        # Tool registry
        self.tools = {
            "analyze_peaks": self._analyze_peaks,
            "predict_risk": self._predict_risk,
            "simulate_whatif": self._simulate_whatif,
            "optimize_schedule": self._optimize_schedule,
            "get_constraints": self._get_constraints,
            "get_flight_status": self._get_flight_status
        }
        
        logger.info("✅ Tool orchestrator initialized")
    
    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
        """Execute a specific tool with parameters."""
        start_time = time_module.time()
        
        if tool_name not in self.tools:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                data={},
                error=f"Tool '{tool_name}' not found",
                execution_time=time_module.time() - start_time
            )
        
        try:
            tool_func = self.tools[tool_name]
            result_data = tool_func(parameters)
            
            return ToolResult(
                tool_name=tool_name,
                success=True,
                data=result_data,
                execution_time=time_module.time() - start_time,
                confidence=0.8
            )
            
        except Exception as e:
            logger.error(f"Tool execution failed: {tool_name} - {str(e)}")
            return ToolResult(
                tool_name=tool_name,
                success=False,
                data={},
                error=str(e),
                execution_time=time_module.time() - start_time
            )
    
    def _analyze_peaks(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute peak analysis tool."""
        airport = params.get('airport_code', 'BOM')
        bucket_minutes = params.get('time_bucket_minutes', 10)
        
        peak_analysis = self.analytics_engine.analyze_peaks(
            airport=airport,
            bucket_minutes=bucket_minutes
        )
        
        return {
            "type": "peak_analysis",
            "airport": airport,
            "bucket_minutes": bucket_minutes,
            "peak_analysis": peak_analysis,
            "overload_windows": len([b for b in peak_analysis.time_buckets if b.overload > 0]),
            "max_utilization": max([b.utilization for b in peak_analysis.time_buckets], default=0)
        }
    
    def _predict_risk(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute risk prediction tool."""
        flight_number = params.get('flight_number')
        if not flight_number:
            raise ValueError("Flight number required for risk analysis")
        
        # Get flight data
        from datetime import date, timedelta
        today = date.today()
        result = self.database_service.query_flights_by_date_range(
            start_date=today - timedelta(days=7),
            end_date=today
        )
        
        flight_data = [f for f in result.data if f.get('flight_number') == flight_number]
        if not flight_data:
            raise ValueError(f"Flight {flight_number} not found")
        
        # Convert to Flight object
        latest_flight_data = flight_data[0]
        flight = Flight(
            flight_id=latest_flight_data['flight_id'],
            flight_no=latest_flight_data['flight_number'],
            date_local=latest_flight_data['date_local'],
            origin=latest_flight_data['origin_code'],
            destination=latest_flight_data['destination_code'],
            aircraft_type=latest_flight_data.get('aircraft_type', 'UNKNOWN'),
            std_utc=latest_flight_data['std_utc'],
            atd_utc=latest_flight_data.get('atd_utc'),
            sta_utc=latest_flight_data['sta_utc'],
            ata_utc=latest_flight_data.get('ata_utc'),
            dep_delay_min=latest_flight_data.get('dep_delay_minutes'),
            arr_delay_min=latest_flight_data.get('arr_delay_minutes')
        )
        
        prediction = self.delay_predictor.predict_departure_delay(flight)
        
        return {
            "type": "risk_prediction",
            "flight_number": flight_number,
            "prediction": prediction,
            "risk_level": prediction.risk_level.value,
            "probability": prediction.delay_probability,
            "expected_delay": prediction.expected_delay_minutes
        }
    
    def _simulate_whatif(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute what-if simulation tool."""
        flight_number = params.get('flight_number')
        time_shift = params.get('time_shift_minutes')
        
        if not flight_number or time_shift is None:
            raise ValueError("Flight number and time shift required")
        
        scenario = WhatIfScenario(
            flight_number=flight_number,
            time_shift_minutes=time_shift
        )
        
        impact = self.whatif_simulator.simulate_scenario(scenario)
        
        return {
            "type": "whatif_analysis",
            "scenario": scenario,
            "impact": impact,
            "delay_change": impact.delay_delta,
            "affected_flights": len(impact.affected_flights)
        }
    
    def _optimize_schedule(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute schedule optimization tool."""
        airport = params.get('airport_code', 'BOM')
        
        # Get current schedule
        from datetime import date, timedelta
        today = date.today()
        result = self.database_service.query_flights_by_date_range(
            start_date=today,
            end_date=today + timedelta(days=1),
            airport_code=airport
        )
        
        # Convert to Flight objects
        flights = []
        for flight_data in result.data:
            flight = Flight(
                flight_id=flight_data['flight_id'],
                flight_no=flight_data['flight_number'],
                date_local=flight_data['date_local'],
                origin=flight_data['origin_code'],
                destination=flight_data['destination_code'],
                aircraft_type=flight_data.get('aircraft_type', 'UNKNOWN'),
                std_utc=flight_data['std_utc'],
                atd_utc=flight_data.get('atd_utc'),
                sta_utc=flight_data['sta_utc'],
                ata_utc=flight_data.get('ata_utc'),
                dep_delay_min=flight_data.get('dep_delay_minutes'),
                arr_delay_min=flight_data.get('arr_delay_minutes')
            )
            flights.append(flight)
        
        optimization_result = self.schedule_optimizer.optimize_schedule(flights)
        
        return {
            "type": "optimization",
            "airport": airport,
            "result": optimization_result,
            "cost_reduction": optimization_result.cost_reduction,
            "affected_flights": len(optimization_result.affected_flights)
        }
    
    def _get_constraints(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get operational constraints."""
        airport = params.get('airport_code', 'BOM')
        
        from .schedule_optimizer import Constraints
        constraints = Constraints(
            runway_capacity={"09/27": 30, "14/32": 25},
            min_turnaround_minutes={"A320": 45, "B737": 45, "B777": 60},
            curfew_hours=[23, 0, 1, 2, 3, 4, 5],
            min_separation_minutes=2.0
        )
        
        return {
            "type": "constraints",
            "airport": airport,
            "constraints": {
                "runway_capacity": constraints.runway_capacity,
                "turnaround_times": constraints.min_turnaround_minutes,
                "curfew_hours": constraints.curfew_hours,
                "min_separation": constraints.min_separation_minutes
            }
        }
    
    def _get_flight_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get flight status information."""
        flight_number = params.get('flight_number')
        if not flight_number:
            raise ValueError("Flight number required")
        
        from datetime import date, timedelta
        today = date.today()
        result = self.database_service.query_flights_by_date_range(
            start_date=today - timedelta(days=7),
            end_date=today + timedelta(days=1)
        )
        
        flight_data = [f for f in result.data if f.get('flight_number') == flight_number]
        if not flight_data:
            raise ValueError(f"Flight {flight_number} not found")
        
        flight_data.sort(key=lambda x: x.get('date_local', ''), reverse=True)
        
        return {
            "type": "flight_status",
            "flight_number": flight_number,
            "flights": flight_data[:5],
            "latest_flight": flight_data[0] if flight_data else None
        }


class AutonomousDecisionEngine:
    """Engine for autonomous decision-making with confidence-based execution."""
    
    def __init__(self, tool_orchestrator: ToolOrchestrator):
        self.tool_orchestrator = tool_orchestrator
        self.action_history = []
        self.confidence_thresholds = {
            ActionType.SEND_ALERT: 0.6,
            ActionType.MONITOR_CONDITION: 0.5,
            ActionType.ADJUST_CAPACITY: 0.8,
            ActionType.OPTIMIZE_SCHEDULE: 0.9,
            ActionType.ESCALATE_TO_HUMAN: 0.3
        }
        
        logger.info("✅ Autonomous decision engine initialized")
    
    def evaluate_autonomous_actions(self, 
                                  intent: QueryIntent, 
                                  parameters: QueryParameters,
                                  context: ConversationContext) -> List[AutonomousAction]:
        """Evaluate if autonomous actions should be taken."""
        actions = []
        
        # Check for autonomous optimization opportunities
        if intent == QueryIntent.ASK_OPTIMIZE and context.autonomous_permissions.get("minor_optimizations", False):
            action = AutonomousAction(
                action_id=f"auto_opt_{datetime.now().timestamp()}",
                action_type=ActionType.OPTIMIZE_SCHEDULE,
                confidence=0.85,
                reasoning="User requested optimization and has autonomous permissions enabled",
                parameters={"airport_code": parameters.airport_code or "BOM"},
                requires_human_approval=False,
                estimated_impact={"delay_reduction": 15.0, "affected_flights": 10}
            )
            actions.append(action)
        
        # Check for alert conditions
        if intent == QueryIntent.ASK_PEAKS:
            # Simulate checking for overload conditions
            action = AutonomousAction(
                action_id=f"alert_{datetime.now().timestamp()}",
                action_type=ActionType.SEND_ALERT,
                confidence=0.7,
                reasoning="Peak analysis detected potential capacity overload",
                parameters={"airport_code": parameters.airport_code or "BOM", "severity": "medium"},
                requires_human_approval=False,
                estimated_impact={"notification_sent": 1}
            )
            actions.append(action)
        
        return actions
    
    def should_execute_action(self, action: AutonomousAction, context: ConversationContext) -> bool:
        """Determine if an action should be executed autonomously."""
        # Check confidence threshold
        threshold = self.confidence_thresholds.get(action.action_type, 0.8)
        if action.confidence < threshold:
            return False
        
        # Check user permissions
        permission_map = {
            ActionType.SEND_ALERT: "send_alerts",
            ActionType.OPTIMIZE_SCHEDULE: "minor_optimizations",
            ActionType.ADJUST_CAPACITY: "adjust_capacity"
        }
        
        permission_key = permission_map.get(action.action_type)
        if permission_key and not context.autonomous_permissions.get(permission_key, False):
            return False
        
        # Check if human approval is required
        if action.requires_human_approval:
            return False
        
        return True
    
    def execute_action(self, action: AutonomousAction) -> Dict[str, Any]:
        """Execute an autonomous action."""
        start_time = time_module.time()
        
        try:
            if action.action_type == ActionType.OPTIMIZE_SCHEDULE:
                result = self.tool_orchestrator.execute_tool("optimize_schedule", action.parameters)
            elif action.action_type == ActionType.SEND_ALERT:
                result = self._send_alert(action.parameters)
            elif action.action_type == ActionType.ADJUST_CAPACITY:
                result = self._adjust_capacity(action.parameters)
            else:
                result = ToolResult(
                    tool_name=action.action_type.value,
                    success=False,
                    data={},
                    error="Action type not implemented"
                )
            
            # Log the action
            self.action_history.append({
                "action_id": action.action_id,
                "action_type": action.action_type.value,
                "executed_at": datetime.now(),
                "success": result.success,
                "execution_time": time_module.time() - start_time
            })
            
            return {
                "success": result.success,
                "data": result.data,
                "error": result.error,
                "execution_time": result.execution_time
            }
            
        except Exception as e:
            logger.error(f"Action execution failed: {action.action_id} - {str(e)}")
            return {
                "success": False,
                "data": {},
                "error": str(e),
                "execution_time": time_module.time() - start_time
            }
    
    def _send_alert(self, parameters: Dict[str, Any]) -> ToolResult:
        """Send alert notification."""
        start_time = time_module.time()
        
        # Simulate sending alert
        airport = parameters.get("airport_code", "BOM")
        severity = parameters.get("severity", "medium")
        
        logger.info(f"🚨 AUTONOMOUS ALERT: {severity.upper()} severity at {airport}")
        
        return ToolResult(
            tool_name="send_alert",
            success=True,
            data={
                "alert_sent": True,
                "airport": airport,
                "severity": severity,
                "timestamp": datetime.now().isoformat()
            },
            execution_time=time_module.time() - start_time
        )
    
    def _adjust_capacity(self, parameters: Dict[str, Any]) -> ToolResult:
        """Adjust runway capacity."""
        start_time = time_module.time()
        
        # Simulate capacity adjustment
        runway = parameters.get("runway", "09/27")
        adjustment = parameters.get("adjustment", 0)
        
        logger.info(f"⚙️ AUTONOMOUS CAPACITY ADJUSTMENT: {runway} {adjustment:+d}")
        
        return ToolResult(
            tool_name="adjust_capacity",
            success=True,
            data={
                "capacity_adjusted": True,
                "runway": runway,
                "adjustment": adjustment,
                "timestamp": datetime.now().isoformat()
            },
            execution_time=time_module.time() - start_time
        )


class EnhancedNLInterface:
    """Enhanced multi-provider NLP interface with autonomous agent orchestration."""
    
    def __init__(self,
                 analytics_engine: AnalyticsEngine,
                 delay_predictor: DelayRiskPredictor,
                 schedule_optimizer: ScheduleOptimizer,
                 whatif_simulator: WhatIfSimulator,
                 database_service: FlightDatabaseService):
        
        # Initialize providers in fallback order
        self.providers = [
            GeminiProvider(),
            PerplexityProvider(),
            OpenAIProvider(),
            LocalProvider()
        ]
        
        # Initialize orchestration components
        self.tool_orchestrator = ToolOrchestrator(
            analytics_engine, delay_predictor, schedule_optimizer,
            whatif_simulator, database_service
        )
        
        self.decision_engine = AutonomousDecisionEngine(self.tool_orchestrator)
        
        # Provider usage statistics
        self.provider_stats = {provider.classify_intent.__self__.__class__.__name__: 
                             {"calls": 0, "successes": 0, "failures": 0} 
                             for provider in self.providers}
        
        logger.info("✅ Enhanced NL Interface initialized with multi-provider chain")
    
    def process_query(self, query: str, context: ConversationContext) -> EnhancedNLResponse:
        """Process query with multi-provider fallback and autonomous capabilities."""
        start_time = time_module.time()
        
        try:
            # Step 1: Classify intent with provider fallback
            intent, provider_used = self._classify_intent_with_fallback(query)
            
            # Step 2: Extract parameters
            parameters = self._extract_parameters_with_fallback(query, intent)
            
            # Step 3: Route to appropriate tool
            tool_result = self._route_to_tool(intent, parameters)
            
            # Step 4: Evaluate autonomous actions
            autonomous_actions = self.decision_engine.evaluate_autonomous_actions(
                intent, parameters, context
            )
            
            # Step 5: Execute autonomous actions if appropriate
            executed_actions = []
            for action in autonomous_actions:
                if self.decision_engine.should_execute_action(action, context):
                    execution_result = self.decision_engine.execute_action(action)
                    action.parameters["execution_result"] = execution_result
                    executed_actions.append(action)
                    logger.info(f"🤖 Executed autonomous action: {action.action_type.value}")
            
            # Step 6: Generate reasoning for actions
            reasoning = self._generate_reasoning_with_fallback(executed_actions)
            
            # Step 7: Format natural language response
            nl_response = self._format_response_with_fallback(
                tool_result.data, intent, parameters
            )
            
            # Step 8: Calculate confidence
            confidence = self._calculate_confidence(intent, parameters, tool_result, provider_used)
            
            # Step 9: Generate suggestions
            suggestions = self._generate_suggestions(intent, parameters, tool_result.data)
            
            # Step 10: Update context
            self._update_context(context, query, intent, parameters)
            
            return EnhancedNLResponse(
                intent=intent,
                parameters=parameters,
                structured_data=tool_result.data,
                natural_language_response=nl_response,
                confidence_score=confidence,
                provider_used=provider_used,
                reasoning=reasoning,
                autonomous_actions=executed_actions,
                suggestions=suggestions,
                error_message=tool_result.error,
                processing_time=time_module.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            return EnhancedNLResponse(
                intent=QueryIntent.UNKNOWN,
                parameters=QueryParameters(),
                structured_data={},
                natural_language_response=f"I encountered an error: {str(e)}",
                confidence_score=0.0,
                provider_used=ProviderType.LOCAL,
                reasoning="Error occurred during processing",
                error_message=str(e),
                processing_time=time_module.time() - start_time
            )
    
    def _classify_intent_with_fallback(self, query: str) -> Tuple[QueryIntent, ProviderType]:
        """Classify intent with provider fallback chain."""
        for provider in self.providers:
            provider_name = provider.__class__.__name__
            self.provider_stats[provider_name]["calls"] += 1
            
            try:
                response = provider.classify_intent(query)
                
                if response.success and not response.rate_limited:
                    intent = QueryIntent(response.response.lower())
                    self.provider_stats[provider_name]["successes"] += 1
                    logger.info(f"✅ Intent classified by {provider_name}: {intent.value}")
                    return intent, response.provider
                else:
                    self.provider_stats[provider_name]["failures"] += 1
                    if response.rate_limited:
                        logger.warning(f"⏳ {provider_name} rate limited, trying next provider")
                    else:
                        logger.warning(f"❌ {provider_name} failed: {response.error}")
                    continue
                    
            except Exception as e:
                self.provider_stats[provider_name]["failures"] += 1
                logger.error(f"❌ {provider_name} exception: {str(e)}")
                continue
        
        # Fallback to UNKNOWN if all providers fail
        logger.warning("🔄 All providers failed for intent classification")
        return QueryIntent.UNKNOWN, ProviderType.LOCAL
    
    def _extract_parameters_with_fallback(self, query: str, intent: QueryIntent) -> QueryParameters:
        """Extract parameters with provider fallback."""
        for provider in self.providers:
            try:
                response = provider.extract_parameters(query, intent)
                
                if response.success and not response.rate_limited:
                    params_dict = json.loads(response.response)
                    
                    # Convert to QueryParameters object
                    params = QueryParameters()
                    for key, value in params_dict.items():
                        if hasattr(params, key) and value is not None:
                            setattr(params, key, value)
                    
                    logger.info(f"✅ Parameters extracted by {provider.__class__.__name__}")
                    return params
                    
            except Exception as e:
                logger.warning(f"Parameter extraction failed with {provider.__class__.__name__}: {e}")
                continue
        
        # Return empty parameters if all fail
        logger.warning("🔄 All providers failed for parameter extraction")
        return QueryParameters()
    
    def _route_to_tool(self, intent: QueryIntent, parameters: QueryParameters) -> ToolResult:
        """Route query to appropriate tool."""
        tool_map = {
            QueryIntent.ASK_PEAKS: "analyze_peaks",
            QueryIntent.ASK_RISK: "predict_risk",
            QueryIntent.ASK_WHATIF: "simulate_whatif",
            QueryIntent.ASK_OPTIMIZE: "optimize_schedule",
            QueryIntent.ASK_CONSTRAINTS: "get_constraints",
            QueryIntent.ASK_STATUS: "get_flight_status"
        }
        
        tool_name = tool_map.get(intent)
        if not tool_name:
            return ToolResult(
                tool_name="unknown",
                success=False,
                data={"error": f"No tool available for intent: {intent.value}"},
                error=f"Intent {intent.value} not supported"
            )
        
        # Convert parameters to dict
        params_dict = {k: v for k, v in parameters.__dict__.items() if v is not None}
        
        return self.tool_orchestrator.execute_tool(tool_name, params_dict)
    
    def _generate_reasoning_with_fallback(self, actions: List[AutonomousAction]) -> str:
        """Generate reasoning for autonomous actions."""
        if not actions:
            return "No autonomous actions were taken."
        
        reasoning_parts = []
        
        for action in actions:
            for provider in self.providers:
                try:
                    response = provider.generate_reasoning(action)
                    if response.success:
                        reasoning_parts.append(response.response)
                        break
                except Exception:
                    continue
            else:
                # Fallback reasoning
                reasoning_parts.append(
                    f"Executed {action.action_type.value} with confidence {action.confidence:.2f}"
                )
        
        return " ".join(reasoning_parts)
    
    def _format_response_with_fallback(self, data: Dict[str, Any], 
                                     intent: QueryIntent, 
                                     parameters: QueryParameters) -> str:
        """Format response with provider fallback."""
        for provider in self.providers:
            try:
                response = provider.format_response(data, intent)
                if response.success and not response.rate_limited:
                    return response.response
            except Exception:
                continue
        
        # Fallback formatting
        if "error" in data:
            return f"I'm sorry, {data['error']}"
        else:
            return "Analysis completed. Please check the structured data for details."
    
    def _calculate_confidence(self, intent: QueryIntent, parameters: QueryParameters,
                            tool_result: ToolResult, provider_used: ProviderType) -> float:
        """Calculate overall confidence score."""
        confidence = 0.3  # Base confidence
        
        # Intent recognition confidence
        if intent != QueryIntent.UNKNOWN:
            confidence += 0.2
        
        # Parameter extraction confidence
        param_count = sum(1 for v in parameters.__dict__.values() if v is not None)
        confidence += min(param_count * 0.1, 0.2)
        
        # Tool execution confidence
        if tool_result.success:
            confidence += 0.2
        
        # Provider reliability bonus
        provider_bonuses = {
            ProviderType.GEMINI: 0.1,
            ProviderType.PERPLEXITY: 0.08,
            ProviderType.OPENAI: 0.08,
            ProviderType.LOCAL: 0.0
        }
        confidence += provider_bonuses.get(provider_used, 0.0)
        
        return max(0.0, min(confidence, 1.0))
    
    def _generate_suggestions(self, intent: QueryIntent, parameters: QueryParameters,
                            data: Dict[str, Any]) -> List[str]:
        """Generate contextual suggestions."""
        suggestions = []
        
        if intent == QueryIntent.ASK_PEAKS:
            suggestions.extend([
                "Try asking about delay risks for specific flights",
                "Consider running schedule optimization",
                "Check what-if scenarios for peak slots"
            ])
        elif intent == QueryIntent.ASK_RISK:
            suggestions.extend([
                "Analyze peak traffic patterns",
                "Run what-if analysis for this flight",
                "Check operational constraints"
            ])
        elif intent == QueryIntent.ASK_WHATIF:
            suggestions.extend([
                "Compare multiple time shift scenarios",
                "Run full schedule optimization",
                "Check cascade impact analysis"
            ])
        
        return suggestions[:3]  # Limit to 3 suggestions
    
    def _update_context(self, context: ConversationContext, query: str,
                       intent: QueryIntent, parameters: QueryParameters):
        """Update conversation context."""
        context.last_query = query
        context.last_intent = intent
        context.last_parameters = parameters
        
        # Update preferences
        if parameters.airport_code:
            context.preferred_airport = parameters.airport_code
        if parameters.time_bucket_minutes:
            context.preferred_time_bucket = parameters.time_bucket_minutes
        
        # Add to history
        context.conversation_history.append({
            "timestamp": datetime.now(),
            "query": query,
            "intent": intent.value,
            "parameters": parameters.__dict__
        })
        
        # Keep last 10 interactions
        if len(context.conversation_history) > 10:
            context.conversation_history = context.conversation_history[-10:]
    
    def get_provider_statistics(self) -> Dict[str, Any]:
        """Get provider usage statistics."""
        return {
            "provider_stats": self.provider_stats,
            "total_calls": sum(stats["calls"] for stats in self.provider_stats.values()),
            "success_rate": sum(stats["successes"] for stats in self.provider_stats.values()) / 
                          max(sum(stats["calls"] for stats in self.provider_stats.values()), 1)
        }
    
    def get_autonomous_action_history(self) -> List[Dict[str, Any]]:
        """Get history of autonomous actions."""
        return self.decision_engine.action_history