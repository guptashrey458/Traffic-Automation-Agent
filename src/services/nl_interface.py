"""Natural language interface using Gemini for flight scheduling queries."""

import os
import json
import re
import time as time_module
from datetime import datetime, date, time, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
from functools import lru_cache
import hashlib

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, environment variables should be set manually

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-generativeai not available. NL interface will use fallback methods.")

from ..models.flight import Flight
from .analytics import AnalyticsEngine, PeakAnalysis, WeatherRegime
from .delay_prediction import DelayRiskPredictor, DelayPrediction
from .schedule_optimizer import ScheduleOptimizer, OptimizationResult
from .whatif_simulator import WhatIfSimulator, WhatIfScenario, ImpactAnalysis
from .database import FlightDatabaseService


class QueryIntent(Enum):
    """Types of natural language query intents."""
    ASK_PEAKS = "ask_peaks"                 # "What are the busiest slots?"
    ASK_RISK = "ask_risk"                   # "What's the delay risk for AI 2739?"
    ASK_WHATIF = "ask_whatif"               # "Move AI 2739 by +10m—impact?"
    ASK_OPTIMIZE = "ask_optimize"           # "Optimize schedule for BOM"
    ASK_CONSTRAINTS = "ask_constraints"     # "What are the operational rules?"
    ASK_STATUS = "ask_status"               # "Show flight status for AI 2739"
    ASK_GENERAL = "ask_general"             # General aviation questions
    UNKNOWN = "unknown"                     # Unrecognized intent


class ParameterType(Enum):
    """Types of parameters that can be extracted from queries."""
    AIRPORT_CODE = "airport_code"           # BOM, DEL, etc.
    FLIGHT_NUMBER = "flight_number"         # AI 2739, 6E 1234
    TIME_SHIFT = "time_shift"               # +10m, -15min, +30
    DATE_RANGE = "date_range"               # today, tomorrow, next week
    TIME_BUCKET = "time_bucket"             # 5min, 10min, 30min
    RUNWAY = "runway"                       # 09/27, 14/32
    AIRCRAFT_TYPE = "aircraft_type"         # A320, B737


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
    """Context for maintaining conversation state."""
    user_id: str
    session_id: str
    last_query: Optional[str] = None
    last_intent: Optional[QueryIntent] = None
    last_parameters: Optional[QueryParameters] = None
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    preferred_airport: Optional[str] = None
    preferred_time_bucket: int = 10  # Default 10-minute buckets


@dataclass
class NLResponse:
    """Structured response from natural language processing."""
    intent: QueryIntent
    parameters: QueryParameters
    structured_data: Dict[str, Any]
    natural_language_response: str
    confidence_score: float
    suggestions: List[str] = field(default_factory=list)
    error_message: Optional[str] = None


class NLInterface:
    """Natural language interface for flight scheduling system."""
    
    def __init__(self, 
                 analytics_engine: AnalyticsEngine,
                 delay_predictor: DelayRiskPredictor,
                 schedule_optimizer: ScheduleOptimizer,
                 whatif_simulator: WhatIfSimulator,
                 database_service: FlightDatabaseService):
        """Initialize NL interface with required services."""
        self.analytics_engine = analytics_engine
        self.delay_predictor = delay_predictor
        self.schedule_optimizer = schedule_optimizer
        self.whatif_simulator = whatif_simulator
        self.database_service = database_service
        
        # Initialize Gemini if available
        self.gemini_model = None
        self.last_gemini_call = 0
        self.gemini_call_interval = 4.5  # 4.5 seconds between calls (13 calls/minute max)
        self.gemini_cache = {}  # Simple cache for repeated queries
        
        if GEMINI_AVAILABLE:
            api_key = os.getenv('GEMINI_API_KEY')
            if api_key:
                genai.configure(api_key=api_key)
                # Use Gemini 1.5 Flash-8B for better rate limits on free tier
                self.gemini_model = genai.GenerativeModel('gemini-1.5-flash-8b')
                print("✅ Gemini 1.5 Flash-8B initialized with rate limiting")
            else:
                print("Warning: GEMINI_API_KEY not found in environment variables")
        
        # Intent classification patterns
        self.intent_patterns = {
            QueryIntent.ASK_PEAKS: [
                r'busiest.*slots?', r'peak.*traffic', r'demand.*heatmap',
                r'overload.*windows?', r'capacity.*utilization', r'traffic.*analysis',
                r'time.*slots.*overload', r'which.*slots', r'busy.*time'
            ],
            QueryIntent.ASK_RISK: [
                r'delay.*risk', r'probability.*delay', r'risk.*assessment',
                r'likely.*delayed?', r'delay.*prediction', r'risk.*score'
            ],
            QueryIntent.ASK_WHATIF: [
                r'move.*by', r'shift.*flight', r'what.*if.*move',
                r'impact.*of.*moving', r'change.*time', r'reschedule'
            ],
            QueryIntent.ASK_OPTIMIZE: [
                r'optimize.*schedule', r'improve.*schedule', r'minimize.*delays',
                r'best.*schedule', r'optimization', r'reduce.*delays'
            ],
            QueryIntent.ASK_CONSTRAINTS: [
                r'operational.*rules', r'constraints', r'restrictions',
                r'turnaround.*time', r'runway.*capacity', r'curfew'
            ],
            QueryIntent.ASK_STATUS: [
                r'flight.*status', r'show.*flight', r'flight.*info',
                r'details.*for', r'information.*about'
            ]
        }
        
        # Parameter extraction patterns
        self.param_patterns = {
            ParameterType.AIRPORT_CODE: r'\b(BOM|DEL|CCU|MAA|HYD|BLR|GOI|PNQ|AMD|JAI|IXC|COK|TRV|IXM|IXR|IXL|IXU|IXJ|IXA|IXB|IXD|IXE|IXG|IXH|IXI|IXK|IXN|IXP|IXQ|IXS|IXV|IXW|IXY|IXZ)\b',  # Indian airport codes
            ParameterType.FLIGHT_NUMBER: r'\b([A-Z0-9]{2}\s*\d{3,4})\b',  # AI 2739, 6E1234
            ParameterType.TIME_SHIFT: r'([+-]?\d+)\s*m(?:in(?:ute)?s?)?',  # +10m, -15min
            ParameterType.TIME_BUCKET: r'(\d+)\s*m(?:in(?:ute)?s?)?.*bucket',  # 5min bucket
        }
    
    def process_query(self, query: str, context: ConversationContext) -> NLResponse:
        """Process a natural language query and return structured response."""
        try:
            # Classify intent
            intent = self.classify_intent(query)
            
            # Extract parameters
            parameters = self.extract_parameters(query, intent, context)
            
            # Route to appropriate tool
            structured_data = self.route_to_tool(intent, parameters)
            
            # Format natural language response
            nl_response = self.format_response(structured_data, intent, parameters)
            
            # Calculate confidence score
            confidence = self.calculate_confidence(intent, parameters, structured_data)
            
            # Update conversation context
            self.update_context(context, query, intent, parameters)
            
            return NLResponse(
                intent=intent,
                parameters=parameters,
                structured_data=structured_data,
                natural_language_response=nl_response,
                confidence_score=confidence,
                suggestions=self.generate_suggestions(intent, parameters)
            )
            
        except Exception as e:
            return NLResponse(
                intent=QueryIntent.UNKNOWN,
                parameters=QueryParameters(),
                structured_data={},
                natural_language_response=f"I encountered an error processing your query: {str(e)}",
                confidence_score=0.0,
                error_message=str(e)
            )
    
    def classify_intent(self, query: str) -> QueryIntent:
        """Classify the intent of a natural language query."""
        query_lower = query.lower()
        
        # Use Gemini Pro for intent classification if available
        if self.gemini_model:
            try:
                return self._classify_intent_with_gemini(query)
            except Exception as e:
                print(f"Gemini classification failed, using fallback: {e}")
        
        # Fallback to pattern matching
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return intent
        
        return QueryIntent.UNKNOWN
    
    def _classify_intent_with_gemini(self, query: str) -> QueryIntent:
        """Use Gemini to classify query intent with rate limiting and caching."""
        # Check cache first
        query_hash = hashlib.md5(query.lower().encode()).hexdigest()
        if query_hash in self.gemini_cache:
            return self.gemini_cache[query_hash]
        
        # Rate limiting - ensure we don't exceed quota
        current_time = time_module.time()
        time_since_last_call = current_time - self.last_gemini_call
        if time_since_last_call < self.gemini_call_interval:
            sleep_time = self.gemini_call_interval - time_since_last_call
            print(f"⏳ Rate limiting: waiting {sleep_time:.1f}s before Gemini call")
            time_module.sleep(sleep_time)
        
        prompt = f"""
        Classify this flight scheduling query into ONE of these intents:
        
        ASK_PEAKS - Questions about busiest time slots, peak traffic, demand analysis
        ASK_RISK - Questions about delay risk, probability of delays for specific flights  
        ASK_WHATIF - Questions about impact of moving/changing flights
        ASK_OPTIMIZE - Requests to optimize schedules or minimize delays
        ASK_CONSTRAINTS - Questions about operational rules, constraints, restrictions
        ASK_STATUS - Questions about specific flight status or information
        ASK_GENERAL - General aviation questions
        UNKNOWN - Unrecognized or unclear queries
        
        Query: "{query}"
        
        Respond with ONLY the intent name (e.g., "ASK_PEAKS").
        """
        
        try:
            self.last_gemini_call = time_module.time()
            response = self.gemini_model.generate_content(prompt)
            intent_str = response.text.strip().upper()
            
            # Parse the response
            intent = QueryIntent(intent_str.lower())
            
            # Cache the result
            self.gemini_cache[query_hash] = intent
            print(f"🤖 Gemini classified: \"{query}\" → {intent.value}")
            
            return intent
            
        except ValueError:
            fallback_intent = QueryIntent.UNKNOWN
            self.gemini_cache[query_hash] = fallback_intent
            return fallback_intent
    
    def extract_parameters(self, query: str, intent: QueryIntent, 
                          context: ConversationContext) -> QueryParameters:
        """Extract parameters from natural language query."""
        params = QueryParameters()
        
        # Extract using regex patterns
        for param_type, pattern in self.param_patterns.items():
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                if param_type == ParameterType.AIRPORT_CODE:
                    params.airport_code = matches[0].upper()
                elif param_type == ParameterType.FLIGHT_NUMBER:
                    params.flight_number = matches[0].replace(' ', '')
                elif param_type == ParameterType.TIME_SHIFT:
                    params.time_shift_minutes = int(matches[0])
                elif param_type == ParameterType.TIME_BUCKET:
                    params.time_bucket_minutes = int(matches[0])
        
        # Use context defaults if parameters not found
        if not params.airport_code and context.preferred_airport:
            params.airport_code = context.preferred_airport
        
        if not params.time_bucket_minutes:
            params.time_bucket_minutes = context.preferred_time_bucket
        
        # Use Gemini Pro for more sophisticated parameter extraction
        if self.gemini_model:
            try:
                enhanced_params = self._extract_parameters_with_gemini(query, intent)
                # Merge enhanced parameters
                for key, value in enhanced_params.items():
                    if value and not getattr(params, key, None):
                        setattr(params, key, value)
            except Exception as e:
                print(f"Gemini parameter extraction failed: {e}")
        
        return params
    
    def _extract_parameters_with_gemini(self, query: str, intent: QueryIntent) -> Dict[str, Any]:
        """Use Gemini to extract parameters from query with rate limiting."""
        # Check cache first
        cache_key = f"params_{hashlib.md5((query + intent.value).encode()).hexdigest()}"
        if cache_key in self.gemini_cache:
            return self.gemini_cache[cache_key]
        
        # Rate limiting
        current_time = time_module.time()
        time_since_last_call = current_time - self.last_gemini_call
        if time_since_last_call < self.gemini_call_interval:
            sleep_time = self.gemini_call_interval - time_since_last_call
            print(f"⏳ Rate limiting: waiting {sleep_time:.1f}s before Gemini call")
            time_module.sleep(sleep_time)
        
        prompt = f"""
        Extract parameters from this flight scheduling query:
        
        Query: "{query}"
        Intent: {intent.value}
        
        Extract these parameters if present (return null if not found):
        - airport_code: 3-letter IATA code (BOM, DEL, CCU, MAA, HYD, BLR, etc.)
        - flight_number: Flight number without spaces (AI2739, 6E1234, etc.)
        - time_shift_minutes: Time shift as integer (+10, -15, etc.)
        - time_bucket_minutes: Time bucket size as integer (5, 10, 30)
        - runway: Runway identifier (09/27, 14/32, etc.)
        - aircraft_type: Aircraft type (A320, B737, etc.)
        
        Respond with valid JSON only:
        {{"airport_code": "BOM", "flight_number": "AI2739", "time_shift_minutes": 10}}
        """
        
        try:
            self.last_gemini_call = time_module.time()
            response = self.gemini_model.generate_content(prompt)
            result = json.loads(response.text.strip())
            
            # Cache the result
            self.gemini_cache[cache_key] = result
            print(f"🤖 Gemini extracted parameters: {result}")
            
            return result
            
        except (json.JSONDecodeError, Exception) as e:
            print(f"⚠️ Gemini parameter extraction failed: {e}")
            return {}
    
    def route_to_tool(self, intent: QueryIntent, parameters: QueryParameters) -> Dict[str, Any]:
        """Route query to appropriate service based on intent."""
        try:
            if intent == QueryIntent.ASK_PEAKS:
                return self._handle_peaks_query(parameters)
            elif intent == QueryIntent.ASK_RISK:
                return self._handle_risk_query(parameters)
            elif intent == QueryIntent.ASK_WHATIF:
                return self._handle_whatif_query(parameters)
            elif intent == QueryIntent.ASK_OPTIMIZE:
                return self._handle_optimize_query(parameters)
            elif intent == QueryIntent.ASK_CONSTRAINTS:
                return self._handle_constraints_query(parameters)
            elif intent == QueryIntent.ASK_STATUS:
                return self._handle_status_query(parameters)
            else:
                return {"error": "Intent not supported yet"}
        except Exception as e:
            return {"error": f"Tool routing failed: {str(e)}"}
    
    def _handle_peaks_query(self, parameters: QueryParameters) -> Dict[str, Any]:
        """Handle peak traffic analysis queries."""
        airport = parameters.airport_code or "BOM"  # Default to BOM
        bucket_minutes = parameters.time_bucket_minutes or 10
        
        # Get peak analysis from analytics engine
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
    
    def _handle_risk_query(self, parameters: QueryParameters) -> Dict[str, Any]:
        """Handle delay risk prediction queries."""
        if not parameters.flight_number:
            return {"error": "Flight number required for risk analysis"}
        
        # Query flight data from database
        from datetime import date, timedelta
        today = date.today()
        result = self.database_service.query_flights_by_date_range(
            start_date=today - timedelta(days=7),
            end_date=today
        )
        
        # Filter for specific flight number
        flight_data = [f for f in result.data if f.get('flight_number') == parameters.flight_number]
        if not flight_data:
            return {"error": f"Flight {parameters.flight_number} not found"}
        
        # Convert to Flight object (use most recent)
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
        
        # Predict delay risk
        prediction = self.delay_predictor.predict_departure_delay(flight)
        
        return {
            "type": "risk_prediction",
            "flight_number": parameters.flight_number,
            "prediction": prediction,
            "risk_level": prediction.risk_level.value,
            "probability": prediction.delay_probability,
            "expected_delay": prediction.expected_delay_minutes
        }
    
    def _handle_whatif_query(self, parameters: QueryParameters) -> Dict[str, Any]:
        """Handle what-if simulation queries."""
        if not parameters.flight_number or parameters.time_shift_minutes is None:
            return {"error": "Flight number and time shift required for what-if analysis"}
        
        # Create what-if scenario
        scenario = WhatIfScenario(
            flight_number=parameters.flight_number,
            time_shift_minutes=parameters.time_shift_minutes
        )
        
        # Run simulation
        impact = self.whatif_simulator.simulate_scenario(scenario)
        
        return {
            "type": "whatif_analysis",
            "scenario": scenario,
            "impact": impact,
            "delay_change": impact.delay_delta,
            "affected_flights": len(impact.affected_flights)
        }
    
    def _handle_optimize_query(self, parameters: QueryParameters) -> Dict[str, Any]:
        """Handle schedule optimization queries."""
        airport = parameters.airport_code or "BOM"
        
        # Get current schedule from database
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
        
        # Run optimization
        optimization_result = self.schedule_optimizer.optimize_schedule(flights)
        
        return {
            "type": "optimization",
            "airport": airport,
            "result": optimization_result,
            "cost_reduction": optimization_result.cost_reduction,
            "affected_flights": len(optimization_result.affected_flights)
        }
    
    def _handle_constraints_query(self, parameters: QueryParameters) -> Dict[str, Any]:
        """Handle operational constraints queries."""
        airport = parameters.airport_code or "BOM"
        
        # Create default constraints for the airport
        from .schedule_optimizer import Constraints
        constraints = Constraints(
            runway_capacity={"09/27": 30, "14/32": 25},  # Default capacities
            min_turnaround_minutes={"A320": 45, "B737": 45, "B777": 60},
            curfew_hours=[23, 0, 1, 2, 3, 4, 5],  # Curfew from 11 PM to 6 AM
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
    
    def _handle_status_query(self, parameters: QueryParameters) -> Dict[str, Any]:
        """Handle flight status queries."""
        if not parameters.flight_number:
            return {"error": "Flight number required for status query"}
        
        # Query recent flights for this flight number
        from datetime import date, timedelta
        today = date.today()
        result = self.database_service.query_flights_by_date_range(
            start_date=today - timedelta(days=7),
            end_date=today + timedelta(days=1)
        )
        
        # Filter for specific flight number
        flight_data = [f for f in result.data if f.get('flight_number') == parameters.flight_number]
        if not flight_data:
            return {"error": f"Flight {parameters.flight_number} not found"}
        
        # Sort by date (most recent first)
        flight_data.sort(key=lambda x: x.get('date_local', ''), reverse=True)
        
        return {
            "type": "flight_status",
            "flight_number": parameters.flight_number,
            "flights": flight_data[:5],  # Return up to 5 recent flights
            "latest_flight": flight_data[0] if flight_data else None
        }
    
    def format_response(self, structured_data: Dict[str, Any], 
                       intent: QueryIntent, parameters: QueryParameters) -> str:
        """Format structured data into natural language response."""
        if "error" in structured_data:
            return f"I'm sorry, {structured_data['error']}"
        
        # Use Gemini Pro for response formatting if available
        if self.gemini_model:
            try:
                return self._format_response_with_gemini(structured_data, intent, parameters)
            except Exception as e:
                print(f"Gemini response formatting failed: {e}")
        
        # Fallback to template-based responses
        return self._format_response_with_templates(structured_data, intent, parameters)
    
    def _format_response_with_gemini(self, structured_data: Dict[str, Any], 
                                   intent: QueryIntent, parameters: QueryParameters) -> str:
        """Use Gemini Pro to format natural language response."""
        prompt = f"""
        Format this flight scheduling data into a clear, professional response:
        
        Intent: {intent.value}
        Parameters: {parameters.__dict__}
        Data: {json.dumps(structured_data, default=str, indent=2)}
        
        Guidelines:
        - Use aviation terminology (A-CDM, CODA, slots, turnaround)
        - Be concise but informative
        - Include specific numbers and metrics
        - Suggest actionable next steps when appropriate
        - Keep response under 200 words
        """
        
        response = self.gemini_model.generate_content(prompt)
        return response.text.strip()
    
    def _format_response_with_templates(self, structured_data: Dict[str, Any], 
                                      intent: QueryIntent, parameters: QueryParameters) -> str:
        """Format response using predefined templates."""
        data_type = structured_data.get("type", "unknown")
        
        if data_type == "peak_analysis":
            airport = structured_data["airport"]
            overload_windows = structured_data["overload_windows"]
            max_util = structured_data["max_utilization"]
            return f"Peak analysis for {airport}: Found {overload_windows} overload windows with maximum utilization of {max_util:.1%}. The busiest slots show significant capacity constraints."
        
        elif data_type == "risk_prediction":
            flight = structured_data["flight_number"]
            risk = structured_data["risk_level"]
            prob = structured_data["probability"]
            return f"Delay risk for {flight}: {risk.upper()} risk level with {prob:.1%} probability of >15min delay. Expected delay: {structured_data['expected_delay']:.1f} minutes."
        
        elif data_type == "whatif_analysis":
            flight = parameters.flight_number
            shift = parameters.time_shift_minutes
            delay_change = structured_data["delay_change"]
            affected = structured_data["affected_flights"]
            return f"What-if analysis: Moving {flight} by {shift:+d} minutes would change delays by {delay_change:+.1f} minutes and affect {affected} other flights."
        
        else:
            return "Analysis completed. Please check the structured data for detailed results."
    
    def calculate_confidence(self, intent: QueryIntent, parameters: QueryParameters, 
                           structured_data: Dict[str, Any]) -> float:
        """Calculate confidence score for the response."""
        confidence = 0.4  # Base confidence
        
        # Increase confidence based on intent recognition
        if intent != QueryIntent.UNKNOWN:
            confidence += 0.3
        
        # Increase confidence based on parameter extraction
        param_count = sum(1 for v in parameters.__dict__.values() if v is not None and v != {})
        confidence += min(param_count * 0.1, 0.2)
        
        # Increase confidence if structured data is complete
        if structured_data and "error" not in structured_data:
            confidence += 0.1
        
        # Decrease confidence for errors
        if structured_data and "error" in structured_data:
            confidence -= 0.2
        
        return max(0.0, min(confidence, 1.0))
    
    def update_context(self, context: ConversationContext, query: str, 
                      intent: QueryIntent, parameters: QueryParameters):
        """Update conversation context with latest interaction."""
        context.last_query = query
        context.last_intent = intent
        context.last_parameters = parameters
        
        # Update preferred settings based on usage
        if parameters.airport_code:
            context.preferred_airport = parameters.airport_code
        if parameters.time_bucket_minutes:
            context.preferred_time_bucket = parameters.time_bucket_minutes
        
        # Add to conversation history
        context.conversation_history.append({
            "timestamp": datetime.now(),
            "query": query,
            "intent": intent.value,
            "parameters": parameters.__dict__
        })
        
        # Keep only last 10 interactions
        if len(context.conversation_history) > 10:
            context.conversation_history = context.conversation_history[-10:]
    
    def generate_suggestions(self, intent: QueryIntent, parameters: QueryParameters) -> List[str]:
        """Generate helpful suggestions for follow-up queries."""
        suggestions = []
        
        if intent == QueryIntent.ASK_PEAKS:
            suggestions.extend([
                "Try asking about delay risks for specific flights in busy slots",
                "Ask for optimization recommendations to reduce peak congestion"
            ])
        elif intent == QueryIntent.ASK_RISK:
            suggestions.extend([
                "Ask what-if questions about moving high-risk flights",
                "Check constraints that might affect this flight"
            ])
        elif intent == QueryIntent.ASK_WHATIF:
            suggestions.extend([
                "Try different time shifts to find optimal slots",
                "Ask about optimization to find the best overall schedule"
            ])
        
        return suggestions[:3]  # Return top 3 suggestions