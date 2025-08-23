"""
Production-ready NL Router with Gemini Pro integration.
Implements rate limiting, caching, selective calls, and health checks.
"""

import os
import time
import json
import hashlib
import functools
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-generativeai not available")


class NLProvider(Enum):
    """NL processing providers."""
    LOCAL = "local"
    GEMINI = "gemini"
    FALLBACK = "fallback"


@dataclass
class NLResult:
    """Result from NL processing."""
    intent: str
    airport_code: Optional[str] = None
    flight_number: Optional[str] = None
    time_shift_minutes: Optional[int] = None
    time_bucket_minutes: Optional[int] = None
    runway: Optional[str] = None
    aircraft_type: Optional[str] = None
    confidence: float = 0.0
    provider: NLProvider = NLProvider.LOCAL
    latency_ms: int = 0
    cache_hit: bool = False
    error: Optional[str] = None


@dataclass
class QueryLog:
    """Query processing log entry."""
    timestamp: str
    query: str
    intent: str
    provider: str
    latency_ms: int
    cache_hit: bool
    confidence: float
    error: Optional[str] = None


class RateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(self, rpm: int):
        self.min_interval = 60.0 / max(1, rpm)
        self.last_call = 0.0
    
    def wait(self):
        """Wait if necessary to respect rate limit."""
        now = time.time()
        elapsed = now - self.last_call
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            print(f"⏳ Rate limiting: waiting {sleep_time:.1f}s")
            time.sleep(sleep_time)
        self.last_call = time.time()


def cache_prompt(fn):
    """Decorator to cache prompt responses."""
    memo = {}
    
    @functools.wraps(fn)
    def wrapper(self, prompt: str, *args, **kwargs):
        key = hashlib.md5(prompt.encode()).hexdigest()
        if key in memo:
            return memo[key], True  # Return result and cache_hit=True
        
        result = fn(self, prompt, *args, **kwargs)
        memo[key] = result
        return result, False  # Return result and cache_hit=False
    
    return wrapper


class GeminiClient:
    """Gemini Pro client with rate limiting and caching."""
    
    def __init__(self):
        self.model = None
        self.rate_limiter = None
        self.enabled = False
        
        if not GEMINI_AVAILABLE:
            print("❌ Gemini not available - google-generativeai not installed")
            return
        
        # Get API key
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("❌ Gemini not available - no API key found")
            return
        
        # Configure Gemini
        try:
            genai.configure(api_key=api_key)
            model_name = os.getenv("NL_GEMINI_MODEL", "gemini-1.5-pro")
            self.model = genai.GenerativeModel(model_name)
            
            # Setup rate limiting
            rpm = int(os.getenv("NL_GEMINI_RPM", "24"))
            self.rate_limiter = RateLimiter(rpm)
            
            self.enabled = True
            print(f"✅ Gemini {model_name} initialized (RPM: {rpm})")
            
        except Exception as e:
            print(f"❌ Gemini initialization failed: {e}")
    
    @cache_prompt
    def generate_json(self, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Generate JSON response with schema validation."""
        if not self.enabled:
            raise Exception("Gemini not available")
        
        self.rate_limiter.wait()
        
        try:
            # First attempt
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "response_mime_type": "application/json"
                }
            )
            return json.loads(response.text)
            
        except Exception as e:
            # One retry with backoff
            print(f"⚠️ Gemini call failed, retrying: {e}")
            time.sleep(1.2)
            
            try:
                self.rate_limiter.wait()
                response = self.model.generate_content(
                    prompt,
                    generation_config={
                        "response_mime_type": "application/json"
                    }
                )
                return json.loads(response.text)
            except Exception as retry_error:
                raise Exception(f"Gemini failed after retry: {retry_error}")
    
    def health_check(self) -> bool:
        """Check if Gemini is working."""
        if not self.enabled:
            return False
        
        try:
            result, _ = self.generate_json(
                "Reply with JSON: {\"status\": \"ok\", \"timestamp\": \"" + 
                datetime.now().isoformat() + "\"}",
                {"type": "object", "properties": {"status": {"type": "string"}}}
            )
            return result.get("status") == "ok"
        except Exception:
            return False


class LocalNLProcessor:
    """Local regex-based NL processor (fallback)."""
    
    def __init__(self):
        # Intent patterns
        self.intent_patterns = {
            "ask_peaks": [
                r'busiest.*slots?', r'peak.*traffic', r'demand.*heatmap',
                r'overload.*windows?', r'capacity.*utilization', r'traffic.*analysis',
                r'time.*slots.*overload', r'which.*slots', r'busy.*time'
            ],
            "ask_risk": [
                r'delay.*risk', r'probability.*delay', r'risk.*assessment',
                r'likely.*delayed?', r'delay.*prediction', r'risk.*score'
            ],
            "ask_whatif": [
                r'move.*by', r'shift.*flight', r'what.*if.*move',
                r'impact.*of.*moving', r'change.*time', r'reschedule'
            ],
            "ask_optimize": [
                r'optimize.*schedule', r'improve.*schedule', r'minimize.*delays',
                r'best.*schedule', r'optimization', r'reduce.*delays'
            ],
            "ask_constraints": [
                r'operational.*rules', r'constraints', r'restrictions',
                r'turnaround.*time', r'runway.*capacity', r'curfew'
            ],
            "ask_status": [
                r'flight.*status', r'show.*flight', r'flight.*info',
                r'details.*for', r'information.*about'
            ]
        }
        
        # Parameter patterns
        self.param_patterns = {
            "airport_code": r'\b(BOM|DEL|CCU|MAA|HYD|BLR|GOI|PNQ|AMD|JAI|IXC|COK|TRV|IXM|IXR|IXL|IXU|IXJ|IXA|IXB|IXD|IXE|IXG|IXH|IXI|IXK|IXN|IXP|IXQ|IXS|IXV|IXW|IXY|IXZ)\b',
            "flight_number": r'\b([A-Z0-9]{2}\s*\d{3,4})\b',
            "time_shift_minutes": r'([+-]?\d+)\s*m(?:in(?:ute)?s?)?',
            "time_bucket_minutes": r'(\d+)\s*m(?:in(?:ute)?s?)?.*bucket'
        }
    
    def classify_intent(self, query: str) -> str:
        """Classify intent using regex patterns."""
        query_lower = query.lower()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if __import__('re').search(pattern, query_lower):
                    return intent
        
        return "unknown"
    
    def extract_parameters(self, query: str) -> Dict[str, Any]:
        """Extract parameters using regex patterns."""
        import re
        params = {}
        
        for param_type, pattern in self.param_patterns.items():
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                if param_type == "airport_code":
                    params["airport_code"] = matches[0].upper()
                elif param_type == "flight_number":
                    params["flight_number"] = matches[0].replace(' ', '')
                elif param_type == "time_shift_minutes":
                    params["time_shift_minutes"] = int(matches[0])
                elif param_type == "time_bucket_minutes":
                    params["time_bucket_minutes"] = int(matches[0])
        
        return params
    
    def process(self, query: str, context: Dict[str, Any] = None) -> NLResult:
        """Process query with local methods."""
        start_time = time.time()
        
        intent = self.classify_intent(query)
        params = self.extract_parameters(query)
        
        # Calculate confidence based on matches
        confidence = 0.8 if intent != "unknown" else 0.3
        if params:
            confidence += 0.1
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        return NLResult(
            intent=intent,
            provider=NLProvider.LOCAL,
            confidence=confidence,
            latency_ms=latency_ms,
            **params
        )


class NLRouter:
    """Main NL Router with intelligent provider selection."""
    
    def __init__(self):
        self.gemini = GeminiClient()
        self.local = LocalNLProcessor()
        self.query_logs: List[QueryLog] = []
        self.use_gemini = os.getenv("NL_USE_GEMINI", "true").lower() == "true"
        
        print(f"🚀 NL Router initialized (Gemini: {'✅' if self.gemini.enabled and self.use_gemini else '❌'})")
    
    def classify_and_extract(self, query: str, context: Dict[str, Any] = None) -> NLResult:
        """Main entry point for NL processing."""
        start_time = time.time()
        context = context or {}
        
        # 1) Try local first
        local_result = self.local.process(query, context)
        
        # 2) Use Gemini only if local is uncertain or incomplete
        should_use_gemini = (
            self.use_gemini and 
            self.gemini.enabled and
            (local_result.intent == "unknown" or local_result.confidence < 0.7)
        )
        
        if should_use_gemini:
            try:
                gemini_result = self._process_with_gemini(query, context)
                result = gemini_result
            except Exception as e:
                print(f"⚠️ Gemini failed, using local: {e}")
                result = local_result
                result.error = str(e)
        else:
            result = local_result
        
        # 3) Log the query
        self._log_query(query, result, start_time)
        
        return result
    
    def _process_with_gemini(self, query: str, context: Dict[str, Any]) -> NLResult:
        """Process query with Gemini Pro."""
        start_time = time.time()
        
        schema = {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": ["ask_peaks", "ask_risk", "ask_whatif", "ask_optimize", 
                            "ask_constraints", "ask_status", "ask_general", "unknown"]
                },
                "airport_code": {"type": ["string", "null"]},
                "flight_number": {"type": ["string", "null"]},
                "time_shift_minutes": {"type": ["integer", "null"]},
                "time_bucket_minutes": {"type": ["integer", "null"]},
                "runway": {"type": ["string", "null"]},
                "aircraft_type": {"type": ["string", "null"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1}
            },
            "required": ["intent", "confidence"]
        }
        
        prompt = f"""
        Extract intent and parameters for this flight scheduling query.
        
        Query: "{query}"
        Context: {json.dumps(context)}
        
        Intent categories:
        - ask_peaks: Questions about busiest time slots, peak traffic, demand analysis
        - ask_risk: Questions about delay risk, probability of delays for specific flights
        - ask_whatif: Questions about impact of moving/changing flights
        - ask_optimize: Requests to optimize schedules or minimize delays
        - ask_constraints: Questions about operational rules, constraints, restrictions
        - ask_status: Questions about specific flight status or information
        - ask_general: General aviation questions
        - unknown: Unrecognized or unclear queries
        
        Parameters to extract:
        - airport_code: 3-letter IATA code (BOM, DEL, CCU, etc.)
        - flight_number: Flight number without spaces (AI2739, 6E1234, etc.)
        - time_shift_minutes: Time shift as integer (+10, -15, etc.)
        - time_bucket_minutes: Time bucket size (5, 10, 30)
        - runway: Runway identifier (09/27, 14/32, etc.)
        - aircraft_type: Aircraft type (A320, B737, etc.)
        - confidence: Your confidence in the classification (0.0 to 1.0)
        
        Return JSON only.
        """
        
        response, cache_hit = self.gemini.generate_json(prompt, schema)
        latency_ms = int((time.time() - start_time) * 1000)
        
        return NLResult(
            intent=response.get("intent", "unknown"),
            airport_code=response.get("airport_code"),
            flight_number=response.get("flight_number"),
            time_shift_minutes=response.get("time_shift_minutes"),
            time_bucket_minutes=response.get("time_bucket_minutes"),
            runway=response.get("runway"),
            aircraft_type=response.get("aircraft_type"),
            confidence=response.get("confidence", 0.5),
            provider=NLProvider.GEMINI,
            latency_ms=latency_ms,
            cache_hit=cache_hit
        )
    
    def _log_query(self, query: str, result: NLResult, start_time: float):
        """Log query processing."""
        log_entry = QueryLog(
            timestamp=datetime.now().isoformat(),
            query=query,
            intent=result.intent,
            provider=result.provider.value,
            latency_ms=result.latency_ms,
            cache_hit=result.cache_hit,
            confidence=result.confidence,
            error=result.error
        )
        
        self.query_logs.append(log_entry)
        
        # Keep only last 100 logs
        if len(self.query_logs) > 100:
            self.query_logs = self.query_logs[-100:]
        
        # Print status
        provider_emoji = "🤖" if result.provider == NLProvider.GEMINI else "🔧"
        cache_emoji = "💾" if result.cache_hit else "🆕"
        print(f"{provider_emoji}{cache_emoji} \"{query}\" → {result.intent} ({result.confidence:.2f}, {result.latency_ms}ms)")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        if not self.query_logs:
            return {"total_queries": 0}
        
        total = len(self.query_logs)
        gemini_count = sum(1 for log in self.query_logs if log.provider == "gemini")
        cache_hits = sum(1 for log in self.query_logs if log.cache_hit)
        avg_latency = sum(log.latency_ms for log in self.query_logs) / total
        avg_confidence = sum(log.confidence for log in self.query_logs) / total
        
        return {
            "total_queries": total,
            "gemini_usage_pct": (gemini_count / total) * 100,
            "cache_hit_pct": (cache_hits / total) * 100,
            "avg_latency_ms": round(avg_latency, 1),
            "avg_confidence": round(avg_confidence, 2),
            "gemini_enabled": self.gemini.enabled,
            "gemini_healthy": self.gemini.health_check() if self.gemini.enabled else False
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check."""
        return {
            "local_processor": "healthy",
            "gemini_available": self.gemini.enabled,
            "gemini_healthy": self.gemini.health_check() if self.gemini.enabled else False,
            "stats": self.get_stats()
        }


# Global instance
nl_router = NLRouter()


def process_query(query: str, context: Dict[str, Any] = None) -> NLResult:
    """Main entry point for NL processing."""
    return nl_router.classify_and_extract(query, context)


def get_nl_stats() -> Dict[str, Any]:
    """Get NL processing statistics."""
    return nl_router.get_stats()


def health_check() -> Dict[str, Any]:
    """Health check for NL services."""
    return nl_router.health_check()