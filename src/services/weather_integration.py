"""Weather integration service for capacity management and optimization."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import json
import math
import random
from pathlib import Path

logger = logging.getLogger(__name__)


class WeatherRegime(Enum):
    """Weather regime classification for capacity management."""
    CALM = "calm"           # Visibility > 8km, Wind < 10kts
    MEDIUM = "medium"       # Visibility 3-8km, Wind 10-20kts
    STRONG = "strong"       # Visibility 1-3km, Wind 20-35kts
    SEVERE = "severe"       # Visibility < 1km, Wind > 35kts


class WeatherCondition(Enum):
    """Weather condition types."""
    CLEAR = "clear"
    CLOUDY = "cloudy"
    RAIN = "rain"
    THUNDERSTORM = "thunderstorm"
    FOG = "fog"
    SNOW = "snow"
    WIND = "wind"


@dataclass
class WeatherData:
    """Current weather observation data."""
    airport_code: str
    timestamp: datetime
    visibility_km: float
    wind_speed_kts: float
    wind_direction: int
    temperature_c: float
    humidity_percent: float
    pressure_hpa: float
    precipitation: bool
    condition: WeatherCondition
    weather_regime: Optional[WeatherRegime] = None
    source: str = "metar"
    
    def __post_init__(self):
        """Calculate weather regime based on conditions."""
        if self.weather_regime is None:
            self.weather_regime = self._classify_regime()
    
    def _classify_regime(self) -> WeatherRegime:
        """Classify weather regime based on visibility and wind."""
        if self.visibility_km >= 8 and self.wind_speed_kts < 10:
            return WeatherRegime.CALM
        elif self.visibility_km >= 3 and self.wind_speed_kts < 20:
            return WeatherRegime.MEDIUM
        elif self.visibility_km >= 1 and self.wind_speed_kts < 35:
            return WeatherRegime.STRONG
        else:
            return WeatherRegime.SEVERE


@dataclass
class WeatherForecast:
    """Weather forecast data."""
    airport_code: str
    forecast_time: datetime
    valid_from: datetime
    valid_to: datetime
    visibility_km: float
    wind_speed_kts: float
    wind_direction: int
    precipitation_probability: float
    condition: WeatherCondition
    confidence: float = 0.8
    
    def get_regime(self) -> WeatherRegime:
        """Get weather regime for this forecast."""
        if self.visibility_km >= 8 and self.wind_speed_kts < 10:
            return WeatherRegime.CALM
        elif self.visibility_km >= 3 and self.wind_speed_kts < 20:
            return WeatherRegime.MEDIUM
        elif self.visibility_km >= 1 and self.wind_speed_kts < 35:
            return WeatherRegime.STRONG
        else:
            return WeatherRegime.SEVERE


@dataclass
class CapacityAdjustment:
    """Runway capacity adjustment based on weather."""
    runway: str
    original_capacity: int
    adjusted_capacity: int
    reduction_factor: float
    weather_regime: WeatherRegime
    valid_from: datetime
    valid_to: datetime
    reason: str
    
    @property
    def capacity_reduction_percent(self) -> float:
        """Calculate capacity reduction percentage."""
        if self.original_capacity == 0:
            return 0.0
        return (1 - self.reduction_factor) * 100


@dataclass
class WeatherScenario:
    """Weather scenario for contingency planning."""
    scenario_id: str
    name: str
    description: str
    airport_code: str
    start_time: datetime
    duration_hours: int
    weather_conditions: List[WeatherForecast]
    capacity_impacts: Dict[str, CapacityAdjustment]
    probability: float = 0.5
    
    def get_total_capacity_impact(self) -> float:
        """Calculate total capacity impact across all runways."""
        if not self.capacity_impacts:
            return 0.0
        
        total_reduction = sum(
            adj.capacity_reduction_percent for adj in self.capacity_impacts.values()
        )
        return total_reduction / len(self.capacity_impacts)


@dataclass
class ImpactForecast:
    """Weather impact forecast for operations."""
    airport_code: str
    forecast_horizon_hours: int
    scenarios: List[WeatherScenario]
    recommended_actions: List[str]
    confidence: float
    generated_at: datetime = field(default_factory=datetime.now)
    
    def get_most_likely_scenario(self) -> Optional[WeatherScenario]:
        """Get the most likely weather scenario."""
        if not self.scenarios:
            return None
        return max(self.scenarios, key=lambda s: s.probability)


class WeatherIntegrationService:
    """Service for integrating weather data into capacity management."""
    
    def __init__(self):
        """Initialize weather integration service."""
        self.current_weather: Dict[str, WeatherData] = {}
        self.forecasts: Dict[str, List[WeatherForecast]] = {}
        self.capacity_adjustments: Dict[str, Dict[str, CapacityAdjustment]] = {}
        
        # Default runway capacities (movements per hour)
        self.default_capacities = {
            "BOM": {"09L/27R": 45, "09R/27L": 45, "14/32": 30},
            "DEL": {"10/28": 50, "11/29": 48, "09/27": 40},
            "MAA": {"07/25": 42, "12/30": 38},
            "BLR": {"09/27": 44, "18/36": 40},
            "HYD": {"09L/27R": 46, "09R/27L": 46}
        }
        
        # Weather regime capacity reduction factors
        self.regime_factors = {
            WeatherRegime.CALM: 1.0,      # No reduction
            WeatherRegime.MEDIUM: 0.85,   # 15% reduction
            WeatherRegime.STRONG: 0.65,   # 35% reduction
            WeatherRegime.SEVERE: 0.40    # 60% reduction
        }
    
    def process_weather_update(self, weather_data: WeatherData) -> None:
        """Process real-time weather data update."""
        logger.info(f"Processing weather update for {weather_data.airport_code}: "
                   f"{weather_data.weather_regime.value}")
        
        self.current_weather[weather_data.airport_code] = weather_data
        
        # Update capacity adjustments for all runways at this airport
        self._update_runway_capacities(weather_data)
    
    def _update_runway_capacities(self, weather_data: WeatherData) -> None:
        """Update runway capacities based on weather conditions."""
        airport_code = weather_data.airport_code
        
        if airport_code not in self.default_capacities:
            logger.warning(f"No default capacities defined for {airport_code}")
            return
        
        if airport_code not in self.capacity_adjustments:
            self.capacity_adjustments[airport_code] = {}
        
        reduction_factor = self.regime_factors.get(weather_data.weather_regime, 0.8)
        
        for runway, default_capacity in self.default_capacities[airport_code].items():
            adjusted_capacity = int(default_capacity * reduction_factor)
            
            adjustment = CapacityAdjustment(
                runway=runway,
                original_capacity=default_capacity,
                adjusted_capacity=adjusted_capacity,
                reduction_factor=reduction_factor,
                weather_regime=weather_data.weather_regime,
                valid_from=weather_data.timestamp,
                valid_to=weather_data.timestamp + timedelta(hours=1),
                reason=f"Weather regime: {weather_data.weather_regime.value}, "
                       f"Visibility: {weather_data.visibility_km}km, "
                       f"Wind: {weather_data.wind_speed_kts}kts"
            )
            
            self.capacity_adjustments[airport_code][runway] = adjustment
            
            logger.info(f"Updated capacity for {airport_code} {runway}: "
                       f"{default_capacity} -> {adjusted_capacity} "
                       f"({adjustment.capacity_reduction_percent:.1f}% reduction)")
    
    def adjust_runway_capacity(self, runway: str, weather: WeatherData) -> CapacityAdjustment:
        """Adjust runway capacity based on weather conditions."""
        airport_code = weather.airport_code
        
        # Get default capacity
        default_capacity = 40  # Default fallback
        if airport_code in self.default_capacities:
            runway_capacities = self.default_capacities[airport_code]
            default_capacity = runway_capacities.get(runway, default_capacity)
        
        # Apply weather reduction factor
        reduction_factor = self.regime_factors.get(weather.weather_regime, 0.8)
        adjusted_capacity = int(default_capacity * reduction_factor)
        
        return CapacityAdjustment(
            runway=runway,
            original_capacity=default_capacity,
            adjusted_capacity=adjusted_capacity,
            reduction_factor=reduction_factor,
            weather_regime=weather.weather_regime,
            valid_from=weather.timestamp,
            valid_to=weather.timestamp + timedelta(hours=1),
            reason=f"Weather adjustment for {weather.weather_regime.value} conditions"
        )
    
    def predict_weather_impact(self, forecast: WeatherForecast, horizon_hours: int) -> ImpactForecast:
        """Predict weather impact on operations."""
        airport_code = forecast.airport_code
        scenarios = []
        
        # Generate scenarios based on forecast confidence
        base_scenario = self._create_weather_scenario(
            f"base_{airport_code}_{forecast.forecast_time.strftime('%H%M')}",
            "Base forecast scenario",
            forecast,
            horizon_hours
        )
        scenarios.append(base_scenario)
        
        # Generate optimistic scenario (better conditions)
        if forecast.confidence < 0.9:
            optimistic_forecast = self._adjust_forecast(forecast, improvement=True)
            optimistic_scenario = self._create_weather_scenario(
                f"optimistic_{airport_code}_{forecast.forecast_time.strftime('%H%M')}",
                "Optimistic weather scenario",
                optimistic_forecast,
                horizon_hours
            )
            optimistic_scenario.probability = 0.3
            scenarios.append(optimistic_scenario)
        
        # Generate pessimistic scenario (worse conditions)
        if forecast.confidence < 0.8:
            pessimistic_forecast = self._adjust_forecast(forecast, improvement=False)
            pessimistic_scenario = self._create_weather_scenario(
                f"pessimistic_{airport_code}_{forecast.forecast_time.strftime('%H%M')}",
                "Pessimistic weather scenario",
                pessimistic_forecast,
                horizon_hours
            )
            pessimistic_scenario.probability = 0.2
            scenarios.append(pessimistic_scenario)
        
        # Normalize probabilities
        total_prob = sum(s.probability for s in scenarios)
        for scenario in scenarios:
            scenario.probability /= total_prob
        
        # Generate recommendations
        recommendations = self._generate_recommendations(scenarios)
        
        return ImpactForecast(
            airport_code=airport_code,
            forecast_horizon_hours=horizon_hours,
            scenarios=scenarios,
            recommended_actions=recommendations,
            confidence=forecast.confidence
        )
    
    def _create_weather_scenario(self, scenario_id: str, name: str, 
                                forecast: WeatherForecast, duration_hours: int) -> WeatherScenario:
        """Create a weather scenario from forecast."""
        airport_code = forecast.airport_code
        
        # Create capacity impacts for all runways
        capacity_impacts = {}
        if airport_code in self.default_capacities:
            for runway in self.default_capacities[airport_code]:
                # Create weather data from forecast
                weather_data = WeatherData(
                    airport_code=airport_code,
                    timestamp=forecast.valid_from,
                    visibility_km=forecast.visibility_km,
                    wind_speed_kts=forecast.wind_speed_kts,
                    wind_direction=forecast.wind_direction,
                    temperature_c=20.0,  # Default
                    humidity_percent=60.0,  # Default
                    pressure_hpa=1013.25,  # Default
                    precipitation=forecast.precipitation_probability > 0.5,
                    condition=forecast.condition
                )
                
                adjustment = self.adjust_runway_capacity(runway, weather_data)
                capacity_impacts[runway] = adjustment
        
        return WeatherScenario(
            scenario_id=scenario_id,
            name=name,
            description=f"Weather scenario with {forecast.get_regime().value} conditions",
            airport_code=airport_code,
            start_time=forecast.valid_from,
            duration_hours=duration_hours,
            weather_conditions=[forecast],
            capacity_impacts=capacity_impacts,
            probability=0.5
        )
    
    def _adjust_forecast(self, base_forecast: WeatherForecast, improvement: bool) -> WeatherForecast:
        """Adjust forecast for scenario generation."""
        visibility_factor = 1.2 if improvement else 0.8
        wind_factor = 0.8 if improvement else 1.2
        
        adjusted_forecast = WeatherForecast(
            airport_code=base_forecast.airport_code,
            forecast_time=base_forecast.forecast_time,
            valid_from=base_forecast.valid_from,
            valid_to=base_forecast.valid_to,
            visibility_km=max(0.5, base_forecast.visibility_km * visibility_factor),
            wind_speed_kts=max(0, base_forecast.wind_speed_kts * wind_factor),
            wind_direction=base_forecast.wind_direction,
            precipitation_probability=max(0, min(1, base_forecast.precipitation_probability * (0.7 if improvement else 1.3))),
            condition=base_forecast.condition,
            confidence=base_forecast.confidence * 0.8
        )
        
        return adjusted_forecast
    
    def _generate_recommendations(self, scenarios: List[WeatherScenario]) -> List[str]:
        """Generate operational recommendations based on weather scenarios."""
        recommendations = []
        
        # Find worst-case scenario
        worst_scenario = min(scenarios, key=lambda s: min(
            adj.reduction_factor for adj in s.capacity_impacts.values()
        ) if s.capacity_impacts else 1.0)
        
        if worst_scenario.capacity_impacts:
            max_reduction = max(
                adj.capacity_reduction_percent for adj in worst_scenario.capacity_impacts.values()
            )
            
            if max_reduction > 50:
                recommendations.append("Consider implementing ground delay program")
                recommendations.append("Activate severe weather contingency plan")
            elif max_reduction > 30:
                recommendations.append("Increase spacing between arrivals")
                recommendations.append("Consider runway configuration changes")
            elif max_reduction > 15:
                recommendations.append("Monitor weather conditions closely")
                recommendations.append("Prepare for potential delays")
        
        # Check for high probability scenarios
        high_prob_scenarios = [s for s in scenarios if s.probability > 0.6]
        if high_prob_scenarios:
            scenario = high_prob_scenarios[0]
            if scenario.get_total_capacity_impact() > 20:
                recommendations.append(f"High probability of {scenario.get_total_capacity_impact():.1f}% capacity reduction")
        
        return recommendations
    
    def generate_weather_scenarios(self, base_schedule: Any) -> List[WeatherScenario]:
        """Generate weather scenarios for contingency planning."""
        scenarios = []
        
        # Get current time for scenario generation
        now = datetime.now()
        
        # Generate scenarios for major airports
        airports = ["BOM", "DEL", "MAA", "BLR", "HYD"]
        
        for airport in airports:
            # Calm weather scenario (baseline)
            calm_scenario = self._generate_scenario(
                airport, "calm", now, 6, WeatherRegime.CALM, 0.4
            )
            scenarios.append(calm_scenario)
            
            # Medium weather scenario
            medium_scenario = self._generate_scenario(
                airport, "medium", now + timedelta(hours=2), 4, WeatherRegime.MEDIUM, 0.3
            )
            scenarios.append(medium_scenario)
            
            # Strong weather scenario
            strong_scenario = self._generate_scenario(
                airport, "strong", now + timedelta(hours=4), 3, WeatherRegime.STRONG, 0.2
            )
            scenarios.append(strong_scenario)
            
            # Severe weather scenario (low probability)
            severe_scenario = self._generate_scenario(
                airport, "severe", now + timedelta(hours=6), 2, WeatherRegime.SEVERE, 0.1
            )
            scenarios.append(severe_scenario)
        
        return scenarios
    
    def _generate_scenario(self, airport_code: str, scenario_type: str, start_time: datetime,
                          duration_hours: int, regime: WeatherRegime, probability: float) -> WeatherScenario:
        """Generate a specific weather scenario."""
        scenario_id = f"{airport_code}_{scenario_type}_{start_time.strftime('%Y%m%d_%H%M')}"
        
        # Create weather conditions based on regime
        visibility_km = {
            WeatherRegime.CALM: 10.0,
            WeatherRegime.MEDIUM: 5.0,
            WeatherRegime.STRONG: 2.0,
            WeatherRegime.SEVERE: 0.8
        }[regime]
        
        wind_speed_kts = {
            WeatherRegime.CALM: 5.0,
            WeatherRegime.MEDIUM: 15.0,
            WeatherRegime.STRONG: 25.0,
            WeatherRegime.SEVERE: 40.0
        }[regime]
        
        condition = {
            WeatherRegime.CALM: WeatherCondition.CLEAR,
            WeatherRegime.MEDIUM: WeatherCondition.CLOUDY,
            WeatherRegime.STRONG: WeatherCondition.RAIN,
            WeatherRegime.SEVERE: WeatherCondition.THUNDERSTORM
        }[regime]
        
        # Create forecast
        forecast = WeatherForecast(
            airport_code=airport_code,
            forecast_time=start_time - timedelta(hours=2),
            valid_from=start_time,
            valid_to=start_time + timedelta(hours=duration_hours),
            visibility_km=visibility_km,
            wind_speed_kts=wind_speed_kts,
            wind_direction=270,  # Default westerly
            precipitation_probability=0.8 if regime in [WeatherRegime.STRONG, WeatherRegime.SEVERE] else 0.2,
            condition=condition
        )
        
        # Create capacity impacts
        capacity_impacts = {}
        if airport_code in self.default_capacities:
            for runway in self.default_capacities[airport_code]:
                weather_data = WeatherData(
                    airport_code=airport_code,
                    timestamp=start_time,
                    visibility_km=visibility_km,
                    wind_speed_kts=wind_speed_kts,
                    wind_direction=270,
                    temperature_c=25.0,
                    humidity_percent=70.0,
                    pressure_hpa=1013.25,
                    precipitation=forecast.precipitation_probability > 0.5,
                    condition=condition,
                    weather_regime=regime
                )
                
                adjustment = self.adjust_runway_capacity(runway, weather_data)
                capacity_impacts[runway] = adjustment
        
        return WeatherScenario(
            scenario_id=scenario_id,
            name=f"{scenario_type.title()} weather at {airport_code}",
            description=f"{regime.value.title()} weather conditions with {visibility_km}km visibility and {wind_speed_kts}kt winds",
            airport_code=airport_code,
            start_time=start_time,
            duration_hours=duration_hours,
            weather_conditions=[forecast],
            capacity_impacts=capacity_impacts,
            probability=probability
        )
    
    def optimize_for_weather(self, scenario: WeatherScenario, flights: Optional[List] = None) -> Dict[str, Any]:
        """Optimize schedule for specific weather scenario."""
        optimization_result = {
            "scenario_id": scenario.scenario_id,
            "airport_code": scenario.airport_code,
            "weather_regime": scenario.weather_conditions[0].get_regime().value if scenario.weather_conditions else "unknown",
            "capacity_adjustments": {},
            "recommended_actions": [],
            "estimated_delay_impact": 0.0,
            "confidence": 0.8,
            "schedule_changes": []
        }
        
        total_capacity_reduction = 0.0
        runway_count = 0
        
        # Process capacity adjustments
        for runway, adjustment in scenario.capacity_impacts.items():
            optimization_result["capacity_adjustments"][runway] = {
                "original_capacity": adjustment.original_capacity,
                "adjusted_capacity": adjustment.adjusted_capacity,
                "reduction_percent": adjustment.capacity_reduction_percent,
                "reason": adjustment.reason
            }
            
            total_capacity_reduction += adjustment.capacity_reduction_percent
            runway_count += 1
        
        # Calculate average capacity reduction
        if runway_count > 0:
            avg_reduction = total_capacity_reduction / runway_count
            optimization_result["estimated_delay_impact"] = avg_reduction * 2.5  # Rough estimate: 2.5 min delay per % capacity reduction
        
        # If flights are provided, generate specific schedule recommendations
        if flights:
            schedule_changes = self._generate_schedule_changes(scenario, flights)
            optimization_result["schedule_changes"] = schedule_changes
        
        # Generate recommendations based on impact
        if total_capacity_reduction > 150:  # Severe impact across multiple runways
            optimization_result["recommended_actions"].extend([
                "Implement ground delay program",
                "Consider airport closure if conditions worsen",
                "Coordinate with ATC for traffic management",
                "Activate emergency response procedures"
            ])
        elif total_capacity_reduction > 100:  # Significant impact
            optimization_result["recommended_actions"].extend([
                "Increase arrival spacing",
                "Consider runway configuration changes",
                "Implement flow control measures",
                "Prepare for extended delays"
            ])
        elif total_capacity_reduction > 50:  # Moderate impact
            optimization_result["recommended_actions"].extend([
                "Monitor conditions closely",
                "Adjust approach procedures",
                "Inform airlines of potential delays"
            ])
        else:  # Minor impact
            optimization_result["recommended_actions"].append("Continue normal operations with weather monitoring")
        
        return optimization_result
    
    def _generate_schedule_changes(self, scenario: WeatherScenario, flights: List) -> List[Dict[str, Any]]:
        """Generate specific schedule change recommendations based on weather scenario."""
        changes = []
        
        # Get weather regime
        weather_regime = scenario.weather_conditions[0].get_regime() if scenario.weather_conditions else WeatherRegime.MEDIUM
        
        # Calculate delay factors based on weather regime
        delay_factors = {
            WeatherRegime.CALM: 1.0,
            WeatherRegime.MEDIUM: 1.2,
            WeatherRegime.STRONG: 1.5,
            WeatherRegime.SEVERE: 2.0
        }
        
        delay_factor = delay_factors.get(weather_regime, 1.3)
        
        # Simulate schedule adjustments for high-risk flights
        for i, flight in enumerate(flights[:10]):  # Process first 10 flights as example
            if hasattr(flight, 'flight_number') and hasattr(flight, 'departure'):
                # Simulate delay based on weather impact
                base_delay = 5 * delay_factor  # Base delay in minutes
                weather_delay = int(base_delay * (1 + i * 0.1))  # Increasing delay for later flights
                
                change = {
                    "flight_number": flight.flight_number,
                    "original_time": flight.departure.scheduled.strftime('%H:%M') if flight.departure.scheduled else "Unknown",
                    "recommended_delay": weather_delay,
                    "new_time": (datetime.combine(datetime.now().date(), flight.departure.scheduled) + 
                               timedelta(minutes=weather_delay)).strftime('%H:%M') if flight.departure.scheduled else "Unknown",
                    "reason": f"Weather delay due to {weather_regime.value} conditions",
                    "priority": "high" if weather_delay > 15 else "medium"
                }
                changes.append(change)
        
        return changes
    
    def get_current_weather(self, airport_code: str) -> Optional[WeatherData]:
        """Get current weather data for airport."""
        return self.current_weather.get(airport_code)
    
    def get_capacity_adjustment(self, airport_code: str, runway: str) -> Optional[CapacityAdjustment]:
        """Get current capacity adjustment for runway."""
        if airport_code in self.capacity_adjustments:
            return self.capacity_adjustments[airport_code].get(runway)
        return None
    
    def get_all_capacity_adjustments(self, airport_code: str) -> Dict[str, CapacityAdjustment]:
        """Get all capacity adjustments for airport."""
        return self.capacity_adjustments.get(airport_code, {})
    
    def implement_weather_forecast_integration(self, forecasts: List[WeatherForecast], 
                                             flights: Optional[List] = None) -> Dict[str, Any]:
        """Implement proactive schedule adjustments based on weather forecasts."""
        integration_result = {
            "forecast_count": len(forecasts),
            "airports_affected": list(set(f.airport_code for f in forecasts)),
            "time_horizon_hours": max((f.valid_to - f.valid_from).total_seconds() / 3600 for f in forecasts) if forecasts else 0,
            "proactive_adjustments": [],
            "capacity_planning": {},
            "alert_triggers": [],
            "confidence": 0.0
        }
        
        if not forecasts:
            return integration_result
        
        # Calculate average confidence
        integration_result["confidence"] = sum(f.confidence for f in forecasts) / len(forecasts)
        
        # Process each forecast
        for forecast in forecasts:
            airport_code = forecast.airport_code
            weather_regime = forecast.get_regime()
            
            # Create capacity planning for this airport
            if airport_code not in integration_result["capacity_planning"]:
                integration_result["capacity_planning"][airport_code] = {
                    "current_regime": weather_regime.value,
                    "capacity_reductions": {},
                    "recommended_actions": []
                }
            
            # Calculate capacity impacts
            if airport_code in self.default_capacities:
                for runway in self.default_capacities[airport_code]:
                    # Create weather data from forecast for capacity calculation
                    weather_data = WeatherData(
                        airport_code=airport_code,
                        timestamp=forecast.valid_from,
                        visibility_km=forecast.visibility_km,
                        wind_speed_kts=forecast.wind_speed_kts,
                        wind_direction=forecast.wind_direction,
                        temperature_c=20.0,
                        humidity_percent=70.0,
                        pressure_hpa=1013.25,
                        precipitation=forecast.precipitation_probability > 0.5,
                        condition=forecast.condition,
                        weather_regime=weather_regime
                    )
                    
                    adjustment = self.adjust_runway_capacity(runway, weather_data)
                    integration_result["capacity_planning"][airport_code]["capacity_reductions"][runway] = {
                        "reduction_percent": adjustment.capacity_reduction_percent,
                        "valid_from": forecast.valid_from.strftime('%H:%M'),
                        "valid_to": forecast.valid_to.strftime('%H:%M')
                    }
            
            # Generate proactive adjustments
            if weather_regime in [WeatherRegime.STRONG, WeatherRegime.SEVERE]:
                hours_ahead = (forecast.valid_from - datetime.now()).total_seconds() / 3600
                
                if hours_ahead > 0:  # Future forecast
                    adjustment = {
                        "airport": airport_code,
                        "action": "proactive_delay",
                        "weather_regime": weather_regime.value,
                        "hours_ahead": round(hours_ahead, 1),
                        "recommended_delay_minutes": 15 if weather_regime == WeatherRegime.STRONG else 30,
                        "reason": f"Proactive adjustment for forecasted {weather_regime.value} weather",
                        "confidence": forecast.confidence
                    }
                    integration_result["proactive_adjustments"].append(adjustment)
            
            # Generate alert triggers
            if weather_regime == WeatherRegime.SEVERE:
                alert = {
                    "airport": airport_code,
                    "severity": "critical",
                    "message": f"Severe weather forecast for {airport_code} at {forecast.valid_from.strftime('%H:%M')}",
                    "action_required": "Implement ground delay program",
                    "lead_time_hours": (forecast.valid_from - datetime.now()).total_seconds() / 3600
                }
                integration_result["alert_triggers"].append(alert)
            elif weather_regime == WeatherRegime.STRONG:
                alert = {
                    "airport": airport_code,
                    "severity": "warning",
                    "message": f"Strong weather forecast for {airport_code} at {forecast.valid_from.strftime('%H:%M')}",
                    "action_required": "Prepare for capacity reduction",
                    "lead_time_hours": (forecast.valid_from - datetime.now()).total_seconds() / 3600
                }
                integration_result["alert_triggers"].append(alert)
        
        return integration_result
    
    def simulate_weather_data(self, airport_code: str, hours_ahead: int = 6) -> List[WeatherData]:
        """Simulate weather data for testing purposes."""
        weather_data = []
        base_time = datetime.now()
        
        for hour in range(hours_ahead):
            timestamp = base_time + timedelta(hours=hour)
            
            # Simulate changing conditions
            visibility = max(0.5, 10.0 - hour * 1.5 + random.uniform(-2, 2))
            wind_speed = min(50, 5 + hour * 3 + random.uniform(-5, 10))
            
            condition = WeatherCondition.CLEAR
            if visibility < 3:
                condition = WeatherCondition.FOG
            elif wind_speed > 25:
                condition = WeatherCondition.THUNDERSTORM
            elif wind_speed > 15:
                condition = WeatherCondition.RAIN
            
            weather = WeatherData(
                airport_code=airport_code,
                timestamp=timestamp,
                visibility_km=visibility,
                wind_speed_kts=wind_speed,
                wind_direction=270 + random.randint(-30, 30),
                temperature_c=25 + random.uniform(-5, 5),
                humidity_percent=60 + random.uniform(-20, 30),
                pressure_hpa=1013.25 + random.uniform(-10, 10),
                precipitation=condition in [WeatherCondition.RAIN, WeatherCondition.THUNDERSTORM],
                condition=condition
            )
            
            weather_data.append(weather)
        
        return weather_data