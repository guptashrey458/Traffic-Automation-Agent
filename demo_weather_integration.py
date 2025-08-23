#!/usr/bin/env python3
"""
Demo script for Weather Integration Service.

This script demonstrates the weather modeling and capacity management capabilities
including weather regime classification, capacity adjustments, impact prediction,
and scenario generation for contingency planning.
"""

import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from services.weather_integration import (
    WeatherIntegrationService,
    WeatherData,
    WeatherForecast,
    WeatherRegime,
    WeatherCondition,
    WeatherScenario
)


def print_header(title: str):
    """Print formatted header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_weather_data(weather: WeatherData):
    """Print formatted weather data."""
    print(f"Airport: {weather.airport_code}")
    print(f"Time: {weather.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Visibility: {weather.visibility_km:.1f} km")
    print(f"Wind: {weather.wind_speed_kts:.0f} kts @ {weather.wind_direction}°")
    print(f"Temperature: {weather.temperature_c:.1f}°C")
    print(f"Condition: {weather.condition.value}")
    print(f"Weather Regime: {weather.weather_regime.value.upper()}")
    print(f"Precipitation: {'Yes' if weather.precipitation else 'No'}")


def print_capacity_adjustment(runway: str, adjustment):
    """Print formatted capacity adjustment."""
    print(f"  {runway}:")
    print(f"    Original Capacity: {adjustment.original_capacity} movements/hour")
    print(f"    Adjusted Capacity: {adjustment.adjusted_capacity} movements/hour")
    print(f"    Reduction: {adjustment.capacity_reduction_percent:.1f}%")
    print(f"    Weather Regime: {adjustment.weather_regime.value}")
    print(f"    Reason: {adjustment.reason}")


def print_scenario(scenario: WeatherScenario):
    """Print formatted weather scenario."""
    print(f"Scenario: {scenario.name}")
    print(f"ID: {scenario.scenario_id}")
    print(f"Airport: {scenario.airport_code}")
    print(f"Duration: {scenario.duration_hours} hours")
    print(f"Probability: {scenario.probability:.1%}")
    print(f"Total Impact: {scenario.get_total_capacity_impact():.1f}% capacity reduction")
    
    if scenario.weather_conditions:
        weather = scenario.weather_conditions[0]
        print(f"Conditions: {weather.get_regime().value} weather")
        print(f"  Visibility: {weather.visibility_km:.1f} km")
        print(f"  Wind: {weather.wind_speed_kts:.0f} kts")
    
    print("Runway Impacts:")
    for runway, adjustment in scenario.capacity_impacts.items():
        print(f"  {runway}: {adjustment.capacity_reduction_percent:.1f}% reduction")


def demo_weather_regime_classification():
    """Demonstrate weather regime classification."""
    print_header("Weather Regime Classification Demo")
    
    # Create different weather conditions
    weather_conditions = [
        {
            "name": "Perfect Flying Weather",
            "visibility": 15.0,
            "wind": 3.0,
            "condition": WeatherCondition.CLEAR,
            "precipitation": False
        },
        {
            "name": "Moderate Conditions",
            "visibility": 6.0,
            "wind": 12.0,
            "condition": WeatherCondition.CLOUDY,
            "precipitation": False
        },
        {
            "name": "Challenging Weather",
            "visibility": 2.5,
            "wind": 28.0,
            "condition": WeatherCondition.RAIN,
            "precipitation": True
        },
        {
            "name": "Severe Thunderstorm",
            "visibility": 0.8,
            "wind": 45.0,
            "condition": WeatherCondition.THUNDERSTORM,
            "precipitation": True
        }
    ]
    
    for i, conditions in enumerate(weather_conditions, 1):
        print(f"\n{i}. {conditions['name']}")
        print("-" * 40)
        
        weather = WeatherData(
            airport_code="BOM",
            timestamp=datetime.now(),
            visibility_km=conditions["visibility"],
            wind_speed_kts=conditions["wind"],
            wind_direction=270,
            temperature_c=25.0,
            humidity_percent=70.0,
            pressure_hpa=1013.25,
            precipitation=conditions["precipitation"],
            condition=conditions["condition"]
        )
        
        print_weather_data(weather)


def demo_capacity_adjustments():
    """Demonstrate runway capacity adjustments."""
    print_header("Runway Capacity Adjustments Demo")
    
    service = WeatherIntegrationService()
    
    # Test different weather conditions at BOM
    weather_scenarios = [
        ("Calm Morning", WeatherRegime.CALM, 10.0, 5.0),
        ("Afternoon Clouds", WeatherRegime.MEDIUM, 5.0, 15.0),
        ("Evening Storms", WeatherRegime.STRONG, 2.0, 25.0),
        ("Severe Weather", WeatherRegime.SEVERE, 0.5, 40.0)
    ]
    
    for scenario_name, regime, visibility, wind_speed in weather_scenarios:
        print(f"\n{scenario_name} ({regime.value.upper()})")
        print("-" * 50)
        
        weather = WeatherData(
            airport_code="BOM",
            timestamp=datetime.now(),
            visibility_km=visibility,
            wind_speed_kts=wind_speed,
            wind_direction=270,
            temperature_c=25.0,
            humidity_percent=70.0,
            pressure_hpa=1013.25,
            precipitation=regime in [WeatherRegime.STRONG, WeatherRegime.SEVERE],
            condition=WeatherCondition.CLEAR if regime == WeatherRegime.CALM else WeatherCondition.THUNDERSTORM,
            weather_regime=regime
        )
        
        # Process weather update
        service.process_weather_update(weather)
        
        # Show capacity adjustments
        adjustments = service.get_all_capacity_adjustments("BOM")
        print("Runway Capacity Adjustments:")
        for runway, adjustment in adjustments.items():
            print_capacity_adjustment(runway, adjustment)


def demo_weather_impact_prediction():
    """Demonstrate weather impact prediction."""
    print_header("Weather Impact Prediction Demo")
    
    service = WeatherIntegrationService()
    
    # Create a forecast for deteriorating conditions
    forecast = WeatherForecast(
        airport_code="BOM",
        forecast_time=datetime.now(),
        valid_from=datetime.now() + timedelta(hours=2),
        valid_to=datetime.now() + timedelta(hours=6),
        visibility_km=3.0,
        wind_speed_kts=22.0,
        wind_direction=270,
        precipitation_probability=0.7,
        condition=WeatherCondition.RAIN,
        confidence=0.75
    )
    
    print("Weather Forecast:")
    print(f"Airport: {forecast.airport_code}")
    print(f"Valid: {forecast.valid_from.strftime('%H:%M')} - {forecast.valid_to.strftime('%H:%M')}")
    print(f"Visibility: {forecast.visibility_km:.1f} km")
    print(f"Wind: {forecast.wind_speed_kts:.0f} kts")
    print(f"Precipitation: {forecast.precipitation_probability:.0%} chance")
    print(f"Confidence: {forecast.confidence:.0%}")
    print(f"Expected Regime: {forecast.get_regime().value.upper()}")
    
    # Predict impact
    impact = service.predict_weather_impact(forecast, 6)
    
    print(f"\nImpact Forecast (Next {impact.forecast_horizon_hours} hours):")
    print(f"Confidence: {impact.confidence:.0%}")
    print(f"Number of Scenarios: {len(impact.scenarios)}")
    
    print("\nRecommended Actions:")
    for i, action in enumerate(impact.recommended_actions, 1):
        print(f"  {i}. {action}")
    
    # Show most likely scenario
    most_likely = impact.get_most_likely_scenario()
    if most_likely:
        print(f"\nMost Likely Scenario ({most_likely.probability:.0%} chance):")
        print(f"  {most_likely.name}")
        print(f"  Total Impact: {most_likely.get_total_capacity_impact():.1f}% capacity reduction")


def demo_scenario_generation():
    """Demonstrate weather scenario generation."""
    print_header("Weather Scenario Generation Demo")
    
    service = WeatherIntegrationService()
    
    # Generate scenarios for contingency planning
    scenarios = service.generate_weather_scenarios(None)
    
    print(f"Generated {len(scenarios)} weather scenarios for contingency planning\n")
    
    # Group scenarios by airport and show summary
    airport_scenarios = {}
    for scenario in scenarios:
        if scenario.airport_code not in airport_scenarios:
            airport_scenarios[scenario.airport_code] = []
        airport_scenarios[scenario.airport_code].append(scenario)
    
    for airport, airport_scenario_list in airport_scenarios.items():
        print(f"{airport} Airport Scenarios:")
        print("-" * 30)
        
        for scenario in sorted(airport_scenario_list, key=lambda s: s.probability, reverse=True):
            regime = scenario.weather_conditions[0].get_regime().value if scenario.weather_conditions else "unknown"
            print(f"  {regime.title():>10}: {scenario.probability:.0%} chance, "
                  f"{scenario.get_total_capacity_impact():.1f}% impact")
        print()


def demo_weather_optimization():
    """Demonstrate weather-aware optimization."""
    print_header("Weather-Aware Optimization Demo")
    
    service = WeatherIntegrationService()
    
    # Create a challenging weather scenario
    scenario = service._generate_scenario(
        "BOM", "strong_weather", datetime.now() + timedelta(hours=1), 
        4, WeatherRegime.STRONG, 0.6
    )
    
    print("Weather Scenario for Optimization:")
    print_scenario(scenario)
    
    # Optimize for this scenario
    optimization_result = service.optimize_for_weather(scenario)
    
    print(f"\nOptimization Results:")
    print(f"Weather Regime: {optimization_result['weather_regime'].upper()}")
    print(f"Estimated Delay Impact: {optimization_result['estimated_delay_impact']:.1f} minutes")
    print(f"Confidence: {optimization_result['confidence']:.0%}")
    
    print("\nRunway Capacity Adjustments:")
    for runway, adjustment in optimization_result["capacity_adjustments"].items():
        print(f"  {runway}:")
        print(f"    {adjustment['original_capacity']} → {adjustment['adjusted_capacity']} movements/hour")
        print(f"    {adjustment['reduction_percent']:.1f}% reduction")
    
    print("\nRecommended Actions:")
    for i, action in enumerate(optimization_result["recommended_actions"], 1):
        print(f"  {i}. {action}")


def demo_real_time_monitoring():
    """Demonstrate real-time weather monitoring."""
    print_header("Real-Time Weather Monitoring Demo")
    
    service = WeatherIntegrationService()
    
    print("Simulating 6-hour weather progression at BOM...")
    
    # Simulate deteriorating weather conditions
    weather_data = service.simulate_weather_data("BOM", 6)
    
    for i, weather in enumerate(weather_data):
        print(f"\nHour {i+1}: {weather.timestamp.strftime('%H:%M')}")
        print(f"  Regime: {weather.weather_regime.value.upper()}")
        print(f"  Visibility: {weather.visibility_km:.1f} km")
        print(f"  Wind: {weather.wind_speed_kts:.0f} kts")
        print(f"  Condition: {weather.condition.value}")
        
        # Process the weather update
        service.process_weather_update(weather)
        
        # Show capacity impact
        adjustments = service.get_all_capacity_adjustments("BOM")
        if adjustments:
            avg_reduction = sum(adj.capacity_reduction_percent for adj in adjustments.values()) / len(adjustments)
            print(f"  Average Capacity Reduction: {avg_reduction:.1f}%")
            
            if avg_reduction > 30:
                print("  ⚠️  ALERT: Significant capacity reduction detected!")
            elif avg_reduction > 15:
                print("  ⚡ WARNING: Moderate capacity impact")


def demo_forecast_integration():
    """Demonstrate weather forecast integration."""
    print_header("Weather Forecast Integration Demo")
    
    service = WeatherIntegrationService()
    
    # Create multiple forecasts for different time periods
    base_time = datetime.now()
    forecasts = []
    
    # Morning forecast - calm
    forecasts.append(WeatherForecast(
        airport_code="DEL",
        forecast_time=base_time,
        valid_from=base_time + timedelta(hours=2),
        valid_to=base_time + timedelta(hours=8),
        visibility_km=12.0,
        wind_speed_kts=8.0,
        wind_direction=270,
        precipitation_probability=0.1,
        condition=WeatherCondition.CLEAR,
        confidence=0.9
    ))
    
    # Afternoon forecast - deteriorating
    forecasts.append(WeatherForecast(
        airport_code="DEL",
        forecast_time=base_time,
        valid_from=base_time + timedelta(hours=8),
        valid_to=base_time + timedelta(hours=14),
        visibility_km=4.0,
        wind_speed_kts=18.0,
        wind_direction=280,
        precipitation_probability=0.6,
        condition=WeatherCondition.RAIN,
        confidence=0.7
    ))
    
    # Evening forecast - severe
    forecasts.append(WeatherForecast(
        airport_code="DEL",
        forecast_time=base_time,
        valid_from=base_time + timedelta(hours=14),
        valid_to=base_time + timedelta(hours=18),
        visibility_km=1.5,
        wind_speed_kts=35.0,
        wind_direction=290,
        precipitation_probability=0.9,
        condition=WeatherCondition.THUNDERSTORM,
        confidence=0.6
    ))
    
    print("24-Hour Weather Forecast for DEL:")
    print("=" * 40)
    
    for i, forecast in enumerate(forecasts):
        period = ["Morning", "Afternoon", "Evening"][i]
        regime = forecast.get_regime()
        
        print(f"\n{period} ({forecast.valid_from.strftime('%H:%M')} - {forecast.valid_to.strftime('%H:%M')}):")
        print(f"  Weather Regime: {regime.value.upper()}")
        print(f"  Visibility: {forecast.visibility_km:.1f} km")
        print(f"  Wind: {forecast.wind_speed_kts:.0f} kts")
        print(f"  Precipitation: {forecast.precipitation_probability:.0%} chance")
        print(f"  Confidence: {forecast.confidence:.0%}")
        
        # Predict impact for this period
        impact = service.predict_weather_impact(forecast, 6)
        most_likely = impact.get_most_likely_scenario()
        
        if most_likely:
            capacity_impact = most_likely.get_total_capacity_impact()
            print(f"  Expected Impact: {capacity_impact:.1f}% capacity reduction")
            
            if capacity_impact > 40:
                print("  🚨 SEVERE IMPACT - Consider ground delay program")
            elif capacity_impact > 20:
                print("  ⚠️  MODERATE IMPACT - Increase spacing")
            elif capacity_impact > 10:
                print("  ⚡ MINOR IMPACT - Monitor closely")
            else:
                print("  ✅ MINIMAL IMPACT - Normal operations")
    
    # Demonstrate proactive forecast integration
    print(f"\n{'='*50}")
    print("PROACTIVE FORECAST INTEGRATION")
    print(f"{'='*50}")
    
    integration_result = service.implement_weather_forecast_integration(forecasts)
    
    print(f"Forecast Integration Summary:")
    print(f"  Forecasts Processed: {integration_result['forecast_count']}")
    print(f"  Airports Affected: {', '.join(integration_result['airports_affected'])}")
    print(f"  Time Horizon: {integration_result['time_horizon_hours']:.1f} hours")
    print(f"  Overall Confidence: {integration_result['confidence']:.0%}")
    
    if integration_result['proactive_adjustments']:
        print(f"\nProactive Schedule Adjustments:")
        for adj in integration_result['proactive_adjustments']:
            print(f"  • {adj['airport']}: {adj['recommended_delay_minutes']}min delay")
            print(f"    Reason: {adj['reason']}")
            print(f"    Lead Time: {adj['hours_ahead']:.1f} hours")
    
    if integration_result['alert_triggers']:
        print(f"\nAlert Triggers:")
        for alert in integration_result['alert_triggers']:
            severity_icon = "🚨" if alert['severity'] == 'critical' else "⚠️"
            print(f"  {severity_icon} {alert['message']}")
            print(f"    Action: {alert['action_required']}")
            print(f"    Lead Time: {alert['lead_time_hours']:.1f} hours")
    
    if integration_result['capacity_planning']:
        print(f"\nCapacity Planning:")
        for airport, planning in integration_result['capacity_planning'].items():
            print(f"  {airport} Airport:")
            print(f"    Weather Regime: {planning['current_regime'].upper()}")
            for runway, reduction in planning['capacity_reductions'].items():
                print(f"    {runway}: {reduction['reduction_percent']:.1f}% reduction "
                      f"({reduction['valid_from']} - {reduction['valid_to']})")


def main():
    """Run all weather integration demos."""
    print("🌦️  Weather Integration Service Demo")
    print("Demonstrating weather modeling and capacity management capabilities")
    
    try:
        # Run all demo functions
        demo_weather_regime_classification()
        demo_capacity_adjustments()
        demo_weather_impact_prediction()
        demo_scenario_generation()
        demo_weather_optimization()
        demo_real_time_monitoring()
        demo_forecast_integration()
        
        print_header("Demo Complete")
        print("✅ All weather integration features demonstrated successfully!")
        print("\nKey Capabilities Shown:")
        print("• Weather regime classification (calm/medium/strong/severe)")
        print("• Runway capacity adjustments based on weather conditions")
        print("• Weather impact prediction with multiple scenarios")
        print("• Contingency planning with scenario generation")
        print("• Weather-aware optimization recommendations")
        print("• Real-time weather monitoring and alerting")
        print("• Forecast integration for proactive planning")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())