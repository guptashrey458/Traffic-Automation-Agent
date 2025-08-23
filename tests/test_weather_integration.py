"""Tests for weather integration service."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.services.weather_integration import (
    WeatherIntegrationService,
    WeatherData,
    WeatherForecast,
    WeatherRegime,
    WeatherCondition,
    CapacityAdjustment,
    WeatherScenario,
    ImpactForecast
)


class TestWeatherRegimeClassification:
    """Test weather regime classification logic."""
    
    def test_calm_weather_classification(self):
        """Test classification of calm weather conditions."""
        weather = WeatherData(
            airport_code="BOM",
            timestamp=datetime.now(),
            visibility_km=10.0,
            wind_speed_kts=5.0,
            wind_direction=270,
            temperature_c=25.0,
            humidity_percent=60.0,
            pressure_hpa=1013.25,
            precipitation=False,
            condition=WeatherCondition.CLEAR
        )
        
        assert weather.weather_regime == WeatherRegime.CALM
    
    def test_medium_weather_classification(self):
        """Test classification of medium weather conditions."""
        weather = WeatherData(
            airport_code="BOM",
            timestamp=datetime.now(),
            visibility_km=5.0,
            wind_speed_kts=15.0,
            wind_direction=270,
            temperature_c=25.0,
            humidity_percent=70.0,
            pressure_hpa=1010.0,
            precipitation=False,
            condition=WeatherCondition.CLOUDY
        )
        
        assert weather.weather_regime == WeatherRegime.MEDIUM
    
    def test_strong_weather_classification(self):
        """Test classification of strong weather conditions."""
        weather = WeatherData(
            airport_code="BOM",
            timestamp=datetime.now(),
            visibility_km=2.0,
            wind_speed_kts=25.0,
            wind_direction=270,
            temperature_c=20.0,
            humidity_percent=80.0,
            pressure_hpa=1005.0,
            precipitation=True,
            condition=WeatherCondition.RAIN
        )
        
        assert weather.weather_regime == WeatherRegime.STRONG
    
    def test_severe_weather_classification(self):
        """Test classification of severe weather conditions."""
        weather = WeatherData(
            airport_code="BOM",
            timestamp=datetime.now(),
            visibility_km=0.5,
            wind_speed_kts=40.0,
            wind_direction=270,
            temperature_c=18.0,
            humidity_percent=90.0,
            pressure_hpa=995.0,
            precipitation=True,
            condition=WeatherCondition.THUNDERSTORM
        )
        
        assert weather.weather_regime == WeatherRegime.SEVERE
    
    def test_edge_case_classifications(self):
        """Test edge cases in weather classification."""
        # Exactly at boundary - should be calm
        weather1 = WeatherData(
            airport_code="BOM",
            timestamp=datetime.now(),
            visibility_km=8.0,
            wind_speed_kts=10.0,
            wind_direction=270,
            temperature_c=25.0,
            humidity_percent=60.0,
            pressure_hpa=1013.25,
            precipitation=False,
            condition=WeatherCondition.CLEAR
        )
        assert weather1.weather_regime == WeatherRegime.MEDIUM
        
        # Low visibility but low wind
        weather2 = WeatherData(
            airport_code="BOM",
            timestamp=datetime.now(),
            visibility_km=0.8,
            wind_speed_kts=5.0,
            wind_direction=270,
            temperature_c=25.0,
            humidity_percent=60.0,
            pressure_hpa=1013.25,
            precipitation=False,
            condition=WeatherCondition.FOG
        )
        assert weather2.weather_regime == WeatherRegime.SEVERE


class TestWeatherIntegrationService:
    """Test weather integration service functionality."""
    
    @pytest.fixture
    def service(self):
        """Create weather integration service instance."""
        return WeatherIntegrationService()
    
    @pytest.fixture
    def sample_weather_data(self):
        """Create sample weather data."""
        return WeatherData(
            airport_code="BOM",
            timestamp=datetime.now(),
            visibility_km=5.0,
            wind_speed_kts=15.0,
            wind_direction=270,
            temperature_c=25.0,
            humidity_percent=70.0,
            pressure_hpa=1010.0,
            precipitation=False,
            condition=WeatherCondition.CLOUDY
        )
    
    def test_service_initialization(self, service):
        """Test service initialization."""
        assert isinstance(service.current_weather, dict)
        assert isinstance(service.forecasts, dict)
        assert isinstance(service.capacity_adjustments, dict)
        assert "BOM" in service.default_capacities
        assert WeatherRegime.CALM in service.regime_factors
    
    def test_process_weather_update(self, service, sample_weather_data):
        """Test processing weather updates."""
        service.process_weather_update(sample_weather_data)
        
        # Check weather data is stored
        assert "BOM" in service.current_weather
        assert service.current_weather["BOM"] == sample_weather_data
        
        # Check capacity adjustments are created
        assert "BOM" in service.capacity_adjustments
        assert len(service.capacity_adjustments["BOM"]) > 0
    
    def test_runway_capacity_adjustment(self, service, sample_weather_data):
        """Test runway capacity adjustment calculations."""
        adjustment = service.adjust_runway_capacity("09L/27R", sample_weather_data)
        
        assert isinstance(adjustment, CapacityAdjustment)
        assert adjustment.runway == "09L/27R"
        assert adjustment.weather_regime == WeatherRegime.MEDIUM
        assert adjustment.reduction_factor == 0.85  # Medium weather factor
        assert adjustment.adjusted_capacity < adjustment.original_capacity
        assert abs(adjustment.capacity_reduction_percent - 15.0) < 0.001
    
    def test_capacity_reduction_factors(self, service):
        """Test capacity reduction factors for different weather regimes."""
        base_time = datetime.now()
        
        # Test all weather regimes
        regimes_and_factors = [
            (WeatherRegime.CALM, 1.0),
            (WeatherRegime.MEDIUM, 0.85),
            (WeatherRegime.STRONG, 0.65),
            (WeatherRegime.SEVERE, 0.40)
        ]
        
        for regime, expected_factor in regimes_and_factors:
            weather = WeatherData(
                airport_code="BOM",
                timestamp=base_time,
                visibility_km=10.0 if regime == WeatherRegime.CALM else 1.0,
                wind_speed_kts=5.0 if regime == WeatherRegime.CALM else 40.0,
                wind_direction=270,
                temperature_c=25.0,
                humidity_percent=60.0,
                pressure_hpa=1013.25,
                precipitation=False,
                condition=WeatherCondition.CLEAR,
                weather_regime=regime
            )
            
            adjustment = service.adjust_runway_capacity("09L/27R", weather)
            assert adjustment.reduction_factor == expected_factor
    
    def test_weather_impact_prediction(self, service):
        """Test weather impact prediction."""
        forecast = WeatherForecast(
            airport_code="BOM",
            forecast_time=datetime.now(),
            valid_from=datetime.now() + timedelta(hours=1),
            valid_to=datetime.now() + timedelta(hours=4),
            visibility_km=3.0,
            wind_speed_kts=20.0,
            wind_direction=270,
            precipitation_probability=0.6,
            condition=WeatherCondition.RAIN,
            confidence=0.7
        )
        
        impact = service.predict_weather_impact(forecast, 6)
        
        assert isinstance(impact, ImpactForecast)
        assert impact.airport_code == "BOM"
        assert impact.forecast_horizon_hours == 6
        assert len(impact.scenarios) >= 1
        assert len(impact.recommended_actions) > 0
        assert 0.0 <= impact.confidence <= 1.0
    
    def test_weather_scenario_generation(self, service):
        """Test weather scenario generation."""
        scenarios = service.generate_weather_scenarios(None)
        
        assert len(scenarios) > 0
        
        # Check that we have scenarios for different regimes
        regimes_found = set()
        for scenario in scenarios:
            if scenario.weather_conditions:
                regime = scenario.weather_conditions[0].get_regime()
                regimes_found.add(regime)
        
        assert len(regimes_found) >= 3  # Should have multiple regime types
        
        # Check scenario structure
        sample_scenario = scenarios[0]
        assert isinstance(sample_scenario, WeatherScenario)
        assert sample_scenario.airport_code in ["BOM", "DEL", "MAA", "BLR", "HYD"]
        assert len(sample_scenario.capacity_impacts) > 0
        assert 0.0 <= sample_scenario.probability <= 1.0
    
    def test_optimize_for_weather(self, service):
        """Test weather-aware optimization."""
        # Create a test scenario
        scenario = service._generate_scenario(
            "BOM", "strong", datetime.now(), 3, WeatherRegime.STRONG, 0.3
        )
        
        result = service.optimize_for_weather(scenario)
        
        assert "scenario_id" in result
        assert "airport_code" in result
        assert "weather_regime" in result
        assert "capacity_adjustments" in result
        assert "recommended_actions" in result
        assert "estimated_delay_impact" in result
        
        # Check that capacity adjustments are present
        assert len(result["capacity_adjustments"]) > 0
        
        # Check that recommendations are provided
        assert len(result["recommended_actions"]) > 0
        
        # Check delay impact is reasonable
        assert result["estimated_delay_impact"] >= 0
    
    def test_severe_weather_recommendations(self, service):
        """Test recommendations for severe weather scenarios."""
        # Create severe weather scenario
        scenario = service._generate_scenario(
            "BOM", "severe", datetime.now(), 2, WeatherRegime.SEVERE, 0.1
        )
        
        result = service.optimize_for_weather(scenario)
        
        # Should have strong recommendations for severe weather
        recommendations = result["recommended_actions"]
        assert any("ground delay" in rec.lower() for rec in recommendations)
        assert result["estimated_delay_impact"] > 50  # Should predict significant delays
    
    def test_get_current_weather(self, service, sample_weather_data):
        """Test getting current weather data."""
        # Initially no data
        assert service.get_current_weather("BOM") is None
        
        # After processing update
        service.process_weather_update(sample_weather_data)
        current = service.get_current_weather("BOM")
        assert current == sample_weather_data
    
    def test_get_capacity_adjustments(self, service, sample_weather_data):
        """Test getting capacity adjustments."""
        service.process_weather_update(sample_weather_data)
        
        # Get specific runway adjustment
        adjustment = service.get_capacity_adjustment("BOM", "09L/27R")
        assert adjustment is not None
        assert adjustment.runway == "09L/27R"
        
        # Get all adjustments for airport
        all_adjustments = service.get_all_capacity_adjustments("BOM")
        assert len(all_adjustments) > 0
        assert "09L/27R" in all_adjustments
    
    def test_simulate_weather_data(self, service):
        """Test weather data simulation."""
        simulated_data = service.simulate_weather_data("BOM", 6)
        
        assert len(simulated_data) == 6
        
        for weather in simulated_data:
            assert isinstance(weather, WeatherData)
            assert weather.airport_code == "BOM"
            assert weather.visibility_km > 0
            assert weather.wind_speed_kts >= 0
            assert isinstance(weather.weather_regime, WeatherRegime)
    
    def test_forecast_regime_calculation(self):
        """Test weather regime calculation for forecasts."""
        # Test calm forecast
        calm_forecast = WeatherForecast(
            airport_code="BOM",
            forecast_time=datetime.now(),
            valid_from=datetime.now(),
            valid_to=datetime.now() + timedelta(hours=3),
            visibility_km=10.0,
            wind_speed_kts=5.0,
            wind_direction=270,
            precipitation_probability=0.1,
            condition=WeatherCondition.CLEAR
        )
        assert calm_forecast.get_regime() == WeatherRegime.CALM
        
        # Test severe forecast
        severe_forecast = WeatherForecast(
            airport_code="BOM",
            forecast_time=datetime.now(),
            valid_from=datetime.now(),
            valid_to=datetime.now() + timedelta(hours=2),
            visibility_km=0.8,
            wind_speed_kts=45.0,
            wind_direction=270,
            precipitation_probability=0.9,
            condition=WeatherCondition.THUNDERSTORM
        )
        assert severe_forecast.get_regime() == WeatherRegime.SEVERE
    
    def test_capacity_adjustment_properties(self):
        """Test capacity adjustment property calculations."""
        adjustment = CapacityAdjustment(
            runway="09L/27R",
            original_capacity=40,
            adjusted_capacity=30,
            reduction_factor=0.75,
            weather_regime=WeatherRegime.MEDIUM,
            valid_from=datetime.now(),
            valid_to=datetime.now() + timedelta(hours=1),
            reason="Test adjustment"
        )
        
        assert adjustment.capacity_reduction_percent == 25.0
        
        # Test zero capacity case
        zero_adjustment = CapacityAdjustment(
            runway="09L/27R",
            original_capacity=0,
            adjusted_capacity=0,
            reduction_factor=0.5,
            weather_regime=WeatherRegime.MEDIUM,
            valid_from=datetime.now(),
            valid_to=datetime.now() + timedelta(hours=1),
            reason="Test zero capacity"
        )
        
        assert zero_adjustment.capacity_reduction_percent == 0.0
    
    def test_weather_scenario_impact_calculation(self):
        """Test weather scenario total impact calculation."""
        # Create scenario with multiple runway impacts
        adjustments = {
            "09L/27R": CapacityAdjustment(
                runway="09L/27R",
                original_capacity=40,
                adjusted_capacity=30,
                reduction_factor=0.75,
                weather_regime=WeatherRegime.MEDIUM,
                valid_from=datetime.now(),
                valid_to=datetime.now() + timedelta(hours=1),
                reason="Test"
            ),
            "09R/27L": CapacityAdjustment(
                runway="09R/27L",
                original_capacity=40,
                adjusted_capacity=20,
                reduction_factor=0.5,
                weather_regime=WeatherRegime.STRONG,
                valid_from=datetime.now(),
                valid_to=datetime.now() + timedelta(hours=1),
                reason="Test"
            )
        }
        
        scenario = WeatherScenario(
            scenario_id="test",
            name="Test scenario",
            description="Test",
            airport_code="BOM",
            start_time=datetime.now(),
            duration_hours=2,
            weather_conditions=[],
            capacity_impacts=adjustments,
            probability=0.5
        )
        
        # Should average the capacity reductions: (25% + 50%) / 2 = 37.5%
        assert scenario.get_total_capacity_impact() == 37.5
        
        # Test empty impacts
        empty_scenario = WeatherScenario(
            scenario_id="empty",
            name="Empty scenario",
            description="Test",
            airport_code="BOM",
            start_time=datetime.now(),
            duration_hours=2,
            weather_conditions=[],
            capacity_impacts={},
            probability=0.5
        )
        
        assert empty_scenario.get_total_capacity_impact() == 0.0
    
    def test_impact_forecast_most_likely_scenario(self):
        """Test finding most likely scenario in impact forecast."""
        scenarios = [
            WeatherScenario(
                scenario_id="low_prob",
                name="Low probability",
                description="Test",
                airport_code="BOM",
                start_time=datetime.now(),
                duration_hours=2,
                weather_conditions=[],
                capacity_impacts={},
                probability=0.2
            ),
            WeatherScenario(
                scenario_id="high_prob",
                name="High probability",
                description="Test",
                airport_code="BOM",
                start_time=datetime.now(),
                duration_hours=2,
                weather_conditions=[],
                capacity_impacts={},
                probability=0.7
            )
        ]
        
        forecast = ImpactForecast(
            airport_code="BOM",
            forecast_horizon_hours=6,
            scenarios=scenarios,
            recommended_actions=["Test action"],
            confidence=0.8
        )
        
        most_likely = forecast.get_most_likely_scenario()
        assert most_likely is not None
        assert most_likely.scenario_id == "high_prob"
        assert most_likely.probability == 0.7
        
        # Test empty scenarios
        empty_forecast = ImpactForecast(
            airport_code="BOM",
            forecast_horizon_hours=6,
            scenarios=[],
            recommended_actions=[],
            confidence=0.8
        )
        
        assert empty_forecast.get_most_likely_scenario() is None


class TestWeatherIntegrationEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def service(self):
        """Create weather integration service instance."""
        return WeatherIntegrationService()
    
    def test_unknown_airport_capacity_adjustment(self, service):
        """Test capacity adjustment for unknown airport."""
        weather = WeatherData(
            airport_code="UNKNOWN",
            timestamp=datetime.now(),
            visibility_km=5.0,
            wind_speed_kts=15.0,
            wind_direction=270,
            temperature_c=25.0,
            humidity_percent=70.0,
            pressure_hpa=1010.0,
            precipitation=False,
            condition=WeatherCondition.CLOUDY
        )
        
        # Should not raise exception
        service.process_weather_update(weather)
        
        # Should still create adjustment with default capacity
        adjustment = service.adjust_runway_capacity("TEST_RUNWAY", weather)
        assert adjustment.original_capacity == 40  # Default fallback
    
    def test_extreme_weather_values(self, service):
        """Test handling of extreme weather values."""
        extreme_weather = WeatherData(
            airport_code="BOM",
            timestamp=datetime.now(),
            visibility_km=0.1,  # Extremely low
            wind_speed_kts=100.0,  # Extremely high
            wind_direction=360,
            temperature_c=-20.0,
            humidity_percent=100.0,
            pressure_hpa=900.0,
            precipitation=True,
            condition=WeatherCondition.THUNDERSTORM
        )
        
        assert extreme_weather.weather_regime == WeatherRegime.SEVERE
        
        adjustment = service.adjust_runway_capacity("09L/27R", extreme_weather)
        assert adjustment.reduction_factor == 0.40  # Severe weather factor
        assert adjustment.adjusted_capacity < adjustment.original_capacity
    
    def test_forecast_adjustment_edge_cases(self, service):
        """Test forecast adjustment edge cases."""
        base_forecast = WeatherForecast(
            airport_code="BOM",
            forecast_time=datetime.now(),
            valid_from=datetime.now(),
            valid_to=datetime.now() + timedelta(hours=3),
            visibility_km=1.0,
            wind_speed_kts=5.0,
            wind_direction=270,
            precipitation_probability=0.5,
            condition=WeatherCondition.RAIN
        )
        
        # Test improvement adjustment
        improved = service._adjust_forecast(base_forecast, improvement=True)
        assert improved.visibility_km > base_forecast.visibility_km
        assert improved.wind_speed_kts < base_forecast.wind_speed_kts
        
        # Test degradation adjustment
        degraded = service._adjust_forecast(base_forecast, improvement=False)
        assert degraded.visibility_km < base_forecast.visibility_km
        assert degraded.wind_speed_kts > base_forecast.wind_speed_kts
        
        # Test minimum bounds
        minimal_forecast = WeatherForecast(
            airport_code="BOM",
            forecast_time=datetime.now(),
            valid_from=datetime.now(),
            valid_to=datetime.now() + timedelta(hours=3),
            visibility_km=0.1,
            wind_speed_kts=0.0,
            wind_direction=270,
            precipitation_probability=0.0,
            condition=WeatherCondition.CLEAR
        )
        
        degraded_minimal = service._adjust_forecast(minimal_forecast, improvement=False)
        assert degraded_minimal.visibility_km >= 0.5  # Should not go below minimum
        assert degraded_minimal.wind_speed_kts >= 0.0  # Should not go negative
    
    def test_forecast_integration(self, service):
        """Test weather forecast integration for proactive planning."""
        # Create test forecasts
        base_time = datetime.now()
        forecasts = [
            WeatherForecast(
                airport_code="BOM",
                forecast_time=base_time,
                valid_from=base_time + timedelta(hours=2),
                valid_to=base_time + timedelta(hours=6),
                visibility_km=8.0,
                wind_speed_kts=12.0,
                wind_direction=270,
                precipitation_probability=0.3,
                condition=WeatherCondition.CLOUDY,
                confidence=0.8
            ),
            WeatherForecast(
                airport_code="BOM",
                forecast_time=base_time,
                valid_from=base_time + timedelta(hours=6),
                valid_to=base_time + timedelta(hours=10),
                visibility_km=1.5,
                wind_speed_kts=30.0,
                wind_direction=280,
                precipitation_probability=0.8,
                condition=WeatherCondition.THUNDERSTORM,
                confidence=0.7
            )
        ]
        
        result = service.implement_weather_forecast_integration(forecasts)
        
        # Check basic structure
        assert result["forecast_count"] == 2
        assert "BOM" in result["airports_affected"]
        assert result["time_horizon_hours"] > 0
        assert 0.0 <= result["confidence"] <= 1.0
        
        # Check capacity planning
        assert "BOM" in result["capacity_planning"]
        bom_planning = result["capacity_planning"]["BOM"]
        assert "capacity_reductions" in bom_planning
        assert len(bom_planning["capacity_reductions"]) > 0
        
        # Check proactive adjustments (should have at least one for strong weather)
        assert len(result["proactive_adjustments"]) > 0
        strong_adjustment = next(
            (adj for adj in result["proactive_adjustments"] 
             if adj["weather_regime"] == "strong"), None
        )
        assert strong_adjustment is not None
        assert strong_adjustment["recommended_delay_minutes"] > 0
        
        # Check alert triggers
        assert len(result["alert_triggers"]) > 0
        warning_alert = next(
            (alert for alert in result["alert_triggers"] 
             if alert["severity"] == "warning"), None
        )
        assert warning_alert is not None
        assert "Strong weather" in warning_alert["message"]
    
    def test_empty_forecast_integration(self, service):
        """Test forecast integration with empty forecast list."""
        result = service.implement_weather_forecast_integration([])
        
        assert result["forecast_count"] == 0
        assert result["airports_affected"] == []
        assert result["time_horizon_hours"] == 0
        assert result["proactive_adjustments"] == []
        assert result["capacity_planning"] == {}
        assert result["alert_triggers"] == []
        assert result["confidence"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__])