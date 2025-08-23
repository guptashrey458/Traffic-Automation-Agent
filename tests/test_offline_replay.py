"""Tests for offline replay mode functionality."""

import pytest
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import pandas as pd

from src.services.offline_replay import (
    OfflineReplayService, 
    WeatherRegime, 
    SimulationEvent, 
    WeatherCondition,
    SimulationState
)
from src.services.console_alerting import (
    ConsoleAlertingService,
    AlertType,
    AlertSeverity,
    ConsoleAlert
)
from src.models.flight import Flight
from src.config.settings import settings


class TestOfflineReplayService:
    """Test cases for OfflineReplayService."""
    
    @pytest.fixture
    def replay_service(self):
        """Create a fresh replay service instance."""
        return OfflineReplayService()
    
    @pytest.fixture
    def sample_flights(self):
        """Create sample flight data for testing."""
        base_time = datetime.now()
        flights = []
        
        for i in range(10):
            flight = Flight(
                flight_id=f"TEST{i:03d}",
                flight_no=f"AI{100+i}",
                date_local=base_time.date(),
                origin="BOM" if i % 2 == 0 else "DEL",
                destination="DEL" if i % 2 == 0 else "BOM",
                aircraft_type="A320",
                tail_number=f"VT-TEST{i}",
                std_utc=base_time + timedelta(hours=i),
                atd_utc=None,
                sta_utc=base_time + timedelta(hours=i+2),
                ata_utc=None,
                dep_delay_min=None,
                arr_delay_min=None,
                runway=None,
                stand=None,
                source_file="test_data.xlsx"
            )
            flights.append(flight)
        
        return flights
    
    @pytest.fixture
    def temp_data_dir(self, sample_flights):
        """Create temporary data directory with test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample Excel file
            excel_path = Path(temp_dir) / "test_flights.xlsx"
            
            # Convert flights to DataFrame
            flight_data = []
            for flight in sample_flights:
                flight_data.append({
                    'Flight No': flight.flight_no,
                    'Date': flight.date_local.strftime('%Y-%m-%d'),
                    'From': flight.origin,
                    'To': flight.destination,
                    'Aircraft': flight.aircraft_type,
                    'STD': flight.std_utc.strftime('%H:%M'),
                    'STA': flight.sta_utc.strftime('%H:%M')
                })
            
            df = pd.DataFrame(flight_data)
            df.to_excel(excel_path, index=False)
            
            yield temp_dir
    
    def test_weather_regime_initialization(self, replay_service):
        """Test weather regime initialization."""
        weather_conditions = replay_service._initialize_weather_conditions()
        
        assert len(weather_conditions) > 0
        assert "BOM" in weather_conditions
        assert "DEL" in weather_conditions
        
        for airport, condition in weather_conditions.items():
            assert isinstance(condition.regime, WeatherRegime)
            assert 0 <= condition.visibility_km <= 10
            assert 0 <= condition.wind_speed_kts <= 50
            assert 0.5 <= condition.capacity_multiplier <= 1.0
    
    def test_runway_capacity_initialization(self, replay_service):
        """Test runway capacity initialization."""
        capacities = replay_service._initialize_runway_capacities()
        
        assert len(capacities) > 0
        assert all(capacity > 0 for capacity in capacities.values())
        assert any("BOM" in runway for runway in capacities.keys())
        assert any("DEL" in runway for runway in capacities.keys())
    
    def test_visibility_for_regime(self, replay_service):
        """Test visibility calculation for different weather regimes."""
        for regime in WeatherRegime:
            visibility = replay_service._get_visibility_for_regime(regime)
            assert visibility > 0
            
            if regime == WeatherRegime.CALM:
                assert visibility >= 8.0
            elif regime == WeatherRegime.SEVERE:
                assert visibility <= 2.0
    
    def test_capacity_multiplier_for_regime(self, replay_service):
        """Test capacity multiplier for different weather regimes."""
        for regime in WeatherRegime:
            multiplier = replay_service._get_capacity_multiplier_for_regime(regime)
            assert 0 < multiplier <= 1.0
            
            if regime == WeatherRegime.CALM:
                assert multiplier == 1.0
            elif regime == WeatherRegime.SEVERE:
                assert multiplier == 0.5
    
    @pytest.mark.asyncio
    async def test_initialize_replay_mode(self, replay_service, temp_data_dir):
        """Test replay mode initialization."""
        # Configure settings for test
        original_enabled = settings.offline_replay.enabled
        settings.offline_replay.enabled = True
        
        try:
            success = await replay_service.initialize_replay_mode(temp_data_dir)
            assert success
            
            assert replay_service.simulation_state is not None
            assert len(replay_service.simulation_state.flights) > 0
            assert len(replay_service.event_queue) > 0
            
            # Check simulation state components
            state = replay_service.simulation_state
            assert isinstance(state.current_time, datetime)
            assert len(state.weather_conditions) > 0
            assert len(state.runway_capacities) > 0
            assert isinstance(state.active_alerts, list)
            assert isinstance(state.autonomous_actions, list)
            
        finally:
            settings.offline_replay.enabled = original_enabled
    
    @pytest.mark.asyncio
    async def test_load_replay_data_empty_directory(self, replay_service):
        """Test loading data from empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            flights = await replay_service._load_replay_data(temp_dir)
            assert flights == []
    
    @pytest.mark.asyncio
    async def test_generate_simulation_events(self, replay_service, sample_flights):
        """Test simulation event generation."""
        # Set up simulation state
        start_time = min(flight.std_utc for flight in sample_flights)
        replay_service.simulation_state = SimulationState(
            current_time=start_time,
            flights=sample_flights,
            weather_conditions=replay_service._initialize_weather_conditions(),
            runway_capacities=replay_service._initialize_runway_capacities(),
            active_alerts=[],
            autonomous_actions=[]
        )
        
        # Configure short simulation for testing
        original_hours = settings.offline_replay.max_simulation_hours
        settings.offline_replay.max_simulation_hours = 2
        
        try:
            await replay_service._generate_simulation_events()
            
            assert len(replay_service.event_queue) > 0
            
            # Check event types
            event_types = {event.event_type for event in replay_service.event_queue}
            assert "flight_departure" in event_types
            assert "monitoring_check" in event_types
            
            # Check events are sorted by timestamp
            timestamps = [event.timestamp for event in replay_service.event_queue]
            assert timestamps == sorted(timestamps)
            
        finally:
            settings.offline_replay.max_simulation_hours = original_hours
    
    def test_simulation_status_not_initialized(self, replay_service):
        """Test simulation status when not initialized."""
        status = replay_service.get_simulation_status()
        assert status["status"] == "not_initialized"
    
    def test_simulation_status_initialized(self, replay_service, sample_flights):
        """Test simulation status when initialized."""
        start_time = datetime.now()
        replay_service.simulation_state = SimulationState(
            current_time=start_time,
            flights=sample_flights,
            weather_conditions={},
            runway_capacities={},
            active_alerts=[],
            autonomous_actions=[]
        )
        
        status = replay_service.get_simulation_status()
        assert status["status"] == "stopped"
        assert status["current_time"] == start_time
        assert status["total_flights"] == len(sample_flights)
        assert status["active_alerts"] == 0
        assert status["autonomous_actions"] == 0
    
    def test_stop_simulation(self, replay_service):
        """Test simulation stop functionality."""
        replay_service.is_running = True
        replay_service.stop_simulation()
        assert not replay_service.is_running


class TestConsoleAlertingService:
    """Test cases for ConsoleAlertingService."""
    
    @pytest.fixture
    def alerting_service(self):
        """Create a fresh alerting service instance."""
        return ConsoleAlertingService()
    
    @pytest.fixture
    def sample_alert(self):
        """Create a sample alert for testing."""
        return ConsoleAlert(
            alert_id="TEST_001",
            timestamp=datetime.now(),
            alert_type=AlertType.CAPACITY_OVERLOAD,
            severity=AlertSeverity.HIGH,
            title="Test Alert",
            message="This is a test alert",
            airport="BOM",
            affected_flights=5,
            recommendations=["Test recommendation 1", "Test recommendation 2"],
            metrics={"test_metric": 42.5}
        )
    
    def test_alert_creation(self, sample_alert):
        """Test alert creation and properties."""
        assert sample_alert.alert_id == "TEST_001"
        assert sample_alert.alert_type == AlertType.CAPACITY_OVERLOAD
        assert sample_alert.severity == AlertSeverity.HIGH
        assert sample_alert.airport == "BOM"
        assert sample_alert.affected_flights == 5
        assert len(sample_alert.recommendations) == 2
        assert "test_metric" in sample_alert.metrics
    
    def test_send_alert(self, alerting_service, sample_alert, capsys):
        """Test sending an alert."""
        alerting_service.send_alert(sample_alert)
        
        # Check alert is stored
        assert sample_alert.alert_id in alerting_service.active_alerts
        assert sample_alert in alerting_service.alert_history
        
        # Check console output
        captured = capsys.readouterr()
        assert "FLIGHT OPERATIONS ALERT" in captured.out
        assert "Test Alert" in captured.out
        assert "BOM" in captured.out
    
    def test_capacity_overload_alert(self, alerting_service, capsys):
        """Test capacity overload alert generation."""
        alert_id = alerting_service.send_capacity_overload_alert(
            airport="BOM",
            overload_percentage=1.3,
            affected_flights=25,
            time_window="30 minutes"
        )
        
        assert alert_id.startswith("CAP_BOM_")
        assert alert_id in alerting_service.active_alerts
        
        alert = alerting_service.active_alerts[alert_id]
        assert alert.alert_type == AlertType.CAPACITY_OVERLOAD
        assert alert.severity == AlertSeverity.HIGH  # > 1.2
        assert alert.airport == "BOM"
        assert alert.affected_flights == 25
        
        captured = capsys.readouterr()
        assert "Capacity Overload" in captured.out
        assert "30.0%" in captured.out  # (1.3-1)*100
    
    def test_weather_impact_alert(self, alerting_service, capsys):
        """Test weather impact alert generation."""
        alert_id = alerting_service.send_weather_impact_alert(
            airport="DEL",
            weather_regime="strong",
            capacity_reduction=0.3,
            affected_flights=15
        )
        
        assert alert_id.startswith("WX_DEL_")
        assert alert_id in alerting_service.active_alerts
        
        alert = alerting_service.active_alerts[alert_id]
        assert alert.alert_type == AlertType.WEATHER_IMPACT
        assert alert.severity == AlertSeverity.HIGH
        assert alert.airport == "DEL"
        
        captured = capsys.readouterr()
        assert "Weather Impact" in captured.out
        assert "Strong weather" in captured.out
        assert "30%" in captured.out
    
    def test_autonomous_action_alert(self, alerting_service, capsys):
        """Test autonomous action alert generation."""
        alert_id = alerting_service.send_autonomous_action_alert(
            action_type="schedule_optimization",
            confidence=0.85,
            affected_flights=12,
            expected_benefit="15 minutes delay reduction"
        )
        
        assert alert_id.startswith("AUTO_")
        assert alert_id in alerting_service.active_alerts
        
        alert = alerting_service.active_alerts[alert_id]
        assert alert.alert_type == AlertType.AUTONOMOUS_ACTION
        assert alert.affected_flights == 12
        
        captured = capsys.readouterr()
        assert "Autonomous Action" in captured.out
        assert "85%" in captured.out
        assert "schedule_optimization" in captured.out
    
    def test_optimization_complete_alert(self, alerting_service, capsys):
        """Test optimization completion alert."""
        alert_id = alerting_service.send_optimization_complete_alert(
            flights_optimized=20,
            delay_reduction=12.5,
            runway_changes=6,
            success_rate=0.95
        )
        
        assert alert_id.startswith("OPT_")
        alert = alerting_service.active_alerts[alert_id]
        assert alert.alert_type == AlertType.SYSTEM_OPTIMIZATION
        assert alert.severity == AlertSeverity.LOW
        
        captured = capsys.readouterr()
        assert "Schedule Optimization Completed" in captured.out
        assert "12.5min delay reduction" in captured.out
    
    def test_delay_cascade_alert(self, alerting_service, capsys):
        """Test delay cascade alert generation."""
        alert_id = alerting_service.send_delay_cascade_alert(
            trigger_flight="AI123",
            cascade_depth=4,
            downstream_flights=8,
            estimated_impact=45.0
        )
        
        assert alert_id.startswith("CASCADE_")
        alert = alerting_service.active_alerts[alert_id]
        assert alert.alert_type == AlertType.DELAY_CASCADE
        assert alert.severity == AlertSeverity.HIGH  # depth > 3
        
        captured = capsys.readouterr()
        assert "Delay Cascade Detected" in captured.out
        assert "AI123" in captured.out
    
    def test_resolve_alert(self, alerting_service, sample_alert, capsys):
        """Test alert resolution."""
        # Send alert first
        alerting_service.send_alert(sample_alert)
        assert sample_alert.alert_id in alerting_service.active_alerts
        
        # Resolve alert
        success = alerting_service.resolve_alert(
            sample_alert.alert_id, 
            "Issue resolved by manual intervention"
        )
        
        assert success
        assert sample_alert.alert_id not in alerting_service.active_alerts
        assert sample_alert in alerting_service.alert_history
        
        captured = capsys.readouterr()
        assert "ALERT RESOLVED" in captured.out
        assert "Issue resolved by manual intervention" in captured.out
    
    def test_resolve_nonexistent_alert(self, alerting_service):
        """Test resolving non-existent alert."""
        success = alerting_service.resolve_alert("NONEXISTENT")
        assert not success
    
    def test_get_active_alerts(self, alerting_service, sample_alert):
        """Test getting active alerts."""
        alerting_service.send_alert(sample_alert)
        active_alerts = alerting_service.get_active_alerts()
        
        assert len(active_alerts) == 1
        assert active_alerts[0] == sample_alert
    
    def test_get_alert_history(self, alerting_service, sample_alert):
        """Test getting alert history."""
        alerting_service.send_alert(sample_alert)
        history = alerting_service.get_alert_history()
        
        assert len(history) == 1
        assert history[0] == sample_alert
    
    def test_alert_summary_no_alerts(self, alerting_service, capsys):
        """Test alert summary with no alerts."""
        alerting_service.print_alert_summary()
        
        captured = capsys.readouterr()
        assert "No alerts generated" in captured.out
    
    def test_alert_summary_with_alerts(self, alerting_service, capsys):
        """Test alert summary with multiple alerts."""
        # Create different types of alerts
        alerting_service.send_capacity_overload_alert("BOM", 1.2, 20, "30min")
        alerting_service.send_weather_impact_alert("DEL", "strong", 0.3, 15)
        alerting_service.send_autonomous_action_alert("optimization", 0.9, 10, "test")
        
        alerting_service.print_alert_summary()
        
        captured = capsys.readouterr()
        assert "ALERT ACTIVITY SUMMARY" in captured.out
        assert "Total Alerts: 3" in captured.out
        assert "By Severity:" in captured.out
        assert "By Type:" in captured.out


class TestOfflineReplayIntegration:
    """Integration tests for offline replay mode."""
    
    @pytest.mark.asyncio
    async def test_full_replay_simulation(self):
        """Test complete replay simulation workflow."""
        # This is a simplified integration test
        replay_service = OfflineReplayService()
        
        # Configure for fast test
        original_hours = settings.offline_replay.max_simulation_hours
        original_speed = settings.offline_replay.simulation_speed_multiplier
        
        settings.offline_replay.max_simulation_hours = 1
        settings.offline_replay.simulation_speed_multiplier = 100.0
        
        try:
            # Create minimal test data
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create empty directory (will use fallback data)
                success = await replay_service.initialize_replay_mode(temp_dir)
                
                # Even with no data, initialization should handle gracefully
                if success:
                    # Run very short simulation
                    replay_service.simulation_state.flights = []  # Empty for fast test
                    await replay_service._generate_simulation_events()
                    
                    # Should complete without errors
                    assert replay_service.get_simulation_status()["status"] != "not_initialized"
        
        finally:
            settings.offline_replay.max_simulation_hours = original_hours
            settings.offline_replay.simulation_speed_multiplier = original_speed
    
    def test_console_alerting_integration(self):
        """Test integration between replay service and console alerting."""
        alerting_service = ConsoleAlertingService()
        
        # Test that alerting service can handle various alert types
        alert_ids = []
        
        alert_ids.append(alerting_service.send_capacity_overload_alert("BOM", 1.1, 15, "20min"))
        alert_ids.append(alerting_service.send_weather_impact_alert("DEL", "medium", 0.2, 10))
        alert_ids.append(alerting_service.send_autonomous_action_alert("optimization", 0.8, 8, "test"))
        
        assert len(alert_ids) == 3
        assert len(alerting_service.get_active_alerts()) == 3
        
        # Resolve all alerts
        for alert_id in alert_ids:
            success = alerting_service.resolve_alert(alert_id)
            assert success
        
        assert len(alerting_service.get_active_alerts()) == 0
        assert len(alerting_service.get_alert_history()) == 3


if __name__ == "__main__":
    pytest.main([__file__])