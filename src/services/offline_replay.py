"""Offline replay mode for reliable demo execution."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import random
from enum import Enum

from ..config.settings import settings
from ..models.flight import Flight
from .data_ingestion import DataIngestionService
from .analytics import AnalyticsEngine
from .schedule_optimizer import ScheduleOptimizer
from .whatif_simulator import WhatIfSimulator

logger = logging.getLogger(__name__)


class WeatherRegime(Enum):
    """Weather regime classifications for simulation."""
    CALM = "calm"
    MEDIUM = "medium"
    STRONG = "strong"
    SEVERE = "severe"


@dataclass
class SimulationEvent:
    """Represents an event in the simulation timeline."""
    timestamp: datetime
    event_type: str  # "flight_update", "weather_change", "capacity_change"
    data: Dict[str, Any]
    priority: int = 0  # Higher priority events processed first


@dataclass
class WeatherCondition:
    """Weather conditions for simulation."""
    regime: WeatherRegime
    visibility_km: float
    wind_speed_kts: float
    precipitation: bool
    capacity_multiplier: float


@dataclass
class SimulationState:
    """Current state of the simulation."""
    current_time: datetime
    flights: List[Flight]
    weather_conditions: Dict[str, WeatherCondition]
    runway_capacities: Dict[str, int]
    active_alerts: List[Dict[str, Any]]
    autonomous_actions: List[Dict[str, Any]]


class OfflineReplayService:
    """Service for running offline replay simulations."""
    
    def __init__(self):
        self.data_ingestion = DataIngestionService()
        self.analytics = AnalyticsEngine()
        self.optimizer = ScheduleOptimizer()
        self.whatif_simulator = WhatIfSimulator()
        
        # Initialize autonomous monitor with dependencies (will be set up later if needed)
        self.autonomous_monitor = None
        
        self.simulation_state: Optional[SimulationState] = None
        self.event_queue: List[SimulationEvent] = []
        self.is_running = False
        
    async def initialize_replay_mode(self, data_source_path: str) -> bool:
        """Initialize offline replay mode with data from specified path."""
        try:
            logger.info(f"Initializing offline replay mode with data from: {data_source_path}")
            
            # Load historical flight data
            flights = await self._load_replay_data(data_source_path)
            if not flights:
                logger.error("No flight data loaded for replay mode")
                return False
            
            # Initialize simulation state
            start_time = min(flight.std_utc for flight in flights)
            self.simulation_state = SimulationState(
                current_time=start_time,
                flights=flights,
                weather_conditions=self._initialize_weather_conditions(),
                runway_capacities=self._initialize_runway_capacities(),
                active_alerts=[],
                autonomous_actions=[]
            )
            
            # Generate simulation events
            await self._generate_simulation_events()
            
            logger.info(f"Replay mode initialized with {len(flights)} flights and {len(self.event_queue)} events")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize replay mode: {e}")
            return False
    
    async def _load_replay_data(self, data_source_path: str) -> List[Flight]:
        """Load flight data from various sources for replay."""
        flights = []
        data_path = Path(data_source_path)
        
        try:
            # Load Excel files
            excel_files = list(data_path.glob("*.xlsx"))
            if excel_files:
                logger.info(f"Loading {len(excel_files)} Excel files")
                for excel_file in excel_files:
                    file_flights = await self.data_ingestion.process_excel_file(str(excel_file))
                    flights.extend(file_flights)
            
            # Load from DuckDB if available
            duckdb_path = data_path / "flights.duckdb"
            if duckdb_path.exists():
                logger.info("Loading flights from DuckDB")
                db_flights = await self.data_ingestion.load_from_database()
                flights.extend(db_flights)
            
            # Remove duplicates based on flight_id
            unique_flights = {}
            for flight in flights:
                unique_flights[flight.flight_id] = flight
            
            return list(unique_flights.values())
            
        except Exception as e:
            logger.error(f"Error loading replay data: {e}")
            return []
    
    def _initialize_weather_conditions(self) -> Dict[str, WeatherCondition]:
        """Initialize weather conditions for major airports."""
        airports = ["BOM", "DEL", "BLR", "MAA", "CCU", "HYD"]
        conditions = {}
        
        for airport in airports:
            # Start with random weather regime
            regime = random.choice(list(WeatherRegime))
            conditions[airport] = WeatherCondition(
                regime=regime,
                visibility_km=self._get_visibility_for_regime(regime),
                wind_speed_kts=self._get_wind_speed_for_regime(regime),
                precipitation=regime in [WeatherRegime.STRONG, WeatherRegime.SEVERE],
                capacity_multiplier=self._get_capacity_multiplier_for_regime(regime)
            )
        
        return conditions
    
    def _initialize_runway_capacities(self) -> Dict[str, int]:
        """Initialize baseline runway capacities."""
        return {
            "BOM_09L": 30,
            "BOM_09R": 30,
            "BOM_14": 25,
            "BOM_32": 25,
            "DEL_10": 35,
            "DEL_28": 35,
            "DEL_11": 30,
            "DEL_29": 30
        }
    
    def _get_visibility_for_regime(self, regime: WeatherRegime) -> float:
        """Get visibility based on weather regime."""
        visibility_map = {
            WeatherRegime.CALM: random.uniform(8.0, 10.0),
            WeatherRegime.MEDIUM: random.uniform(5.0, 8.0),
            WeatherRegime.STRONG: random.uniform(2.0, 5.0),
            WeatherRegime.SEVERE: random.uniform(0.5, 2.0)
        }
        return visibility_map[regime]
    
    def _get_wind_speed_for_regime(self, regime: WeatherRegime) -> float:
        """Get wind speed based on weather regime."""
        wind_map = {
            WeatherRegime.CALM: random.uniform(0, 10),
            WeatherRegime.MEDIUM: random.uniform(10, 20),
            WeatherRegime.STRONG: random.uniform(20, 35),
            WeatherRegime.SEVERE: random.uniform(35, 50)
        }
        return wind_map[regime]
    
    def _get_capacity_multiplier_for_regime(self, regime: WeatherRegime) -> float:
        """Get capacity multiplier based on weather regime."""
        multiplier_map = {
            WeatherRegime.CALM: 1.0,
            WeatherRegime.MEDIUM: 0.9,
            WeatherRegime.STRONG: 0.7,
            WeatherRegime.SEVERE: 0.5
        }
        return multiplier_map[regime]
    
    async def _generate_simulation_events(self):
        """Generate events for the simulation timeline."""
        if not self.simulation_state:
            return
        
        flights = self.simulation_state.flights
        start_time = self.simulation_state.current_time
        end_time = start_time + timedelta(hours=settings.offline_replay.max_simulation_hours)
        
        # Generate flight events
        for flight in flights:
            if start_time <= flight.std_utc <= end_time:
                # Flight departure event
                self.event_queue.append(SimulationEvent(
                    timestamp=flight.std_utc,
                    event_type="flight_departure",
                    data={"flight": flight, "action": "departure"},
                    priority=1
                ))
                
                # Flight arrival event
                if flight.sta_utc:
                    self.event_queue.append(SimulationEvent(
                        timestamp=flight.sta_utc,
                        event_type="flight_arrival",
                        data={"flight": flight, "action": "arrival"},
                        priority=1
                    ))
        
        # Generate weather change events
        current_time = start_time
        while current_time < end_time:
            # Weather changes every 2-4 hours
            next_weather_change = current_time + timedelta(hours=random.uniform(2, 4))
            if next_weather_change <= end_time:
                self.event_queue.append(SimulationEvent(
                    timestamp=next_weather_change,
                    event_type="weather_change",
                    data={"airports": list(self.simulation_state.weather_conditions.keys())},
                    priority=2
                ))
            current_time = next_weather_change
        
        # Generate monitoring events (every 5 minutes)
        current_time = start_time
        while current_time < end_time:
            self.event_queue.append(SimulationEvent(
                timestamp=current_time,
                event_type="monitoring_check",
                data={},
                priority=3
            ))
            current_time += timedelta(minutes=5)
        
        # Sort events by timestamp and priority
        self.event_queue.sort(key=lambda x: (x.timestamp, x.priority))
        
        logger.info(f"Generated {len(self.event_queue)} simulation events")
    
    async def run_simulation(self) -> Dict[str, Any]:
        """Run the offline replay simulation."""
        if not self.simulation_state:
            raise ValueError("Simulation not initialized")
        
        self.is_running = True
        simulation_results = {
            "start_time": self.simulation_state.current_time,
            "events_processed": 0,
            "alerts_generated": 0,
            "autonomous_actions": 0,
            "optimization_runs": 0,
            "weather_changes": 0
        }
        
        try:
            logger.info("Starting offline replay simulation")
            
            for event in self.event_queue:
                if not self.is_running:
                    break
                
                # Update simulation time
                self.simulation_state.current_time = event.timestamp
                
                # Process event
                await self._process_simulation_event(event, simulation_results)
                
                # Add realistic delay between events
                if settings.offline_replay.simulation_speed_multiplier > 0:
                    await asyncio.sleep(0.1 / settings.offline_replay.simulation_speed_multiplier)
            
            logger.info(f"Simulation completed. Results: {simulation_results}")
            return simulation_results
            
        except Exception as e:
            logger.error(f"Simulation error: {e}")
            raise
        finally:
            self.is_running = False
    
    async def _process_simulation_event(self, event: SimulationEvent, results: Dict[str, Any]):
        """Process a single simulation event."""
        try:
            results["events_processed"] += 1
            
            if event.event_type == "flight_departure":
                await self._handle_flight_event(event, "departure")
            
            elif event.event_type == "flight_arrival":
                await self._handle_flight_event(event, "arrival")
            
            elif event.event_type == "weather_change":
                await self._handle_weather_change(event)
                results["weather_changes"] += 1
            
            elif event.event_type == "monitoring_check":
                alerts, actions = await self._handle_monitoring_check()
                results["alerts_generated"] += len(alerts)
                results["autonomous_actions"] += len(actions)
                
                # Trigger optimization if needed
                if actions:
                    optimization_result = await self._trigger_autonomous_optimization()
                    if optimization_result:
                        results["optimization_runs"] += 1
            
            # Log progress every 100 events
            if results["events_processed"] % 100 == 0:
                logger.info(f"Processed {results['events_processed']} events at {event.timestamp}")
                
        except Exception as e:
            logger.error(f"Error processing event {event.event_type}: {e}")
    
    async def _handle_flight_event(self, event: SimulationEvent, action: str):
        """Handle flight departure or arrival events."""
        flight = event.data["flight"]
        
        # Simulate potential delays based on weather and capacity
        airport = flight.origin if action == "departure" else flight.destination
        weather = self.simulation_state.weather_conditions.get(airport)
        
        if weather and weather.regime in [WeatherRegime.STRONG, WeatherRegime.SEVERE]:
            # Higher chance of delays in bad weather
            if random.random() < 0.3:  # 30% chance of delay
                delay_minutes = random.uniform(15, 60)
                logger.info(f"Simulated {delay_minutes:.1f}min delay for {flight.flight_no} due to {weather.regime.value} weather")
    
    async def _handle_weather_change(self, event: SimulationEvent):
        """Handle weather regime changes."""
        airports = event.data["airports"]
        
        for airport in airports:
            # Randomly change weather regime
            new_regime = random.choice(list(WeatherRegime))
            old_regime = self.simulation_state.weather_conditions[airport].regime
            
            self.simulation_state.weather_conditions[airport] = WeatherCondition(
                regime=new_regime,
                visibility_km=self._get_visibility_for_regime(new_regime),
                wind_speed_kts=self._get_wind_speed_for_regime(new_regime),
                precipitation=new_regime in [WeatherRegime.STRONG, WeatherRegime.SEVERE],
                capacity_multiplier=self._get_capacity_multiplier_for_regime(new_regime)
            )
            
            if old_regime != new_regime:
                logger.info(f"Weather change at {airport}: {old_regime.value} → {new_regime.value}")
                
                # Update runway capacities based on new weather
                await self._update_runway_capacities_for_weather(airport, new_regime)
    
    async def _update_runway_capacities_for_weather(self, airport: str, regime: WeatherRegime):
        """Update runway capacities based on weather regime."""
        multiplier = self._get_capacity_multiplier_for_regime(regime)
        
        for runway_id in self.simulation_state.runway_capacities:
            if runway_id.startswith(airport):
                base_capacity = self.simulation_state.runway_capacities[runway_id]
                new_capacity = int(base_capacity * multiplier)
                self.simulation_state.runway_capacities[runway_id] = new_capacity
    
    async def _handle_monitoring_check(self) -> tuple[List[Dict], List[Dict]]:
        """Handle autonomous monitoring checks."""
        alerts = []
        actions = []
        
        try:
            # Check for capacity overloads
            current_time = self.simulation_state.current_time
            time_window = timedelta(minutes=30)
            
            # Get flights in current time window
            window_flights = [
                f for f in self.simulation_state.flights
                if current_time <= f.std_utc <= current_time + time_window
            ]
            
            if len(window_flights) > 20:  # Threshold for overload
                alert = {
                    "timestamp": current_time,
                    "type": "capacity_overload",
                    "severity": "high",
                    "message": f"High traffic detected: {len(window_flights)} flights in next 30 minutes",
                    "affected_flights": len(window_flights)
                }
                alerts.append(alert)
                self.simulation_state.active_alerts.append(alert)
                
                # Console alert
                if settings.offline_replay.console_alerts_enabled:
                    self._print_console_alert(alert)
                
                # Autonomous action
                action = {
                    "timestamp": current_time,
                    "type": "schedule_optimization",
                    "trigger": "capacity_overload",
                    "confidence": 0.85,
                    "affected_flights": len(window_flights)
                }
                actions.append(action)
                self.simulation_state.autonomous_actions.append(action)
        
        except Exception as e:
            logger.error(f"Error in monitoring check: {e}")
        
        return alerts, actions
    
    def _print_console_alert(self, alert: Dict[str, Any]):
        """Print alert to console in Slack-style format."""
        severity_emoji = {
            "low": "🟢",
            "medium": "🟡", 
            "high": "🔴",
            "critical": "🚨"
        }
        
        emoji = severity_emoji.get(alert["severity"], "ℹ️")
        timestamp = alert["timestamp"].strftime("%H:%M:%S")
        
        print(f"\n{emoji} ALERT [{timestamp}] - {alert['type'].upper()}")
        print(f"   Severity: {alert['severity'].upper()}")
        print(f"   Message: {alert['message']}")
        if "affected_flights" in alert:
            print(f"   Affected Flights: {alert['affected_flights']}")
        print("-" * 60)
    
    async def _trigger_autonomous_optimization(self) -> Optional[Dict[str, Any]]:
        """Trigger autonomous schedule optimization."""
        try:
            current_time = self.simulation_state.current_time
            
            # Get flights for optimization (next 2 hours)
            optimization_window = timedelta(hours=2)
            optimization_flights = [
                f for f in self.simulation_state.flights
                if current_time <= f.std_utc <= current_time + optimization_window
            ]
            
            if len(optimization_flights) < 5:
                return None
            
            logger.info(f"Running autonomous optimization for {len(optimization_flights)} flights")
            
            # Simulate optimization (simplified)
            optimization_result = {
                "timestamp": current_time,
                "flights_optimized": len(optimization_flights),
                "estimated_delay_reduction": random.uniform(5, 15),
                "runway_changes": random.randint(2, 8),
                "success": True
            }
            
            print(f"\n🤖 AUTONOMOUS OPTIMIZATION [{current_time.strftime('%H:%M:%S')}]")
            print(f"   Flights Optimized: {optimization_result['flights_optimized']}")
            print(f"   Est. Delay Reduction: {optimization_result['estimated_delay_reduction']:.1f} minutes")
            print(f"   Runway Changes: {optimization_result['runway_changes']}")
            print("-" * 60)
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"Error in autonomous optimization: {e}")
            return None
    
    def stop_simulation(self):
        """Stop the running simulation."""
        self.is_running = False
        logger.info("Simulation stop requested")
    
    def get_simulation_status(self) -> Dict[str, Any]:
        """Get current simulation status."""
        if not self.simulation_state:
            return {"status": "not_initialized"}
        
        return {
            "status": "running" if self.is_running else "stopped",
            "current_time": self.simulation_state.current_time,
            "total_flights": len(self.simulation_state.flights),
            "active_alerts": len(self.simulation_state.active_alerts),
            "autonomous_actions": len(self.simulation_state.autonomous_actions),
            "events_remaining": len([e for e in self.event_queue if e.timestamp > self.simulation_state.current_time])
        }


# Global instance
offline_replay_service = OfflineReplayService()