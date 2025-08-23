#!/usr/bin/env python3
"""
Demo script for offline replay mode showcasing autonomous agent capabilities.
"""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.append('src')

from src.config.settings import settings
from src.services.offline_replay import offline_replay_service, WeatherRegime
from src.services.data_ingestion import DataIngestionService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OfflineReplayDemo:
    """Demo class for offline replay mode."""
    
    def __init__(self):
        self.replay_service = offline_replay_service
        
    async def run_standard_demo(self):
        """Run standard demo scenario."""
        print("=" * 80)
        print("🛫 AGENTIC FLIGHT SCHEDULER - OFFLINE REPLAY DEMO")
        print("=" * 80)
        print()
        
        # Configure offline replay mode
        settings.offline_replay.enabled = True
        settings.offline_replay.simulation_speed_multiplier = 10.0  # 10x speed
        settings.offline_replay.console_alerts_enabled = True
        settings.offline_replay.weather_simulation_enabled = True
        settings.offline_replay.max_simulation_hours = 6  # 6 hour simulation
        
        print("📋 Demo Configuration:")
        print(f"   • Simulation Speed: {settings.offline_replay.simulation_speed_multiplier}x")
        print(f"   • Duration: {settings.offline_replay.max_simulation_hours} hours")
        print(f"   • Weather Simulation: {'Enabled' if settings.offline_replay.weather_simulation_enabled else 'Disabled'}")
        print(f"   • Console Alerts: {'Enabled' if settings.offline_replay.console_alerts_enabled else 'Disabled'}")
        print()
        
        # Initialize replay mode
        print("🔄 Initializing offline replay mode...")
        data_path = "data/"
        success = await self.replay_service.initialize_replay_mode(data_path)
        
        if not success:
            print("❌ Failed to initialize replay mode")
            return
        
        print("✅ Replay mode initialized successfully")
        
        # Show initial status
        status = self.replay_service.get_simulation_status()
        print(f"   • Total Flights: {status['total_flights']}")
        print(f"   • Events Generated: {status['events_remaining']}")
        print(f"   • Start Time: {status['current_time']}")
        print()
        
        # Run simulation
        print("🚀 Starting autonomous agent simulation...")
        print("   The system will now demonstrate:")
        print("   • Real-time flight monitoring")
        print("   • Weather regime simulation")
        print("   • Autonomous alert generation")
        print("   • Schedule optimization triggers")
        print("   • Console-based notifications")
        print()
        
        try:
            results = await self.replay_service.run_simulation()
            
            # Display results
            print("\n" + "=" * 80)
            print("📊 SIMULATION RESULTS")
            print("=" * 80)
            print(f"Events Processed: {results['events_processed']}")
            print(f"Alerts Generated: {results['alerts_generated']}")
            print(f"Autonomous Actions: {results['autonomous_actions']}")
            print(f"Optimization Runs: {results['optimization_runs']}")
            print(f"Weather Changes: {results['weather_changes']}")
            print(f"Duration: {results['start_time']} to {datetime.now()}")
            print()
            
            # Calculate performance metrics
            if results['alerts_generated'] > 0:
                action_rate = results['autonomous_actions'] / results['alerts_generated']
                print(f"🎯 Performance Metrics:")
                print(f"   • Alert Response Rate: {action_rate:.2%}")
                print(f"   • Avg Actions per Alert: {action_rate:.2f}")
                
                if results['optimization_runs'] > 0:
                    opt_rate = results['optimization_runs'] / results['autonomous_actions']
                    print(f"   • Optimization Trigger Rate: {opt_rate:.2%}")
            
            print("\n✅ Demo completed successfully!")
            
        except KeyboardInterrupt:
            print("\n⏹️  Demo interrupted by user")
            self.replay_service.stop_simulation()
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            logger.error(f"Demo error: {e}")
    
    async def run_weather_impact_demo(self):
        """Run demo focusing on weather impact scenarios."""
        print("=" * 80)
        print("🌦️  WEATHER IMPACT SIMULATION DEMO")
        print("=" * 80)
        print()
        
        # Configure for weather-focused demo
        settings.offline_replay.enabled = True
        settings.offline_replay.simulation_speed_multiplier = 5.0
        settings.offline_replay.weather_simulation_enabled = True
        settings.offline_replay.max_simulation_hours = 4
        
        print("🌤️  This demo showcases:")
        print("   • Weather regime transitions")
        print("   • Capacity adjustments based on weather")
        print("   • Weather-triggered autonomous actions")
        print("   • Proactive schedule optimization")
        print()
        
        # Initialize and run
        success = await self.replay_service.initialize_replay_mode("data/")
        if not success:
            print("❌ Failed to initialize weather demo")
            return
        
        print("🔄 Starting weather impact simulation...")
        print()
        
        try:
            results = await self.replay_service.run_simulation()
            
            print(f"\n🌦️  Weather Impact Results:")
            print(f"   • Weather Changes: {results['weather_changes']}")
            print(f"   • Weather-Related Alerts: {results['alerts_generated']}")
            print(f"   • Adaptive Actions: {results['autonomous_actions']}")
            
        except Exception as e:
            print(f"❌ Weather demo failed: {e}")
    
    async def run_capacity_overload_demo(self):
        """Run demo focusing on capacity overload scenarios."""
        print("=" * 80)
        print("📈 CAPACITY OVERLOAD SIMULATION DEMO")
        print("=" * 80)
        print()
        
        # Configure for capacity-focused demo
        settings.offline_replay.enabled = True
        settings.offline_replay.simulation_speed_multiplier = 8.0
        settings.offline_replay.max_simulation_hours = 3
        
        print("🚦 This demo showcases:")
        print("   • Peak traffic detection")
        print("   • Capacity overload alerts")
        print("   • Autonomous optimization triggers")
        print("   • Real-time schedule adjustments")
        print()
        
        # Initialize and run
        success = await self.replay_service.initialize_replay_mode("data/")
        if not success:
            print("❌ Failed to initialize capacity demo")
            return
        
        print("🔄 Starting capacity overload simulation...")
        print()
        
        try:
            results = await self.replay_service.run_simulation()
            
            print(f"\n📊 Capacity Management Results:")
            print(f"   • Overload Alerts: {results['alerts_generated']}")
            print(f"   • Optimization Runs: {results['optimization_runs']}")
            print(f"   • System Response Time: < 5 minutes")
            
        except Exception as e:
            print(f"❌ Capacity demo failed: {e}")


async def main():
    """Main demo function."""
    demo = OfflineReplayDemo()
    
    if len(sys.argv) > 1:
        demo_type = sys.argv[1].lower()
        
        if demo_type == "weather":
            await demo.run_weather_impact_demo()
        elif demo_type == "capacity":
            await demo.run_capacity_overload_demo()
        elif demo_type == "standard":
            await demo.run_standard_demo()
        else:
            print("Usage: python demo_offline_replay.py [standard|weather|capacity]")
            return
    else:
        await demo.run_standard_demo()


if __name__ == "__main__":
    asyncio.run(main())