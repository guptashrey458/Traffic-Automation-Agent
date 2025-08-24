#!/usr/bin/env python3
"""
Enhanced Demo API server for the Flight Scheduler Dashboard
This provides comprehensive mock data with robust error handling.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
from datetime import datetime, date, timedelta
import uvicorn
import random
import json
import asyncio
import logging
import traceback
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state for demo data
demo_state = {
    "alerts": [],
    "flights": {},
    "optimization_history": [],
    "system_health": "healthy"
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("🚀 Starting Enhanced Flight Scheduler API...")
    # Initialize demo data
    initialize_demo_data()
    yield
    logger.info("🛑 Shutting down Enhanced Flight Scheduler API...")

app = FastAPI(
    title="Enhanced Agentic Flight Scheduler API",
    description="Comprehensive demo API for flight scheduling dashboard with robust error handling",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def initialize_demo_data():
    """Initialize demo data on startup"""
    logger.info("Initializing demo data...")
    
    # Initialize sample flights
    flight_ids = [f"AI{2700 + i}" for i in range(20)] + [f"6E{100 + i}" for i in range(10)] + [f"UK{950 + i}" for i in range(5)]
    
    for flight_id in flight_ids:
        demo_state["flights"][flight_id] = {
            "flight_id": flight_id,
            "status": random.choice(["scheduled", "delayed", "on_time", "boarding"]),
            "delay_minutes": random.randint(0, 45) if random.random() < 0.3 else 0,
            "aircraft_type": random.choice(["A320", "A321", "B737", "B777", "A350"]),
            "origin": random.choice(["BOM", "DEL", "BLR", "MAA", "CCU", "HYD"]),
            "destination": random.choice(["BOM", "DEL", "BLR", "MAA", "CCU", "HYD"])
        }
    
    logger.info(f"Initialized {len(demo_state['flights'])} demo flights")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc) if app.debug else "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )

def generate_realistic_time_buckets(bucket_minutes: int = 10, airport: str = "BOM"):
    """Generate realistic time bucket data with airport-specific patterns"""
    buckets = []
    start_time = datetime.now().replace(hour=6, minute=0, second=0, microsecond=0)
    
    # Airport-specific capacity
    airport_capacity = {
        "BOM": 35, "DEL": 40, "BLR": 30, "MAA": 25, "CCU": 20, "HYD": 28
    }
    base_capacity = airport_capacity.get(airport, 30)
    
    for i in range(72):  # 12 hours of data
        bucket_start = start_time + timedelta(minutes=i * bucket_minutes)
        bucket_end = bucket_start + timedelta(minutes=bucket_minutes)
        hour = bucket_start.hour
        
        # Realistic traffic patterns
        if 6 <= hour <= 8 or 18 <= hour <= 20:  # Peak hours
            demand_multiplier = random.uniform(1.2, 1.5)
        elif 9 <= hour <= 11 or 14 <= hour <= 17:  # Busy hours
            demand_multiplier = random.uniform(0.8, 1.1)
        elif 12 <= hour <= 13:  # Lunch lull
            demand_multiplier = random.uniform(0.6, 0.8)
        else:  # Off-peak
            demand_multiplier = random.uniform(0.3, 0.6)
        
        base_demand = base_capacity * 0.7  # 70% of capacity as base
        demand = int(base_demand * demand_multiplier)
        capacity = base_capacity
        
        # Weather impact
        weather_impact = random.uniform(0.9, 1.1)
        effective_capacity = int(capacity * weather_impact)
        
        utilization = min(demand / effective_capacity, 1.5)
        overload = max(0, demand - effective_capacity)
        
        # Delay calculation based on overload
        if overload > 0:
            avg_delay = min(45, overload * 3 + random.uniform(5, 15))
        else:
            avg_delay = random.uniform(0, 8)
        
        buckets.append({
            "start_time": bucket_start.isoformat(),
            "end_time": bucket_end.isoformat(),
            "scheduled_departures": demand // 2,
            "actual_departures": max(0, (demand // 2) - random.randint(0, overload)),
            "scheduled_arrivals": demand // 2,
            "actual_arrivals": max(0, (demand // 2) - random.randint(0, overload)),
            "total_demand": demand,
            "capacity": effective_capacity,
            "utilization": utilization,
            "overload": overload,
            "avg_delay": avg_delay,
            "delayed_flights": min(demand, int(overload * 1.5 + random.randint(0, 3))),
            "traffic_level": (
                "critical" if utilization > 1.2 else
                "high" if utilization > 1.0 else
                "medium" if utilization > 0.7 else
                "low"
            )
        })
    
    return buckets

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Enhanced Agentic Flight Scheduler API",
        "version": "1.0.0",
        "status": "ready",
        "features": [
            "Real-time flight analytics",
            "AI-powered delay prediction",
            "Schedule optimization",
            "What-if analysis",
            "Alert management",
            "Algorithm transparency"
        ],
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "analytics": "/flights/peaks",
            "optimization": "/optimize",
            "delay_prediction": "/flights/risks",
            "whatif": "/whatif",
            "alerts": "/alerts/active"
        }
    }

@app.get("/health")
async def health_check():
    """Enhanced health check with system details"""
    return {
        "status": "healthy",
        "environment": "demo",
        "timestamp": datetime.now().isoformat(),
        "uptime": "running",
        "version": "1.0.0",
        "services": {
            "api": "healthy",
            "database": "healthy",
            "analytics": "healthy",
            "ml_models": "healthy"
        }
    }

@app.get("/airports")
async def get_airports():
    """Get supported airports with detailed information"""
    return {
        "airports": [
            {
                "code": "BOM",
                "name": "Chhatrapati Shivaji Maharaj International Airport",
                "city": "Mumbai",
                "country": "India",
                "timezone": "Asia/Kolkata",
                "runways": ["09L/27R", "09R/27L"],
                "capacity_per_hour": 35
            },
            {
                "code": "DEL",
                "name": "Indira Gandhi International Airport",
                "city": "Delhi",
                "country": "India",
                "timezone": "Asia/Kolkata",
                "runways": ["10/28", "11/29", "09/27"],
                "capacity_per_hour": 40
            },
            {
                "code": "BLR",
                "name": "Kempegowda International Airport",
                "city": "Bangalore",
                "country": "India",
                "timezone": "Asia/Kolkata",
                "runways": ["09/27", "09R/27L"],
                "capacity_per_hour": 30
            },
            {
                "code": "MAA",
                "name": "Chennai International Airport",
                "city": "Chennai",
                "country": "India",
                "timezone": "Asia/Kolkata",
                "runways": ["07/25", "12/30"],
                "capacity_per_hour": 25
            },
            {
                "code": "CCU",
                "name": "Netaji Subhas Chandra Bose International Airport",
                "city": "Kolkata",
                "country": "India",
                "timezone": "Asia/Kolkata",
                "runways": ["01L/19R", "01R/19L"],
                "capacity_per_hour": 20
            },
            {
                "code": "HYD",
                "name": "Rajiv Gandhi International Airport",
                "city": "Hyderabad",
                "country": "India",
                "timezone": "Asia/Kolkata",
                "runways": ["09L/27R", "09R/27L"],
                "capacity_per_hour": 28
            }
        ]
    }

@app.get("/status")
async def get_system_status():
    """Get comprehensive system status"""
    return {
        "status": demo_state["system_health"],
        "services": {
            "database": "healthy",
            "analytics_engine": "healthy",
            "optimization_engine": "healthy",
            "delay_predictor": "healthy",
            "whatif_simulator": "healthy",
            "alert_manager": "healthy",
            "notification_service": "healthy"
        },
        "metrics": {
            "active_flights": len([f for f in demo_state["flights"].values() if f["status"] in ["scheduled", "boarding"]]),
            "delayed_flights": len([f for f in demo_state["flights"].values() if f["delay_minutes"] > 0]),
            "active_alerts": len(demo_state["alerts"]),
            "optimizations_today": len(demo_state["optimization_history"])
        },
        "timestamp": datetime.now().isoformat(),
        "api_version": "1.0.0"
    }

@app.get("/flights/peaks")
async def get_flight_peaks(
    airport: str = "BOM",
    bucket_minutes: int = 10,
    date: Optional[str] = None,
    weather_regime: str = "calm"
):
    """Enhanced peak analysis with realistic data"""
    try:
        logger.info(f"Generating peak analysis for {airport} with {bucket_minutes}min buckets")
        
        time_buckets = generate_realistic_time_buckets(bucket_minutes, airport)
        
        # Generate overload windows
        overload_windows = []
        for i, bucket in enumerate(time_buckets):
            if bucket["overload"] > 0:
                severity = "critical" if bucket["overload"] > 10 else "high" if bucket["overload"] > 5 else "medium"
                overload_windows.append({
                    "start_time": bucket["start_time"],
                    "end_time": bucket["end_time"],
                    "duration_minutes": bucket_minutes,
                    "peak_overload": bucket["overload"],
                    "affected_flights": min(bucket["overload"] * 2, 15),
                    "severity": severity,
                    "estimated_delay": bucket["avg_delay"]
                })
        
        # Calculate comprehensive metrics
        total_utilization = sum(bucket["utilization"] for bucket in time_buckets)
        avg_utilization = total_utilization / len(time_buckets)
        peak_utilization = max(bucket["utilization"] for bucket in time_buckets)
        total_delays = sum(bucket["avg_delay"] for bucket in time_buckets)
        avg_delay = total_delays / len(time_buckets)
        
        # Generate intelligent recommendations
        recommendations = []
        if peak_utilization > 1.2:
            recommendations.append("Critical: Consider implementing ground delay programs during peak hours")
        if avg_utilization > 0.9:
            recommendations.append("High utilization detected: Optimize turnaround times to increase effective capacity")
        if len(overload_windows) > 3:
            recommendations.append("Multiple overload periods: Implement dynamic slot allocation")
        if avg_delay > 15:
            recommendations.append("High average delays: Review scheduling patterns and add buffer time")
        
        recommendations.extend([
            f"Weather regime '{weather_regime}': Adjust capacity planning accordingly",
            "Consider implementing Collaborative Decision Making (CDM) processes",
            "Monitor real-time conditions and adjust operations proactively"
        ])
        
        return {
            "airport": airport,
            "bucket_minutes": bucket_minutes,
            "analysis_date": date or datetime.now().date().isoformat(),
            "time_buckets": time_buckets,
            "overload_windows": overload_windows,
            "capacity_utilization": avg_utilization,
            "peak_utilization": peak_utilization,
            "average_delay": avg_delay,
            "total_overload_periods": len(overload_windows),
            "recommendations": recommendations,
            "weather_regime": weather_regime,
            "analysis_metadata": {
                "generated_at": datetime.now().isoformat(),
                "data_quality": "high",
                "confidence": 0.92
            }
        }
        
    except Exception as e:
        logger.error(f"Error in peak analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate peak analysis: {str(e)}")

@app.post("/optimize")
async def optimize_schedule(request: Dict[str, Any]):
    """Enhanced optimization with detailed algorithm information"""
    try:
        logger.info("Starting schedule optimization...")
        
        # Simulate processing time
        await asyncio.sleep(random.uniform(1.5, 3.0))
        
        airport = request.get("airport", "BOM")
        objectives = request.get("objectives", {})
        
        # Generate realistic optimization results
        original_metrics = {
            "total_flights": random.randint(200, 300),
            "avg_delay_minutes": random.uniform(15, 25),
            "on_time_performance": random.uniform(0.65, 0.80),
            "fuel_cost_usd": random.randint(100000, 150000),
            "co2_emissions_kg": random.randint(40000, 60000),
            "runway_utilization": random.uniform(0.80, 0.95)
        }
        
        # Calculate improvements
        delay_improvement = random.uniform(0.2, 0.4)
        otp_improvement = random.uniform(0.1, 0.2)
        fuel_improvement = random.uniform(0.08, 0.15)
        co2_improvement = random.uniform(0.08, 0.15)
        
        optimized_metrics = {
            "total_flights": original_metrics["total_flights"],
            "avg_delay_minutes": original_metrics["avg_delay_minutes"] * (1 - delay_improvement),
            "on_time_performance": min(0.95, original_metrics["on_time_performance"] + otp_improvement),
            "fuel_cost_usd": original_metrics["fuel_cost_usd"] * (1 - fuel_improvement),
            "co2_emissions_kg": original_metrics["co2_emissions_kg"] * (1 - co2_improvement),
            "runway_utilization": min(1.0, original_metrics["runway_utilization"] + random.uniform(0.02, 0.08))
        }
        
        # Generate detailed recommended changes
        recommended_changes = []
        num_changes = random.randint(12, 25)
        
        for i in range(num_changes):
            flight_id = random.choice(list(demo_state["flights"].keys()))
            change_type = random.choice(["time_shift", "runway_change", "gate_change", "aircraft_swap"])
            
            # Generate realistic time changes
            if change_type == "time_shift":
                time_change = random.randint(-30, 30)
                original_time = (datetime.now() + timedelta(hours=random.randint(1, 12))).strftime("%H:%M")
                recommended_time = (datetime.now() + timedelta(hours=random.randint(1, 12), minutes=time_change)).strftime("%H:%M")
                change_value = f"{time_change:+d} minutes"
            else:
                original_time = "Current"
                recommended_time = "Optimized"
                change_value = random.choice(["09L", "09R", "27L", "27R", "T1-A3", "T2-B5", "VT-ABC", "VT-XYZ"])
            
            # Calculate optimization scores
            delay_score = random.uniform(0.6, 0.95)
            fuel_score = random.uniform(0.7, 0.9)
            capacity_score = random.uniform(0.8, 0.95)
            environmental_score = random.uniform(0.75, 0.92)
            
            # Weighted optimization score
            weights = {"delay": 0.4, "fuel": 0.25, "capacity": 0.2, "environmental": 0.15}
            optimization_score = (
                delay_score * weights["delay"] +
                fuel_score * weights["fuel"] +
                capacity_score * weights["capacity"] +
                environmental_score * weights["environmental"]
            )
            
            recommended_changes.append({
                "flight_id": flight_id,
                "change_type": change_type,
                "change_value": change_value,
                "original_time": original_time,
                "recommended_time": recommended_time,
                "impact_description": f"Optimize {change_type.replace('_', ' ')} to reduce system-wide delays",
                "delay_reduction_minutes": random.uniform(3, 18),
                "fuel_savings_usd": random.uniform(200, 800),
                "co2_reduction_kg": random.uniform(50, 300),
                "optimization_score": optimization_score,
                "algorithm_factors": {
                    "delay_score": delay_score,
                    "fuel_score": fuel_score,
                    "capacity_score": capacity_score,
                    "environmental_score": environmental_score
                },
                "formula": f"Score = (Delay×{weights['delay']}) + (Fuel×{weights['fuel']}) + (Capacity×{weights['capacity']}) + (Env×{weights['environmental']})",
                "calculation": f"({delay_score:.3f}×{weights['delay']}) + ({fuel_score:.3f}×{weights['fuel']}) + ({capacity_score:.3f}×{weights['capacity']}) + ({environmental_score:.3f}×{weights['environmental']}) = {optimization_score:.3f}",
                "confidence": random.uniform(0.8, 0.95)
            })
        
        # Sort by optimization score
        recommended_changes.sort(key=lambda x: x["optimization_score"], reverse=True)
        
        optimization_id = f"opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        cost_reduction = (original_metrics["fuel_cost_usd"] - optimized_metrics["fuel_cost_usd"])
        
        result = {
            "optimization_id": optimization_id,
            "status": "completed",
            "airport": airport,
            "original_metrics": original_metrics,
            "optimized_metrics": optimized_metrics,
            "recommended_changes": recommended_changes,
            "cost_reduction": cost_reduction,
            "execution_time_seconds": random.uniform(2.1, 3.8),
            "algorithm_info": {
                "method": "Multi-Objective Genetic Algorithm with Simulated Annealing",
                "objective_function": "Minimize: α×Delay + β×Fuel_Cost + γ×CO2 + δ×Capacity_Violation + ε×Passenger_Impact",
                "constraints": [
                    "Minimum_Turnaround_Time ≥ Aircraft_Type_Requirement",
                    "Runway_Capacity ≤ Weather_Adjusted_Maximum",
                    "Gate_Availability = True",
                    "Crew_Rest_Time ≥ Regulatory_Minimum",
                    "Passenger_Connection_Time ≥ Minimum_Connect_Time"
                ],
                "parameters": {
                    "population_size": 150,
                    "generations": 75,
                    "mutation_rate": 0.12,
                    "crossover_rate": 0.85,
                    "elite_size": 15,
                    "temperature_initial": 1000,
                    "cooling_rate": 0.95
                },
                "convergence_criteria": "Improvement < 0.05% for 15 consecutive generations",
                "solution_quality": f"Near-optimal (within {random.uniform(1.2, 2.8):.1f}% of theoretical optimum)",
                "features_considered": [
                    "Historical delay patterns",
                    "Weather forecast impact",
                    "Aircraft performance characteristics",
                    "Crew scheduling constraints",
                    "Passenger connection requirements",
                    "Fuel consumption models",
                    "Environmental impact factors"
                ]
            },
            "performance_gains": {
                "delay_reduction_percent": delay_improvement * 100,
                "otp_improvement_percent": otp_improvement * 100,
                "fuel_savings_percent": fuel_improvement * 100,
                "co2_reduction_percent": co2_improvement * 100,
                "estimated_passenger_satisfaction_increase": random.uniform(8, 15)
            }
        }
        
        # Store in history
        demo_state["optimization_history"].append({
            "id": optimization_id,
            "timestamp": datetime.now().isoformat(),
            "airport": airport,
            "improvements": result["performance_gains"]
        })
        
        logger.info(f"Optimization completed: {optimization_id}")
        return result
        
    except Exception as e:
        logger.error(f"Error in optimization: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")@app
.post("/whatif")
async def whatif_analysis(request: Dict[str, Any]):
    """Enhanced what-if analysis with detailed impact modeling"""
    try:
        flight_id = request.get("flight_id", "AI2739")
        change_type = request.get("change_type", "time_shift")
        change_value = request.get("change_value", "+15")
        airport = request.get("airport", "BOM")
        
        logger.info(f"Running what-if analysis: {flight_id} {change_type} {change_value}")
        
        # Simulate processing time
        await asyncio.sleep(random.uniform(0.8, 1.5))
        
        # Parse change value for time shifts
        if change_type == "time_shift":
            try:
                time_change = float(change_value.replace('+', '').replace('-', '').replace('m', ''))
                if change_value.startswith('-'):
                    time_change = -time_change
            except:
                time_change = 15
        else:
            time_change = random.uniform(-10, 20)
        
        # Calculate cascading effects
        base_delay_change = time_change
        
        # Network effect multiplier based on flight importance
        network_multiplier = random.uniform(0.3, 0.8)
        system_delay_change = base_delay_change * network_multiplier
        
        # Calculate affected flights based on change magnitude
        if abs(time_change) > 20:
            affected_flights_count = random.randint(8, 15)
        elif abs(time_change) > 10:
            affected_flights_count = random.randint(4, 10)
        else:
            affected_flights_count = random.randint(2, 6)
        
        # Environmental impact calculation
        co2_per_minute = 2.3  # kg CO2 per minute of delay
        co2_change = system_delay_change * co2_per_minute
        
        # Fuel cost impact
        fuel_cost_per_minute = 12.5  # USD per minute
        fuel_cost_change = abs(system_delay_change) * fuel_cost_per_minute
        
        # Passenger impact
        avg_passengers_per_flight = 150
        passenger_impact = affected_flights_count * avg_passengers_per_flight
        
        # Generate affected flights list
        available_flights = list(demo_state["flights"].keys())
        affected_flights = random.sample(available_flights, min(affected_flights_count, len(available_flights)))
        
        # Calculate confidence based on change type and magnitude
        if change_type == "time_shift" and abs(time_change) <= 30:
            confidence = random.uniform(0.85, 0.95)
        elif change_type in ["runway_change", "gate_change"]:
            confidence = random.uniform(0.75, 0.88)
        else:
            confidence = random.uniform(0.70, 0.85)
        
        result = {
            "flight_id": flight_id,
            "change_type": change_type,
            "change_value": change_value,
            "change_description": f"Flight {flight_id} {change_type.replace('_', ' ')}: {change_value}",
            "impact_summary": {
                "delay_change": system_delay_change,
                "co2_change": co2_change,
                "fuel_cost_change": fuel_cost_change if system_delay_change > 0 else -fuel_cost_change,
                "confidence": confidence,
                "affected_flights": affected_flights_count,
                "passenger_impact": passenger_impact
            },
            "before_metrics": {
                "avg_delay": random.uniform(12, 18),
                "peak_overload": random.randint(2, 5),
                "on_time_performance": random.uniform(0.75, 0.85),
                "fuel_consumption": random.randint(8000, 12000)
            },
            "after_metrics": {
                "avg_delay": max(0, random.uniform(12, 18) + system_delay_change),
                "peak_overload": max(0, random.randint(2, 5) + (1 if system_delay_change > 10 else 0)),
                "on_time_performance": max(0.5, min(0.95, random.uniform(0.75, 0.85) - (system_delay_change * 0.01))),
                "fuel_consumption": random.randint(8000, 12000) + int(abs(system_delay_change) * 50)
            },
            "affected_flights": affected_flights,
            "detailed_impacts": {
                "primary_flight": {
                    "flight_id": flight_id,
                    "direct_delay_change": base_delay_change,
                    "probability_of_delay": min(0.95, max(0.05, 0.3 + abs(time_change) * 0.02))
                },
                "secondary_effects": [
                    {
                        "category": "Downstream flights",
                        "impact": f"{affected_flights_count - 2} flights affected by gate/crew constraints",
                        "severity": "medium" if affected_flights_count < 8 else "high"
                    },
                    {
                        "category": "Airport capacity",
                        "impact": f"Runway utilization change: {system_delay_change * 0.1:.1f}%",
                        "severity": "low" if abs(system_delay_change) < 10 else "medium"
                    },
                    {
                        "category": "Passenger connections",
                        "impact": f"~{int(passenger_impact * 0.15)} passengers may miss connections",
                        "severity": "medium" if passenger_impact > 500 else "low"
                    }
                ]
            },
            "algorithm_details": {
                "model": "Cascading Impact Network Analysis",
                "factors_considered": [
                    "Aircraft turnaround constraints",
                    "Crew scheduling dependencies",
                    "Gate availability windows",
                    "Passenger connection requirements",
                    "Downstream flight impacts",
                    "Airport capacity utilization"
                ],
                "propagation_formula": "Impact = Direct_Change × Network_Multiplier × Confidence_Factor",
                "network_multiplier": network_multiplier,
                "confidence_factors": {
                    "time_accuracy": 0.92,
                    "network_modeling": 0.87,
                    "external_factors": 0.83
                }
            },
            "recommendations": [
                f"Monitor {flight_id} closely for actual vs predicted impact",
                "Consider notifying affected passengers proactively",
                "Prepare contingency plans for downstream flights",
                "Coordinate with ground operations for resource allocation"
            ],
            "analysis_metadata": {
                "generated_at": datetime.now().isoformat(),
                "analysis_duration_ms": random.randint(800, 1500),
                "data_sources": ["flight_schedule", "historical_patterns", "real_time_status"],
                "model_version": "2.1.3"
            }
        }
        
        logger.info(f"What-if analysis completed for {flight_id}")
        return result
        
    except Exception as e:
        logger.error(f"Error in what-if analysis: {e}")
        raise HTTPException(status_code=500, detail=f"What-if analysis failed: {str(e)}")

@app.get("/flights/risks")
async def get_delay_risks(
    airport: str = "BOM",
    date: Optional[str] = None,
    flight_ids: Optional[str] = None,
    risk_threshold: float = 0.2
):
    """Enhanced delay risk prediction with comprehensive algorithm details"""
    try:
        logger.info(f"Generating delay risk predictions for {airport}")
        
        # Get flight list
        if flight_ids:
            flight_list = [fid.strip() for fid in flight_ids.split(',')]
        else:
            # Get flights for the specified airport
            airport_flights = [fid for fid, flight in demo_state["flights"].items() 
                             if flight["origin"] == airport or flight["destination"] == airport]
            flight_list = random.sample(airport_flights, min(12, len(airport_flights)))
        
        risks = []
        
        for flight_id in flight_list:
            flight_data = demo_state["flights"].get(flight_id, {})
            
            # Generate realistic risk factors
            base_delay_prob = random.uniform(0.15, 0.85)
            
            # Environmental factors
            weather_factor = random.uniform(0.8, 1.4)
            traffic_factor = random.uniform(0.85, 1.6)
            time_factor = random.uniform(0.7, 1.3)
            aircraft_factor = random.uniform(0.9, 1.15)
            airline_factor = random.uniform(0.85, 1.2)
            route_factor = random.uniform(0.9, 1.1)
            
            # Calculate departure risk using enhanced formula
            departure_risk = min(0.95, base_delay_prob * weather_factor * traffic_factor * time_factor * aircraft_factor)
            
            # Arrival risk includes route complexity
            arrival_risk = min(0.95, departure_risk * random.uniform(0.8, 1.3) * route_factor)
            
            # Expected delay calculations with realistic bounds
            expected_dep_delay = max(0, departure_risk * 50 + random.uniform(-8, 12))
            expected_arr_delay = max(0, arrival_risk * 55 + random.uniform(-10, 15))
            
            # Confidence calculation based on data quality
            data_quality_factors = {
                "historical_data": random.uniform(0.85, 0.95),
                "weather_forecast": random.uniform(0.80, 0.92),
                "traffic_prediction": random.uniform(0.75, 0.88),
                "aircraft_status": random.uniform(0.90, 0.98)
            }
            
            confidence = sum(data_quality_factors.values()) / len(data_quality_factors)
            
            # Risk level classification
            def classify_risk(risk_score):
                if risk_score > 0.8:
                    return "critical"
                elif risk_score > 0.6:
                    return "high"
                elif risk_score > 0.35:
                    return "medium"
                else:
                    return "low"
            
            # Contributing factors based on risk level
            dep_factors = ["weather", "traffic_congestion", "aircraft_availability"]
            arr_factors = ["downstream_delays", "weather", "airspace_congestion"]
            
            if departure_risk > 0.6:
                dep_factors.extend(["crew_scheduling", "maintenance_delays"])
            if arrival_risk > 0.6:
                arr_factors.extend(["destination_weather", "atc_delays"])
            
            # Generate recommendations
            recommendations = []
            if departure_risk > 0.7:
                recommendations.extend([
                    "Consider 20-minute departure buffer",
                    "Monitor weather conditions closely",
                    "Prepare backup aircraft if available"
                ])
            elif departure_risk > 0.4:
                recommendations.extend([
                    "Consider 10-minute departure buffer",
                    "Monitor traffic conditions"
                ])
            else:
                recommendations.append("Standard monitoring sufficient")
            
            if departure_risk >= risk_threshold:
                risks.append({
                    "flight_id": flight_id,
                    "aircraft_type": flight_data.get("aircraft_type", "A320"),
                    "route": f"{flight_data.get('origin', 'BOM')} → {flight_data.get('destination', 'DEL')}",
                    "departure_risk": {
                        "risk_score": departure_risk,
                        "expected_delay_minutes": expected_dep_delay,
                        "confidence": confidence,
                        "risk_level": classify_risk(departure_risk),
                        "contributing_factors": dep_factors,
                        "algorithm_factors": {
                            "base_probability": base_delay_prob,
                            "weather_factor": weather_factor,
                            "traffic_factor": traffic_factor,
                            "time_factor": time_factor,
                            "aircraft_factor": aircraft_factor,
                            "airline_factor": airline_factor
                        },
                        "formula": "Risk = Base_Prob × Weather × Traffic × Time × Aircraft × Airline",
                        "calculation": f"{base_delay_prob:.3f} × {weather_factor:.3f} × {traffic_factor:.3f} × {time_factor:.3f} × {aircraft_factor:.3f} × {airline_factor:.3f} = {departure_risk:.3f}",
                        "delay_range": {
                            "min": max(0, expected_dep_delay - 10),
                            "max": expected_dep_delay + 15,
                            "p90": expected_dep_delay + 8
                        }
                    },
                    "arrival_risk": {
                        "risk_score": arrival_risk,
                        "expected_delay_minutes": expected_arr_delay,
                        "confidence": confidence * 0.95,  # Slightly lower for arrival
                        "risk_level": classify_risk(arrival_risk),
                        "contributing_factors": arr_factors,
                        "formula": "Arrival_Risk = Departure_Risk × Route_Complexity × Propagation_Factor",
                        "calculation": f"{departure_risk:.3f} × {route_factor:.3f} × {random.uniform(0.8, 1.3):.3f} = {arrival_risk:.3f}",
                        "delay_range": {
                            "min": max(0, expected_arr_delay - 12),
                            "max": expected_arr_delay + 18,
                            "p90": expected_arr_delay + 10
                        }
                    },
                    "risk_factors": [
                        "Peak hour operations" if random.random() < 0.6 else "Off-peak operations",
                        "Weather conditions",
                        "Traffic density",
                        "Aircraft maintenance status"
                    ],
                    "recommendations": recommendations,
                    "model_info": {
                        "algorithm": "Enhanced LightGBM with Ensemble Methods",
                        "features_used": [
                            "time_of_day", "day_of_week", "weather_conditions",
                            "traffic_density", "aircraft_type", "airline_performance",
                            "route_complexity", "historical_patterns", "crew_scheduling",
                            "maintenance_status", "passenger_load", "fuel_planning"
                        ],
                        "training_accuracy": random.uniform(0.88, 0.96),
                        "validation_accuracy": random.uniform(0.85, 0.93),
                        "last_updated": "2024-01-15T08:00:00Z",
                        "model_version": "3.2.1",
                        "training_data_size": "2.3M flight records",
                        "feature_importance": {
                            "weather_conditions": 0.23,
                            "traffic_density": 0.19,
                            "time_of_day": 0.16,
                            "aircraft_type": 0.12,
                            "route_complexity": 0.11,
                            "historical_patterns": 0.10,
                            "other_factors": 0.09
                        }
                    },
                    "prediction_metadata": {
                        "generated_at": datetime.now().isoformat(),
                        "prediction_horizon": "4 hours",
                        "data_freshness": "< 5 minutes",
                        "external_data_sources": ["weather_api", "atc_system", "airline_ops"],
                        "uncertainty_bounds": {
                            "departure": f"±{random.uniform(5, 12):.1f} minutes",
                            "arrival": f"±{random.uniform(8, 15):.1f} minutes"
                        }
                    }
                })
        
        logger.info(f"Generated {len(risks)} delay risk predictions")
        
        return {
            "airport": airport,
            "analysis_date": date or datetime.now().date().isoformat(),
            "risk_threshold": risk_threshold,
            "total_flights_analyzed": len(flight_list),
            "high_risk_flights": len([r for r in risks if r["departure_risk"]["risk_level"] in ["high", "critical"]]),
            "risks": risks,
            "summary_statistics": {
                "average_departure_risk": sum(r["departure_risk"]["risk_score"] for r in risks) / len(risks) if risks else 0,
                "average_arrival_risk": sum(r["arrival_risk"]["risk_score"] for r in risks) / len(risks) if risks else 0,
                "total_expected_delay_minutes": sum(r["departure_risk"]["expected_delay_minutes"] for r in risks),
                "model_confidence": sum(r["departure_risk"]["confidence"] for r in risks) / len(risks) if risks else 0
            },
            "analysis_metadata": {
                "generated_at": datetime.now().isoformat(),
                "processing_time_ms": random.randint(200, 800),
                "api_version": "1.0.0"
            }
        }
        
    except Exception as e:
        logger.error(f"Error in delay risk prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Delay risk prediction failed: {str(e)}")@
app.get("/alerts/active")
async def get_active_alerts(airport: Optional[str] = None, severity: Optional[str] = None):
    """Get active alerts with comprehensive details"""
    try:
        # Generate dynamic alerts based on current conditions
        current_time = datetime.now()
        alerts = []
        
        # Capacity overload alert
        if random.random() < 0.7:  # 70% chance
            alerts.append({
                "alert_id": f"cap_{current_time.strftime('%H%M%S')}",
                "alert_type": "capacity_overload",
                "severity": random.choice(["high", "medium"]),
                "title": "Runway Capacity Approaching Limits",
                "description": f"Runway 09L at {airport or 'BOM'} experiencing {random.randint(95, 125)}% capacity utilization",
                "airport": airport or "BOM",
                "timestamp": (current_time - timedelta(minutes=random.randint(5, 30))).isoformat(),
                "affected_flights": [f"AI{2700 + i}" for i in range(random.randint(3, 8))],
                "recommendations": [
                    {
                        "action": "Implement 3-minute spacing increase",
                        "impact": "Reduce overload to manageable levels",
                        "priority": 1,
                        "estimated_improvement": "20% delay reduction",
                        "implementation_time": "5 minutes"
                    },
                    {
                        "action": "Activate secondary runway",
                        "impact": "Increase total capacity by 40%",
                        "priority": 2,
                        "estimated_improvement": "Eliminate overload",
                        "implementation_time": "15 minutes"
                    }
                ],
                "metrics": {
                    "current_utilization": random.randint(95, 125),
                    "threshold_utilization": 100,
                    "affected_flights_count": random.randint(8, 15),
                    "estimated_delay_increase": random.randint(5, 20)
                },
                "resolved": False,
                "escalated": False,
                "auto_actions_available": True
            })
        
        # Weather impact alert
        if random.random() < 0.5:  # 50% chance
            alerts.append({
                "alert_id": f"wx_{current_time.strftime('%H%M%S')}",
                "alert_type": "weather_impact",
                "severity": random.choice(["medium", "low"]),
                "title": "Weather Conditions Affecting Operations",
                "description": f"Crosswinds at {random.randint(15, 35)} knots reducing effective runway capacity",
                "airport": airport or "BOM",
                "timestamp": (current_time - timedelta(minutes=random.randint(2, 15))).isoformat(),
                "affected_flights": [f"6E{100 + i}" for i in range(random.randint(2, 5))],
                "recommendations": [
                    {
                        "action": "Switch to alternate runway configuration",
                        "impact": "Maintain 90% of normal capacity",
                        "priority": 1,
                        "estimated_improvement": "Prevent 15% capacity reduction",
                        "implementation_time": "10 minutes"
                    }
                ],
                "metrics": {
                    "wind_speed_kts": random.randint(15, 35),
                    "wind_direction": random.randint(180, 270),
                    "visibility_km": random.uniform(5, 10),
                    "capacity_reduction_percent": random.randint(10, 25)
                },
                "resolved": False,
                "escalated": False,
                "weather_forecast": {
                    "next_hour": "Conditions expected to improve",
                    "trend": "improving",
                    "confidence": 0.85
                }
            })
        
        # System performance alert
        if random.random() < 0.3:  # 30% chance
            alerts.append({
                "alert_id": f"sys_{current_time.strftime('%H%M%S')}",
                "alert_type": "system_performance",
                "severity": "low",
                "title": "ML Model Performance Degradation",
                "description": "Delay prediction accuracy dropped below 90% threshold",
                "airport": airport or "ALL",
                "timestamp": (current_time - timedelta(minutes=random.randint(10, 45))).isoformat(),
                "affected_flights": [],
                "recommendations": [
                    {
                        "action": "Retrain model with recent data",
                        "impact": "Restore prediction accuracy",
                        "priority": 3,
                        "estimated_improvement": "Return to >92% accuracy",
                        "implementation_time": "30 minutes"
                    }
                ],
                "metrics": {
                    "current_accuracy": random.uniform(0.85, 0.89),
                    "threshold_accuracy": 0.90,
                    "data_drift_score": random.uniform(0.15, 0.35),
                    "last_retrain": "2024-01-10T14:30:00Z"
                },
                "resolved": False,
                "escalated": False,
                "auto_resolution": {
                    "available": True,
                    "estimated_time": "25 minutes",
                    "confidence": 0.92
                }
            })
        
        # Filter alerts
        filtered_alerts = alerts
        if airport and airport != "ALL":
            filtered_alerts = [a for a in filtered_alerts if a["airport"] == airport]
        if severity:
            filtered_alerts = [a for a in filtered_alerts if a["severity"] == severity]
        
        # Update global state
        demo_state["alerts"] = filtered_alerts
        
        return filtered_alerts
        
    except Exception as e:
        logger.error(f"Error getting active alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")

@app.get("/alerts/summary")
async def get_alert_summary(airport: Optional[str] = None):
    """Get comprehensive alert summary"""
    try:
        # Get current alerts
        alerts = await get_active_alerts(airport)
        
        # Calculate summary statistics
        total_alerts = len(alerts)
        by_severity = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        by_type = {}
        escalated_count = 0
        
        oldest_alert = None
        newest_alert = None
        
        for alert in alerts:
            # Count by severity
            severity = alert.get("severity", "unknown")
            if severity in by_severity:
                by_severity[severity] += 1
            
            # Count by type
            alert_type = alert.get("alert_type", "unknown")
            by_type[alert_type] = by_type.get(alert_type, 0) + 1
            
            # Count escalated
            if alert.get("escalated", False):
                escalated_count += 1
            
            # Track oldest and newest
            alert_time = datetime.fromisoformat(alert["timestamp"].replace('Z', '+00:00'))
            if oldest_alert is None or alert_time < oldest_alert:
                oldest_alert = alert_time
            if newest_alert is None or alert_time > newest_alert:
                newest_alert = alert_time
        
        return {
            "total_active_alerts": total_alerts,
            "by_severity": by_severity,
            "by_type": by_type,
            "escalated_alerts": escalated_count,
            "oldest_alert": oldest_alert.isoformat() if oldest_alert else None,
            "most_recent_alert": newest_alert.isoformat() if newest_alert else None,
            "airport_filter": airport,
            "summary_generated_at": datetime.now().isoformat(),
            "trends": {
                "alerts_last_hour": random.randint(0, 5),
                "average_resolution_time": f"{random.randint(15, 45)} minutes",
                "auto_resolved_percentage": random.uniform(0.6, 0.8)
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting alert summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get alert summary: {str(e)}")

@app.post("/alerts/check")
async def check_alerts(request: Dict[str, Any]):
    """Check for new alerts with intelligent detection"""
    try:
        airport = request.get("airport", "BOM")
        force_check = request.get("force_check", False)
        
        logger.info(f"Checking for new alerts at {airport}")
        
        # Simulate alert detection logic
        new_alerts = []
        
        if force_check or random.random() < 0.3:  # 30% chance of new alerts
            # Generate a new alert
            alert_types = [
                {
                    "type": "sudden_weather_change",
                    "severity": "high",
                    "title": "Sudden Weather Deterioration",
                    "description": "Unexpected thunderstorm cell approaching airport"
                },
                {
                    "type": "equipment_failure",
                    "severity": "medium",
                    "title": "Ground Equipment Malfunction",
                    "description": "Jetbridge 12 experiencing technical difficulties"
                },
                {
                    "type": "traffic_spike",
                    "severity": "medium",
                    "title": "Unexpected Traffic Increase",
                    "description": "Diverted flights increasing airport demand by 25%"
                }
            ]
            
            selected_alert = random.choice(alert_types)
            new_alert = {
                "alert_id": f"new_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "alert_type": selected_alert["type"],
                "severity": selected_alert["severity"],
                "title": selected_alert["title"],
                "description": selected_alert["description"],
                "airport": airport,
                "timestamp": datetime.now().isoformat(),
                "affected_flights": [f"AI{2700 + i}" for i in range(random.randint(1, 4))],
                "recommendations": [
                    {
                        "action": "Immediate assessment required",
                        "impact": "Prevent operational disruption",
                        "priority": 1,
                        "estimated_improvement": "Maintain normal operations"
                    }
                ],
                "resolved": False,
                "escalated": False,
                "detection_method": "automated_monitoring"
            }
            
            new_alerts.append(new_alert)
            demo_state["alerts"].append(new_alert)
        
        return {
            "new_alerts_found": len(new_alerts),
            "new_alerts": new_alerts,
            "check_timestamp": datetime.now().isoformat(),
            "airport": airport,
            "next_check_recommended": (datetime.now() + timedelta(minutes=5)).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error checking alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Alert check failed: {str(e)}")

@app.post("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str):
    """Resolve an alert with detailed logging"""
    try:
        logger.info(f"Resolving alert: {alert_id}")
        
        # Find and update alert in demo state
        for alert in demo_state["alerts"]:
            if alert["alert_id"] == alert_id:
                alert["resolved"] = True
                alert["resolved_at"] = datetime.now().isoformat()
                alert["resolved_by"] = "user_action"
                break
        
        return {
            "status": "resolved",
            "alert_id": alert_id,
            "resolved_at": datetime.now().isoformat(),
            "resolution_method": "manual",
            "follow_up_required": random.choice([True, False])
        }
        
    except Exception as e:
        logger.error(f"Error resolving alert: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to resolve alert: {str(e)}")

@app.post("/alerts/{alert_id}/escalate")
async def escalate_alert(alert_id: str):
    """Escalate an alert with notification"""
    try:
        logger.info(f"Escalating alert: {alert_id}")
        
        # Find and update alert in demo state
        for alert in demo_state["alerts"]:
            if alert["alert_id"] == alert_id:
                alert["escalated"] = True
                alert["escalated_at"] = datetime.now().isoformat()
                alert["escalation_level"] = "management"
                break
        
        return {
            "status": "escalated",
            "alert_id": alert_id,
            "escalated_at": datetime.now().isoformat(),
            "escalation_level": "management",
            "notification_sent": True,
            "expected_response_time": "15 minutes"
        }
        
    except Exception as e:
        logger.error(f"Error escalating alert: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to escalate alert: {str(e)}")

@app.post("/alerts/test-notification")
async def test_notification():
    """Enhanced test notification with comprehensive details"""
    try:
        logger.info("Sending test notification...")
        
        # Simulate processing time
        await asyncio.sleep(random.uniform(0.5, 1.2))
        
        test_alert_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Realistic notification scenarios
        notification_scenarios = [
            {
                "type": "capacity_alert",
                "message": "Test: Runway capacity approaching 95% utilization during peak hours",
                "severity": "warning",
                "urgency": "medium"
            },
            {
                "type": "delay_alert",
                "message": "Test: Multiple flights experiencing delays exceeding 15 minutes",
                "severity": "medium",
                "urgency": "high"
            },
            {
                "type": "weather_alert",
                "message": "Test: Weather conditions may impact operations in next 30 minutes",
                "severity": "info",
                "urgency": "low"
            },
            {
                "type": "system_alert",
                "message": "Test: ML model performance requires attention",
                "severity": "low",
                "urgency": "low"
            }
        ]
        
        selected_scenario = random.choice(notification_scenarios)
        delivery_time = random.randint(150, 800)
        
        # Simulate multi-channel delivery
        channels = [
            {"name": "dashboard", "status": "delivered", "delivery_time_ms": delivery_time},
            {"name": "email", "status": "delivered", "delivery_time_ms": delivery_time + random.randint(200, 500)},
            {"name": "slack", "status": "delivered", "delivery_time_ms": delivery_time + random.randint(100, 300)},
            {"name": "sms", "status": "pending", "delivery_time_ms": None}
        ]
        
        return {
            "status": "success",
            "message": f"Test notification sent successfully: {selected_scenario['message']}",
            "alert_id": test_alert_id,
            "notification_details": {
                "type": selected_scenario["type"],
                "severity": selected_scenario["severity"],
                "urgency": selected_scenario["urgency"],
                "timestamp": datetime.now().isoformat(),
                "channels": [ch["name"] for ch in channels],
                "delivery_status": "delivered",
                "delivery_summary": {
                    "total_channels": len(channels),
                    "successful_deliveries": len([ch for ch in channels if ch["status"] == "delivered"]),
                    "pending_deliveries": len([ch for ch in channels if ch["status"] == "pending"]),
                    "failed_deliveries": len([ch for ch in channels if ch["status"] == "failed"])
                }
            },
            "test_data": {
                "notification_id": test_alert_id,
                "sent_at": datetime.now().isoformat(),
                "recipient_count": 3,
                "delivery_time_ms": delivery_time,
                "channel_details": channels,
                "test_scenario": selected_scenario["type"]
            },
            "performance_metrics": {
                "api_response_time_ms": random.randint(50, 150),
                "notification_processing_time_ms": random.randint(200, 400),
                "total_time_ms": delivery_time + random.randint(250, 550)
            }
        }
        
    except Exception as e:
        logger.error(f"Error in test notification: {e}")
        raise HTTPException(status_code=500, detail=f"Test notification failed: {str(e)}")

@app.get("/ai/recommendations")
async def get_ai_recommendations(airport: str = "BOM"):
    """Get AI-powered operational recommendations"""
    try:
        logger.info(f"Generating AI recommendations for {airport}")
        
        # Generate realistic AI recommendations based on current conditions
        recommendations = [
            {
                "id": "rec_001",
                "type": "optimization",
                "title": "Optimize Gate Assignment",
                "description": f"AI detected suboptimal gate assignments at {airport} causing 12min average taxi delays",
                "impact": "Reduce delays by 15-20%",
                "confidence": 0.87,
                "action": "Reassign gates A1-A5 for international flights",
                "priority": "high",
                "estimated_savings": "$45K/day",
                "affected_flights": 23,
                "algorithm": "Hungarian Algorithm with Dynamic Constraints",
                "formula": "minimize(Σ(taxi_time × fuel_cost + delay_penalty))",
                "implementation_time": "15 minutes"
            },
            {
                "id": "rec_002", 
                "type": "prediction",
                "title": "Weather Impact Alert",
                "description": "Monsoon patterns suggest 40% delay probability in next 4 hours",
                "impact": "Potential 25min average delays",
                "confidence": 0.92,
                "action": "Preemptively adjust schedule for flights after 2 PM",
                "priority": "high",
                "estimated_impact": "47 flights affected",
                "weather_model": "ECMWF Ensemble Forecast",
                "prediction_window": "4 hours",
                "mitigation_options": ["Schedule buffer", "Route alternatives", "Ground holds"]
            },
            {
                "id": "rec_003",
                "type": "efficiency", 
                "title": "Turnaround Optimization",
                "description": "Ground crew scheduling can be optimized for 18% faster turnarounds",
                "impact": "Save 8min per flight",
                "confidence": 0.78,
                "action": "Implement staggered crew shifts",
                "priority": "medium",
                "estimated_savings": "$28K/week",
                "optimization_method": "Linear Programming with Resource Constraints",
                "crew_efficiency_gain": "18%",
                "implementation_complexity": "Medium"
            },
            {
                "id": "rec_004",
                "type": "alert",
                "title": "Runway Capacity Warning", 
                "description": f"Peak hour congestion detected at {airport} - runway utilization at 95%",
                "impact": "Prevent cascade delays",
                "confidence": 0.95,
                "action": "Activate secondary runway protocols",
                "priority": "high",
                "urgency": "immediate",
                "capacity_model": "Queuing Theory Analysis",
                "current_utilization": "95%",
                "recommended_utilization": "85%"
            },
            {
                "id": "rec_005",
                "type": "optimization",
                "title": "Fuel Efficiency Opportunity",
                "description": "Route optimization can reduce fuel consumption by 12%",
                "impact": "Save 340L fuel per day",
                "confidence": 0.83,
                "action": "Update flight paths for routes BOM-DEL, BOM-BLR",
                "priority": "medium",
                "estimated_savings": "$15K/month",
                "optimization_algorithm": "Genetic Algorithm with Wind Pattern Analysis",
                "fuel_savings": "340L/day",
                "co2_reduction": "850kg/day"
            },
            {
                "id": "rec_006",
                "type": "prediction",
                "title": "Maintenance Window Optimization",
                "description": "AI predicts optimal maintenance scheduling to minimize disruptions",
                "impact": "Reduce maintenance delays by 30%",
                "confidence": 0.81,
                "action": "Schedule maintenance during low-traffic periods (2-5 AM)",
                "priority": "medium",
                "predictive_model": "LSTM Neural Network",
                "maintenance_efficiency": "+30%",
                "cost_avoidance": "$22K/month"
            }
        ]
        
        # Filter recommendations based on current time and conditions
        current_hour = datetime.now().hour
        if current_hour < 6 or current_hour > 22:
            # Night time - fewer operational recommendations
            recommendations = [r for r in recommendations if r["type"] in ["prediction", "efficiency", "optimization"]]
        
        # Add dynamic scoring based on current conditions
        for rec in recommendations:
            # Add real-time relevance score
            rec["relevance_score"] = random.uniform(0.7, 1.0)
            rec["last_updated"] = datetime.now().isoformat()
            rec["status"] = "active"
            
            # Add implementation tracking
            rec["implementation"] = {
                "estimated_duration": f"{random.randint(10, 120)} minutes",
                "complexity": random.choice(["Low", "Medium", "High"]),
                "prerequisites": ["System access", "Stakeholder approval"] if rec["priority"] == "high" else ["System access"],
                "rollback_plan": "Available" if rec["type"] == "optimization" else "N/A"
            }
        
        return {
            "recommendations": recommendations,
            "airport": airport,
            "timestamp": datetime.now().isoformat(),
            "total_count": len(recommendations),
            "high_priority_count": len([r for r in recommendations if r["priority"] == "high"]),
            "medium_priority_count": len([r for r in recommendations if r["priority"] == "medium"]),
            "low_priority_count": len([r for r in recommendations if r["priority"] == "low"]),
            "ai_system_info": {
                "model_version": "4.2.1",
                "last_training": "2024-01-10T00:00:00Z",
                "confidence_threshold": 0.7,
                "recommendation_engine": "Multi-Agent Reinforcement Learning",
                "data_sources": ["operational_data", "weather_feeds", "historical_patterns", "real_time_metrics"],
                "update_frequency": "Every 5 minutes"
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting AI recommendations: {e}")
        return {"recommendations": [], "error": str(e)}

if __name__ == "__main__":
    print("🚀 Starting Enhanced Agentic Flight Scheduler Demo API...")
    print("📡 Server will be available at: http://localhost:8000")
    print("📊 API Documentation: http://localhost:8000/docs")
    print("✅ CORS enabled for frontend development")
    print("🔧 Enhanced error handling and logging enabled")
    print("📈 Comprehensive algorithm transparency included")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False,
        access_log=True
    )