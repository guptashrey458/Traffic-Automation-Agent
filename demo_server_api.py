#!/usr/bin/env python3
"""
Demo API server for the Flight Scheduler Dashboard
This provides mock data for demonstration purposes.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
from datetime import datetime, date, timedelta
import uvicorn
import random
import json
import asyncio

app = FastAPI(
    title="Agentic Flight Scheduler API - Demo",
    description="Demo API for the flight scheduling dashboard",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock data generators
def generate_time_buckets(bucket_minutes: int = 10):
    """Generate mock time bucket data"""
    buckets = []
    start_time = datetime.now().replace(hour=6, minute=0, second=0, microsecond=0)
    
    for i in range(72):  # 12 hours of data
        bucket_start = start_time + timedelta(minutes=i * bucket_minutes)
        bucket_end = bucket_start + timedelta(minutes=bucket_minutes)
        
        # Simulate traffic patterns
        hour = bucket_start.hour
        if 7 <= hour <= 9 or 17 <= hour <= 19:  # Peak hours
            demand = random.randint(15, 25)
            capacity = 20
        elif 10 <= hour <= 16:  # Busy hours
            demand = random.randint(8, 15)
            capacity = 20
        else:  # Off-peak
            demand = random.randint(2, 8)
            capacity = 20
            
        utilization = min(demand / capacity, 1.2)
        overload = max(0, demand - capacity)
        
        buckets.append({
            "start_time": bucket_start.isoformat(),
            "end_time": bucket_end.isoformat(),
            "scheduled_departures": demand // 2,
            "actual_departures": demand // 2,
            "scheduled_arrivals": demand // 2,
            "actual_arrivals": demand // 2,
            "total_demand": demand,
            "capacity": capacity,
            "utilization": utilization,
            "overload": overload,
            "avg_delay": random.uniform(0, 30) if overload > 0 else random.uniform(0, 5),
            "delayed_flights": random.randint(0, overload * 2),
            "traffic_level": "critical" if utilization > 1.0 else "high" if utilization > 0.8 else "medium" if utilization > 0.5 else "low"
        })
    
    return buckets

@app.get("/")
async def root():
    return {
        "message": "Agentic Flight Scheduler API - Demo Mode",
        "version": "0.1.0",
        "status": "ready"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "environment": "demo"}

@app.get("/airports")
async def get_airports():
    return {
        "airports": [
            {"code": "BOM", "name": "Mumbai", "city": "Mumbai"},
            {"code": "DEL", "name": "Delhi", "city": "Delhi"},
            {"code": "BLR", "name": "Bangalore", "city": "Bangalore"},
            {"code": "MAA", "name": "Chennai", "city": "Chennai"},
            {"code": "CCU", "name": "Kolkata", "city": "Kolkata"},
            {"code": "HYD", "name": "Hyderabad", "city": "Hyderabad"}
        ]
    }

@app.get("/status")
async def get_system_status():
    return {
        "status": "healthy",
        "services": {
            "database": "healthy",
            "analytics": "healthy",
            "optimizer": "healthy",
            "whatif_simulator": "healthy",
            "delay_predictor": "healthy"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/flights/peaks")
async def get_flight_peaks(
    airport: str = "BOM",
    bucket_minutes: int = 10,
    date: Optional[str] = None,
    weather_regime: str = "calm"
):
    """Mock peak analysis endpoint"""
    time_buckets = generate_time_buckets(bucket_minutes)
    
    # Generate overload windows
    overload_windows = []
    for i, bucket in enumerate(time_buckets):
        if bucket["overload"] > 0:
            overload_windows.append({
                "start_time": bucket["start_time"],
                "end_time": bucket["end_time"],
                "duration_minutes": bucket_minutes,
                "peak_overload": bucket["overload"],
                "affected_flights": bucket["overload"] * 2,
                "severity": "high" if bucket["overload"] > 5 else "medium"
            })
    
    # Calculate capacity utilization
    total_utilization = sum(bucket["utilization"] for bucket in time_buckets)
    avg_utilization = total_utilization / len(time_buckets)
    
    recommendations = [
        "Consider increasing runway capacity during peak hours (7-9 AM, 5-7 PM)",
        "Implement dynamic slot allocation to reduce overload periods",
        "Optimize turnaround times to improve overall efficiency",
        "Consider weather-based capacity adjustments for better planning"
    ]
    
    return {
        "airport": airport,
        "bucket_minutes": bucket_minutes,
        "analysis_date": date or datetime.now().date().isoformat(),
        "time_buckets": time_buckets,
        "overload_windows": overload_windows,
        "capacity_utilization": avg_utilization,
        "recommendations": recommendations,
        "weather_regime": weather_regime
    }

@app.post("/optimize")
async def optimize_schedule(request: Dict[str, Any]):
    """Enhanced optimization endpoint with algorithm formulas"""
    await asyncio.sleep(2)  # Simulate processing time
    
    # Mock optimization results
    original_metrics = {
        "total_flights": 245,
        "avg_delay_minutes": 18.5,
        "on_time_performance": 0.72,
        "fuel_cost_usd": 125000,
        "co2_emissions_kg": 45000,
        "runway_utilization": 0.85
    }
    
    optimized_metrics = {
        "total_flights": 245,
        "avg_delay_minutes": 12.1,
        "on_time_performance": 0.89,
        "fuel_cost_usd": 112000,
        "co2_emissions_kg": 41500,
        "runway_utilization": 0.92
    }
    
    # Generate recommended changes with algorithm details
    recommended_changes = []
    for i in range(15):
        flight_id = f"AI{2700 + i}"
        change_type = random.choice(["time_shift", "runway_change", "gate_change"])
        original_time = (datetime.now() + timedelta(hours=random.randint(1, 12))).strftime("%H:%M")
        recommended_time = (datetime.now() + timedelta(hours=random.randint(1, 12), minutes=random.randint(-30, 30))).strftime("%H:%M")
        
        # Calculate optimization scores
        delay_score = random.uniform(0.6, 0.95)
        fuel_score = random.uniform(0.7, 0.9)
        capacity_score = random.uniform(0.8, 0.95)
        
        # Weighted optimization score
        optimization_score = (delay_score * 0.4 + fuel_score * 0.3 + capacity_score * 0.3)
        
        recommended_changes.append({
            "flight_id": flight_id,
            "change_type": change_type,
            "original_time": original_time,
            "recommended_time": recommended_time,
            "impact_description": f"Reduce delay by moving to less congested slot",
            "delay_reduction_minutes": random.uniform(5, 20),
            "co2_reduction_kg": random.uniform(50, 200),
            "optimization_score": optimization_score,
            "algorithm_factors": {
                "delay_score": delay_score,
                "fuel_score": fuel_score,
                "capacity_score": capacity_score
            },
            "formula": "Score = (Delay_Score × 0.4) + (Fuel_Score × 0.3) + (Capacity_Score × 0.3)",
            "calculation": f"({delay_score:.3f} × 0.4) + ({fuel_score:.3f} × 0.3) + ({capacity_score:.3f} × 0.3) = {optimization_score:.3f}"
        })
    
    return {
        "optimization_id": f"opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "status": "completed",
        "original_metrics": original_metrics,
        "optimized_metrics": optimized_metrics,
        "recommended_changes": recommended_changes,
        "cost_reduction": 13000,
        "execution_time_seconds": 2.3,
        "algorithm_info": {
            "method": "Multi-Objective Genetic Algorithm + Simulated Annealing",
            "objective_function": "Minimize: α×Delay + β×Fuel_Cost + γ×CO2 + δ×Capacity_Violation",
            "constraints": [
                "Turnaround_Time ≥ Minimum_Required",
                "Runway_Capacity ≤ Maximum_Hourly",
                "Gate_Availability = True",
                "Crew_Rest_Time ≥ Regulatory_Minimum"
            ],
            "parameters": {
                "alpha_delay_weight": 0.4,
                "beta_fuel_weight": 0.3,
                "gamma_co2_weight": 0.2,
                "delta_capacity_weight": 0.1,
                "population_size": 100,
                "generations": 50,
                "mutation_rate": 0.1
            },
            "convergence_criteria": "Improvement < 0.1% for 10 consecutive generations",
            "solution_quality": "Near-optimal (within 2% of theoretical optimum)"
        }
    }

@app.post("/whatif")
async def whatif_analysis(request: Dict[str, Any]):
    """Mock what-if analysis endpoint"""
    flight_id = request.get("flight_id", "AI2739")
    change_type = request.get("change_type", "time_shift")
    change_value = request.get("change_value", "+10")
    
    # Simulate impact based on change
    if change_type == "time_shift":
        delay_change = float(change_value.replace('+', '').replace('m', ''))
        if change_value.startswith('-'):
            delay_change = -delay_change
    else:
        delay_change = random.uniform(-5, 15)
    
    return {
        "flight_id": flight_id,
        "change_description": f"Flight {flight_id} {change_type}: {change_value}",
        "impact_summary": {
            "delay_change": delay_change,
            "co2_change": delay_change * 2.5,
            "confidence": 0.87,
            "affected_flights": random.randint(3, 12)
        },
        "before_metrics": {
            "avg_delay": 15.2,
            "peak_overload": 3
        },
        "after_metrics": {
            "avg_delay": 15.2 + delay_change,
            "peak_overload": 3 + (1 if delay_change > 10 else 0)
        },
        "affected_flights": [f"AI{2700 + i}" for i in range(random.randint(3, 8))],
        "co2_impact_kg": delay_change * 2.5
    }

@app.get("/flights/risks")
async def get_delay_risks(
    airport: str = "BOM",
    date: str = None,
    flight_ids: Optional[str] = None,
    risk_threshold: float = 0.2
):
    """Enhanced delay risk prediction endpoint with algorithm formulas"""
    risks = []
    
    flight_list = flight_ids.split(',') if flight_ids else [f"AI{2700 + i}" for i in range(10)]
    
    for flight_id in flight_list:
        # Simulate realistic delay prediction algorithm
        base_delay_prob = random.uniform(0.1, 0.9)
        
        # Weather impact factor (0.8 - 1.3)
        weather_factor = random.uniform(0.8, 1.3)
        
        # Traffic congestion factor (0.9 - 1.5)
        traffic_factor = random.uniform(0.9, 1.5)
        
        # Time of day factor (0.7 - 1.4)
        time_factor = random.uniform(0.7, 1.4)
        
        # Aircraft type factor (0.85 - 1.15)
        aircraft_factor = random.uniform(0.85, 1.15)
        
        # Calculate final risk using weighted formula
        departure_risk = min(0.95, base_delay_prob * weather_factor * traffic_factor * time_factor * aircraft_factor)
        arrival_risk = min(0.95, departure_risk * random.uniform(0.8, 1.2))
        
        # Expected delay calculation (minutes)
        expected_dep_delay = departure_risk * 45 + random.uniform(-5, 10)
        expected_arr_delay = arrival_risk * 50 + random.uniform(-8, 15)
        
        if departure_risk >= risk_threshold:
            risks.append({
                "flight_id": flight_id.strip(),
                "departure_risk": {
                    "risk_score": departure_risk,
                    "expected_delay_minutes": max(0, expected_dep_delay),
                    "confidence": random.uniform(0.7, 0.95),
                    "risk_level": "critical" if departure_risk > 0.8 else "high" if departure_risk > 0.6 else "medium" if departure_risk > 0.3 else "low",
                    "contributing_factors": ["weather", "traffic_congestion", "aircraft_availability"],
                    "algorithm_factors": {
                        "base_probability": base_delay_prob,
                        "weather_factor": weather_factor,
                        "traffic_factor": traffic_factor,
                        "time_factor": time_factor,
                        "aircraft_factor": aircraft_factor
                    },
                    "formula": "Risk = Base_Prob × Weather_Factor × Traffic_Factor × Time_Factor × Aircraft_Factor",
                    "calculation": f"{base_delay_prob:.3f} × {weather_factor:.3f} × {traffic_factor:.3f} × {time_factor:.3f} × {aircraft_factor:.3f} = {departure_risk:.3f}"
                },
                "arrival_risk": {
                    "risk_score": arrival_risk,
                    "expected_delay_minutes": max(0, expected_arr_delay),
                    "confidence": random.uniform(0.7, 0.95),
                    "risk_level": "critical" if arrival_risk > 0.8 else "high" if arrival_risk > 0.6 else "medium" if arrival_risk > 0.3 else "low",
                    "contributing_factors": ["downstream_delays", "weather", "airspace_congestion"],
                    "formula": "Arrival_Risk = Departure_Risk × Propagation_Factor + Route_Complexity",
                    "calculation": f"{departure_risk:.3f} × {random.uniform(0.8, 1.2):.3f} = {arrival_risk:.3f}"
                },
                "risk_factors": ["Peak hour departure", "Weather conditions", "High traffic volume"],
                "recommendations": ["Consider 15-minute buffer", "Monitor weather updates", "Prepare alternate runway"],
                "model_info": {
                    "algorithm": "LightGBM Gradient Boosting + Heuristic Rules",
                    "features_used": ["time_of_day", "weather_conditions", "traffic_density", "aircraft_type", "airline_performance", "route_complexity"],
                    "training_accuracy": random.uniform(0.85, 0.94),
                    "last_updated": "2024-01-15T08:00:00Z"
                }
            })
    
    return risks

@app.get("/alerts/active")
async def get_active_alerts(airport: Optional[str] = None, severity: Optional[str] = None):
    """Mock active alerts endpoint"""
    alerts = [
        {
            "alert_id": "alert_001",
            "alert_type": "capacity_overload",
            "severity": "high",
            "title": "Runway Capacity Overload Detected",
            "description": "Runway 09L experiencing 120% capacity utilization during peak hours",
            "airport": "BOM",
            "timestamp": (datetime.now() - timedelta(minutes=15)).isoformat(),
            "affected_flights": ["AI2739", "6E123", "UK955"],
            "recommendations": [
                {
                    "action": "Implement 5-minute spacing increase",
                    "impact": "Reduce overload to 95% capacity",
                    "priority": 1,
                    "estimated_improvement": "15% delay reduction"
                }
            ],
            "metrics": {"overload_percentage": 120, "affected_flights_count": 12},
            "resolved": False,
            "escalated": False
        },
        {
            "alert_id": "alert_002",
            "alert_type": "weather_impact",
            "severity": "medium",
            "title": "Weather Conditions Affecting Operations",
            "description": "Moderate crosswinds reducing runway capacity by 15%",
            "airport": "BOM",
            "timestamp": (datetime.now() - timedelta(minutes=8)).isoformat(),
            "affected_flights": ["AI2740", "6E124"],
            "recommendations": [
                {
                    "action": "Switch to alternate runway configuration",
                    "impact": "Maintain normal capacity",
                    "priority": 2,
                    "estimated_improvement": "Prevent 10% capacity reduction"
                }
            ],
            "metrics": {"capacity_reduction": 15, "wind_speed": 25},
            "resolved": False,
            "escalated": False
        }
    ]
    
    # Filter by airport and severity if provided
    filtered_alerts = alerts
    if airport:
        filtered_alerts = [a for a in filtered_alerts if a["airport"] == airport]
    if severity:
        filtered_alerts = [a for a in filtered_alerts if a["severity"] == severity]
    
    return filtered_alerts

@app.get("/alerts/summary")
async def get_alert_summary(airport: Optional[str] = None):
    """Mock alert summary endpoint"""
    return {
        "total_active_alerts": 2,
        "by_severity": {
            "critical": 0,
            "high": 1,
            "medium": 1,
            "low": 0
        },
        "by_type": {
            "capacity_overload": 1,
            "weather_impact": 1,
            "system_error": 0
        },
        "escalated_alerts": 0,
        "oldest_alert": (datetime.now() - timedelta(minutes=15)).isoformat(),
        "most_recent_alert": (datetime.now() - timedelta(minutes=8)).isoformat()
    }

@app.post("/alerts/check")
async def check_alerts(request: Dict[str, Any]):
    """Mock alert check endpoint"""
    return []  # No new alerts for demo

@app.post("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str):
    """Mock alert resolution endpoint"""
    return {"status": "resolved", "alert_id": alert_id}

@app.post("/alerts/{alert_id}/escalate")
async def escalate_alert(alert_id: str):
    """Mock alert escalation endpoint"""
    return {"status": "escalated", "alert_id": alert_id}

@app.post("/alerts/test-notification")
async def test_notification():
    """Enhanced test notification endpoint"""
    # Simulate some processing time
    await asyncio.sleep(1)
    
    # Generate a realistic test alert
    test_alert_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Simulate different notification types
    notification_types = [
        {
            "type": "capacity_alert",
            "message": "Test: Runway capacity approaching 95% utilization",
            "severity": "warning"
        },
        {
            "type": "delay_alert", 
            "message": "Test: Multiple flights experiencing delays > 15 minutes",
            "severity": "medium"
        },
        {
            "type": "weather_alert",
            "message": "Test: Weather conditions may impact operations",
            "severity": "info"
        }
    ]
    
    selected_notification = random.choice(notification_types)
    
    return {
        "status": "success",
        "message": f"Test notification sent successfully: {selected_notification['message']}",
        "alert_id": test_alert_id,
        "notification_details": {
            "type": selected_notification["type"],
            "severity": selected_notification["severity"],
            "timestamp": datetime.now().isoformat(),
            "channels": ["dashboard", "email", "slack"],
            "delivery_status": "delivered"
        },
        "test_data": {
            "notification_id": test_alert_id,
            "sent_at": datetime.now().isoformat(),
            "recipient_count": 3,
            "delivery_time_ms": random.randint(150, 500)
        }
    }

if __name__ == "__main__":
    print("🚀 Starting Agentic Flight Scheduler Demo API...")
    print("📡 Server will be available at: http://localhost:8000")
    print("📊 API Documentation: http://localhost:8000/docs")
    print("✅ CORS enabled for frontend development")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False
    )
