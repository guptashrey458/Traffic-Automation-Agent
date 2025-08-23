"""FastAPI application main module."""

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
from datetime import datetime, date
from pydantic import BaseModel, Field
import logging

from src.config.environments import get_config
from src.services.analytics import AnalyticsEngine, WeatherRegime
from src.services.schedule_optimizer import ScheduleOptimizer, ObjectiveWeights, Constraints
from src.services.whatif_simulator import WhatIfSimulator
from src.services.delay_prediction import DelayRiskPredictor, OperationalContext
from src.services.database import FlightDatabaseService

# Get configuration
config = get_config()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Agentic Flight Scheduler API",
    description="AI-powered flight scheduling optimization system",
    version="0.1.0",
    debug=config.api.debug,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
db_service = FlightDatabaseService()
analytics_engine = AnalyticsEngine(db_service)
schedule_optimizer = ScheduleOptimizer(db_service)
whatif_simulator = WhatIfSimulator(db_service)
delay_predictor = DelayRiskPredictor(db_service)

# Initialize alerting service
from src.services.alerting import AlertingService
alerting_service = AlertingService(
    analytics_engine=analytics_engine,
    schedule_optimizer=schedule_optimizer
)

# Pydantic models for request/response schemas
class PeakAnalysisResponse(BaseModel):
    airport: str
    bucket_minutes: int
    analysis_date: date
    time_buckets: List[Dict[str, Any]]
    overload_windows: List[Dict[str, Any]]
    capacity_utilization: float
    recommendations: List[str]
    weather_regime: str

class OptimizationRequest(BaseModel):
    airport: str
    date: date
    flights: Optional[List[str]] = None  # Flight IDs to optimize
    objectives: Optional[Dict[str, float]] = None  # Objective weights
    constraints: Optional[Dict[str, Any]] = None  # Custom constraints

class OptimizationResponse(BaseModel):
    optimization_id: str
    status: str
    original_metrics: Dict[str, Any]
    optimized_metrics: Dict[str, Any]
    recommended_changes: List[Dict[str, Any]]
    cost_reduction: float
    execution_time_seconds: float

class WhatIfRequest(BaseModel):
    flight_id: str
    change_type: str  # "time_shift", "runway_change", "cancellation"
    change_value: Optional[str] = None  # "+10m", "RW09", etc.
    airport: str
    date: date

class WhatIfResponse(BaseModel):
    flight_id: str
    change_description: str
    impact_summary: Dict[str, Any]
    before_metrics: Dict[str, Any]
    after_metrics: Dict[str, Any]
    affected_flights: List[str]
    co2_impact_kg: float

class DelayRiskResponse(BaseModel):
    flight_id: str
    departure_risk: Dict[str, Any]
    arrival_risk: Dict[str, Any]
    risk_factors: List[str]
    recommendations: List[str]

class ConstraintsResponse(BaseModel):
    airport: str
    operational_rules: Dict[str, Any]
    capacity_limits: Dict[str, Any]
    weather_adjustments: Dict[str, Any]
    curfew_hours: Dict[str, Any]

class AlertResponse(BaseModel):
    alert_id: str
    alert_type: str
    severity: str
    title: str
    description: str
    airport: str
    timestamp: datetime
    affected_flights: List[str]
    recommendations: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    resolved: bool
    escalated: bool

class AlertSummaryResponse(BaseModel):
    total_active_alerts: int
    by_severity: Dict[str, int]
    by_type: Dict[str, int]
    escalated_alerts: int
    oldest_alert: Optional[datetime]
    most_recent_alert: Optional[datetime]

class AlertCheckRequest(BaseModel):
    airport: str
    date: Optional[date] = None
    force_check: bool = False


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Agentic Flight Scheduler API",
        "version": "0.1.0",
        "environment": config.environment,
        "status": "ready"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "environment": config.environment}

@app.get("/flights/peaks", response_model=PeakAnalysisResponse)
async def get_flight_peaks(
    airport: str = Query(..., description="Airport code (e.g., BOM, DEL)"),
    bucket_minutes: int = Query(10, description="Time bucket size in minutes (5, 10, 15, 30)"),
    date: Optional[date] = Query(None, description="Analysis date (defaults to latest available)"),
    weather_regime: Optional[str] = Query("calm", description="Weather regime: calm, medium, strong, severe")
):
    """
    Analyze peak traffic periods for a specific airport.
    
    Returns demand heatmaps, overload windows, and capacity utilization metrics.
    """
    try:
        logger.info(f"Analyzing peaks for {airport}, bucket_minutes={bucket_minutes}")
        
        # Validate parameters
        if bucket_minutes not in [5, 10, 15, 30]:
            raise HTTPException(status_code=400, detail="bucket_minutes must be 5, 10, 15, or 30")
        
        if airport not in ["BOM", "DEL", "BLR", "MAA", "CCU", "HYD"]:
            raise HTTPException(status_code=400, detail="Unsupported airport code")
        
        # Set weather regime
        try:
            weather = WeatherRegime(weather_regime.lower())
        except ValueError:
            weather = WeatherRegime.CALM
        
        # Perform peak analysis
        analysis = analytics_engine.analyze_peaks(
            airport=airport,
            bucket_minutes=bucket_minutes,
            analysis_date=date,
            weather_regime=weather
        )
        
        # Format response
        return PeakAnalysisResponse(
            airport=analysis.airport,
            bucket_minutes=analysis.bucket_minutes,
            analysis_date=analysis.analysis_date,
            time_buckets=[bucket.to_dict() for bucket in analysis.time_buckets],
            overload_windows=[window.to_dict() for window in analysis.overload_windows],
            capacity_utilization=analysis.capacity_utilization,
            recommendations=analysis.recommendations,
            weather_regime=weather.value
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions (validation errors, etc.)
        raise
    except Exception as e:
        logger.error(f"Error in peak analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Peak analysis failed: {str(e)}")


@app.post("/optimize", response_model=OptimizationResponse)
async def optimize_schedule(request: OptimizationRequest):
    """
    Optimize flight schedules using constraint-based algorithms.
    
    Returns recommended schedule changes with impact metrics.
    """
    try:
        logger.info(f"Optimizing schedule for {request.airport} on {request.date}")
        
        # Set up optimization parameters
        weights = ObjectiveWeights()
        if request.objectives:
            weights.delay_weight = request.objectives.get("delay_weight", 1.0)
            weights.taxi_weight = request.objectives.get("taxi_weight", 0.3)
            weights.runway_change_weight = request.objectives.get("runway_change_weight", 0.2)
        
        # Set up constraints
        constraints = Constraints()
        if request.constraints:
            # Apply custom constraints if provided
            pass
        
        # Run optimization
        result = schedule_optimizer.optimize_schedule(
            airport=request.airport,
            date=request.date,
            flight_ids=request.flights,
            weights=weights,
            constraints=constraints
        )
        
        return OptimizationResponse(
            optimization_id=result.optimization_id,
            status=result.status,
            original_metrics=result.original_metrics.to_dict(),
            optimized_metrics=result.optimized_metrics.to_dict(),
            recommended_changes=[change.to_dict() for change in result.recommended_changes],
            cost_reduction=result.cost_reduction,
            execution_time_seconds=result.execution_time_seconds
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in schedule optimization: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Schedule optimization failed: {str(e)}")


@app.post("/whatif", response_model=WhatIfResponse)
async def whatif_analysis(request: WhatIfRequest):
    """
    Perform what-if analysis for single flight changes.
    
    Returns impact metrics showing before/after comparison.
    """
    try:
        logger.info(f"What-if analysis for flight {request.flight_id}: {request.change_type}")
        
        # Parse time change from request
        time_change_minutes = 0
        if request.change_value and request.change_type == "time_shift":
            try:
                time_change_minutes = int(request.change_value.replace('+', '').replace('m', ''))
                if '-' in request.change_value:
                    time_change_minutes = -time_change_minutes
            except ValueError:
                time_change_minutes = 0
        
        # Run what-if simulation
        result = whatif_simulator.analyze_single_flight_change(
            flight_id=request.flight_id,
            time_change_minutes=time_change_minutes,
            airport=request.airport,
            analysis_date=request.date
        )
        
        return WhatIfResponse(
            flight_id=request.flight_id,
            change_description=result.scenario_description,
            impact_summary={
                "delay_change": result.delay_change_minutes,
                "co2_change": result.co2_change_kg,
                "confidence": result.confidence_level,
                "affected_flights": result.affected_flights_count
            },
            before_metrics={
                "avg_delay": 0.0,  # Base metrics would be calculated separately
                "peak_overload": 0
            },
            after_metrics={
                "avg_delay": result.delay_change_minutes,
                "peak_overload": result.peak_overload_change
            },
            affected_flights=[],  # Individual flight IDs would need separate tracking
            co2_impact_kg=result.co2_change_kg
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in what-if analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"What-if analysis failed: {str(e)}")


@app.get("/flights/risks", response_model=List[DelayRiskResponse])
async def get_delay_risks(
    airport: str = Query(..., description="Airport code"),
    date: date = Query(..., description="Analysis date"),
    flight_ids: Optional[str] = Query(None, description="Comma-separated flight IDs"),
    risk_threshold: float = Query(0.2, description="Minimum risk threshold (0.0-1.0)")
):
    """
    Get delay risk predictions for flights.
    
    Returns risk assessments with contributing factors and recommendations.
    """
    try:
        logger.info(f"Analyzing delay risks for {airport} on {date}")
        
        # Parse flight IDs if provided
        target_flights = None
        if flight_ids:
            target_flights = [fid.strip() for fid in flight_ids.split(",")]
        
        # Get risk predictions
        risks = delay_predictor.predict_delay_risks(
            airport=airport,
            date=date,
            flight_ids=target_flights,
            risk_threshold=risk_threshold
        )
        
        # Format response
        responses = []
        for risk in risks:
            responses.append(DelayRiskResponse(
                flight_id=risk.flight_id,
                departure_risk=risk.departure_risk.to_dict(),
                arrival_risk=risk.arrival_risk.to_dict(),
                risk_factors=risk.risk_factors,
                recommendations=risk.recommendations
            ))
        
        return responses
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in delay risk analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Delay risk analysis failed: {str(e)}")


@app.get("/constraints", response_model=ConstraintsResponse)
async def get_operational_constraints(
    airport: str = Query(..., description="Airport code"),
    date: Optional[date] = Query(None, description="Date for time-specific constraints")
):
    """
    Get operational constraints and rules for an airport.
    
    Returns capacity limits, operational rules, and constraint parameters.
    """
    try:
        logger.info(f"Getting constraints for {airport}")
        
        # Get constraints from optimizer
        constraints = schedule_optimizer.get_constraints(airport=airport, date=date)
        
        return ConstraintsResponse(
            airport=airport,
            operational_rules=constraints.operational_rules,
            capacity_limits=constraints.capacity_limits,
            weather_adjustments=constraints.weather_adjustments,
            curfew_hours=constraints.curfew_hours
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting constraints: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get constraints: {str(e)}")


# Additional utility endpoints
@app.get("/airports")
async def get_supported_airports():
    """Get list of supported airports."""
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
    """Get system status and service health."""
    try:
        # Check database connectivity
        try:
            db_status = "healthy" if hasattr(db_service, 'health_check') else "unknown"
        except:
            db_status = "degraded"
        
        # Check service availability
        services_status = {
            "database": db_status,
            "analytics": "healthy",
            "optimizer": "healthy",
            "whatif_simulator": "healthy",
            "delay_predictor": "healthy"
        }
        
        return {
            "status": "healthy",
            "services": services_status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"System status check failed: {str(e)}")
        return {
            "status": "degraded",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Alerting and notification endpoints
@app.post("/alerts/check", response_model=List[AlertResponse])
async def check_alerts(request: AlertCheckRequest):
    """
    Check for alert conditions and generate new alerts.
    
    Analyzes current flight data for capacity overloads, delay cascades,
    and critical flight delays. Returns any new alerts generated.
    """
    try:
        logger.info(f"Checking alerts for {request.airport}")
        
        # Get flights for the specified date
        if request.date:
            flights = db_service.get_flights_by_date(request.airport, request.date)
        else:
            # Get latest available flights
            flights = db_service.get_recent_flights(request.airport, days=1)
        
        if not flights:
            logger.warning(f"No flights found for {request.airport}")
            return []
        
        # Check for alerts
        new_alerts = alerting_service.check_for_alerts(request.airport, flights)
        
        # Send notifications for new alerts
        for alert in new_alerts:
            try:
                alerting_service.send_alert_notification(alert)
            except Exception as e:
                logger.error(f"Failed to send notification for alert {alert.alert_id}: {e}")
        
        # Convert to response format
        alert_responses = []
        for alert in new_alerts:
            alert_responses.append(AlertResponse(
                alert_id=alert.alert_id,
                alert_type=alert.alert_type.value,
                severity=alert.severity.value,
                title=alert.title,
                description=alert.description,
                airport=alert.airport,
                timestamp=alert.timestamp,
                affected_flights=alert.affected_flights,
                recommendations=[
                    {
                        "action": rec.action,
                        "impact": rec.impact,
                        "priority": rec.priority,
                        "estimated_improvement": rec.estimated_improvement
                    }
                    for rec in alert.recommendations
                ],
                metrics=alert.metrics,
                resolved=alert.resolved,
                escalated=alert.escalated
            ))
        
        logger.info(f"Generated {len(new_alerts)} new alerts for {request.airport}")
        return alert_responses
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking alerts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to check alerts: {str(e)}")


@app.get("/alerts/active", response_model=List[AlertResponse])
async def get_active_alerts(
    airport: Optional[str] = Query(None, description="Filter by airport code"),
    severity: Optional[str] = Query(None, description="Filter by severity: low, medium, high, critical")
):
    """
    Get all active alerts with optional filtering.
    
    Returns currently active alerts that have not been resolved.
    Can be filtered by airport and/or severity level.
    """
    try:
        # Parse severity filter
        severity_filter = None
        if severity:
            from src.services.alerting import AlertSeverity
            try:
                severity_filter = AlertSeverity(severity.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid severity: {severity}")
        
        # Get active alerts
        active_alerts = alerting_service.get_active_alerts(
            airport=airport,
            severity=severity_filter
        )
        
        # Convert to response format
        alert_responses = []
        for alert in active_alerts:
            alert_responses.append(AlertResponse(
                alert_id=alert.alert_id,
                alert_type=alert.alert_type.value,
                severity=alert.severity.value,
                title=alert.title,
                description=alert.description,
                airport=alert.airport,
                timestamp=alert.timestamp,
                affected_flights=alert.affected_flights,
                recommendations=[
                    {
                        "action": rec.action,
                        "impact": rec.impact,
                        "priority": rec.priority,
                        "estimated_improvement": rec.estimated_improvement
                    }
                    for rec in alert.recommendations
                ],
                metrics=alert.metrics,
                resolved=alert.resolved,
                escalated=alert.escalated
            ))
        
        logger.info(f"Retrieved {len(active_alerts)} active alerts")
        return alert_responses
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting active alerts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get active alerts: {str(e)}")


@app.get("/alerts/summary", response_model=AlertSummaryResponse)
async def get_alert_summary(
    airport: Optional[str] = Query(None, description="Filter by airport code")
):
    """
    Get summary statistics of alert status.
    
    Returns counts by severity and type, escalation status,
    and timing information for active alerts.
    """
    try:
        summary = alerting_service.get_alert_summary(airport=airport)
        
        return AlertSummaryResponse(
            total_active_alerts=summary["total_active_alerts"],
            by_severity=summary["by_severity"],
            by_type=summary["by_type"],
            escalated_alerts=summary["escalated_alerts"],
            oldest_alert=summary["oldest_alert"],
            most_recent_alert=summary["most_recent_alert"]
        )
        
    except Exception as e:
        logger.error(f"Error getting alert summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get alert summary: {str(e)}")


@app.post("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str):
    """
    Mark an alert as resolved.
    
    Removes the alert from active status and sends a resolution notification.
    """
    try:
        success = alerting_service.resolve_alert(alert_id)
        
        if success:
            logger.info(f"Alert {alert_id} resolved")
            return {"status": "resolved", "alert_id": alert_id}
        else:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resolving alert {alert_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to resolve alert: {str(e)}")


@app.post("/alerts/{alert_id}/escalate")
async def escalate_alert(alert_id: str):
    """
    Escalate an alert to higher severity level.
    
    Increases the alert severity and sends an escalation notification.
    """
    try:
        success = alerting_service.escalate_alert(alert_id)
        
        if success:
            logger.info(f"Alert {alert_id} escalated")
            return {"status": "escalated", "alert_id": alert_id}
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Alert {alert_id} cannot be escalated (not found, already critical, or already escalated)"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error escalating alert {alert_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to escalate alert: {str(e)}")


@app.post("/alerts/test-notification")
async def test_alert_notification():
    """
    Send a test alert notification to verify webhook configuration.
    
    Useful for testing Slack integration and notification delivery.
    """
    try:
        from src.services.alerting import Alert, AlertType, AlertSeverity, AlertRecommendation
        
        # Create test alert
        test_alert = Alert(
            alert_id=f"test_alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            alert_type=AlertType.SYSTEM_ERROR,
            severity=AlertSeverity.LOW,
            title="🧪 Test Alert - System Check",
            description="This is a test alert to verify notification delivery. No action required.",
            airport="TEST",
            timestamp=datetime.now(),
            recommendations=[
                AlertRecommendation(
                    action="Verify alert system is working correctly",
                    impact="Confirms notification delivery",
                    priority=1,
                    estimated_improvement="System validation complete"
                )
            ],
            metrics={
                "test_metric": "success",
                "notification_test": True
            }
        )
        
        # Send test notification
        success = alerting_service.send_alert_notification(test_alert)
        
        if success:
            return {
                "status": "success",
                "message": "Test notification sent successfully",
                "alert_id": test_alert.alert_id
            }
        else:
            return {
                "status": "failed",
                "message": "Test notification failed - check webhook configuration",
                "alert_id": test_alert.alert_id
            }
            
    except Exception as e:
        logger.error(f"Error sending test notification: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to send test notification: {str(e)}")