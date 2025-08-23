# FastAPI Backend Documentation

## Overview

The Agentic Flight Scheduler API provides REST endpoints for flight scheduling optimization, delay risk analysis, and what-if simulations. The API is built with FastAPI and includes comprehensive validation, error handling, and response schemas.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication. In production, JWT token-based authentication should be implemented.

## Endpoints

### Basic Endpoints

#### GET /
Root endpoint providing API information.

**Response:**
```json
{
  "message": "Agentic Flight Scheduler API",
  "version": "0.1.0",
  "environment": "development",
  "status": "ready"
}
```

#### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "environment": "development"
}
```

#### GET /airports
Get list of supported airports.

**Response:**
```json
{
  "airports": [
    {"code": "BOM", "name": "Mumbai", "city": "Mumbai"},
    {"code": "DEL", "name": "Delhi", "city": "Delhi"},
    {"code": "BLR", "name": "Bangalore", "city": "Bangalore"},
    {"code": "MAA", "name": "Chennai", "city": "Chennai"},
    {"code": "CCU", "name": "Kolkata", "city": "Kolkata"},
    {"code": "HYD", "name": "Hyderabad", "city": "Hyderabad"}
  ]
}
```

#### GET /status
Get system status and service health.

**Response:**
```json
{
  "status": "healthy",
  "services": {
    "database": "healthy",
    "analytics": "healthy",
    "optimizer": "healthy",
    "whatif_simulator": "healthy",
    "delay_predictor": "healthy"
  },
  "timestamp": "2024-01-01T12:00:00"
}
```

### Flight Analysis Endpoints

#### GET /flights/peaks
Analyze peak traffic periods for a specific airport.

**Parameters:**
- `airport` (required): Airport code (BOM, DEL, BLR, MAA, CCU, HYD)
- `bucket_minutes` (optional): Time bucket size in minutes (5, 10, 15, 30). Default: 10
- `date` (optional): Analysis date in YYYY-MM-DD format. Default: latest available
- `weather_regime` (optional): Weather regime (calm, medium, strong, severe). Default: calm

**Example Request:**
```
GET /flights/peaks?airport=BOM&bucket_minutes=10&date=2024-01-01&weather_regime=calm
```

**Response:**
```json
{
  "airport": "BOM",
  "bucket_minutes": 10,
  "analysis_date": "2024-01-01",
  "time_buckets": [
    {
      "start_time": "2024-01-01T06:00:00",
      "end_time": "2024-01-01T06:10:00",
      "total_demand": 15,
      "capacity": 12,
      "overload": 3
    }
  ],
  "overload_windows": [
    {
      "start_time": "2024-01-01T06:00:00",
      "end_time": "2024-01-01T06:30:00",
      "severity": "high",
      "overload_flights": 8
    }
  ],
  "capacity_utilization": 0.85,
  "recommendations": ["Consider redistributing flights from 06:00-06:30"],
  "weather_regime": "calm"
}
```

#### GET /flights/risks
Get delay risk predictions for flights.

**Parameters:**
- `airport` (required): Airport code
- `date` (required): Analysis date in YYYY-MM-DD format
- `flight_ids` (optional): Comma-separated flight IDs
- `risk_threshold` (optional): Minimum risk threshold (0.0-1.0). Default: 0.2

**Example Request:**
```
GET /flights/risks?airport=BOM&date=2024-01-01&risk_threshold=0.3
```

**Response:**
```json
[
  {
    "flight_id": "AI2739",
    "departure_risk": {
      "probability": 0.35,
      "expected_delay": 8.2,
      "risk_level": "moderate"
    },
    "arrival_risk": {
      "probability": 0.28,
      "expected_delay": 6.1,
      "risk_level": "moderate"
    },
    "risk_factors": ["Peak hour slot", "Weather conditions"],
    "recommendations": ["Consider earlier slot", "Monitor weather"]
  }
]
```

### Optimization Endpoints

#### POST /optimize
Optimize flight schedules using constraint-based algorithms.

**Request Body:**
```json
{
  "airport": "BOM",
  "date": "2024-01-01",
  "flights": ["AI2739", "6E1234"],  // Optional: specific flights to optimize
  "objectives": {  // Optional: objective weights
    "delay_weight": 1.0,
    "taxi_weight": 0.3,
    "runway_change_weight": 0.2
  },
  "constraints": {}  // Optional: custom constraints
}
```

**Response:**
```json
{
  "optimization_id": "opt_123",
  "status": "completed",
  "original_metrics": {
    "avg_delay": 12.5
  },
  "optimized_metrics": {
    "avg_delay": 8.2
  },
  "recommended_changes": [
    {
      "flight_id": "AI2739",
      "original_time": "2024-01-01T06:30:00",
      "new_time": "2024-01-01T06:40:00",
      "reason": "Reduce peak congestion"
    }
  ],
  "cost_reduction": 0.35,
  "execution_time_seconds": 2.1
}
```

#### POST /whatif
Perform what-if analysis for single flight changes.

**Request Body:**
```json
{
  "flight_id": "AI2739",
  "change_type": "time_shift",  // "time_shift", "runway_change", "cancellation"
  "change_value": "+10m",  // "+10m", "-5m", "RW09", etc.
  "airport": "BOM",
  "date": "2024-01-01"
}
```

**Response:**
```json
{
  "flight_id": "AI2739",
  "change_description": "Move flight AI2739 by +10 minutes",
  "impact_summary": {
    "delay_change": -2.3,
    "co2_change": -15.2,
    "confidence": "high",
    "affected_flights": 2
  },
  "before_metrics": {
    "avg_delay": 12.5,
    "peak_overload": 3
  },
  "after_metrics": {
    "avg_delay": 10.2,
    "peak_overload": 2
  },
  "affected_flights": [],
  "co2_impact_kg": -15.2
}
```

#### GET /constraints
Get operational constraints and rules for an airport.

**Parameters:**
- `airport` (required): Airport code
- `date` (optional): Date for time-specific constraints

**Example Request:**
```
GET /constraints?airport=BOM
```

**Response:**
```json
{
  "airport": "BOM",
  "operational_rules": {
    "min_turnaround_time": 45,
    "wake_turbulence_separation": true
  },
  "capacity_limits": {
    "runway_09": 30,
    "runway_27": 25
  },
  "weather_adjustments": {
    "calm": 1.0,
    "medium": 0.8,
    "strong": 0.6
  },
  "curfew_hours": {
    "start": "23:00",
    "end": "06:00"
  }
}
```

## Error Handling

The API uses standard HTTP status codes:

- `200`: Success
- `400`: Bad Request (validation errors)
- `422`: Unprocessable Entity (Pydantic validation errors)
- `500`: Internal Server Error

**Error Response Format:**
```json
{
  "detail": "Error message describing what went wrong"
}
```

## Rate Limiting

Currently, no rate limiting is implemented. In production, consider implementing rate limiting based on API keys or IP addresses.

## CORS

CORS is enabled for all origins in development. Configure appropriately for production.

## Running the API

### Development
```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Production
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Testing

Run the API integration tests:
```bash
python -m pytest tests/test_api_integration.py -v
```

Test API startup:
```bash
python test_api_startup.py
```

## Interactive Documentation

FastAPI automatically generates interactive API documentation:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json