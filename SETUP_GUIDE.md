# Agentic Flight Scheduler - Setup Guide

## Quick Start

### 1. Environment Setup

1. **Copy the environment file:**

   ```bash
   cp .env.example .env
   ```

2. **Create required directories:**

   ```bash
   mkdir -p data/parquet data/backups models logs
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   **Note on OR-Tools**: The system includes support for advanced optimization algorithms using Google OR-Tools. If OR-Tools installation fails or causes compatibility issues on your system, the schedule optimizer will automatically fall back to heuristic methods that provide good results without external dependencies.

### 2. Basic Configuration

The `.env` file contains all necessary configuration with sensible defaults. For basic usage, you don't need to change anything.

**Key settings you might want to adjust:**

```bash
# Environment (development/production/testing)
ENVIRONMENT=development

# API settings
API__PORT=8000
API__DEBUG=true

# Database location
DATABASE__DUCKDB_PATH=data/flights.duckdb

# Logging level
LOGGING__LEVEL=INFO
```

### 3. Running the Application

#### Option A: Run the API Server

```bash
python main.py
```

The API will be available at `http://localhost:8000`

#### Option B: Run Analytics Demo

```bash
python demo_analytics.py
```

This demonstrates the peak traffic analysis engine with sample data.

#### Option C: Run Schedule Optimization Demo

```bash
python demo_schedule_optimization.py
```

This demonstrates the constraint-based schedule optimization engine with multi-objective optimization.

#### Option D: Run Tests

```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/test_analytics.py -v
pytest tests/test_schedule_optimizer.py -v
pytest tests/test_analytics_integration.py -v
```

## Advanced Configuration

### External APIs (Optional)

#### Weather API Integration

For real-time weather data affecting runway capacity:

1. Get API key from [OpenWeatherMap](https://openweathermap.org/api)
2. Add to `.env`:
   ```bash
   WEATHER_API_KEY=your_api_key_here
   ```

#### AI Insights with Gemini Pro

For AI-powered scheduling insights:

1. Get API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Add to `.env`:
   ```bash
   GEMINI_API_KEY=your_api_key_here
   ```

### Slack Alerts (Optional)

For operational alerts:

1. Create Slack webhook at [Slack API](https://api.slack.com/messaging/webhooks)
2. Add to `.env`:
   ```bash
   SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
   ```

### Production Configuration

For production deployment, update these settings in `.env`:

```bash
ENVIRONMENT=production
API__DEBUG=false
API__RELOAD=false
API__WORKERS=4
LOGGING__LEVEL=INFO
LOGGING__FORMAT=json
LOGGING__LOG_FILE=logs/app.log
```

## Directory Structure

After setup, your project should look like:

```
agentic-flight-scheduler/
├── .env                    # Your environment configuration
├── .env.example           # Template configuration
├── data/
│   ├── flights.duckdb     # Main database (created automatically)
│   ├── parquet/           # Exported data files
│   └── backups/           # Database backups
├── models/                # ML models storage
├── logs/                  # Application logs (if enabled)
├── src/                   # Source code
├── tests/                 # Test files
└── demo_analytics.py      # Analytics demo
```

## Testing Your Setup

### 1. Run Analytics Demo

```bash
python demo_analytics.py
```

Should show peak traffic analysis with sample Mumbai airport data.

### 2. Run Schedule Optimization Demo

```bash
python demo_schedule_optimization.py
```

Should demonstrate constraint-based optimization with multi-objective algorithms.

### 3. Test API Server

```bash
# Start server
python main.py

# In another terminal, test endpoints
curl http://localhost:8000/
curl http://localhost:8000/health
```

### 4. Run Test Suite

```bash
pytest tests/ -v
```

Should show all tests passing.

## Common Issues & Solutions

### Issue: "ModuleNotFoundError"

**Solution:** Ensure you're in the project root directory and have installed dependencies:

```bash
pip install -r requirements.txt
```

### Issue: "Permission denied" for data directories

**Solution:** Create directories with proper permissions:

```bash
mkdir -p data/parquet data/backups models logs
chmod 755 data/ models/ logs/
```

### Issue: Database connection errors

**Solution:** Ensure the data directory exists and is writable:

```bash
mkdir -p data
touch data/flights.duckdb
```

### Issue: Port already in use

**Solution:** Change the port in `.env`:

```bash
API__PORT=8001
```

## Environment Variables Reference

### Required Settings

- `ENVIRONMENT`: Environment name (development/production/testing)
- `DATABASE__DUCKDB_PATH`: Database file path
- `API__PORT`: API server port

### Optional Settings

- `WEATHER_API_KEY`: For weather-based capacity adjustments
- `GEMINI_API_KEY`: For AI-powered insights
- `SLACK_WEBHOOK_URL`: For operational alerts

### Performance Tuning

- `API__WORKERS`: Number of worker processes
- `BATCH_SIZE`: Data processing batch size
- `ML__PREDICTION_CACHE_TTL`: Prediction cache duration

## Next Steps

1. **Load Flight Data**: Use the data ingestion service to load your Excel files
2. **Run Analytics**: Use the analytics engine to analyze peak traffic patterns
3. **Set Up Monitoring**: Configure alerts for capacity overloads and delays
4. **Customize Capacity**: Update airport capacity configurations as needed

For more detailed information, see the implementation documentation in each module.
