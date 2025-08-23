# Agentic AI Flight Scheduling System

An intelligent flight scheduling optimization system that combines machine learning, optimization algorithms, and natural language processing to provide smart flight schedule management for busy aviation hubs.

## Features

- **Data Ingestion**: Process multiple Excel files with flight data (STD/ATD/STA/ATA)
- **Peak Traffic Analysis**: Identify busy time slots and capacity bottlenecks
- **Delay Risk Prediction**: ML-powered delay risk assessment
- **Schedule Optimization**: Constraint-based optimization algorithms
- **Natural Language Interface**: Query system using natural language
- **What-If Analysis**: Simulate impact of schedule changes
- **Real-time Alerts**: Automated notifications for capacity overloads

## Project Structure

```
├── src/
│   ├── api/           # FastAPI endpoints
│   ├── config/        # Configuration management
│   ├── models/        # Data models and schemas
│   ├── services/      # Business logic services
│   └── utils/         # Utility functions
├── tests/             # Test files
├── data/              # Data storage
│   ├── parquet/       # Processed data files
│   └── backups/       # Database backups
├── models/            # ML model files
└── logs/              # Application logs
```

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Run the Application**:
   ```bash
   python main.py
   ```

## Development

- **Code Formatting**: `black src/ tests/`
- **Import Sorting**: `isort src/ tests/`
- **Type Checking**: `mypy src/`
- **Linting**: `flake8 src/ tests/`
- **Testing**: `pytest`

## Configuration

The system supports multiple environments (development, production, testing) with environment-specific configurations. See `.env.example` for available settings.

## Requirements Coverage

This setup addresses the following requirements:
- **1.1**: Foundation for processing multiple Excel files with flight data
- **1.2**: Environment configuration for different deployment scenarios