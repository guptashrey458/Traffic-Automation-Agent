"""FastAPI application main module."""

from fastapi import FastAPI
from src.config.environments import get_config

# Get configuration
config = get_config()

# Create FastAPI app
app = FastAPI(
    title="Agentic Flight Scheduler API",
    description="AI-powered flight scheduling optimization system",
    version="0.1.0",
    debug=config.api.debug,
)


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