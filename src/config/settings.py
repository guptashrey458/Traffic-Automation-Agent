"""Configuration management for different environments."""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    duckdb_path: str = Field(default="data/flights.duckdb", description="Path to DuckDB database file")
    parquet_path: str = Field(default="data/parquet/", description="Path to Parquet files directory")
    backup_path: str = Field(default="data/backups/", description="Path to database backups")


class APISettings(BaseSettings):
    """API configuration settings."""
    
    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, description="API port")
    debug: bool = Field(default=False, description="Debug mode")
    reload: bool = Field(default=False, description="Auto-reload on changes")
    workers: int = Field(default=1, description="Number of worker processes")


class MLSettings(BaseSettings):
    """Machine learning configuration settings."""
    
    model_path: str = Field(default="models/", description="Path to ML models directory")
    retrain_interval_hours: int = Field(default=24, description="Model retraining interval in hours")
    prediction_cache_ttl: int = Field(default=300, description="Prediction cache TTL in seconds")
    lightgbm_params: dict = Field(
        default={
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": 0
        },
        description="LightGBM model parameters"
    )


class OptimizationSettings(BaseSettings):
    """Optimization configuration settings."""
    
    max_optimization_time_seconds: int = Field(default=300, description="Maximum optimization time")
    what_if_timeout_seconds: int = Field(default=5, description="What-if analysis timeout")
    constraint_weights: dict = Field(
        default={
            "delay_weight": 1.0,
            "taxi_weight": 0.5,
            "fairness_weight": 0.3,
            "curfew_weight": 2.0
        },
        description="Optimization objective weights"
    )


class LoggingSettings(BaseSettings):
    """Logging configuration settings."""
    
    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(default="json", description="Log format (json or text)")
    log_file: Optional[str] = Field(default=None, description="Log file path")


class Settings(BaseSettings):
    """Main application settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False
    )
    
    # Environment
    environment: str = Field(default="development", description="Environment name")
    
    # Component settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    api: APISettings = Field(default_factory=APISettings)
    ml: MLSettings = Field(default_factory=MLSettings)
    optimization: OptimizationSettings = Field(default_factory=OptimizationSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    
    # Data processing
    max_file_size_mb: int = Field(default=100, description="Maximum Excel file size in MB")
    batch_size: int = Field(default=1000, description="Batch size for data processing")
    
    # External APIs
    weather_api_key: Optional[str] = Field(default=None, description="Weather API key")
    gemini_api_key: Optional[str] = Field(default=None, description="Gemini Pro API key")
    
    # Alerting
    slack_webhook_url: Optional[str] = Field(default=None, description="Slack webhook URL for alerts")
    alert_thresholds: dict = Field(
        default={
            "capacity_overload_threshold": 0.9,
            "delay_threshold_minutes": 15,
            "cascade_impact_threshold": 5
        },
        description="Alert threshold configurations"
    )


# Global settings instance
settings = Settings()