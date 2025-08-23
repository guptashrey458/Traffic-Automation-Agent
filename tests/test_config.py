"""Test configuration management."""

import pytest
from src.config.settings import Settings
from src.config.environments import get_config, DevelopmentConfig, ProductionConfig, TestingConfig


def test_default_settings():
    """Test default settings initialization."""
    settings = Settings()
    
    assert settings.environment == "development"
    assert settings.database.duckdb_path == "data/flights.duckdb"
    assert settings.api.host == "0.0.0.0"
    assert settings.api.port == 8000
    assert settings.ml.model_path == "models/"
    assert settings.logging.level == "INFO"


def test_development_config():
    """Test development environment configuration."""
    config = DevelopmentConfig()
    
    assert config.api.debug is True
    assert config.api.reload is True
    assert config.logging.level == "DEBUG"
    assert config.ml.retrain_interval_hours == 1


def test_production_config():
    """Test production environment configuration."""
    config = ProductionConfig()
    
    assert config.api.debug is False
    assert config.api.reload is False
    assert config.api.workers == 4
    assert config.logging.level == "INFO"
    assert config.logging.format == "json"


def test_testing_config():
    """Test testing environment configuration."""
    config = TestingConfig()
    
    assert config.database.duckdb_path == ":memory:"
    assert config.database.parquet_path == "tests/data/parquet/"
    assert config.ml.model_path == "tests/models/"
    assert config.logging.level == "WARNING"
    assert config.optimization.max_optimization_time_seconds == 10


def test_get_config():
    """Test environment-specific config retrieval."""
    dev_config = get_config("development")
    assert isinstance(dev_config, DevelopmentConfig)
    
    prod_config = get_config("production")
    assert isinstance(prod_config, ProductionConfig)
    
    test_config = get_config("testing")
    assert isinstance(test_config, TestingConfig)
    
    # Test default fallback
    default_config = get_config("unknown")
    assert isinstance(default_config, DevelopmentConfig)