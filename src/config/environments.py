"""Environment-specific configuration overrides."""

from typing import Dict, Any
from .settings import Settings


class DevelopmentConfig(Settings):
    """Development environment configuration."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api.debug = True
        self.api.reload = True
        self.logging.level = "DEBUG"
        self.ml.retrain_interval_hours = 1  # More frequent retraining in dev


class ProductionConfig(Settings):
    """Production environment configuration."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api.debug = False
        self.api.reload = False
        self.api.workers = 4
        self.logging.level = "INFO"
        self.logging.format = "json"


class TestingConfig(Settings):
    """Testing environment configuration."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.database.duckdb_path = ":memory:"  # In-memory database for tests
        self.database.parquet_path = "tests/data/parquet/"
        self.ml.model_path = "tests/models/"
        self.logging.level = "WARNING"
        self.optimization.max_optimization_time_seconds = 10  # Faster tests


def get_config(environment: str = None) -> Settings:
    """Get configuration for the specified environment."""
    
    if environment is None:
        # Try to get from environment variable or default
        import os
        environment = os.getenv("ENVIRONMENT", "development")
    
    config_map: Dict[str, type] = {
        "development": DevelopmentConfig,
        "production": ProductionConfig,
        "testing": TestingConfig,
    }
    
    config_class = config_map.get(environment.lower(), DevelopmentConfig)
    return config_class()