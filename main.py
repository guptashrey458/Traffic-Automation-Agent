"""Main application entry point."""

import uvicorn
from src.config.environments import get_config


def main():
    """Run the application."""
    config = get_config()
    
    uvicorn.run(
        "src.api.main:app",
        host=config.api.host,
        port=config.api.port,
        debug=config.api.debug,
        reload=config.api.reload,
        workers=config.api.workers if not config.api.reload else 1,
    )


if __name__ == "__main__":
    main()