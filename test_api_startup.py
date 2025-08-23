#!/usr/bin/env python3
"""Test script to verify the FastAPI application can start successfully."""

import sys
import traceback
from src.api.main import app

def test_api_startup():
    """Test that the FastAPI app can be imported and initialized."""
    try:
        print("✓ FastAPI app imported successfully")
        
        # Check that the app has the expected routes
        routes = [route.path for route in app.routes]
        expected_routes = [
            "/",
            "/health", 
            "/flights/peaks",
            "/optimize",
            "/whatif",
            "/flights/risks",
            "/constraints",
            "/airports",
            "/status"
        ]
        
        print(f"✓ Found {len(routes)} routes")
        
        for expected_route in expected_routes:
            if expected_route in routes:
                print(f"✓ Route {expected_route} found")
            else:
                print(f"✗ Route {expected_route} missing")
                return False
        
        print("✓ All expected routes are present")
        print("✓ FastAPI application is ready to start")
        return True
        
    except Exception as e:
        print(f"✗ Error during API startup test: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_api_startup()
    sys.exit(0 if success else 1)