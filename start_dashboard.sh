#!/bin/bash

# Flight Scheduler Dashboard Startup Script
echo "🚀 Starting Flight Scheduler Dashboard..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18+ first."
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "flight-dashboard/package.json" ]; then
    echo "❌ flight-dashboard directory not found. Please run this script from the project root."
    exit 1
fi

# Navigate to dashboard directory
cd flight-dashboard

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies"
        exit 1
    fi
fi

# Check if backend is running
echo "🔍 Checking if backend API is running..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Backend API is running on port 8000"
else
    echo "⚠️  Backend API is not running on port 8000"
    echo "   Please start the backend first with: python enhanced_demo_api_server.py"
    echo "   Or run the complete system with: ./start_complete_system.sh"
    echo ""
    echo "   Continuing anyway - dashboard will show connection errors until backend is started..."
fi

# Start the dashboard
echo "🌐 Starting dashboard on http://localhost:3000..."
echo "   Press Ctrl+C to stop"
echo ""

npm run dev