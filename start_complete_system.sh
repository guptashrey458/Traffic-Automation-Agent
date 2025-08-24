#!/bin/bash

echo "🚀 Starting Complete Agentic Flight Scheduler System..."
echo "======================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}$1${NC}"
}

# Kill any existing processes
print_status "Cleaning up existing processes..."
pkill -f "enhanced_demo_api_server.py" 2>/dev/null
pkill -f "demo_api_server.py" 2>/dev/null
pkill -f "next dev" 2>/dev/null
pkill -f "npm run dev" 2>/dev/null
sleep 2

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        print_error "Python is not installed or not in PATH"
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

print_status "Using Python command: $PYTHON_CMD"

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    print_error "Node.js is not installed or not in PATH"
    print_error "Please install Node.js from https://nodejs.org/"
    exit 1
fi

if ! command -v npm &> /dev/null; then
    print_error "npm is not installed or not in PATH"
    exit 1
fi

print_status "Node.js version: $(node --version)"
print_status "npm version: $(npm --version)"

# Install Python dependencies
print_header "Installing Python Dependencies..."
$PYTHON_CMD -m pip install fastapi uvicorn pandas numpy --break-system-packages --quiet 2>/dev/null || \
$PYTHON_CMD -m pip install fastapi uvicorn pandas numpy --user --quiet 2>/dev/null || \
print_warning "Could not install Python dependencies, but they may already be available"

print_status "Python dependencies check completed"

# Check if dashboard directory exists
if [ ! -d "dashboard" ]; then
    print_error "Dashboard directory not found!"
    exit 1
fi

# Install Node.js dependencies
print_header "Installing Node.js Dependencies..."
cd dashboard

if [ ! -f "package.json" ]; then
    print_error "package.json not found in dashboard directory!"
    exit 1
fi

# Install dependencies
npm install --silent
if [ $? -eq 0 ]; then
    print_status "Node.js dependencies installed successfully"
else
    print_error "Failed to install Node.js dependencies"
    exit 1
fi

cd ..

# Start the backend API
print_header "Starting Backend API..."
if [ -f "working_demo_api_server.py" ]; then
    print_status "Starting working demo backend..."
    $PYTHON_CMD working_demo_api_server.py &
    BACKEND_PID=$!
else
    print_status "Starting original backend..."
    $PYTHON_CMD demo_api_server.py &
    BACKEND_PID=$!
fi

# Wait for backend to start
print_status "Waiting for backend to initialize..."
sleep 5

# Test backend health
print_status "Testing backend connection..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    print_status "✅ Backend API is running on http://localhost:8000"
    print_status "📊 API Documentation: http://localhost:8000/docs"
else
    print_error "❌ Backend failed to start or is not responding"
    print_error "Checking backend process..."
    if kill -0 $BACKEND_PID 2>/dev/null; then
        print_warning "Backend process is running but not responding to health check"
        print_warning "It may still be starting up..."
    else
        print_error "Backend process has died"
        exit 1
    fi
fi

# Start the frontend dashboard
print_header "Starting Frontend Dashboard..."
cd dashboard

# Check if .env.local exists, create if not
if [ ! -f ".env.local" ]; then
    print_status "Creating .env.local file..."
    cat > .env.local << EOF
# API Configuration
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000

# Development
NODE_ENV=development

# Optional: Enable debug logging
DEBUG=false
EOF
fi

# Start the development server
npm run dev &
FRONTEND_PID=$!

cd ..

# Wait a bit for frontend to start
sleep 8

# Test frontend
print_status "Testing frontend connection..."
FRONTEND_PORT=3000
if curl -s http://localhost:$FRONTEND_PORT > /dev/null 2>&1; then
    print_status "✅ Frontend is running on http://localhost:$FRONTEND_PORT"
elif curl -s http://localhost:3001 > /dev/null 2>&1; then
    FRONTEND_PORT=3001
    print_status "✅ Frontend is running on http://localhost:$FRONTEND_PORT"
elif curl -s http://localhost:3002 > /dev/null 2>&1; then
    FRONTEND_PORT=3002
    print_status "✅ Frontend is running on http://localhost:$FRONTEND_PORT"
else
    print_warning "⚠️ Frontend may still be starting up..."
    print_status "Please wait a moment and check http://localhost:3000"
fi

echo ""
print_header "🎉 SYSTEM STARTUP COMPLETE!"
echo "======================================================"
print_status "📡 Backend API: http://localhost:8000"
print_status "🌐 Frontend Dashboard: http://localhost:$FRONTEND_PORT"
print_status "📚 API Documentation: http://localhost:8000/docs"
print_status "🔧 API Health Check: http://localhost:8000/health"
echo ""
print_header "🔍 AVAILABLE FEATURES:"
echo "   • Real-time flight analytics with interactive charts"
echo "   • AI-powered delay prediction with algorithm transparency"
echo "   • Schedule optimization with step-by-step formulas"
echo "   • What-if analysis for scenario simulation"
echo "   • Alert management with notification testing"
echo "   • Live algorithm execution preview"
echo "   • Dark/Light mode with proper contrast"
echo ""
print_header "🎯 QUICK START:"
echo "   1. Open http://localhost:$FRONTEND_PORT in your browser"
echo "   2. Try the 'Test Notification' button in Alerts page"
echo "   3. Run delay prediction to see algorithm formulas"
echo "   4. Use optimization to see step-by-step calculations"
echo "   5. Experiment with what-if analysis scenarios"
echo ""
print_status "Press Ctrl+C to stop both servers"
echo "======================================================"

# Function to cleanup on exit
cleanup() {
    echo ""
    print_header "🛑 Shutting down system..."
    print_status "Stopping backend server (PID: $BACKEND_PID)..."
    kill $BACKEND_PID 2>/dev/null
    print_status "Stopping frontend server (PID: $FRONTEND_PID)..."
    kill $FRONTEND_PID 2>/dev/null
    
    # Kill any remaining processes
    pkill -f "enhanced_demo_api_server.py" 2>/dev/null
    pkill -f "next dev" 2>/dev/null
    pkill -f "npm run dev" 2>/dev/null
    
    sleep 2
    print_status "✅ Cleanup complete"
    echo "Thank you for using the Agentic Flight Scheduler! 🛩️"
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Keep the script running and show live status
while true; do
    sleep 30
    
    # Check if processes are still running
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        print_error "Backend process has stopped unexpectedly!"
        break
    fi
    
    if ! kill -0 $FRONTEND_PID 2>/dev/null; then
        print_error "Frontend process has stopped unexpectedly!"
        break
    fi
    
    # Optional: Show a heartbeat
    # print_status "System running... Backend PID: $BACKEND_PID, Frontend PID: $FRONTEND_PID"
done

# If we get here, something went wrong
print_error "One or more processes have stopped. Cleaning up..."
cleanup