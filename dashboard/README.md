# Retool Dashboard Interface

This directory contains the complete Retool dashboard configuration for the Agentic Flight Scheduler system. The dashboard provides an intuitive interface for airport operations staff to analyze flight schedules, run optimizations, and perform what-if analysis.

## 📁 Directory Structure

```
dashboard/
├── main_dashboard.json          # Main dashboard configuration
├── retool_config.json          # Global dashboard settings
├── components/                  # Individual component configurations
│   ├── demand_heatmap.json     # Traffic demand visualization
│   ├── flight_gantt.json       # Flight schedule timeline
│   ├── recommendation_cards.json # Optimization suggestions
│   ├── optimize_button.json    # Optimization trigger
│   ├── whatif_interface.json   # What-if analysis modal
│   ├── control_panel.json      # Dashboard controls
│   ├── metrics_summary.json    # Performance metrics
│   └── tooltips_glossary.json  # A-CDM/CODA terminology
├── modals/                      # Modal dialog configurations
│   ├── flight_details.json     # Flight information modal
│   └── help_modal.json         # User guide and help
├── tests/                       # Test suite
│   └── dashboard_tests.js       # Comprehensive test suite
└── README.md                    # This file
```

## 🚀 Quick Start

### Prerequisites

1. **Retool Account**: Sign up at [retool.com](https://retool.com)
2. **API Backend**: Ensure the FastAPI backend is running on `http://localhost:8000`
3. **Database**: DuckDB with flight data loaded

### Deployment Steps

1. **Import Dashboard Configuration**
   ```bash
   # In Retool, create a new app and import main_dashboard.json
   # Or use Retool CLI if available
   retool import main_dashboard.json
   ```

2. **Configure API Connection**
   - Set `API_BASE_URL` environment variable in Retool
   - Default: `http://localhost:8000`
   - Production: Update to your deployed API URL

3. **Import Components**
   - Import each component JSON file from the `components/` directory
   - Import modal configurations from the `modals/` directory

4. **Test Dashboard**
   ```bash
   # Run the test suite
   node dashboard/tests/dashboard_tests.js
   ```

## 🎯 Key Features

### 1. Demand Heatmap Visualization
- **Purpose**: Visualize flight demand vs runway capacity
- **Features**:
  - Color-coded intensity (green=low, red=overload)
  - Configurable time buckets (5, 10, 15, 30 minutes)
  - Weather regime adjustments
  - Interactive time slot selection

### 2. Flight Gantt Chart
- **Purpose**: Timeline view of flight schedules
- **Features**:
  - Color-coded delay risk indicators
  - Runway-based grouping
  - Interactive flight selection
  - Context menu for quick actions

### 3. Optimization Engine Interface
- **Purpose**: Generate schedule improvements
- **Features**:
  - One-click optimization trigger
  - Real-time progress indicators
  - Configurable objective weights
  - Constraint parameter adjustment

### 4. What-If Analysis
- **Purpose**: Simulate impact of schedule changes
- **Features**:
  - Flight time adjustments
  - Runway reassignments
  - Impact metrics calculation
  - Before/after comparisons

### 5. Recommendation Cards
- **Purpose**: Display AI-generated suggestions
- **Features**:
  - Impact metrics (delay reduction, CO₂ savings)
  - Confidence scores
  - One-click application
  - Detailed explanations

## 🎨 Component Details

### Demand Heatmap (`demand_heatmap.json`)

**Configuration Options:**
- `bucket_minutes`: Time interval (5, 10, 15, 30)
- `weather_regime`: Capacity adjustments (calm, medium, strong, severe)
- `color_scale`: Visual intensity mapping

**API Integration:**
```javascript
GET /flights/peaks?airport={airport}&bucket_minutes={bucket}&date={date}&weather_regime={weather}
```

**Interactions:**
- Click time slots to filter related flights
- Hover for detailed capacity information
- Color coding indicates overload severity

### Flight Gantt Chart (`flight_gantt.json`)

**Configuration Options:**
- `time_axis`: 24-hour timeline with configurable intervals
- `resource_axis`: Runway-based grouping
- `color_coding`: Risk-based flight coloring

**API Integration:**
```javascript
GET /flights/risks?airport={airport}&date={date}&risk_threshold={threshold}
```

**Interactions:**
- Click flights for detailed information
- Double-click for what-if analysis
- Right-click for context menu

### Optimization Button (`optimize_button.json`)

**Configuration Options:**
- `objective_weights`: Delay, taxi time, runway change weights
- `constraints`: Max delay, min turnaround, curfew enforcement
- `progress_simulation`: Multi-stage progress indicators

**API Integration:**
```javascript
POST /optimize
{
  "airport": "BOM",
  "date": "2024-01-15",
  "objectives": {...},
  "constraints": {...}
}
```

**Progress Stages:**
1. Loading flight data (10%)
2. Analyzing demand patterns (25%)
3. Building constraint model (40%)
4. Running optimization algorithm (60%)
5. Validating solutions (80%)
6. Generating recommendations (95%)

### What-If Interface (`whatif_interface.json`)

**Configuration Options:**
- `change_types`: Time shift, runway change, gate change, cancellation
- `time_adjustments`: -120 to +120 minutes in 5-minute increments
- `impact_metrics`: Delay, CO₂, affected flights

**API Integration:**
```javascript
POST /whatif
{
  "flight_id": "AI2739_20240115",
  "change_type": "time_shift",
  "change_value": "+10m",
  "airport": "BOM",
  "date": "2024-01-15"
}
```

## 🔧 Customization

### Adding New Metrics

1. **Update Metrics Summary** (`metrics_summary.json`):
```json
{
  "name": "new_metric",
  "title": "New Metric",
  "data_source": "api_endpoint.field",
  "format": "number|percentage|currency",
  "color": "conditional_color_logic",
  "icon": "material_icon_name"
}
```

2. **Add API Endpoint** (if needed):
```python
@app.get("/new-endpoint")
async def get_new_metric():
    return {"new_field": calculated_value}
```

### Customizing Color Schemes

**Heatmap Colors** (`demand_heatmap.json`):
```json
"color_scale": {
  "colors": ["#custom1", "#custom2", "#custom3", "#custom4"],
  "domain": [0, 0.7, 0.9, 1.2],
  "labels": ["Low", "Medium", "High", "Critical"]
}
```

**Risk Level Colors** (`flight_gantt.json`):
```json
"color_coding": {
  "scheme": {
    "low": "#4caf50",
    "medium": "#ff9800",
    "high": "#f44336",
    "critical": "#9c27b0"
  }
}
```

### Adding New Tooltips

**Update Glossary** (`tooltips_glossary.json`):
```json
"NEW_TERM": {
  "term": "New Aviation Term",
  "definition": "Detailed explanation of the term",
  "context": "How it applies to A-CDM/CODA",
  "example": "Example usage or value"
}
```

## 📱 Responsive Design

### Breakpoints
- **Mobile**: < 768px
- **Tablet**: 768px - 1024px
- **Desktop**: > 1024px

### Mobile Optimizations
- Component stacking
- Simplified controls
- Touch-friendly interactions
- Reduced data density

### Tablet Optimizations
- Hybrid layout
- Collapsible panels
- Gesture support

## 🧪 Testing

### Running Tests

```bash
# Install dependencies
npm install jest

# Run test suite
npm test dashboard/tests/dashboard_tests.js

# Run specific test category
npm test -- --grep "Component Loading"
```

### Test Categories

1. **Component Loading**: Verify all components load correctly
2. **User Interactions**: Test clicks, selections, form inputs
3. **API Integration**: Validate API calls and data handling
4. **Responsiveness**: Check mobile/tablet/desktop layouts
5. **Performance**: Measure load times and data processing
6. **Error Handling**: Test failure scenarios and recovery

### Manual Testing Checklist

- [ ] Dashboard loads without errors
- [ ] All components display data correctly
- [ ] Airport/date selection updates all components
- [ ] Optimization button triggers and shows progress
- [ ] What-if analysis modal opens and functions
- [ ] Tooltips display on hover
- [ ] Mobile layout adapts properly
- [ ] Error messages display for API failures
- [ ] Performance acceptable with large datasets

## 🚀 Deployment

### Development Environment

1. **Local API**: `http://localhost:8000`
2. **Retool Development**: Use Retool's development environment
3. **Test Data**: Use sample flight data for testing

### Production Environment

1. **API Endpoint**: Update `API_BASE_URL` to production URL
2. **Authentication**: Configure API keys/tokens
3. **Performance**: Enable caching and optimization
4. **Monitoring**: Set up error tracking and analytics

### Environment Variables

```javascript
// In Retool app settings
API_BASE_URL=https://api.yourcompany.com
API_KEY=your_api_key_here
ENVIRONMENT=production
DEBUG_MODE=false
```

## 🔒 Security Considerations

### API Security
- Use HTTPS for all API communications
- Implement proper authentication (JWT tokens)
- Validate all user inputs
- Rate limiting on API endpoints

### Data Privacy
- Mask sensitive flight information
- Implement role-based access control
- Audit logging for all actions
- Secure data transmission

## 📊 Performance Optimization

### Data Loading
- Implement pagination for large datasets
- Use caching for frequently accessed data
- Lazy loading for non-critical components
- Optimize API response sizes

### UI Performance
- Minimize component re-renders
- Use virtual scrolling for large lists
- Optimize image and asset loading
- Implement proper loading states

## 🆘 Troubleshooting

### Common Issues

**Dashboard Not Loading**
- Check API connectivity
- Verify Retool app configuration
- Review browser console for errors

**Components Not Updating**
- Check variable bindings
- Verify API response format
- Review component refresh triggers

**Performance Issues**
- Reduce data query frequency
- Optimize component configurations
- Check for memory leaks

**Mobile Layout Issues**
- Test responsive breakpoints
- Verify mobile-specific configurations
- Check touch interaction handling

## 📞 Support

For technical support or questions:
- **Documentation**: Review this README and component configurations
- **Testing**: Run the test suite to identify issues
- **API Issues**: Check the FastAPI backend logs
- **Retool Support**: Contact Retool support for platform-specific issues

## 🔄 Version History

- **v1.0.0**: Initial dashboard implementation
  - Core components (heatmap, gantt, optimization)
  - What-if analysis interface
  - Mobile responsiveness
  - A-CDM/CODA terminology tooltips
  - Comprehensive test suite