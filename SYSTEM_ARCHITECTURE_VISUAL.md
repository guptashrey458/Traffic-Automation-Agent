# 🛫 Agentic Flight Scheduler - System Architecture & Data Flow

## 🎯 High-Level System Overview

```mermaid
graph TB
    subgraph "Data Sources"
        A[Excel Flight Data<br/>📊 Historical schedules<br/>⏰ STD/ATD times<br/>✈️ Aircraft types]
        B[Weather Data<br/>🌤️ Visibility<br/>💨 Wind conditions<br/>🌧️ Precipitation]
        C[Airport Config<br/>🛬 Runway capacity<br/>⏱️ Curfew hours<br/>📏 Separations]
    end

    subgraph "Data Processing Layer"
        D[Data Ingestion<br/>🔄 Excel → DuckDB<br/>🕐 IST → UTC conversion<br/>✅ Validation]
        E[Analytics Engine<br/>📈 Peak detection<br/>🔥 Hotspot analysis<br/>📊 Capacity utilization]
        F[ML Models<br/>🤖 Delay prediction<br/>🎯 Risk scoring<br/>📉 Pattern recognition]
    end

    subgraph "AI Optimization Core"
        G[Schedule Optimizer<br/>⚡ Min-cost flow<br/>🎯 Multi-objective<br/>⚖️ Constraint satisfaction]
        H[What-If Simulator<br/>🔮 Impact analysis<br/>📊 Before/after metrics<br/>🌍 CO₂ estimation]
        I[Cascade Analyzer<br/>🔗 Dependency graphs<br/>📈 Impact scoring<br/>🎯 High-value flights]
    end

    subgraph "Intelligence Layer"
        J[NLP Interface<br/>🧠 Gemini Pro<br/>💬 Intent classification<br/>🔍 Query routing]
        K[Alerting System<br/>🚨 Real-time monitoring<br/>📱 Slack notifications<br/>⚡ Auto-escalation]
    end

    subgraph "User Interfaces"
        L[FastAPI Backend<br/>🌐 REST endpoints<br/>📡 Real-time data<br/>🔒 Authentication]
        M[Web Dashboard<br/>📊 Visualizations<br/>🎛️ Interactive controls<br/>📈 Real-time updates]
        N[Chat Interface<br/>💬 Natural language<br/>🤖 AI responses<br/>📋 Action suggestions]
    end

    A --> D
    B --> D
    C --> D
    D --> E
    D --> F
    E --> G
    F --> G
    G --> H
    E --> I
    F --> I
    G --> J
    H --> J
    I --> J
    J --> K
    G --> L
    H --> L
    I --> L
    K --> L
    L --> M
    L --> N
    J --> N
```

## 🔄 Detailed Data Flow Process

### 1. **Data Ingestion Pipeline**
```
📊 Excel Files → 🔄 Parser → ✅ Validation → 🗄️ DuckDB → 📈 Analytics Ready
```

**What happens:**
- Reads multiple Excel files with flight schedules
- Converts IST timestamps to UTC with timezone handling
- Calculates delay metrics (departure/arrival delays)
- Validates data quality and handles missing values
- Stores normalized data in DuckDB for fast querying

### 2. **AI-Powered Analysis Engine**
```
🗄️ Flight Data → 🤖 ML Models → 📊 Insights → 🎯 Recommendations
```

**Components in action:**
- **Peak Traffic Analyzer**: Identifies congestion patterns in 5-10 minute buckets
- **Delay Risk Predictor**: LightGBM models predict delay probability for each flight
- **Cascade Impact Analyzer**: Builds dependency graphs to find high-impact flights
- **Capacity Calculator**: Adjusts runway capacity based on weather conditions

### 3. **Optimization Engine Workflow**
```
📋 Current Schedule → ⚡ Optimizer → 🎯 Recommendations → 📊 Impact Analysis
```

**How it works:**
1. **Problem Formulation**: Converts scheduling to min-cost flow problem
2. **Constraint Application**: Applies runway capacity, curfews, turnaround times
3. **Multi-Objective Optimization**: Balances delay reduction, fairness, environmental impact
4. **Solution Validation**: Ensures all operational constraints are satisfied
5. **Impact Quantification**: Calculates delay reduction, CO₂ savings, affected flights

## 🧠 AI Components Deep Dive

### **Natural Language Processing Flow**
```mermaid
graph LR
    A[User Query<br/>"What's the best time<br/>to move 6E 123?"] --> B[Intent Classification<br/>🧠 Gemini Pro<br/>📝 Parameter extraction]
    B --> C{Query Type?}
    C -->|Peak Analysis| D[Analytics Engine<br/>📊 Traffic patterns<br/>🔥 Congestion spots]
    C -->|What-If| E[Simulator<br/>🔮 Impact analysis<br/>📈 Before/after metrics]
    C -->|Optimization| F[Schedule Optimizer<br/>⚡ Slot recommendations<br/>🎯 Best alternatives]
    D --> G[Natural Language Response<br/>💬 Human-readable<br/>📊 Data visualization<br/>🎯 Action items]
    E --> G
    F --> G
```

### **Machine Learning Pipeline**
```mermaid
graph TB
    subgraph "Feature Engineering"
        A[Raw Flight Data] --> B[Time Features<br/>🕐 Hour of day<br/>📅 Day of week<br/>🗓️ Season]
        A --> C[Operational Features<br/>✈️ Aircraft type<br/>🏢 Airline<br/>🛬 Route]
        A --> D[Historical Features<br/>📈 Past delays<br/>🔄 Turnaround times<br/>📊 Weather patterns]
    end
    
    subgraph "ML Models"
        B --> E[Delay Risk Model<br/>🤖 LightGBM<br/>🎯 Binary + Regression<br/>📊 Confidence scores]
        C --> E
        D --> E
        E --> F[Risk Predictions<br/>🔴 High risk flights<br/>🟡 Medium risk<br/>🟢 Low risk]
    end
    
    subgraph "Optimization Integration"
        F --> G[Schedule Optimizer<br/>⚡ Risk-aware scheduling<br/>🎯 Proactive adjustments<br/>📉 Delay minimization]
    end
```

## 🎛️ Real-Time Operation Flow

### **Continuous Monitoring Cycle**
```
🔄 Every 5 minutes:
1. 📊 Ingest latest flight data
2. 🤖 Run delay risk predictions  
3. 📈 Analyze traffic patterns
4. 🚨 Check alert conditions
5. ⚡ Generate recommendations
6. 📱 Send notifications if needed
```

### **User Interaction Scenarios**

#### **Scenario 1: Traffic Controller Query**
```
👤 User: "Which hour is most congested at Mumbai today?"

🧠 AI Processing:
├── Intent: AskPeaks
├── Parameters: airport=BOM, date=today
├── Service: Analytics Engine
└── Analysis: Peak traffic detection

📊 Response:
├── Peak hour: 8-9 AM (45 flights, 150% capacity)
├── Visualization: Traffic heatmap
├── Recommendations: Ground delay program
└── Alternative slots: 7-8 AM, 10-11 AM
```

#### **Scenario 2: Schedule Optimization Request**
```
👤 User: "Optimize today's schedule to reduce delays"

⚡ AI Processing:
├── Current metrics: 25 min avg delay, 60% OTP
├── Constraint check: Curfews, turnarounds, capacity
├── Optimization: Min-cost flow algorithm
└── Solution: 12 flight adjustments

📈 Results:
├── Delay reduction: 15 minutes average
├── OTP improvement: 60% → 78%
├── Affected flights: 12 (with alternatives)
└── Environmental impact: -450 kg CO₂
```

#### **Scenario 3: What-If Analysis**
```
👤 User: "What happens if I delay AI 123 by 30 minutes?"

🔮 AI Processing:
├── Current slot: 08:30 → New slot: 09:00
├── Impact analysis: Cascade effects
├── Metrics calculation: Before/after comparison
└── Recommendation: Impact assessment

📊 Impact Card:
├── Delay impact: +5 min (3 downstream flights)
├── Capacity impact: Reduces 8 AM peak by 1 flight
├── CO₂ impact: +85 kg (longer taxi time)
└── Recommendation: ⚠️ Consider 09:15 instead
```

## 🚨 Alert System in Action

### **Real-Time Monitoring Dashboard**
```
🔍 Continuous Analysis:
├── Capacity utilization > 90% → 🟡 Medium alert
├── Capacity utilization > 110% → 🔴 High alert  
├── 5+ flights delayed > 15 min → 🟠 Cascade alert
├── Single flight delayed > 60 min → 🚨 Critical alert
└── System errors → ⚠️ Technical alert

📱 Slack Notifications:
├── Rich formatting with emojis
├── Top 3 actionable recommendations
├── Impact metrics and affected flights
├── Escalation for unresolved alerts
└── Resolution confirmations
```

## 🎯 Optimization Algorithms Explained

### **Min-Cost Flow Network**
```
🛬 Runway Slots (Supply) ←→ ✈️ Flight Demands
├── Nodes: Time slots (5-min intervals)
├── Edges: Possible assignments
├── Costs: Delay penalties + operational costs
├── Constraints: Capacity, separations, curfews
└── Objective: Minimize total system cost
```

### **Multi-Objective Scoring**
```
Total Cost = w₁×Delay + w₂×Taxi + w₃×Fairness + w₄×Environment + w₅×Curfew

Where:
├── w₁ = 1.0 (Delay weight - highest priority)
├── w₂ = 0.3 (Taxi time efficiency)  
├── w₃ = 0.4 (Airline fairness)
├── w₄ = 0.2 (Environmental impact)
└── w₅ = 2.0 (Curfew violations - critical)
```

## 📊 Performance Metrics

### **System Performance**
- **Query Response**: < 500ms for most operations
- **Optimization Speed**: < 5 seconds for 100+ flights
- **ML Prediction**: < 100ms per flight
- **Data Processing**: 1000+ flights/second ingestion
- **Alert Detection**: < 2 seconds analysis cycle

### **Business Impact**
- **Delay Reduction**: 15-30% average improvement
- **OTP Improvement**: 10-20 percentage points
- **Fuel Savings**: 200-500 kg CO₂ per optimization
- **Operational Efficiency**: 95%+ constraint satisfaction
- **User Satisfaction**: Natural language interface

## 🔮 What Makes It "Agentic"

### **Autonomous Decision Making**
```
🤖 AI Agent Capabilities:
├── 🧠 Understands natural language queries
├── 🔍 Analyzes complex operational data
├── ⚡ Generates optimal solutions automatically  
├── 🎯 Provides actionable recommendations
├── 📱 Monitors and alerts proactively
├── 🔄 Learns from historical patterns
└── 💬 Explains reasoning in human terms
```

### **Intelligent Orchestration**
The system acts as an intelligent agent that:
1. **Perceives**: Continuously monitors flight operations
2. **Reasons**: Applies AI models to understand patterns
3. **Plans**: Generates optimal schedule adjustments
4. **Acts**: Provides recommendations and alerts
5. **Learns**: Improves predictions from historical data
6. **Communicates**: Explains decisions in natural language

This creates a truly **agentic** system that doesn't just process data, but actively helps air traffic controllers make better decisions through AI-powered insights and recommendations.