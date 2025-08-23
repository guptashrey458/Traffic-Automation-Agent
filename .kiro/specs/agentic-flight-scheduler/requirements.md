# Requirements Document

## Introduction

The Agentic AI Flight Scheduling system addresses critical capacity and delay management challenges at busy Indian aviation hubs (Mumbai BOM, Delhi DEL). The system provides intelligent flight scheduling optimization through natural language interaction, real-time what-if analysis, and automated delay cascade prevention. By analyzing historical flight data (STD/ATD/STA/ATA patterns), the system identifies optimal takeoff/landing slots, predicts delay risks, and provides actionable recommendations to reduce operational disruptions.

## Requirements

### Requirement 1: Data Ingestion and Processing

**User Story:** As an airport operations manager, I want to import and process multiple Excel files containing flight data, so that I can analyze historical patterns and optimize future schedules.

#### Acceptance Criteria

1. WHEN multiple Excel files (up to 59 files) are uploaded THEN the system SHALL parse flight data including Flight No, Date, From, To, Aircraft, Flight Time, STD, ATD, STA, ATA
2. WHEN processing flight times THEN the system SHALL convert IST timestamps to UTC and calculate departure/arrival delays
3. WHEN data contains missing or invalid entries THEN the system SHALL handle gracefully with fallback to scheduled times and mark imputed data
4. WHEN ingestion is complete THEN the system SHALL store normalized data in DuckDB/Parquet format for fast querying

### Requirement 2: Peak Traffic Analysis

**User Story:** As a traffic flow manager, I want to identify the busiest time slots and overload windows, so that I can avoid scheduling conflicts and capacity bottlenecks.

#### Acceptance Criteria

1. WHEN analyzing flight data THEN the system SHALL generate demand heatmaps in 5-minute or 10-minute buckets
2. WHEN demand exceeds capacity THEN the system SHALL highlight overload windows with severity indicators
3. WHEN requested THEN the system SHALL provide separate analysis for arrivals and departures
4. WHEN weather conditions change THEN the system SHALL adjust capacity curves accordingly (calm/medium/strong wind regimes)

### Requirement 3: Delay Risk Prediction

**User Story:** As a flight dispatcher, I want to predict delay risks for individual flights, so that I can proactively adjust schedules to minimize disruptions.

#### Acceptance Criteria

1. WHEN analyzing a flight THEN the system SHALL predict probability of departure delay > 15 minutes
2. WHEN calculating risk THEN the system SHALL consider factors including slot demand, time-of-day, airline, aircraft type, and late inbound connections
3. WHEN providing predictions THEN the system SHALL include both binary risk classification and expected delay minutes
4. WHEN turnaround analysis is requested THEN the system SHALL calculate P90 turnaround times for same-tail operations

### Requirement 4: Cascade Impact Analysis

**User Story:** As an operations controller, I want to identify high-impact flights that create the biggest delay cascades, so that I can prioritize optimization efforts on the most critical operations.

#### Acceptance Criteria

1. WHEN analyzing flight dependencies THEN the system SHALL build cascade graphs showing late arrival impacts on subsequent departures
2. WHEN identifying high-impact flights THEN the system SHALL rank by cascade centrality and downstream effects
3. WHEN a delay occurs THEN the system SHALL trace impact through same-tail operations, stand turnovers, and runway banks
4. WHEN requested THEN the system SHALL provide top-10 high-impact flights with justification

### Requirement 5: Schedule Optimization

**User Story:** As a slot coordinator, I want to optimize flight schedules using constraint-based algorithms, so that I can minimize overall delays while respecting operational constraints.

#### Acceptance Criteria

1. WHEN optimizing schedules THEN the system SHALL use min-cost flow or CP-SAT algorithms with weighted objectives
2. WHEN calculating costs THEN the system SHALL consider expected delays, taxi times, runway changes, fairness, and curfew penalties
3. WHEN applying constraints THEN the system SHALL respect P90 turnaround times, runway availability, wake turbulence rules, and weather capacity
4. WHEN optimization completes THEN the system SHALL provide recommended schedule changes with impact metrics

### Requirement 6: What-If Simulation

**User Story:** As a flight planner, I want to simulate the impact of moving individual flights, so that I can understand the consequences before making schedule changes.

#### Acceptance Criteria

1. WHEN a flight time change is proposed THEN the system SHALL calculate immediate impact on delay metrics
2. WHEN simulation runs THEN the system SHALL show changes in peak overload, 95-percentile delays, and CO₂ impact
3. WHEN multiple scenarios are tested THEN the system SHALL allow comparison of different optimization strategies
4. WHEN results are generated THEN the system SHALL provide clear impact cards with before/after metrics

### Requirement 7: Natural Language Interface

**User Story:** As an airport operations staff member, I want to ask questions in natural language about flight schedules, so that I can quickly get insights without learning complex query syntax.

#### Acceptance Criteria

1. WHEN a natural language query is received THEN the system SHALL classify intent (AskPeaks, AskRisk, AskWhatIf, AskOptimize, AskConstraints)
2. WHEN processing queries THEN the system SHALL support questions like "What are the busiest 30-min slots at BOM?" and "Move AI 2739 by +10m—impact?"
3. WHEN providing responses THEN the system SHALL include both structured data and natural language explanations
4. WHEN explanations are needed THEN the system SHALL provide clear reasoning with A-CDM/CODA terminology

### Requirement 8: Dashboard and Visualization

**User Story:** As an operations manager, I want interactive dashboards showing flight schedules and optimization recommendations, so that I can make informed decisions quickly.

#### Acceptance Criteria

1. WHEN accessing the dashboard THEN the system SHALL display demand heatmaps, flight Gantt charts, and recommendation cards
2. WHEN optimization runs THEN the system SHALL provide "Run Optimize" button with real-time progress
3. WHEN viewing recommendations THEN the system SHALL show impact metrics and explanation tooltips
4. WHEN changes are proposed THEN the system SHALL highlight affected flights with color coding

### Requirement 9: Alerting and Notifications

**User Story:** As a duty manager, I want automated alerts when capacity overloads are detected, so that I can take corrective action before delays cascade.

#### Acceptance Criteria

1. WHEN overload thresholds are exceeded THEN the system SHALL send Slack notifications with severity levels
2. WHEN alerts are triggered THEN the system SHALL include top 3 recommended changes to resolve the issue
3. WHEN critical situations arise THEN the system SHALL escalate through appropriate notification channels
4. WHEN alerts are resolved THEN the system SHALL send confirmation notifications

### Requirement 10: API Integration

**User Story:** As a system integrator, I want RESTful APIs to access flight data and optimization functions, so that I can integrate with existing airport systems.

#### Acceptance Criteria

1. WHEN API calls are made THEN the system SHALL provide endpoints for peaks analysis, risk assessment, what-if simulation, and optimization
2. WHEN requesting data THEN the system SHALL return structured JSON responses with appropriate HTTP status codes
3. WHEN authentication is required THEN the system SHALL implement secure API key or token-based access
4. WHEN rate limiting is needed THEN the system SHALL implement appropriate throttling mechanisms