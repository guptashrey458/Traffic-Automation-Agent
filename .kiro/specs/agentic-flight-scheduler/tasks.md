# Implementation Plan

- [x] 1. Set up project structure and data ingestion foundation

  - Create directory structure for services, models, and APIs
  - Set up Python environment with required dependencies (pandas, duckdb, fastapi, lightgbm)
  - Create configuration management for different environments
  - _Requirements: 1.1, 1.2_

- [x] 2. Implement core data models and validation

  - Define Flight dataclass with all required fields (flight_id, std_utc, atd_utc, etc.)
  - Create validation functions for timestamp conversion and data quality checks
  - Implement data normalization utilities for airline/airport codes
  - Write unit tests for data model validation
  - _Requirements: 1.1, 1.2, 1.3_

- [x] 3. Build Excel data ingestion pipeline

  - Create DataIngestionService class to process multiple Excel files
  - Implement IST to UTC timestamp conversion with timezone handling
  - Add delay calculation logic (dep_delay_min, arr_delay_min)
  - Handle missing data gracefully with fallback to scheduled times
  - Write tests with sample Excel data to validate ingestion accuracy
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [x] 4. Set up DuckDB storage and querying

  - Configure DuckDB database with optimized schema for flight data
  - Implement data storage functions to save normalized flight data
  - Create indexing strategy for fast time-based queries
  - Add data export to Parquet format for analytics
  - Write integration tests for database operations
  - _Requirements: 1.4_

- [x] 5. Implement peak traffic analysis engine

  - Create AnalyticsEngine class with peak detection algorithms
  - Build time bucketing functions (5-min, 10-min intervals)
  - Implement demand vs capacity calculation logic
  - Generate heatmap data structures for visualization
  - Add capacity adjustment logic for weather regimes
  - Write tests to validate peak detection accuracy
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [x] 6. Build delay risk prediction models

  - Create DelayRiskPredictor class with LightGBM integration
  - Implement feature engineering for delay prediction (time-of-day, airline, aircraft type)
  - Train departure delay risk model (binary classification + regression)
  - Train arrival delay risk model with similar approach
  - Add model evaluation metrics and validation
  - Write tests for prediction accuracy and edge cases
  - _Requirements: 3.1, 3.2, 3.3_

- [x] 7. Implement turnaround time analysis

  - Add turnaround time calculation for same-tail operations
  - Implement P90 quantile estimation for turnaround times
  - Create taxi time estimation functions (EXOT/EXIN)
  - Build validation logic for feasible departure slots
  - Write tests for turnaround time accuracy
  - _Requirements: 3.4_

- [x] 8. Create cascade impact analysis system

  - Build dependency graph construction for flight cascades
  - Implement cascade centrality scoring algorithms
  - Create high-impact flight identification logic
  - Add downstream impact tracing for same-tail and stand operations
  - Generate top-10 high-impact flights ranking
  - Write tests for cascade graph accuracy
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [x] 9. Build schedule optimization engine

  - Create ScheduleOptimizer class with min-cost flow implementation
  - Define cost function with weighted objectives (delay, taxi, fairness, curfew)
  - Implement constraint satisfaction for operational rules
  - Add runway capacity and weather regime constraints
  - Create optimization result formatting and validation
  - Write tests for optimization correctness and constraint satisfaction
  - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [x] 10. Implement what-if simulation system

  - Create what-if analysis functions for single flight changes
  - Build impact calculation for delay metrics and peak overload
  - Implement before/after comparison logic
  - Add CO₂ impact estimation for schedule changes
  - Generate impact cards with clear metrics
  - Write tests for simulation accuracy and edge cases
  - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [x] 11. Build natural language interface with Gemini Pro

  - Create NLInterface class with Gemini Pro integration
  - Implement intent classification for query types (AskPeaks, AskRisk, AskWhatIf, etc.)
  - Add parameter extraction from natural language queries
  - Build tool orchestration logic to route intents to appropriate services
  - Create response formatting with natural language explanations
  - Write tests for intent classification accuracy
  - _Requirements: 7.1, 7.2, 7.3, 7.4_

- [x] 12. Create FastAPI backend with REST endpoints

  - Set up FastAPI application with proper routing structure
  - Implement GET /flights/peaks endpoint with airport and bucket parameters
  - Create POST /optimize endpoint for schedule optimization requests
  - Add POST /whatif endpoint for single flight impact analysis
  - Implement GET /flights/risks endpoint for delay risk queries
  - Add GET /constraints endpoint for operational rules
  - Write API integration tests and validate response schemas
  - _Requirements: 10.1, 10.2, 10.3, 10.4_

- [x] 13. Build Retool dashboard interface

  - Create demand heatmap visualization component
  - Build flight Gantt chart display with color coding
  - Implement recommendation cards with impact metrics
  - Add "Run Optimize" button with progress indicators
  - Create what-if analysis interface for flight adjustments
  - Add explanation tooltips with A-CDM/CODA terminology
  - Test dashboard responsiveness and user interaction flows
  - _Requirements: 8.1, 8.2, 8.3, 8.4_

- [x] 14. Implement alerting and notification system

  - Create alert detection logic for capacity overload thresholds
  - Build Zapier integration for Slack notifications
  - Implement severity level classification for alerts
  - Add top-3 recommendations in alert messages
  - Create escalation logic for critical situations
  - Add confirmation notifications when alerts are resolved
  - Write tests for alert triggering and notification delivery
  - _Requirements: 9.1, 9.2, 9.3, 9.4_

- [x] 15. Implement weighted graph optimization engine

  - Create BipartiteGraph class to model flights ↔ runway-time slots
  - Implement multi-objective cost function with runway-dependent weights (delay, taxi, fairness, CO₂, curfew)
  - Build CP-SAT solver integration with heuristic fallback for constraint satisfaction
  - Add feasibility checking for turnaround times, wake separation, and curfew constraints
  - Create graph-based runway assignment with capacity constraints
  - Write tests for graph construction and optimization correctness
  - _Requirements: 13.1, 13.2, 13.3, 13.4_

- [x] 16. Build autonomous monitoring and policy engine

  - Create AutonomousMonitor class with policy-based condition evaluation
  - Implement threshold detection for utilization overload and delay cascades
  - Build autonomous decision-making engine with confidence scoring
  - Add guardrail checking for maximum changes and fairness constraints
  - Create escalation logic for complex scenarios requiring human approval
  - Implement audit logging for all autonomous decisions and reasoning
  - Write tests for policy evaluation and autonomous action execution
  - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.5, 15.6_

- [x] 17. Implement FlightRadar24/FlightAware data ingestion for hackathon compliance

  - Create FlightAware AeroAPI integration for official schedule data ingestion
  - Build FlightRadar24 HTML parser for saved page data extraction
  - Implement data normalization to match existing pipeline schema (flight_no, std_utc, atd_utc, etc.)
  - Add BOM/DEL airport data collection for 1-week historical schedules
  - Create unified data loader that works with Excel, FlightAware, and FR24 sources
  - Build command-line interface for switching between data sources
  - Write tests for data ingestion accuracy and schema compliance
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [x] 18. Implement offline-replay mode for reliable demo execution

  - Create configuration system to switch between offline-replay and live modes
  - Build replay data ingestion for XLSX files with time-based simulation
  - Implement simulated real-time monitoring using historical data patterns
  - Add weather regime simulation and capacity adjustment modeling
  - Create demo scripts that showcase autonomous agent capabilities using multiple data sources
  - Build console-based alert system that mimics Slack notifications
  - Write integration tests for offline-replay mode functionality
  - _Requirements: 11.6, 12.5, 14.4_

- [x] 19. Enhance multi-provider NLP resilience and agent orchestration

  - Implement multi-provider NLP chain (Gemini → Perplexity → OpenAI → Local)
  - Add automatic fallback handling for rate limits and API failures
  - Create tool orchestration engine for autonomous agent decision-making
  - Build transparent reasoning and explanation generation for agent actions
  - Implement confidence-based action execution with human escalation
  - Add context-aware conversation management for complex queries
  - Write tests for NLP provider fallbacks and tool orchestration
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 15.1, 15.6_

- [x] 20. Build continuous learning pipeline foundation

  - Create feature engineering pipeline with online/offline parity
  - Implement model performance monitoring and drift detection
  - Build automated retraining triggers based on performance degradation
  - Add ensemble prediction methods combining multiple model approaches
  - Create model versioning and deployment pipeline with rollback capabilities
  - Implement incremental learning capabilities for real-time model updates
  - Write tests for model training, evaluation, and deployment processes
  - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6_

- [x] 21. Integrate weather modeling and capacity management

  - Create WeatherIntegrationService for weather-based capacity adjustments
  - Implement weather regime classification (calm/medium/strong/severe)
  - Build predictive weather impact modeling for capacity planning
  - Add weather scenario generation for contingency planning
  - Create weather-aware optimization with capacity reduction factors
  - Implement weather forecast integration for proactive schedule adjustments
  - Write tests for weather impact calculations and capacity adjustments
  - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5, 14.6_

- [ ] 22. Create comprehensive demo and hackathon presentation system

  - Build end-to-end demo script showcasing autonomous agent capabilities with multiple data sources
  - Create impact cards with quantified metrics (delay reduction, OTP improvement, CO₂ savings)
  - Implement console-based monitoring dashboard for live demo execution
  - Add what-if analysis demonstrations with before/after comparisons using FR24/FlightAware data
  - Create autonomous alert generation with Slack-style formatting
  - Build presentation materials explaining data source compliance and autonomous agent architecture
  - Write comprehensive demo documentation and troubleshooting guide
  - _Requirements: All requirements demonstration and validation_
