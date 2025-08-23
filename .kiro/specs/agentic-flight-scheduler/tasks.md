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

- [ ] 6. Build delay risk prediction models

  - Create DelayRiskPredictor class with LightGBM integration
  - Implement feature engineering for delay prediction (time-of-day, airline, aircraft type)
  - Train departure delay risk model (binary classification + regression)
  - Train arrival delay risk model with similar approach
  - Add model evaluation metrics and validation
  - Write tests for prediction accuracy and edge cases
  - _Requirements: 3.1, 3.2, 3.3_

- [ ] 7. Implement turnaround time analysis

  - Add turnaround time calculation for same-tail operations
  - Implement P90 quantile estimation for turnaround times
  - Create taxi time estimation functions (EXOT/EXIN)
  - Build validation logic for feasible departure slots
  - Write tests for turnaround time accuracy
  - _Requirements: 3.4_

- [ ] 8. Create cascade impact analysis system

  - Build dependency graph construction for flight cascades
  - Implement cascade centrality scoring algorithms
  - Create high-impact flight identification logic
  - Add downstream impact tracing for same-tail and stand operations
  - Generate top-10 high-impact flights ranking
  - Write tests for cascade graph accuracy
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [ ] 9. Build schedule optimization engine

  - Create ScheduleOptimizer class with min-cost flow implementation
  - Define cost function with weighted objectives (delay, taxi, fairness, curfew)
  - Implement constraint satisfaction for operational rules
  - Add runway capacity and weather regime constraints
  - Create optimization result formatting and validation
  - Write tests for optimization correctness and constraint satisfaction
  - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [ ] 10. Implement what-if simulation system

  - Create what-if analysis functions for single flight changes
  - Build impact calculation for delay metrics and peak overload
  - Implement before/after comparison logic
  - Add CO₂ impact estimation for schedule changes
  - Generate impact cards with clear metrics
  - Write tests for simulation accuracy and edge cases
  - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [ ] 11. Build natural language interface with Gemini Pro

  - Create NLInterface class with Gemini Pro integration
  - Implement intent classification for query types (AskPeaks, AskRisk, AskWhatIf, etc.)
  - Add parameter extraction from natural language queries
  - Build tool orchestration logic to route intents to appropriate services
  - Create response formatting with natural language explanations
  - Write tests for intent classification accuracy
  - _Requirements: 7.1, 7.2, 7.3, 7.4_

- [ ] 12. Create FastAPI backend with REST endpoints

  - Set up FastAPI application with proper routing structure
  - Implement GET /flights/peaks endpoint with airport and bucket parameters
  - Create POST /optimize endpoint for schedule optimization requests
  - Add POST /whatif endpoint for single flight impact analysis
  - Implement GET /flights/risks endpoint for delay risk queries
  - Add GET /constraints endpoint for operational rules
  - Write API integration tests and validate response schemas
  - _Requirements: 10.1, 10.2, 10.3, 10.4_

- [ ] 13. Build Retool dashboard interface

  - Create demand heatmap visualization component
  - Build flight Gantt chart display with color coding
  - Implement recommendation cards with impact metrics
  - Add "Run Optimize" button with progress indicators
  - Create what-if analysis interface for flight adjustments
  - Add explanation tooltips with A-CDM/CODA terminology
  - Test dashboard responsiveness and user interaction flows
  - _Requirements: 8.1, 8.2, 8.3, 8.4_

- [ ] 14. Implement alerting and notification system

  - Create alert detection logic for capacity overload thresholds
  - Build Zapier integration for Slack notifications
  - Implement severity level classification for alerts
  - Add top-3 recommendations in alert messages
  - Create escalation logic for critical situations
  - Add confirmation notifications when alerts are resolved
  - Write tests for alert triggering and notification delivery
  - _Requirements: 9.1, 9.2, 9.3, 9.4_

- [ ] 15. Add authentication and security features

  - Implement JWT token-based authentication for API endpoints
  - Add rate limiting and input validation for all endpoints
  - Create role-based access control for different user types
  - Implement data encryption for sensitive flight information
  - Add audit logging for optimization decisions and data access
  - Write security tests and vulnerability assessments
  - _Requirements: 10.3, 10.4_

- [ ] 16. Create comprehensive test suite and validation

  - Build end-to-end test scenarios from query to dashboard display
  - Create performance tests for concurrent user simulation
  - Add data volume tests with full 59-file dataset
  - Implement load testing for optimization scalability
  - Create user acceptance tests with domain expert validation
  - Add monitoring and alerting for system health
  - _Requirements: All requirements validation_

- [ ] 17. Package and deploy the complete system
  - Create Docker containers for all microservices
  - Set up Kubernetes deployment configurations
  - Implement CI/CD pipeline for automated testing and deployment
  - Create environment-specific configuration management
  - Add monitoring dashboards with Prometheus and Grafana
  - Create deployment documentation and runbooks
  - _Requirements: System deployment and operations_
