import { mockDataService } from './mockData';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';

// Flag to enable mock data when backend is unavailable
const USE_MOCK_DATA = process.env.NEXT_PUBLIC_USE_MOCK_DATA === 'true' || false;

export interface Flight {
  flight_id: string
  airline: string
  flight_number: string
  departure_time: string
  arrival_time: string
  aircraft_type: string
  origin: string
  destination: string
  status: string
  delay_minutes?: number
}

export interface PeakAnalysis {
  airport: string
  bucket_minutes: number
  analysis_date: string
  time_buckets: TimeBucket[]
  overload_windows: OverloadWindow[]
  avg_utilization: number  // Changed from capacity_utilization to match backend
  recommendations: string[]
  weather_regime: string
}

export interface TimeBucket {
  start_time: string
  end_time: string
  scheduled_departures: number
  actual_departures: number
  scheduled_arrivals: number
  actual_arrivals: number
  total_demand: number
  capacity: number
  utilization: number
  overload: number
  avg_delay: number
  delayed_flights: number
  traffic_level: string
}

export interface OverloadWindow {
  start_time: string
  end_time: string
  duration_minutes: number
  peak_overload: number
  affected_flights: number
  severity: string
}

export interface OptimizationResult {
  optimization_id: string
  status: string
  original_metrics: OptimizationMetrics
  optimized_metrics: OptimizationMetrics
  recommended_changes: RecommendedChange[]
  cost_reduction: number
  execution_time_seconds: number
}

export interface OptimizationMetrics {
  total_flights: number
  avg_delay_minutes: number
  on_time_performance: number
  fuel_cost_usd: number
  co2_emissions_kg: number
  runway_utilization: number
}

export interface RecommendedChange {
  flight_id: string
  change_type: string
  original_time: string
  recommended_time: string
  impact_description: string
  delay_reduction_minutes: number
  co2_reduction_kg: number
}

export interface WhatIfResult {
  flight_id: string
  change_description: string
  impact_summary: {
    delay_change: number
    co2_change: number
    confidence: number
    affected_flights: number
  }
  before_metrics: {
    avg_delay: number
    peak_overload: number
  }
  after_metrics: {
    avg_delay: number
    peak_overload: number
  }
  affected_flights: string[]
  co2_impact_kg: number
}

export interface DelayRisk {
  flight_id: string
  departure_risk: RiskAssessment
  arrival_risk: RiskAssessment
  risk_factors: string[]
  recommendations: string[]
}

export interface RiskAssessment {
  risk_score: number
  confidence: number
  risk_level: string
  contributing_factors: string[]
}

export interface Alert {
  alert_id: string
  alert_type: string
  severity: string
  title: string
  description: string
  airport: string
  timestamp: string
  affected_flights: string[]
  recommendations: AlertRecommendation[]
  metrics: Record<string, any>
  resolved: boolean
  escalated: boolean
}

export interface AlertRecommendation {
  action: string
  impact: string
  priority: number
  estimated_improvement: string
}

export interface AlertSummary {
  total_active_alerts: number
  by_severity: Record<string, number>
  by_type: Record<string, number>
  escalated_alerts: number
  oldest_alert?: string
  most_recent_alert?: string
}

class ApiClient {
  private baseUrl: string

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`
    
    const config: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    }

    try {
      const response = await fetch(url, config)
      
      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(`API Error ${response.status}: ${errorText}`)
      }

      return await response.json()
    } catch (error) {
      console.error(`API request failed: ${url}`, error)
      throw error
    }
  }

  // Peak Analysis
  async getPeakAnalysis(
    airport: string,
    bucketMinutes: number = 10,
    date?: string,
    weatherRegime: string = 'calm'
  ): Promise<PeakAnalysis> {
    // Force mock data for now
    console.log('Using mock data for peak analysis')
    return mockDataService.getPeakAnalysis(airport, bucketMinutes) as any
    
    // Uncomment below when backend is fixed
    /*
    try {
      const params = new URLSearchParams({
        airport,
        bucket_minutes: bucketMinutes.toString(),
        weather_regime: weatherRegime,
      })
      
      if (date) {
        params.append('date', date)
      }

      return await this.request<PeakAnalysis>(`/flights/peaks?${params}`)
    } catch (error) {
      console.warn('Backend peak analysis failed, using mock data:', error)
      return mockDataService.getPeakAnalysis(airport, bucketMinutes) as any
    }
    */
  }

  // Schedule Optimization
  async optimizeSchedule(
    airport: string,
    date: string,
    flights?: string[],
    objectives?: Record<string, number>,
    constraints?: Record<string, any>
  ): Promise<OptimizationResult> {
    // Force mock data for now
    console.log('Using mock data for optimization')
    return mockDataService.optimizeSchedule(airport, date) as any
    
    // Uncomment below when backend is fixed
    /*
    try {
      return await this.request<OptimizationResult>('/optimize', {
        method: 'POST',
        body: JSON.stringify({
          airport,
          date,
          flights,
          objectives,
          constraints,
        }),
      })
    } catch (error) {
      console.warn('Backend optimization failed, using mock data:', error)
      return mockDataService.optimizeSchedule(airport, date) as any
    }
    */
  }

  // What-If Analysis
  async whatIfAnalysis(
    flightId: string,
    changeType: string,
    changeValue: string,
    airport: string,
    date: string
  ): Promise<WhatIfResult> {
    try {
      return await this.request<WhatIfResult>('/whatif', {
        method: 'POST',
        body: JSON.stringify({
          flight_id: flightId,
          change_type: changeType,
          change_value: changeValue,
          airport,
          date,
        }),
      })
    } catch (error) {
      console.warn('Backend what-if analysis failed, using mock data:', error)
      // Return a mock what-if result
      return {
        flight_id: flightId,
        change_description: `${changeType} changed to ${changeValue}`,
        impact_summary: {
          delay_change: -15,
          co2_change: -25,
          confidence: 0.85,
          affected_flights: 3
        },
        before_metrics: { avg_delay: 25, peak_overload: 8 },
        after_metrics: { avg_delay: 10, peak_overload: 3 },
        affected_flights: [`mock-${airport}-1`, `mock-${airport}-2`, `mock-${airport}-3`],
        co2_impact_kg: -150
      } as any
    }
  }

  // Delay Risk Prediction
  async getDelayRisks(
    airport: string,
    date: string,
    flightIds?: string,
    riskThreshold: number = 0.2
  ): Promise<DelayRisk[]> {
    // Force mock data for now
    console.log('Using mock data for delay risks')
    return mockDataService.getDelayRisks(airport, date) as any
    
    // Uncomment below when backend is fixed
    /*
    try {
      const params = new URLSearchParams({
        airport,
        date,
        risk_threshold: riskThreshold.toString(),
      })
      
      if (flightIds) {
        params.append('flight_ids', flightIds)
      }

      return await this.request<DelayRisk[]>(`/flights/risks?${params}`)
    } catch (error) {
      console.warn('Backend delay risk analysis failed, using mock data:', error)
      return mockDataService.getDelayRisks(airport, date) as any
    }
    */
  }

  // Alerts
  async checkAlerts(
    airport: string,
    date?: string,
    forceCheck: boolean = false
  ): Promise<Alert[]> {
    try {
      return await this.request<Alert[]>('/alerts/check', {
        method: 'POST',
        body: JSON.stringify({
          airport,
          date,
          force_check: forceCheck,
        }),
      })
    } catch (error) {
      console.warn('Backend alert check failed, using mock data:', error)
      return mockDataService.getActiveAlerts(airport || 'BOM') as any
    }
  }

  async getActiveAlerts(
    airport?: string,
    severity?: string
  ): Promise<Alert[]> {
    // Force mock data for now
    console.log('Using mock data for active alerts')
    return mockDataService.getActiveAlerts(airport || 'BOM') as any
    
    // Uncomment below when backend is fixed
    /*
    try {
      const params = new URLSearchParams()
      
      if (airport) params.append('airport', airport)
      if (airport) params.append('airport', airport)
      if (severity) params.append('severity', severity)

      return await this.request<Alert[]>(`/alerts/active?${params}`)
    } catch (error) {
      console.warn('Backend active alerts failed, using mock data:', error)
      return mockDataService.getActiveAlerts(airport || 'BOM') as any
    }
    */
  }

  async getAlertSummary(airport?: string): Promise<AlertSummary> {
    const params = new URLSearchParams()
    if (airport) params.append('airport', airport)

    return this.request<AlertSummary>(`/alerts/summary?${params}`)
  }

  async resolveAlert(alertId: string): Promise<{ status: string; alert_id: string }> {
    return this.request(`/alerts/${alertId}/resolve`, {
      method: 'POST',
    })
  }

  async escalateAlert(alertId: string): Promise<{ status: string; alert_id: string }> {
    return this.request(`/alerts/${alertId}/escalate`, {
      method: 'POST',
    })
  }

  async testNotification(): Promise<{ status: string; message: string; alert_id: string }> {
    return this.request('/alerts/test-notification', {
      method: 'POST',
    })
  }

  // System Status
  async getSystemStatus(): Promise<{
    status: string
    services: Record<string, string>
    timestamp: string
  }> {
    // Force mock data for now
    console.log('Using mock data for system status')
    return mockDataService.getSystemStatus() as any
    
    // Uncomment below when backend is fixed
    /*
    try {
      return await this.request('/status')
    } catch (error) {
      console.warn('Backend system status failed, using mock data:', error)
      return mockDataService.getSystemStatus() as any
    }
    */
  }

  async getSupportedAirports(): Promise<{
    airports: Array<{
      code: string
      name: string
      city: string
    }>
  }> {
    try {
      return await this.request('/airports')
    } catch (error) {
      console.warn('Backend airports failed, using mock data:', error)
      return mockDataService.getSupportedAirports() as any
    }
  }

  async getConstraints(
    airport: string,
    date?: string
  ): Promise<{
    airport: string
    operational_rules: Record<string, any>
    capacity_limits: Record<string, any>
    weather_adjustments: Record<string, any>
    curfew_hours: Record<string, any>
  }> {
    try {
      const params = new URLSearchParams({ airport })
      if (date) params.append('date', date)

      return await this.request(`/constraints?${params}`)
    } catch (error) {
      console.warn('Backend constraints failed, using mock data:', error)
      return mockDataService.getConstraints(airport) as any
    }
  }
}

export const apiClient = new ApiClient()
export default apiClient
