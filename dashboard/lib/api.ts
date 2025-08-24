const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000'

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
  capacity_utilization: number
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
    const params = new URLSearchParams({
      airport,
      bucket_minutes: bucketMinutes.toString(),
      weather_regime: weatherRegime,
    })
    
    if (date) {
      params.append('date', date)
    }

    return this.request<PeakAnalysis>(`/flights/peaks?${params}`)
  }

  // Schedule Optimization
  async optimizeSchedule(
    airport: string,
    date: string,
    flights?: string[],
    objectives?: Record<string, number>,
    constraints?: Record<string, any>
  ): Promise<OptimizationResult> {
    return this.request<OptimizationResult>('/optimize', {
      method: 'POST',
      body: JSON.stringify({
        airport,
        date,
        flights,
        objectives,
        constraints,
      }),
    })
  }

  // What-If Analysis
  async whatIfAnalysis(
    flightId: string,
    changeType: string,
    changeValue: string,
    airport: string,
    date: string
  ): Promise<WhatIfResult> {
    return this.request<WhatIfResult>('/whatif', {
      method: 'POST',
      body: JSON.stringify({
        flight_id: flightId,
        change_type: changeType,
        change_value: changeValue,
        airport,
        date,
      }),
    })
  }

  // Delay Risk Prediction
  async getDelayRisks(
    airport: string,
    date: string,
    flightIds?: string,
    riskThreshold: number = 0.2
  ): Promise<DelayRisk[]> {
    const params = new URLSearchParams({
      airport,
      date,
      risk_threshold: riskThreshold.toString(),
    })
    
    if (flightIds) {
      params.append('flight_ids', flightIds)
    }

    return this.request<DelayRisk[]>(`/flights/risks?${params}`)
  }

  // Alerts
  async checkAlerts(
    airport: string,
    date?: string,
    forceCheck: boolean = false
  ): Promise<Alert[]> {
    return this.request<Alert[]>('/alerts/check', {
      method: 'POST',
      body: JSON.stringify({
        airport,
        date,
        force_check: forceCheck,
      }),
    })
  }

  async getActiveAlerts(
    airport?: string,
    severity?: string
  ): Promise<Alert[]> {
    const params = new URLSearchParams()
    
    if (airport) params.append('airport', airport)
    if (severity) params.append('severity', severity)

    return this.request<Alert[]>(`/alerts/active?${params}`)
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
    return this.request('/status')
  }

  async getSupportedAirports(): Promise<{
    airports: Array<{
      code: string
      name: string
      city: string
    }>
  }> {
    return this.request('/airports')
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
    const params = new URLSearchParams({ airport })
    if (date) params.append('date', date)

    return this.request(`/constraints?${params}`)
  }
}

export const apiClient = new ApiClient()
export default apiClient
