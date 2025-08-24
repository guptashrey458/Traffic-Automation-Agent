// Mock data service for demonstration purposes
// This provides realistic flight data when the backend is not fully operational

export interface MockFlight {
  flight_id: string
  flight_number: string
  airline_code: string
  origin: string
  destination: string
  aircraft_type: string
  scheduled_departure: string
  scheduled_arrival: string
  actual_departure?: string
  actual_arrival?: string
  status: 'scheduled' | 'departed' | 'arrived' | 'delayed' | 'cancelled'
  delay_minutes: number
  gate: string
  runway: string
}

export interface MockTimeBucket {
  start_time: string
  end_time: string
  total_demand: number
  capacity: number
  utilization: number
  overload: number
  avg_delay: number
  delayed_flights: number
  traffic_level: 'low' | 'medium' | 'high' | 'critical'
}

export interface MockOverloadWindow {
  start_time: string
  end_time: string
  duration_minutes: number
  peak_overload: number
  affected_flights: number
  severity: 'low' | 'medium' | 'high' | 'critical'
}

export interface MockPeakAnalysis {
  airport: string
  bucket_minutes: number
  analysis_date: string
  time_buckets: MockTimeBucket[]
  overload_windows: MockOverloadWindow[]
  avg_utilization: number
  recommendations: string[]
  weather_regime: string
}

export interface MockAlert {
  alert_id: string
  alert_type: string
  severity: string
  title: string
  description: string
  airport: string
  timestamp: string
  affected_flights: string[]
  recommendations: string[]
  metrics: Record<string, any>
  resolved: boolean
  escalated: boolean
}

export interface MockDelayRisk {
  flight_id: string
  departure_risk: {
    risk_score: number
    confidence: number
    risk_level: string
    contributing_factors: string[]
  }
  arrival_risk: {
    risk_score: number
    confidence: number
    risk_level: string
    contributing_factors: string[]
  }
  risk_factors: string[]
  recommendations: string[]
}

export interface MockOptimizationResult {
  optimization_id: string
  status: string
  original_metrics: {
    total_flights: number
    avg_delay_minutes: number
    on_time_performance: number
    fuel_cost_usd: number
    co2_emissions_kg: number
    runway_utilization: number
  }
  optimized_metrics: {
    total_flights: number
    avg_delay_minutes: number
    on_time_performance: number
    fuel_cost_usd: number
    co2_emissions_kg: number
    runway_utilization: number
  }
  recommended_changes: Array<{
    flight_id: string
    change_type: string
    original_time: string
    recommended_time: string
    impact_description: string
    delay_reduction_minutes: number
    co2_reduction_kg: number
  }>
  cost_reduction: number
  execution_time_seconds: number
}

// Generate realistic flight data based on actual routes from FlightRadar24
const generateMockFlights = (airport: string): MockFlight[] => {
  const baseTime = new Date()
  const flights: MockFlight[] = []
  
  // Top routes from FlightRadar24 data
  const routes = airport === 'BOM' ? [
    { dest: 'DEL', flights: 353, airline: 'AI' }, // Air India
    { dest: 'BLR', flights: 232, airline: '6E' }, // IndiGo
    { dest: 'HYD', flights: 136, airline: 'AI' },
    { dest: 'AMD', flights: 129, airline: '6E' },
    { dest: 'CCU', flights: 124, airline: 'AI' },
    { dest: 'MAA', flights: 117, airline: '6E' },
    { dest: 'DXB', flights: 107, airline: 'EK' }, // Emirates
    { dest: 'GOI', flights: 85, airline: 'AI' },
    { dest: 'COK', flights: 72, airline: '6E' },
    { dest: 'AUH', flights: 69, airline: 'EY' }  // Etihad
  ] : [
    { dest: 'BOM', flights: 358, airline: 'AI' },
    { dest: 'BLR', flights: 279, airline: '6E' },
    { dest: 'HYD', flights: 175, airline: 'AI' },
    { dest: 'PNQ', flights: 157, airline: '6E' },
    { dest: 'CCU', flights: 146, airline: 'AI' },
    { dest: 'AMD', flights: 132, airline: '6E' },
    { dest: 'MAA', flights: 132, airline: 'AI' },
    { dest: 'SXR', flights: 91, airline: '6E' },
    { dest: 'PAT', flights: 91, airline: 'AI' },
    { dest: 'DXB', flights: 90, airline: 'EK' }
  ]

  let flightId = 1
  
  routes.forEach((route, routeIndex) => {
    // Generate multiple flights per route based on frequency
    const flightsPerRoute = Math.ceil(route.flights / 7) // Daily flights
    
    for (let i = 0; i < flightsPerRoute; i++) {
      const departureHour = 6 + (routeIndex * 2) + (i * 3) % 12 // Spread flights throughout day
      const departureTime = new Date(baseTime)
      departureTime.setHours(departureHour, (i * 15) % 60, 0, 0)
      
      const arrivalTime = new Date(departureTime)
      const flightDuration = 1.5 + (Math.random() * 2) // 1.5-3.5 hours
      arrivalTime.setHours(arrivalTime.getHours() + Math.floor(flightDuration))
      arrivalTime.setMinutes(arrivalTime.getMinutes() + Math.round((flightDuration % 1) * 60))
      
      // Add some realistic delays
      const delayMinutes = Math.random() > 0.7 ? Math.floor(Math.random() * 45) : 0
      const actualDeparture = delayMinutes > 0 ? new Date(departureTime.getTime() + delayMinutes * 60000) : undefined
      
      const status = delayMinutes > 0 ? 'delayed' : 
                    departureTime < baseTime ? 'departed' : 'scheduled'
      
      flights.push({
        flight_id: `mock-${airport}-${flightId++}`,
        flight_number: `${route.airline}${1000 + i + (routeIndex * 100)}`,
        airline_code: route.airline,
        origin: airport,
        destination: route.dest,
        aircraft_type: ['A320', 'A321', 'B737', 'B738', 'A350', 'B787'][Math.floor(Math.random() * 6)],
        scheduled_departure: departureTime.toISOString(),
        scheduled_arrival: arrivalTime.toISOString(),
        actual_departure: actualDeparture?.toISOString(),
        actual_arrival: actualDeparture ? new Date(actualDeparture.getTime() + flightDuration * 3600000).toISOString() : undefined,
        status,
        delay_minutes: delayMinutes,
        gate: `${String.fromCharCode(65 + (i % 26))}${Math.floor(Math.random() * 20) + 1}`,
        runway: `${Math.floor(Math.random() * 2) + 1}${['4', '6', '8'][Math.floor(Math.random() * 3)]}`
      })
    }
  })
  
  return flights.sort((a, b) => new Date(a.scheduled_departure).getTime() - new Date(b.scheduled_departure).getTime())
}

// Generate time buckets for peak analysis
const generateTimeBuckets = (flights: MockFlight[]): MockTimeBucket[] => {
  const buckets: MockTimeBucket[] = []
  
  for (let hour = 6; hour < 24; hour++) {
    const startTime = new Date()
    startTime.setHours(hour, 0, 0, 0)
    const endTime = new Date(startTime.getTime() + 60 * 60 * 1000)
    
    const hourFlights = flights.filter(f => {
      const flightHour = new Date(f.scheduled_departure).getHours()
      return flightHour === hour
    })
    
    const totalDemand = hourFlights.length
    const capacity = 35 // Base capacity per hour
    const utilization = totalDemand / capacity
    const overload = Math.max(0, totalDemand - capacity)
    const delayedFlights = hourFlights.filter(f => f.delay_minutes > 0).length
    const avgDelay = delayedFlights > 0 ? 
      hourFlights.filter(f => f.delay_minutes > 0).reduce((sum, f) => sum + f.delay_minutes, 0) / delayedFlights : 0
    
    let trafficLevel: MockTimeBucket['traffic_level'] = 'low'
    if (utilization >= 1.2) trafficLevel = 'critical'
    else if (utilization >= 1.0) trafficLevel = 'high'
    else if (utilization >= 0.7) trafficLevel = 'medium'
    
    buckets.push({
      start_time: startTime.toISOString(),
      end_time: endTime.toISOString(),
      total_demand: totalDemand,
      capacity,
      utilization,
      overload,
      avg_delay: Math.round(avgDelay * 10) / 10,
      delayed_flights: delayedFlights,
      traffic_level: trafficLevel
    })
  }
  
  return buckets
}

// Generate overload windows
const generateOverloadWindows = (buckets: MockTimeBucket[]): MockOverloadWindow[] => {
  const windows: MockOverloadWindow[] = []
  
  for (let i = 0; i < buckets.length; i++) {
    if (buckets[i].overload > 0) {
      // Find consecutive overloaded buckets
      let endIndex = i
      while (endIndex < buckets.length && buckets[endIndex].overload > 0) {
        endIndex++
      }
      
      const startTime = new Date(buckets[i].start_time)
      const endTime = new Date(buckets[endIndex - 1].end_time)
      const durationMinutes = (endTime.getTime() - startTime.getTime()) / (1000 * 60)
      
      const peakOverload = Math.max(...buckets.slice(i, endIndex).map(b => b.overload))
      const affectedFlights = buckets.slice(i, endIndex).reduce((sum, b) => sum + b.total_demand, 0)
      
      let severity: MockOverloadWindow['severity'] = 'low'
      if (peakOverload >= 15) severity = 'critical'
      else if (peakOverload >= 10) severity = 'high'
      else if (peakOverload >= 5) severity = 'medium'
      
      windows.push({
        start_time: startTime.toISOString(),
        end_time: endTime.toISOString(),
        duration_minutes: Math.round(durationMinutes),
        peak_overload: peakOverload,
        affected_flights: affectedFlights,
        severity
      })
      
      i = endIndex - 1 // Skip processed buckets
    }
  }
  
  return windows
}

// Generate realistic alerts
const generateAlerts = (airport: string, flights: MockFlight[]): MockAlert[] => {
  const alerts: MockAlert[] = []
  
  // Check for capacity overload
  const peakUtilization = Math.max(...generateTimeBuckets(flights).map(b => b.utilization))
  if (peakUtilization > 1.2) {
    alerts.push({
      alert_id: `alert-${airport}-capacity-${Date.now()}`,
      alert_type: 'capacity_overload',
      severity: 'high',
      title: 'Critical Capacity Overload Detected',
      description: `Peak utilization at ${(peakUtilization * 100).toFixed(1)}% exceeds critical threshold`,
      airport,
      timestamp: new Date().toISOString(),
      affected_flights: flights.filter(f => f.delay_minutes > 15).slice(0, 5).map(f => f.flight_id),
      recommendations: [
        'Consider redistributing flights from peak hours',
        'Activate additional runway capacity if available',
        'Review ground handling operations'
      ],
      metrics: { peak_utilization: peakUtilization },
      resolved: false,
      escalated: false
    })
  }
  
  // Check for delay cascade
  const delayedFlights = flights.filter(f => f.delay_minutes > 30)
  if (delayedFlights.length > 3) {
    alerts.push({
      alert_id: `alert-${airport}-delays-${Date.now()}`,
      alert_type: 'delay_cascade',
      severity: 'medium',
      title: 'Delay Cascade Detected',
      description: `${delayedFlights.length} flights experiencing significant delays`,
      airport,
      timestamp: new Date().toISOString(),
      affected_flights: delayedFlights.map(f => f.flight_id),
      recommendations: [
        'Analyze delay propagation patterns',
        'Review turnaround time efficiency',
        'Consider schedule adjustments'
      ],
      metrics: { delayed_flights: delayedFlights.length, avg_delay: delayedFlights.reduce((sum, f) => sum + f.delay_minutes, 0) / delayedFlights.length },
      resolved: false,
      escalated: false
    })
  }
  
  return alerts
}

// Generate delay risk assessments
const generateDelayRisks = (flights: MockFlight[]): MockDelayRisk[] => {
  return flights.slice(0, 10).map(flight => {
    const departureRisk = Math.random() * 0.8
    const arrivalRisk = Math.random() * 0.6
    
    const getRiskLevel = (risk: number) => {
      if (risk >= 0.6) return 'high'
      if (risk >= 0.3) return 'medium'
      return 'low'
    }
    
    const getContributingFactors = (risk: number) => {
      const factors = []
      if (risk > 0.5) factors.push('Weather conditions')
      if (risk > 0.4) factors.push('Air traffic congestion')
      if (risk > 0.3) factors.push('Aircraft availability')
      if (risk > 0.2) factors.push('Ground operations')
      return factors
    }
    
    return {
      flight_id: flight.flight_id,
      departure_risk: {
        risk_score: Math.round(departureRisk * 100) / 100,
        confidence: Math.round((0.7 + Math.random() * 0.3) * 100) / 100,
        risk_level: getRiskLevel(departureRisk),
        contributing_factors: getContributingFactors(departureRisk)
      },
      arrival_risk: {
        risk_score: Math.round(arrivalRisk * 100) / 100,
        confidence: Math.round((0.8 + Math.random() * 0.2) * 100) / 100,
        risk_level: getRiskLevel(arrivalRisk),
        contributing_factors: getContributingFactors(arrivalRisk)
      },
      risk_factors: ['Historical delay patterns', 'Weather forecast', 'Airport congestion'],
      recommendations: [
        'Monitor weather conditions closely',
        'Prepare for potential delays',
        'Consider alternative routing if necessary'
      ]
    }
  })
}

// Generate optimization results
const generateOptimizationResult = (airport: string, flights: MockFlight[]): MockOptimizationResult => {
  const totalFlights = flights.length
  const avgDelay = flights.reduce((sum, f) => sum + f.delay_minutes, 0) / totalFlights
  const onTimePerformance = (flights.filter(f => f.delay_minutes <= 15).length / totalFlights) * 100
  
  const originalMetrics = {
    total_flights: totalFlights,
    avg_delay_minutes: Math.round(avgDelay * 10) / 10,
    on_time_performance: Math.round(onTimePerformance * 10) / 10,
    fuel_cost_usd: Math.round(totalFlights * 2500 + Math.random() * 10000),
    co2_emissions_kg: Math.round(totalFlights * 5000 + Math.random() * 20000),
    runway_utilization: Math.round((flights.length / 35) * 100 * 10) / 10
  }
  
  // Simulate optimization improvements
  const improvementFactor = 0.15 + Math.random() * 0.25 // 15-40% improvement
  
  const optimizedMetrics = {
    ...originalMetrics,
    avg_delay_minutes: Math.round(originalMetrics.avg_delay_minutes * (1 - improvementFactor) * 10) / 10,
    on_time_performance: Math.round(Math.min(100, originalMetrics.on_time_performance * (1 + improvementFactor * 0.5) * 10) / 10),
    fuel_cost_usd: Math.round(originalMetrics.fuel_cost_usd * (1 - improvementFactor * 0.3)),
    co2_emissions_kg: Math.round(originalMetrics.co2_emissions_kg * (1 - improvementFactor * 0.3))
  }
  
  const recommendedChanges = flights.filter(f => f.delay_minutes > 15).slice(0, 5).map(flight => ({
    flight_id: flight.flight_id,
    change_type: 'time_shift',
    original_time: new Date(flight.scheduled_departure).toLocaleTimeString(),
    recommended_time: new Date(new Date(flight.scheduled_departure).getTime() - 15 * 60000).toLocaleTimeString(),
    impact_description: 'Reduce delay and improve connection efficiency',
    delay_reduction_minutes: Math.min(flight.delay_minutes, 15),
    co2_reduction_kg: Math.round(Math.random() * 100 + 50)
  }))
  
  return {
    optimization_id: `opt-${airport}-${Date.now()}`,
    status: 'completed',
    original_metrics: originalMetrics,
    optimized_metrics: optimizedMetrics,
    recommended_changes: recommendedChanges,
    cost_reduction: Math.round((originalMetrics.fuel_cost_usd - optimizedMetrics.fuel_cost_usd) * 10) / 10,
    execution_time_seconds: Math.round(45 + Math.random() * 30)
  }
}

// Main mock data service
export class MockDataService {
  private static instance: MockDataService
  private cache: Map<string, any> = new Map()
  private cacheTimeout = 30000 // 30 seconds
  
  static getInstance(): MockDataService {
    if (!MockDataService.instance) {
      MockDataService.instance = new MockDataService()
    }
    return MockDataService.instance
  }
  
  private getCachedData<T>(key: string): T | null {
    const cached = this.cache.get(key)
    if (cached && Date.now() - cached.timestamp < this.cacheTimeout) {
      return cached.data
    }
    return null
  }
  
  private setCachedData<T>(key: string, data: T): void {
    this.cache.set(key, { data, timestamp: Date.now() })
  }
  
  getPeakAnalysis(airport: string, bucketMinutes: number = 10): MockPeakAnalysis {
    const cacheKey = `peak_analysis_${airport}_${bucketMinutes}`
    const cached = this.getCachedData<MockPeakAnalysis>(cacheKey)
    if (cached) return cached
    
    const flights = generateMockFlights(airport)
    const timeBuckets = generateTimeBuckets(flights)
    const overloadWindows = generateOverloadWindows(timeBuckets)
    
    const result: MockPeakAnalysis = {
      airport,
      bucket_minutes: bucketMinutes,
      analysis_date: new Date().toISOString().split('T')[0],
      time_buckets: timeBuckets,
      overload_windows: overloadWindows,
      avg_utilization: timeBuckets.reduce((sum, b) => sum + b.utilization, 0) / timeBuckets.length,
      recommendations: [
        'Consider redistributing flights from 18:00-20:00 peak hours',
        'Optimize ground handling operations during high-traffic periods',
        'Review runway allocation for better capacity utilization'
      ],
      weather_regime: 'calm'
    }
    
    this.setCachedData(cacheKey, result)
    return result
  }
  
  getActiveAlerts(airport: string): MockAlert[] {
    const cacheKey = `alerts_${airport}`
    const cached = this.getCachedData<MockAlert[]>(cacheKey)
    if (cached) return cached
    
    const flights = generateMockFlights(airport)
    const alerts = generateAlerts(airport, flights)
    
    this.setCachedData(cacheKey, alerts)
    return alerts
  }
  
  getDelayRisks(airport: string, date: string): MockDelayRisk[] {
    const cacheKey = `delay_risks_${airport}_${date}`
    const cached = this.getCachedData<MockDelayRisk[]>(cacheKey)
    if (cached) return cached
    
    const flights = generateMockFlights(airport)
    const risks = generateDelayRisks(flights)
    
    this.setCachedData(cacheKey, risks)
    return risks
  }
  
  optimizeSchedule(airport: string, date: string): MockOptimizationResult {
    const cacheKey = `optimization_${airport}_${date}`
    const cached = this.getCachedData<MockOptimizationResult>(cacheKey)
    if (cached) return cached
    
    const flights = generateMockFlights(airport)
    const result = generateOptimizationResult(airport, flights)
    
    this.setCachedData(cacheKey, result)
    return result
  }
  
  getSystemStatus() {
    return {
      status: 'healthy',
      services: {
        'peak_analysis': 'healthy',
        'delay_prediction': 'healthy',
        'schedule_optimization': 'healthy',
        'alert_system': 'healthy',
        'weather_integration': 'healthy'
      },
      timestamp: new Date().toISOString()
    }
  }
  
  getConstraints(airport: string) {
    return {
      airport,
      operational_rules: {
        'minimum_turnaround': '30 minutes',
        'wake_separation': 'ICAO standards',
        'curfew_hours': '23:00-06:00'
      },
      capacity_limits: {
        'runway_capacity_per_hour': 35,
        'gate_capacity': 45,
        'taxiway_capacity': 25
      },
      weather_adjustments: {
        'calm': 1.0,
        'medium': 0.85,
        'strong': 0.65,
        'severe': 0.3
      },
      curfew_hours: {
        'start': '23:00',
        'end': '06:00',
        'exceptions': ['emergency', 'government', 'medical']
      }
    }
  }
  
  getSupportedAirports() {
    return {
      airports: [
        { code: 'BOM', name: 'Mumbai Chhatrapati Shivaji International Airport', city: 'Mumbai' },
        { code: 'DEL', name: 'Delhi Indira Gandhi International Airport', city: 'Delhi' },
        { code: 'BLR', name: 'Bangalore Kempegowda International Airport', city: 'Bangalore' },
        { code: 'HYD', name: 'Hyderabad Rajiv Gandhi International Airport', city: 'Hyderabad' },
        { code: 'CCU', name: 'Kolkata Netaji Subhas Chandra Bose International Airport', city: 'Kolkata' },
        { code: 'MAA', name: 'Chennai International Airport', city: 'Chennai' }
      ]
    }
  }
}

export const mockDataService = MockDataService.getInstance()
