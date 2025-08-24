'use client'

import React, { useMemo, useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Plane,
  TrendingUp,
  AlertTriangle,
  Target,
  Brain,
  Zap,
  Activity,
  Clock,
  BarChart3,
  Settings,
  Sun,
  Moon,
  RefreshCw,
  CheckCircle,
  XCircle,
  Info
} from 'lucide-react'
import {
  ResponsiveContainer,
  LineChart,
  Line,
  BarChart,
  Bar,
  AreaChart,
  Area,
  PieChart,
  Pie,
  Cell,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Legend
} from 'recharts'
import { Card, CardContent, CardHeader } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { StatusBadge } from '@/components/ui/StatusBadge'
import { LoadingSpinner } from '@/components/ui/LoadingSpinner'
import { AlgorithmPreview } from '@/components/ui/AlgorithmPreview'
import { NotificationDemo } from '@/components/ui/NotificationDemo'

// ===================== CONFIG =====================
const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000'

// Motion variants
const fadeIn = {
  hidden: { opacity: 0, y: 12 },
  show: { opacity: 1, y: 0, transition: { duration: 0.35 } }
}

const cardCls = 'rounded-2xl border border-gray-200/70 dark:border-gray-700/60 bg-white/70 dark:bg-zinc-900/60 backdrop-blur-xl shadow-sm'

// Theme toggle component
const ThemeToggle: React.FC = () => {
  const [isDark, setIsDark] = useState(() => 
    typeof window !== 'undefined' && document.documentElement.classList.contains('dark')
  )

  const flip = () => {
    const next = !isDark
    setIsDark(next)
    if (next) document.documentElement.classList.add('dark')
    else document.documentElement.classList.remove('dark')
  }

  return (
    <motion.button
      onClick={flip}
      whileTap={{ scale: 0.95 }}
      className="inline-flex items-center gap-2 rounded-xl border border-gray-200 dark:border-gray-700 px-3 py-2 text-sm text-gray-700 dark:text-gray-200 hover:bg-gray-50 dark:hover:bg-zinc-800"
      aria-label="Toggle theme"
    >
      {isDark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
      <span>{isDark ? 'Light' : 'Dark'}</span>
    </motion.button>
  )
}

// Helper function
function clsx(...xs: (string | false | undefined | null)[]) {
  return xs.filter(Boolean).join(' ')
}

// ===================== MAIN DASHBOARD =====================
export default function FlightSchedulerDashboard() {
  const [activeTab, setActiveTab] = useState<'overview' | 'analytics' | 'optimization' | 'prediction' | 'whatif'>('overview')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  
  // Data states
  const [systemStatus, setSystemStatus] = useState<any>(null)
  const [peakAnalysis, setPeakAnalysis] = useState<any>(null)
  const [activeAlerts, setActiveAlerts] = useState<any[]>([])
  const [delayRisks, setDelayRisks] = useState<any[]>([])
  const [optimizationResult, setOptimizationResult] = useState<any>(null)
  const [whatIfResult, setWhatIfResult] = useState<any>(null)
  
  // UI states
  const [showNotificationDemo, setShowNotificationDemo] = useState(false)
  const [selectedAirport, setSelectedAirport] = useState('BOM')
  const [timeRange, setTimeRange] = useState('24h')

  // Fetch system status
  const fetchSystemStatus = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/status`)
      const data = await response.json()
      setSystemStatus(data)
    } catch (error) {
      console.error('Failed to fetch system status:', error)
    }
  }

  // Fetch peak analysis
  const fetchPeakAnalysis = async () => {
    setLoading(true)
    try {
      const response = await fetch(`${API_BASE_URL}/flights/peaks?airport=${selectedAirport}&bucket_minutes=10`)
      const data = await response.json()
      setPeakAnalysis(data)
    } catch (error) {
      setError('Failed to fetch peak analysis')
    } finally {
      setLoading(false)
    }
  }

  // Fetch active alerts
  const fetchActiveAlerts = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/alerts/active?airport=${selectedAirport}`)
      const data = await response.json()
      setActiveAlerts(data)
    } catch (error) {
      console.error('Failed to fetch alerts:', error)
    }
  }

  // Fetch delay risks
  const fetchDelayRisks = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/flights/risks?airport=${selectedAirport}`)
      const data = await response.json()
      setDelayRisks(data)
    } catch (error) {
      console.error('Failed to fetch delay risks:', error)
    }
  }

  // Run optimization
  const runOptimization = async () => {
    setLoading(true)
    try {
      const response = await fetch(`${API_BASE_URL}/optimize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          airport: selectedAirport,
          time_range: timeRange,
          objectives: ['delay_reduction', 'fuel_efficiency', 'capacity_utilization']
        })
      })
      const data = await response.json()
      setOptimizationResult(data)
    } catch (error) {
      setError('Failed to run optimization')
    } finally {
      setLoading(false)
    }
  }

  // Run what-if analysis
  const runWhatIfAnalysis = async (flightId: string, changeType: string, changeValue: string) => {
    setLoading(true)
    try {
      const response = await fetch(`${API_BASE_URL}/whatif`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          flight_id: flightId,
          change_type: changeType,
          change_value: changeValue
        })
      })
      const data = await response.json()
      setWhatIfResult(data)
    } catch (error) {
      setError('Failed to run what-if analysis')
    } finally {
      setLoading(false)
    }
  }

  // Load initial data
  useEffect(() => {
    fetchSystemStatus()
    fetchPeakAnalysis()
    fetchActiveAlerts()
    fetchDelayRisks()
  }, [selectedAirport])

  // Chart data preparation
  const timeBucketsData = useMemo(() => {
    if (!peakAnalysis?.time_buckets) return []
    return peakAnalysis.time_buckets.slice(0, 24).map((bucket: any, index: number) => ({
      time: new Date(bucket.start_time).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
      demand: bucket.total_demand,
      capacity: bucket.capacity,
      utilization: bucket.utilization * 100,
      delays: bucket.avg_delay
    }))
  }, [peakAnalysis])

  const delayRisksData = useMemo(() => {
    return delayRisks.map(risk => ({
      flight: risk.flight_id,
      departure: risk.departure_risk.risk_score * 100,
      arrival: risk.arrival_risk.risk_score * 100
    }))
  }, [delayRisks])

  const optimizationMetrics = useMemo(() => {
    if (!optimizationResult) return []
    return [
      { metric: 'Delay Reduction', before: optimizationResult.original_metrics.avg_delay_minutes, after: optimizationResult.optimized_metrics.avg_delay_minutes },
      { metric: 'On-Time Performance', before: optimizationResult.original_metrics.on_time_performance * 100, after: optimizationResult.optimized_metrics.on_time_performance * 100 },
      { metric: 'Fuel Cost ($)', before: optimizationResult.original_metrics.fuel_cost_usd / 1000, after: optimizationResult.optimized_metrics.fuel_cost_usd / 1000 },
      { metric: 'CO2 Emissions (kg)', before: optimizationResult.original_metrics.co2_emissions_kg / 1000, after: optimizationResult.optimized_metrics.co2_emissions_kg / 1000 }
    ]
  }, [optimizationResult])

  // Tab navigation component
  const TabNavigation = (
    <div className="flex flex-wrap gap-2 mb-6">
      {[
        { id: 'overview', label: 'Overview', icon: <BarChart3 className="h-4 w-4" /> },
        { id: 'analytics', label: 'Analytics', icon: <TrendingUp className="h-4 w-4" /> },
        { id: 'optimization', label: 'Optimization', icon: <Target className="h-4 w-4" /> },
        { id: 'prediction', label: 'Delay Prediction', icon: <Brain className="h-4 w-4" /> },
        { id: 'whatif', label: 'What-If Analysis', icon: <Zap className="h-4 w-4" /> }
      ].map(tab => (
        <button
          key={tab.id}
          onClick={() => setActiveTab(tab.id as any)}
          className={clsx(
            'inline-flex items-center gap-2 rounded-2xl border px-4 py-2 text-sm transition shadow-sm',
            activeTab === tab.id
              ? 'bg-blue-500 text-white border-blue-500'
              : 'border-gray-200 dark:border-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-zinc-800'
          )}
        >
          {tab.icon}
          <span>{tab.label}</span>
        </button>
      ))}
    </div>
  )

  // Overview tab content
  const OverviewTab = (
    <div className="space-y-6">
      {/* System Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {systemStatus?.services && Object.entries(systemStatus.services).map(([service, status]: [string, any]) => (
          <Card key={service} className="p-4">
            <CardContent className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400 capitalize">{service.replace('_', ' ')}</p>
                <StatusBadge status={status} />
              </div>
              <div className={`w-3 h-3 rounded-full ${status === 'healthy' ? 'bg-green-500' : 'bg-red-500'}`} />
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Peak Analysis Chart */}
      <Card className="p-6">
        <CardHeader>
          <h3 className="text-lg font-semibold">Traffic Pattern Analysis</h3>
          <p className="text-sm text-gray-600 dark:text-gray-400">24-hour demand vs capacity utilization</p>
        </CardHeader>
        <CardContent>
          <div className="h-80 w-full">
            <ResponsiveContainer>
              <AreaChart data={timeBucketsData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Area type="monotone" dataKey="demand" stackId="1" stroke="#8884d8" fill="#8884d8" />
                <Area type="monotone" dataKey="capacity" stackId="1" stroke="#82ca9d" fill="#82ca9d" />
                <Line type="monotone" dataKey="utilization" stroke="#ff7300" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* Active Alerts */}
      <Card className="p-6">
        <CardHeader>
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold">Active Alerts</h3>
            <Button onClick={() => setShowNotificationDemo(true)} size="sm">
              <AlertTriangle className="h-4 w-4 mr-2" />
              Test Notifications
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {activeAlerts.length === 0 ? (
              <p className="text-gray-500 text-center py-4">No active alerts</p>
            ) : (
              activeAlerts.map(alert => (
                <div key={alert.alert_id} className="flex items-center justify-between p-3 border border-gray-200 dark:border-gray-700 rounded-lg">
                  <div className="flex items-center space-x-3">
                    <AlertTriangle className={`h-5 w-5 ${alert.severity === 'high' ? 'text-red-500' : 'text-yellow-500'}`} />
                    <div>
                      <h4 className="font-medium">{alert.title}</h4>
                      <p className="text-sm text-gray-600 dark:text-gray-400">{alert.description}</p>
                    </div>
                  </div>
                  <StatusBadge status={alert.severity} />
                </div>
              ))
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  )

  // Analytics tab content
  const AnalyticsTab = (
    <div className="space-y-6">
      {/* Peak Analysis Details */}
      <Card className="p-6">
        <CardHeader>
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold">Peak Traffic Analysis</h3>
            <div className="flex items-center space-x-4">
              <select
                value={selectedAirport}
                onChange={(e) => setSelectedAirport(e.target.value)}
                className="rounded-lg border border-gray-300 dark:border-gray-600 px-3 py-2 bg-white dark:bg-gray-800"
              >
                <option value="BOM">Mumbai (BOM)</option>
                <option value="DEL">Delhi (DEL)</option>
                <option value="BLR">Bangalore (BLR)</option>
              </select>
              <Button onClick={fetchPeakAnalysis} disabled={loading}>
                <RefreshCw className={`h-4 w-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
                Refresh
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {loading ? (
            <LoadingSpinner />
          ) : peakAnalysis ? (
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="text-center p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                  <p className="text-sm text-blue-600 dark:text-blue-400">Capacity Utilization</p>
                  <p className="text-2xl font-bold text-blue-700 dark:text-blue-300">
                    {(peakAnalysis.capacity_utilization * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="text-center p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
                  <p className="text-sm text-yellow-600 dark:text-yellow-400">Overload Windows</p>
                  <p className="text-2xl font-bold text-yellow-700 dark:text-yellow-300">
                    {peakAnalysis.overload_windows?.length || 0}
                  </p>
                </div>
                <div className="text-center p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                  <p className="text-sm text-green-600 dark:text-green-400">Recommendations</p>
                  <p className="text-2xl font-bold text-green-700 dark:text-green-300">
                    {peakAnalysis.recommendations?.length || 0}
                  </p>
                </div>
              </div>
              
              <div className="h-80 w-full">
                <ResponsiveContainer>
                  <LineChart data={timeBucketsData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="utilization" stroke="#8884d8" strokeWidth={2} />
                    <Line type="monotone" dataKey="delays" stroke="#ff7300" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          ) : (
            <p className="text-gray-500 text-center py-4">No data available</p>
          )}
        </CardContent>
      </Card>
    </div>
  )

  // Optimization tab content
  const OptimizationTab = (
    <div className="space-y-6">
      {/* Optimization Controls */}
      <Card className="p-6">
        <CardHeader>
          <h3 className="text-lg font-semibold">Schedule Optimization</h3>
          <p className="text-sm text-gray-600 dark:text-gray-400">Run AI-powered optimization to improve efficiency</p>
        </CardHeader>
        <CardContent>
          <div className="flex items-center space-x-4 mb-4">
            <select
              value={timeRange}
              onChange={(e) => setTimeRange(e.target.value)}
              className="rounded-lg border border-gray-300 dark:border-gray-600 px-3 py-2 bg-white dark:bg-gray-800"
            >
              <option value="24h">24 Hours</option>
              <option value="48h">48 Hours</option>
              <option value="7d">7 Days</option>
            </select>
            <Button onClick={runOptimization} disabled={loading} className="bg-blue-500 hover:bg-blue-600">
              <Target className="h-4 w-4 mr-2" />
              {loading ? 'Optimizing...' : 'Run Optimization'}
            </Button>
          </div>

          {optimizationResult && (
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
                  <h4 className="font-medium mb-3">Before Optimization</h4>
                  <div className="space-y-2">
                    <p className="text-sm">Avg Delay: <span className="font-medium">{optimizationResult.original_metrics.avg_delay_minutes} min</span></p>
                    <p className="text-sm">On-Time: <span className="font-medium">{(optimizationResult.original_metrics.on_time_performance * 100).toFixed(1)}%</span></p>
                    <p className="text-sm">Fuel Cost: <span className="font-medium">${(optimizationResult.original_metrics.fuel_cost_usd / 1000).toFixed(0)}k</span></p>
                  </div>
                </div>
                <div className="p-4 border border-green-200 dark:border-green-700 rounded-lg bg-green-50 dark:bg-green-900/20">
                  <h4 className="font-medium mb-3">After Optimization</h4>
                  <div className="space-y-2">
                    <p className="text-sm">Avg Delay: <span className="font-medium">{optimizationResult.optimized_metrics.avg_delay_minutes} min</span></p>
                    <p className="text-sm">On-Time: <span className="font-medium">{(optimizationResult.optimized_metrics.on_time_performance * 100).toFixed(1)}%</span></p>
                    <p className="text-sm">Fuel Cost: <span className="font-medium">${(optimizationResult.optimized_metrics.fuel_cost_usd / 1000).toFixed(0)}k</span></p>
                  </div>
                </div>
              </div>

              <div className="h-80 w-full">
                <ResponsiveContainer>
                  <BarChart data={optimizationMetrics}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="metric" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="before" fill="#8884d8" name="Before" />
                    <Bar dataKey="after" fill="#82ca9d" name="After" />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Algorithm Preview */}
              <AlgorithmPreview type="optimization" data={optimizationResult} isVisible={true} />
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )

  // Delay Prediction tab content
  const PredictionTab = (
    <div className="space-y-6">
      {/* Delay Risk Analysis */}
      <Card className="p-6">
        <CardHeader>
          <h3 className="text-lg font-semibold">Delay Risk Prediction</h3>
          <p className="text-sm text-gray-600 dark:text-gray-400">AI-powered delay risk assessment for flights</p>
        </CardHeader>
        <CardContent>
          <div className="h-80 w-full mb-6">
            <ResponsiveContainer>
              <BarChart data={delayRisksData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="flight" />
                <YAxis domain={[0, 100]} />
                <Tooltip />
                <Legend />
                <Bar dataKey="departure" fill="#8884d8" name="Departure Risk (%)" />
                <Bar dataKey="arrival" fill="#82ca9d" name="Arrival Risk (%)" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Risk Details */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {delayRisks.slice(0, 6).map(risk => (
              <div key={risk.flight_id} className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-medium">{risk.flight_id}</h4>
                  <StatusBadge 
                    status={risk.departure_risk.risk_level} 
                  />
                </div>
                <div className="space-y-2 text-sm">
                  <p>Departure: <span className="font-medium">{(risk.departure_risk.risk_score * 100).toFixed(1)}%</span></p>
                  <p>Arrival: <span className="font-medium">{(risk.arrival_risk.risk_score * 100).toFixed(1)}%</span></p>
                  <p>Confidence: <span className="font-medium">{(risk.departure_risk.confidence * 100).toFixed(1)}%</span></p>
                </div>
              </div>
            ))}
          </div>

          {/* Algorithm Preview */}
          <AlgorithmPreview type="delay_prediction" data={delayRisks} isVisible={true} />
        </CardContent>
      </Card>
    </div>
  )

  // What-If Analysis tab content
  const WhatIfTab = (
    <div className="space-y-6">
      {/* What-If Controls */}
      <Card className="p-6">
        <CardHeader>
          <h3 className="text-lg font-semibold">What-If Analysis</h3>
          <p className="text-sm text-gray-600 dark:text-gray-400">Simulate changes and analyze their impact</p>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <div>
              <label className="block text-sm font-medium mb-2">Flight ID</label>
              <input
                type="text"
                placeholder="AI2739"
                className="w-full rounded-lg border border-gray-300 dark:border-gray-600 px-3 py-2 bg-white dark:bg-gray-800"
                defaultValue="AI2739"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">Change Type</label>
              <select className="w-full rounded-lg border border-gray-300 dark:border-gray-600 px-3 py-2 bg-white dark:bg-gray-800">
                <option value="time_shift">Time Shift</option>
                <option value="runway_change">Runway Change</option>
                <option value="gate_change">Gate Change</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">Change Value</label>
              <input
                type="text"
                placeholder="+15"
                className="w-full rounded-lg border border-gray-300 dark:border-gray-600 px-3 py-2 bg-white dark:bg-gray-800"
                defaultValue="+15"
              />
            </div>
          </div>

          <Button 
            onClick={() => runWhatIfAnalysis('AI2739', 'time_shift', '+15')} 
            disabled={loading}
            className="bg-purple-500 hover:bg-purple-600"
          >
            <Zap className="h-4 w-4 mr-2" />
            {loading ? 'Analyzing...' : 'Run Analysis'}
          </Button>

          {whatIfResult && (
            <div className="mt-6 space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
                  <h4 className="font-medium mb-3">Before Change</h4>
                  <div className="space-y-2">
                    <p className="text-sm">Avg Delay: <span className="font-medium">{whatIfResult.before_metrics.avg_delay} min</span></p>
                    <p className="text-sm">Peak Overload: <span className="font-medium">{whatIfResult.before_metrics.peak_overload}</span></p>
                  </div>
                </div>
                <div className="p-4 border border-purple-200 dark:border-purple-700 rounded-lg bg-purple-50 dark:bg-purple-900/20">
                  <h4 className="font-medium mb-3">After Change</h4>
                  <div className="space-y-2">
                    <p className="text-sm">Avg Delay: <span className="font-medium">{whatIfResult.after_metrics.avg_delay} min</span></p>
                    <p className="text-sm">Peak Overload: <span className="font-medium">{whatIfResult.after_metrics.peak_overload}</span></p>
                  </div>
                </div>
              </div>

              <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                <h4 className="font-medium mb-2">Impact Summary</h4>
                <p className="text-sm">Delay Change: <span className="font-medium">{whatIfResult.impact_summary.delay_change} min</span></p>
                <p className="text-sm">CO2 Impact: <span className="font-medium">{whatIfResult.impact_summary.co2_change} kg</span></p>
                <p className="text-sm">Confidence: <span className="font-medium">{(whatIfResult.impact_summary.confidence * 100).toFixed(1)}%</span></p>
              </div>

              {/* Algorithm Preview */}
              <AlgorithmPreview type="whatif" data={whatIfResult} isVisible={true} />
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )

  return (
    <div className="min-h-screen bg-gradient-to-b from-white to-gray-50 dark:from-black dark:to-zinc-950 text-gray-900 dark:text-white">
      {/* Header */}
      <header className="sticky top-0 z-30 bg-white/70 dark:bg-black/40 backdrop-blur-xl border-b border-gray-200/60 dark:border-gray-800/60">
        <div className="mx-auto max-w-7xl px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="inline-flex h-10 w-10 items-center justify-center rounded-xl bg-blue-500/20 text-blue-600 dark:text-blue-400">
              <Plane className="h-6 w-6" />
            </div>
            <div className="leading-tight">
              <div className="text-sm text-gray-500 dark:text-gray-400">Flight Scheduler</div>
              <h1 className="text-lg font-semibold">Traffic Automation Agent</h1>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 text-sm">
              <div className="w-2 h-2 rounded-full bg-green-500"></div>
              <span className="text-gray-600 dark:text-gray-400">API Connected</span>
            </div>
            <ThemeToggle />
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="mx-auto max-w-7xl px-4 py-8">
        <motion.section
          initial="hidden"
          animate="show"
          variants={{ show: { transition: { staggerChildren: 0.06 } } }}
        >
          {TabNavigation}

          <AnimatePresence mode="wait">
            {activeTab === 'overview' && (
              <motion.div key="overview" variants={fadeIn}>
                {OverviewTab}
              </motion.div>
            )}
            {activeTab === 'analytics' && (
              <motion.div key="analytics" variants={fadeIn}>
                {AnalyticsTab}
              </motion.div>
            )}
            {activeTab === 'optimization' && (
              <motion.div key="optimization" variants={fadeIn}>
                {OptimizationTab}
              </motion.div>
            )}
            {activeTab === 'prediction' && (
              <motion.div key="prediction" variants={fadeIn}>
                {PredictionTab}
              </motion.div>
            )}
            {activeTab === 'whatif' && (
              <motion.div key="whatif" variants={fadeIn}>
                {WhatIfTab}
              </motion.div>
            )}
          </AnimatePresence>
        </motion.section>
      </main>

      {/* Notification Demo Modal */}
      <NotificationDemo
        isVisible={showNotificationDemo}
        onClose={() => setShowNotificationDemo(false)}
        notificationData={{
          type: "capacity_alert",
          message: "Runway capacity approaching 95% utilization during peak hours",
          severity: "warning",
          timestamp: new Date().toISOString(),
          channels: ["dashboard", "email", "slack"],
          delivery_status: "delivered"
        }}
      />

      {/* Footer */}
      <footer className="border-t border-gray-200/60 dark:border-gray-800/60 py-8 mt-16">
        <div className="mx-auto max-w-7xl px-4 text-center text-sm text-gray-600 dark:text-gray-400">
          <span className="inline-flex items-center gap-2">
            <Plane className="h-4 w-4 text-blue-500" /> 
            Flight Scheduler Dashboard - Powered by AI & Real-time Analytics
          </span>
        </div>
      </footer>
    </div>
  )
}