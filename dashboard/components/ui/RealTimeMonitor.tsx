'use client'

import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Plane, AlertTriangle, TrendingUp, Clock, Activity, CheckCircle } from 'lucide-react'
import { Card, CardContent, CardHeader } from './Card'
import { StatusBadge } from './StatusBadge'
import { LoadingSpinner } from './LoadingSpinner'
import apiClient from '@/lib/api'

interface RealTimeMonitorProps {
  airport: string
  refreshInterval?: number
}

interface SystemMetrics {
  totalFlights: number
  delayedFlights: number
  onTimePercentage: number
  avgDelay: number
  peakUtilization: number
  activeAlerts: number
}

export const RealTimeMonitor: React.FC<RealTimeMonitorProps> = ({ 
  airport, 
  refreshInterval = 30000 
}) => {
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date())

  const fetchRealTimeData = async () => {
    try {
      setLoading(true)
      
      // Fetch multiple data sources in parallel
      const [peakAnalysis, alerts, delayRisks] = await Promise.all([
        apiClient.getPeakAnalysis(airport, 10),
        apiClient.getActiveAlerts(airport),
        apiClient.getDelayRisks(airport, new Date().toISOString().split('T')[0])
      ])

      // Calculate system metrics
      const totalFlights = peakAnalysis.time_buckets.reduce((sum, bucket) => sum + bucket.total_demand, 0)
      const delayedFlights = peakAnalysis.time_buckets.reduce((sum, bucket) => sum + bucket.delayed_flights, 0)
      const onTimePercentage = totalFlights > 0 ? ((totalFlights - delayedFlights) / totalFlights) * 100 : 0
      const avgDelay = peakAnalysis.time_buckets.reduce((sum, bucket) => sum + bucket.avg_delay, 0) / Math.max(peakAnalysis.time_buckets.length, 1)
      
      const systemMetrics: SystemMetrics = {
        totalFlights,
        delayedFlights,
        onTimePercentage: Math.round(onTimePercentage * 10) / 10,
        avgDelay: Math.round(avgDelay * 10) / 10,
        peakUtilization: Math.round(peakAnalysis.avg_utilization * 100 * 10) / 10,
        activeAlerts: alerts.length
      }

      setMetrics(systemMetrics)
      setLastUpdate(new Date())
      setError(null)
    } catch (err) {
      setError('Failed to fetch real-time data')
      console.error('Real-time data fetch error:', err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchRealTimeData()
    const interval = setInterval(fetchRealTimeData, refreshInterval)
    return () => clearInterval(interval)
  }, [airport, refreshInterval])

  if (loading && !metrics) {
    return (
      <Card className="h-64">
        <CardContent className="flex items-center justify-center h-full">
          <LoadingSpinner />
        </CardContent>
      </Card>
    )
  }

  if (error && !metrics) {
    return (
      <Card className="h-64">
        <CardContent className="flex items-center justify-center h-full">
          <div className="text-center text-red-600">
            <AlertTriangle className="h-8 w-8 mx-auto mb-2" />
            <p>{error}</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (!metrics) return null

  const getUtilizationColor = (utilization: number) => {
    if (utilization >= 90) return 'text-red-600'
    if (utilization >= 75) return 'text-orange-600'
    if (utilization >= 60) return 'text-yellow-600'
    return 'text-green-600'
  }

  const getDelayColor = (delay: number) => {
    if (delay >= 30) return 'text-red-600'
    if (delay >= 15) return 'text-orange-600'
    if (delay >= 5) return 'text-yellow-600'
    return 'text-green-600'
  }

  return (
    <Card>
      <CardHeader className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Activity className="h-5 w-5 text-blue-600" />
          <h3 className="text-lg font-semibold">Real-Time Monitor</h3>
        </div>
        <div className="flex items-center gap-2 text-sm text-gray-500">
          <Clock className="h-4 w-4" />
          <span>Last update: {lastUpdate.toLocaleTimeString()}</span>
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          {/* Total Flights */}
          <motion.div 
            className="text-center p-4 rounded-lg bg-blue-50 dark:bg-blue-900/20"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3 }}
          >
            <Plane className="h-6 w-6 mx-auto mb-2 text-blue-600" />
            <p className="text-sm text-blue-600 dark:text-blue-400">Total Flights</p>
            <p className="text-2xl font-bold text-blue-700 dark:text-blue-300">
              {metrics.totalFlights}
            </p>
          </motion.div>

          {/* On-Time Performance */}
          <motion.div 
            className="text-center p-4 rounded-lg bg-green-50 dark:bg-green-900/20"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3, delay: 0.1 }}
          >
            <TrendingUp className="h-6 w-6 mx-auto mb-2 text-green-600" />
            <p className="text-sm text-green-600 dark:text-green-400">On-Time %</p>
            <p className="text-2xl font-bold text-green-700 dark:text-green-300">
              {metrics.onTimePercentage}%
            </p>
          </motion.div>

          {/* Average Delay */}
          <motion.div 
            className="text-center p-4 rounded-lg bg-orange-50 dark:bg-orange-900/20"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3, delay: 0.2 }}
          >
            <Clock className="h-6 w-6 mx-auto mb-2 text-orange-600" />
            <p className="text-sm text-orange-600 dark:text-orange-400">Avg Delay</p>
            <p className={`text-2xl font-bold ${getDelayColor(metrics.avgDelay)}`}>
              {metrics.avgDelay}m
            </p>
          </motion.div>

          {/* Peak Utilization */}
          <motion.div 
            className="text-center p-4 rounded-lg bg-purple-50 dark:bg-purple-900/20"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3, delay: 0.3 }}
          >
            <Activity className="h-6 w-6 mx-auto mb-2 text-purple-600" />
            <p className="text-sm text-purple-600 dark:text-purple-400">Peak Utilization</p>
            <p className={`text-2xl font-bold ${getUtilizationColor(metrics.peakUtilization)}`}>
              {metrics.peakUtilization}%
            </p>
          </motion.div>

          {/* Active Alerts */}
          <motion.div 
            className="text-center p-4 rounded-lg bg-red-50 dark:bg-red-900/20"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3, delay: 0.4 }}
          >
            <AlertTriangle className="h-6 w-6 mx-auto mb-2 text-red-600" />
            <p className="text-sm text-red-600 dark:text-red-400">Active Alerts</p>
            <p className="text-2xl font-bold text-red-700 dark:text-red-300">
              {metrics.activeAlerts}
            </p>
          </motion.div>

          {/* Delayed Flights */}
          <motion.div 
            className="text-center p-4 rounded-lg bg-yellow-50 dark:bg-yellow-900/20"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3, delay: 0.5 }}
          >
            <Clock className="h-6 w-6 mx-auto mb-2 text-yellow-600" />
            <p className="text-sm text-yellow-600 dark:text-yellow-400">Delayed</p>
            <p className="text-2xl font-bold text-yellow-700 dark:text-yellow-300">
              {metrics.delayedFlights}
            </p>
          </motion.div>
        </div>

        {/* Business Logic Indicators */}
        <div className="mt-6 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            System Status & Business Logic
          </h4>
          <div className="space-y-2 text-sm">
            {metrics.peakUtilization >= 90 && (
              <div className="flex items-center gap-2 text-red-600">
                <AlertTriangle className="h-4 w-4" />
                <span>Critical: Peak utilization at {metrics.peakUtilization}% - Consider schedule adjustments</span>
              </div>
            )}
            {metrics.avgDelay >= 30 && (
              <div className="flex items-center gap-2 text-orange-600">
                <Clock className="h-4 w-4" />
                <span>Warning: Average delay of {metrics.avgDelay} minutes - Cascade analysis recommended</span>
              </div>
            )}
            {metrics.onTimePercentage < 70 && (
              <div className="flex items-center gap-2 text-yellow-600">
                <TrendingUp className="h-4 w-4" />
                <span>Alert: On-time performance below 70% - Optimization needed</span>
              </div>
            )}
            {metrics.activeAlerts > 0 && (
              <div className="flex items-center gap-2 text-blue-600">
                <AlertTriangle className="h-4 w-4" />
                <span>Info: {metrics.activeAlerts} active alerts require attention</span>
              </div>
            )}
            {metrics.peakUtilization < 60 && metrics.avgDelay < 15 && metrics.onTimePercentage > 85 && (
              <div className="flex items-center gap-2 text-green-600">
                <CheckCircle className="h-4 w-4" />
                <span>Optimal: All metrics within acceptable ranges</span>
              </div>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
