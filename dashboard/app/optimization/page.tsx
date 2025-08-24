'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  Target,
  Zap,
  TrendingUp,
  TrendingDown,
  Clock,
  Fuel,
  Leaf,
  DollarSign,
  Play,
  CheckCircle,
  AlertCircle,
  Settings
} from 'lucide-react'

import { Sidebar } from '@/components/layout/Sidebar'
import { Header } from '@/components/layout/Header'
import { Card, CardContent, CardHeader } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { StatusBadge } from '@/components/ui/StatusBadge'
import { LoadingSpinner } from '@/components/ui/LoadingSpinner'
import { AlgorithmPreview } from '@/components/ui/AlgorithmPreview'
import { formatNumber, formatCurrency, formatPercentage, formatDuration } from '@/lib/utils'
import apiClient, { OptimizationResult, RecommendedChange } from '@/lib/api'
import toast from 'react-hot-toast'

const fadeIn = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0, transition: { duration: 0.5 } }
}

export default function OptimizationPage() {
  const [result, setResult] = useState<OptimizationResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [selectedAirport, setSelectedAirport] = useState('BOM')
  const [selectedDate, setSelectedDate] = useState(new Date().toISOString().split('T')[0])
  const [objectives, setObjectives] = useState({
    delay_weight: 1.0,
    taxi_weight: 0.3,
    runway_change_weight: 0.2
  })

  const runOptimization = async () => {
    setLoading(true)
    try {
      const optimizationResult = await apiClient.optimizeSchedule(
        selectedAirport,
        selectedDate,
        undefined, // All flights
        objectives
      )
      setResult(optimizationResult)
      toast.success('Schedule optimization completed successfully!')
    } catch (error) {
      console.error('Optimization failed:', error)
      toast.error('Optimization failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const improvements = result ? {
    delayReduction: result.original_metrics.avg_delay_minutes - result.optimized_metrics.avg_delay_minutes,
    otpImprovement: result.optimized_metrics.on_time_performance - result.original_metrics.on_time_performance,
    fuelSavings: result.original_metrics.fuel_cost_usd - result.optimized_metrics.fuel_cost_usd,
    co2Reduction: result.original_metrics.co2_emissions_kg - result.optimized_metrics.co2_emissions_kg,
    utilizationImprovement: result.optimized_metrics.runway_utilization - result.original_metrics.runway_utilization
  } : null

  return (
    <div className="flex h-screen bg-gray-50 dark:bg-gray-950">
      <Sidebar />
      
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header />
        
        <main className="flex-1 overflow-y-auto p-6">
          <motion.div
            initial="hidden"
            animate="show"
            variants={{ show: { transition: { staggerChildren: 0.1 } } }}
            className="space-y-6"
          >
            {/* Header */}
            <motion.div variants={fadeIn} className="flex items-center justify-between">
              <div>
                <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
                  Schedule Optimization
                </h1>
                <p className="text-gray-600 dark:text-gray-400 mt-2">
                  AI-powered multi-criteria schedule optimization engine
                </p>
              </div>
              
              <Button 
                onClick={runOptimization} 
                loading={loading}
                className="bg-green-600 hover:bg-green-700"
              >
                <Play className="w-4 h-4 mr-2" />
                Run Optimization
              </Button>
            </motion.div>

            {/* Algorithm Preview */}
            {result && (
              <motion.div variants={fadeIn}>
                <AlgorithmPreview 
                  type="optimization" 
                  data={result}
                  isVisible={true}
                />
              </motion.div>
            )}

            {/* Configuration */}
            <motion.div variants={fadeIn}>
              <Card>
                <CardHeader>
                  <div className="flex items-center space-x-2">
                    <Settings className="w-5 h-5 text-gray-500" />
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                      Optimization Configuration
                    </h3>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div>
                      <label className="block text-sm font-medium text-gray-900 dark:text-white mb-3">
                        Airport
                      </label>
                      <select
                        value={selectedAirport}
                        onChange={(e) => setSelectedAirport(e.target.value)}
                        className="w-full px-4 py-3 bg-white dark:bg-gray-800 border-2 border-gray-200 dark:border-gray-600 rounded-lg text-sm font-medium text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-green-500 transition-colors"
                      >
                        <option value="BOM">Mumbai (BOM)</option>
                        <option value="DEL">Delhi (DEL)</option>
                        <option value="BLR">Bangalore (BLR)</option>
                        <option value="MAA">Chennai (MAA)</option>
                        <option value="CCU">Kolkata (CCU)</option>
                        <option value="HYD">Hyderabad (HYD)</option>
                      </select>
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-900 dark:text-white mb-3">
                        Analysis Date
                      </label>
                      <input
                        type="date"
                        value={selectedDate}
                        onChange={(e) => setSelectedDate(e.target.value)}
                        min="2025-07-19"
                        max="2025-07-25"
                        className="w-full px-4 py-3 bg-white dark:bg-gray-800 border-2 border-gray-200 dark:border-gray-600 rounded-lg text-sm font-medium text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-green-500 transition-colors"
                      />
                      <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                        Available data: July 19-25, 2025
                      </p>
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Status
                      </label>
                      <div className="flex items-center space-x-2 mt-2">
                        {result ? (
                          <StatusBadge status="success" />
                        ) : (
                          <StatusBadge status="ready" />
                        )}
                        <span className="text-sm text-gray-600 dark:text-gray-400">
                          {result ? 'Optimization Complete' : 'Ready to Optimize'}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Objective Weights */}
                  <div className="mt-6">
                    <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-4">
                      Objective Weights
                    </h4>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div>
                        <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">
                          Delay Weight: {objectives.delay_weight}
                        </label>
                        <input
                          type="range"
                          min="0"
                          max="2"
                          step="0.1"
                          value={objectives.delay_weight}
                          onChange={(e) => setObjectives(prev => ({ ...prev, delay_weight: parseFloat(e.target.value) }))}
                          className="w-full"
                        />
                      </div>
                      <div>
                        <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">
                          Taxi Weight: {objectives.taxi_weight}
                        </label>
                        <input
                          type="range"
                          min="0"
                          max="1"
                          step="0.1"
                          value={objectives.taxi_weight}
                          onChange={(e) => setObjectives(prev => ({ ...prev, taxi_weight: parseFloat(e.target.value) }))}
                          className="w-full"
                        />
                      </div>
                      <div>
                        <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">
                          Runway Change Weight: {objectives.runway_change_weight}
                        </label>
                        <input
                          type="range"
                          min="0"
                          max="1"
                          step="0.1"
                          value={objectives.runway_change_weight}
                          onChange={(e) => setObjectives(prev => ({ ...prev, runway_change_weight: parseFloat(e.target.value) }))}
                          className="w-full"
                        />
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>

            {loading ? (
              <div className="flex items-center justify-center py-12">
                <div className="text-center">
                  <LoadingSpinner size="lg" />
                  <p className="mt-4 text-gray-600 dark:text-gray-400">
                    Running optimization algorithm...
                  </p>
                </div>
              </div>
            ) : result ? (
              <>
                {/* Results Overview */}
                <motion.div variants={fadeIn} className="grid grid-cols-1 md:grid-cols-5 gap-6">
                  <Card className="bg-gradient-to-r from-green-500 to-green-600 text-white">
                    <CardContent className="p-6">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-green-100 text-sm font-medium">Delay Reduction</p>
                          <p className="text-2xl font-bold">
                            {improvements?.delayReduction ? `${improvements.delayReduction.toFixed(1)}m` : '0m'}
                          </p>
                        </div>
                        <TrendingDown className="w-8 h-8 text-green-200" />
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="bg-gradient-to-r from-blue-500 to-blue-600 text-white">
                    <CardContent className="p-6">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-blue-100 text-sm font-medium">OTP Improvement</p>
                          <p className="text-2xl font-bold">
                            +{improvements ? formatPercentage(improvements.otpImprovement, 1) : '0%'}
                          </p>
                        </div>
                        <TrendingUp className="w-8 h-8 text-blue-200" />
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="bg-gradient-to-r from-yellow-500 to-yellow-600 text-white">
                    <CardContent className="p-6">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-yellow-100 text-sm font-medium">Fuel Savings</p>
                          <p className="text-2xl font-bold">
                            {improvements ? formatCurrency(improvements.fuelSavings) : '$0'}
                          </p>
                        </div>
                        <Fuel className="w-8 h-8 text-yellow-200" />
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="bg-gradient-to-r from-emerald-500 to-emerald-600 text-white">
                    <CardContent className="p-6">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-emerald-100 text-sm font-medium">CO₂ Reduction</p>
                          <p className="text-2xl font-bold">
                            {improvements ? `${formatNumber(improvements.co2Reduction)}kg` : '0kg'}
                          </p>
                        </div>
                        <Leaf className="w-8 h-8 text-emerald-200" />
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="bg-gradient-to-r from-purple-500 to-purple-600 text-white">
                    <CardContent className="p-6">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-purple-100 text-sm font-medium">Total Savings</p>
                          <p className="text-2xl font-bold">
                            {formatCurrency(result.cost_reduction)}
                          </p>
                        </div>
                        <DollarSign className="w-8 h-8 text-purple-200" />
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>

                {/* Before/After Comparison */}
                <motion.div variants={fadeIn}>
                  <Card>
                    <CardHeader>
                      <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                        Before vs After Comparison
                      </h3>
                    </CardHeader>
                    <CardContent>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                        {/* Before */}
                        <div>
                          <h4 className="text-md font-medium text-gray-700 dark:text-gray-300 mb-4 flex items-center">
                            <AlertCircle className="w-4 h-4 mr-2 text-red-500" />
                            Original Schedule
                          </h4>
                          <div className="space-y-3">
                            <div className="flex justify-between items-center p-3 bg-red-50 dark:bg-red-900/20 rounded-lg">
                              <span className="text-sm text-gray-600 dark:text-gray-400">Total Flights</span>
                              <span className="font-medium">{formatNumber(result.original_metrics.total_flights)}</span>
                            </div>
                            <div className="flex justify-between items-center p-3 bg-red-50 dark:bg-red-900/20 rounded-lg">
                              <span className="text-sm text-gray-600 dark:text-gray-400">Avg Delay</span>
                              <span className="font-medium">{result.original_metrics.avg_delay_minutes.toFixed(1)}m</span>
                            </div>
                            <div className="flex justify-between items-center p-3 bg-red-50 dark:bg-red-900/20 rounded-lg">
                              <span className="text-sm text-gray-600 dark:text-gray-400">On-Time Performance</span>
                              <span className="font-medium">{formatPercentage(result.original_metrics.on_time_performance)}</span>
                            </div>
                            <div className="flex justify-between items-center p-3 bg-red-50 dark:bg-red-900/20 rounded-lg">
                              <span className="text-sm text-gray-600 dark:text-gray-400">Fuel Cost</span>
                              <span className="font-medium">{formatCurrency(result.original_metrics.fuel_cost_usd)}</span>
                            </div>
                            <div className="flex justify-between items-center p-3 bg-red-50 dark:bg-red-900/20 rounded-lg">
                              <span className="text-sm text-gray-600 dark:text-gray-400">CO₂ Emissions</span>
                              <span className="font-medium">{formatNumber(result.original_metrics.co2_emissions_kg)}kg</span>
                            </div>
                          </div>
                        </div>

                        {/* After */}
                        <div>
                          <h4 className="text-md font-medium text-gray-700 dark:text-gray-300 mb-4 flex items-center">
                            <CheckCircle className="w-4 h-4 mr-2 text-green-500" />
                            Optimized Schedule
                          </h4>
                          <div className="space-y-3">
                            <div className="flex justify-between items-center p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                              <span className="text-sm text-gray-600 dark:text-gray-400">Total Flights</span>
                              <span className="font-medium">{formatNumber(result.optimized_metrics.total_flights)}</span>
                            </div>
                            <div className="flex justify-between items-center p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                              <span className="text-sm text-gray-600 dark:text-gray-400">Avg Delay</span>
                              <span className="font-medium text-green-600">{result.optimized_metrics.avg_delay_minutes.toFixed(1)}m</span>
                            </div>
                            <div className="flex justify-between items-center p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                              <span className="text-sm text-gray-600 dark:text-gray-400">On-Time Performance</span>
                              <span className="font-medium text-green-600">{formatPercentage(result.optimized_metrics.on_time_performance)}</span>
                            </div>
                            <div className="flex justify-between items-center p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                              <span className="text-sm text-gray-600 dark:text-gray-400">Fuel Cost</span>
                              <span className="font-medium text-green-600">{formatCurrency(result.optimized_metrics.fuel_cost_usd)}</span>
                            </div>
                            <div className="flex justify-between items-center p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                              <span className="text-sm text-gray-600 dark:text-gray-400">CO₂ Emissions</span>
                              <span className="font-medium text-green-600">{formatNumber(result.optimized_metrics.co2_emissions_kg)}kg</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>

                {/* Recommended Changes */}
                <motion.div variants={fadeIn}>
                  <Card>
                    <CardHeader>
                      <div className="flex items-center justify-between">
                        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                          Recommended Schedule Changes
                        </h3>
                        <StatusBadge status={`${result.recommended_changes.length} changes`} />
                      </div>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        {result.recommended_changes.slice(0, 10).map((change, index) => (
                          <div key={index} className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                            <div className="flex items-center space-x-4">
                              <div className="w-8 h-8 bg-blue-100 dark:bg-blue-900 rounded-lg flex items-center justify-center">
                                <span className="text-sm font-bold text-blue-600 dark:text-blue-400">
                                  {index + 1}
                                </span>
                              </div>
                              <div>
                                <p className="font-medium text-gray-900 dark:text-white">
                                  Flight {change.flight_id}
                                </p>
                                <p className="text-sm text-gray-500 dark:text-gray-400">
                                  {change.change_type}: {change.original_time} → {change.recommended_time}
                                </p>
                              </div>
                            </div>
                            <div className="text-right">
                              <p className="text-sm font-medium text-green-600 dark:text-green-400">
                                -{change.delay_reduction_minutes}m delay
                              </p>
                              <p className="text-xs text-gray-500 dark:text-gray-400">
                                -{change.co2_reduction_kg}kg CO₂
                              </p>
                            </div>
                          </div>
                        ))}
                        
                        {result.recommended_changes.length > 10 && (
                          <div className="text-center pt-4">
                            <Button variant="outline" size="sm">
                              View All {result.recommended_changes.length} Changes
                            </Button>
                          </div>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>

                {/* Execution Summary */}
                <motion.div variants={fadeIn}>
                  <Card>
                    <CardHeader>
                      <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                        Optimization Summary
                      </h3>
                    </CardHeader>
                    <CardContent>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                        <div className="text-center">
                          <div className="w-12 h-12 bg-green-100 dark:bg-green-900 rounded-lg flex items-center justify-center mx-auto mb-2">
                            <CheckCircle className="w-6 h-6 text-green-600 dark:text-green-400" />
                          </div>
                          <p className="text-sm text-gray-600 dark:text-gray-400">Status</p>
                          <p className="font-medium text-gray-900 dark:text-white">{result.status}</p>
                        </div>
                        
                        <div className="text-center">
                          <div className="w-12 h-12 bg-blue-100 dark:bg-blue-900 rounded-lg flex items-center justify-center mx-auto mb-2">
                            <Clock className="w-6 h-6 text-blue-600 dark:text-blue-400" />
                          </div>
                          <p className="text-sm text-gray-600 dark:text-gray-400">Execution Time</p>
                          <p className="font-medium text-gray-900 dark:text-white">
                            {result.execution_time_seconds.toFixed(2)}s
                          </p>
                        </div>
                        
                        <div className="text-center">
                          <div className="w-12 h-12 bg-purple-100 dark:bg-purple-900 rounded-lg flex items-center justify-center mx-auto mb-2">
                            <Target className="w-6 h-6 text-purple-600 dark:text-purple-400" />
                          </div>
                          <p className="text-sm text-gray-600 dark:text-gray-400">Optimization ID</p>
                          <p className="font-medium text-gray-900 dark:text-white text-xs">
                            {result.optimization_id}
                          </p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              </>
            ) : (
              <div className="text-center py-12">
                <Target className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                  Ready to Optimize
                </h3>
                <p className="text-gray-500 dark:text-gray-400 mb-6">
                  Configure your optimization parameters and click "Run Optimization" to get started.
                </p>
                <Button onClick={runOptimization} className="bg-green-600 hover:bg-green-700">
                  <Play className="w-4 h-4 mr-2" />
                  Run Optimization
                </Button>
              </div>
            )}
          </motion.div>
        </main>
      </div>
    </div>
  )
}