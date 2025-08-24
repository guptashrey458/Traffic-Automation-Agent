'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  Zap,
  TrendingUp,
  TrendingDown,
  Play,
  RotateCcw,
  Settings,
  BarChart3,
  Clock,
  Plane,
  Target
} from 'lucide-react'
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  LineChart,
  Line,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Legend
} from 'recharts'

import { Sidebar } from '@/components/layout/Sidebar'
import { Header } from '@/components/layout/Header'
import { Card, CardContent, CardHeader } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { LoadingSpinner } from '@/components/ui/LoadingSpinner'
import { AlgorithmPreview } from '@/components/ui/AlgorithmPreview'
import { formatNumber, formatPercentage } from '@/lib/utils'
import apiClient, { WhatIfResult } from '@/lib/api'
import toast from 'react-hot-toast'

const fadeIn = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0, transition: { duration: 0.5 } }
}

export default function WhatIfPage() {
  const [result, setResult] = useState<WhatIfResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [selectedAirport, setSelectedAirport] = useState('BOM')
  const [selectedDate, setSelectedDate] = useState(new Date().toISOString().split('T')[0])
  const [showAlgorithmDetails, setShowAlgorithmDetails] = useState(false)
  const [showImpactCards, setShowImpactCards] = useState(true)
  
  // What-if parameters
  const [flightId, setFlightId] = useState('AI2739')
  const [changeType, setChangeType] = useState('time_shift')
  const [changeValue, setChangeValue] = useState('+15')

  const runWhatIfAnalysis = async () => {
    setLoading(true)
    try {
      const analysisResult = await apiClient.whatIfAnalysis(
        flightId,
        changeType,
        changeValue,
        selectedAirport,
        selectedDate
      )
      setResult(analysisResult)
      toast.success('What-if analysis completed successfully!')
    } catch (error) {
      console.error('What-if analysis failed:', error)
      toast.error('Analysis failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const resetAnalysis = () => {
    setResult(null)
    setFlightId('AI2739')
    setChangeType('time_shift')
    setChangeValue('+15')
  }

  // Enhanced scenarios based on our comprehensive system
  const sampleScenarios = [
    { 
      flight: 'AI2739', 
      type: 'time_shift', 
      value: '+10', 
      description: 'Peak Hour Congestion Relief',
      impact: 'Reduces peak utilization by 29%',
      confidence: 0.94
    },
    { 
      flight: '6E1234', 
      type: 'time_shift', 
      value: '-5', 
      description: 'Weather Window Optimization',
      impact: 'Utilizes favorable weather window',
      confidence: 0.89
    },
    { 
      flight: 'UK987', 
      type: 'time_shift', 
      value: '-5', 
      description: 'Cascade Prevention',
      impact: 'Prevents downstream delay cascade',
      confidence: 0.91
    },
    { 
      flight: 'SG456', 
      type: 'runway_change', 
      value: '09R', 
      description: 'Capacity Balancing',
      impact: 'Balances runway utilization',
      confidence: 0.87
    }
  ]

  const impactData = result ? [
    {
      metric: 'Average Delay',
      before: result.before_metrics.avg_delay,
      after: result.after_metrics.avg_delay,
      change: result.after_metrics.avg_delay - result.before_metrics.avg_delay
    },
    {
      metric: 'Peak Overload',
      before: result.before_metrics.peak_overload,
      after: result.after_metrics.peak_overload,
      change: result.after_metrics.peak_overload - result.before_metrics.peak_overload
    }
  ] : []

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
                  What-If Analysis
                </h1>
                <p className="text-gray-600 dark:text-gray-400 mt-2">
                  Advanced scenario simulation with autonomous agent impact quantification
                </p>
              </div>
              
              <div className="flex items-center space-x-3">
                <Button 
                  onClick={() => setShowAlgorithmDetails(!showAlgorithmDetails)}
                  variant="outline"
                  className={showAlgorithmDetails ? "bg-blue-100 dark:bg-blue-900" : ""}
                >
                  <Settings className="w-4 h-4 mr-2" />
                  AI Details
                </Button>
                <Button onClick={resetAnalysis} variant="outline">
                  <RotateCcw className="w-4 h-4 mr-2" />
                  Reset
                </Button>
                <Button 
                  onClick={runWhatIfAnalysis} 
                  loading={loading}
                  className="bg-purple-600 hover:bg-purple-700"
                >
                  <Play className="w-4 h-4 mr-2" />
                  Run Analysis
                </Button>
              </div>
            </motion.div>

            {/* Configuration */}
            <motion.div variants={fadeIn}>
              <Card>
                <CardHeader>
                  <div className="flex items-center space-x-2">
                    <Settings className="w-5 h-5 text-gray-500" />
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                      Scenario Configuration
                    </h3>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                    <div>
                      <label className="block text-sm font-medium text-gray-900 dark:text-white mb-3">
                        Airport
                      </label>
                      <select
                        value={selectedAirport}
                        onChange={(e) => setSelectedAirport(e.target.value)}
                        className="w-full px-4 py-3 bg-white dark:bg-gray-800 border-2 border-gray-200 dark:border-gray-600 rounded-lg text-sm font-medium text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-purple-500 transition-colors"
                      >
                        <option value="BOM">Mumbai (BOM)</option>
                        <option value="DEL">Delhi (DEL)</option>
                        <option value="BLR">Bangalore (BLR)</option>
                        <option value="MAA">Chennai (MAA)</option>
                      </select>
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-900 dark:text-white mb-3">
                        Flight ID
                      </label>
                      <input
                        type="text"
                        value={flightId}
                        onChange={(e) => setFlightId(e.target.value)}
                        placeholder="AI2739"
                        className="w-full px-4 py-3 bg-white dark:bg-gray-800 border-2 border-gray-200 dark:border-gray-600 rounded-lg text-sm font-medium text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-purple-500 transition-colors"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-900 dark:text-white mb-3">
                        Change Type
                      </label>
                      <select
                        value={changeType}
                        onChange={(e) => setChangeType(e.target.value)}
                        className="w-full px-4 py-3 bg-white dark:bg-gray-800 border-2 border-gray-200 dark:border-gray-600 rounded-lg text-sm font-medium text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-purple-500 transition-colors"
                      >
                        <option value="time_shift">Time Shift</option>
                        <option value="runway_change">Runway Change</option>
                        <option value="gate_change">Gate Change</option>
                        <option value="aircraft_swap">Aircraft Swap</option>
                      </select>
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-900 dark:text-white mb-3">
                        Change Value
                      </label>
                      <input
                        type="text"
                        value={changeValue}
                        onChange={(e) => setChangeValue(e.target.value)}
                        placeholder="+15"
                        className="w-full px-4 py-3 bg-white dark:bg-gray-800 border-2 border-gray-200 dark:border-gray-600 rounded-lg text-sm font-medium text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-purple-500 transition-colors"
                      />
                    </div>
                  </div>

                  {/* Quick Scenarios */}
                  <div className="mt-6">
                    <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-4">
                      Quick Scenarios
                    </h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
                      {sampleScenarios.map((scenario, index) => (
                        <button
                          key={index}
                          onClick={() => {
                            setFlightId(scenario.flight)
                            setChangeType(scenario.type)
                            setChangeValue(scenario.value)
                          }}
                          className="p-3 text-left border border-gray-200 dark:border-gray-700 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
                        >
                          <p className="text-sm font-medium text-gray-900 dark:text-white">
                            {scenario.flight}
                          </p>
                          <p className="text-xs text-gray-500 dark:text-gray-400">
                            {scenario.description}
                          </p>
                        </button>
                      ))}
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
                    Running what-if analysis...
                  </p>
                </div>
              </div>
            ) : result ? (
              <>
                {/* Algorithm Preview */}
                <motion.div variants={fadeIn}>
                  <AlgorithmPreview 
                    type="whatif" 
                    data={result}
                    isVisible={true}
                  />
                </motion.div>

                {/* Impact Summary */}
                <motion.div variants={fadeIn}>
                  <Card>
                    <CardHeader>
                      <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                        Impact Summary
                      </h3>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        {result.change_description}
                      </p>
                    </CardHeader>
                    <CardContent>
                      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                        <div className="text-center p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                          <Clock className="w-8 h-8 text-blue-500 mx-auto mb-2" />
                          <p className="text-sm text-blue-600 dark:text-blue-400">Delay Change</p>
                          <p className="text-2xl font-bold text-blue-700 dark:text-blue-300">
                            {result.impact_summary.delay_change > 0 ? '+' : ''}{result.impact_summary.delay_change.toFixed(1)}m
                          </p>
                        </div>
                        
                        <div className="text-center p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                          <Plane className="w-8 h-8 text-green-500 mx-auto mb-2" />
                          <p className="text-sm text-green-600 dark:text-green-400">CO₂ Impact</p>
                          <p className="text-2xl font-bold text-green-700 dark:text-green-300">
                            {result.impact_summary.co2_change > 0 ? '+' : ''}{result.impact_summary.co2_change.toFixed(1)}kg
                          </p>
                        </div>
                        
                        <div className="text-center p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                          <Target className="w-8 h-8 text-purple-500 mx-auto mb-2" />
                          <p className="text-sm text-purple-600 dark:text-purple-400">Confidence</p>
                          <p className="text-2xl font-bold text-purple-700 dark:text-purple-300">
                            {formatPercentage(result.impact_summary.confidence, 1)}
                          </p>
                        </div>
                        
                        <div className="text-center p-4 bg-orange-50 dark:bg-orange-900/20 rounded-lg">
                          <BarChart3 className="w-8 h-8 text-orange-500 mx-auto mb-2" />
                          <p className="text-sm text-orange-600 dark:text-orange-400">Affected Flights</p>
                          <p className="text-2xl font-bold text-orange-700 dark:text-orange-300">
                            {result.impact_summary.affected_flights}
                          </p>
                        </div>
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
                            <TrendingDown className="w-4 h-4 mr-2 text-red-500" />
                            Before Change
                          </h4>
                          <div className="space-y-3">
                            <div className="flex justify-between items-center p-3 bg-red-50 dark:bg-red-900/20 rounded-lg">
                              <span className="text-sm text-gray-600 dark:text-gray-400">Average Delay</span>
                              <span className="font-medium">{result.before_metrics.avg_delay} min</span>
                            </div>
                            <div className="flex justify-between items-center p-3 bg-red-50 dark:bg-red-900/20 rounded-lg">
                              <span className="text-sm text-gray-600 dark:text-gray-400">Peak Overload</span>
                              <span className="font-medium">{result.before_metrics.peak_overload}</span>
                            </div>
                          </div>
                        </div>

                        {/* After */}
                        <div>
                          <h4 className="text-md font-medium text-gray-700 dark:text-gray-300 mb-4 flex items-center">
                            <TrendingUp className="w-4 h-4 mr-2 text-green-500" />
                            After Change
                          </h4>
                          <div className="space-y-3">
                            <div className="flex justify-between items-center p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                              <span className="text-sm text-gray-600 dark:text-gray-400">Average Delay</span>
                              <span className="font-medium text-green-600">{result.after_metrics.avg_delay} min</span>
                            </div>
                            <div className="flex justify-between items-center p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                              <span className="text-sm text-gray-600 dark:text-gray-400">Peak Overload</span>
                              <span className="font-medium text-green-600">{result.after_metrics.peak_overload}</span>
                            </div>
                          </div>
                        </div>
                      </div>

                      {/* Impact Chart */}
                      <div className="mt-8">
                        <h4 className="text-md font-medium text-gray-700 dark:text-gray-300 mb-4">
                          Impact Visualization
                        </h4>
                        <div className="h-64 w-full">
                          <ResponsiveContainer>
                            <BarChart data={impactData}>
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis dataKey="metric" />
                              <YAxis />
                              <Tooltip />
                              <Legend />
                              <Bar dataKey="before" fill="#ef4444" name="Before" />
                              <Bar dataKey="after" fill="#22c55e" name="After" />
                            </BarChart>
                          </ResponsiveContainer>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>

                {/* Affected Flights */}
                <motion.div variants={fadeIn}>
                  <Card>
                    <CardHeader>
                      <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                        Affected Flights
                      </h3>
                    </CardHeader>
                    <CardContent>
                      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
                        {result.affected_flights.map((flight, index) => (
                          <div
                            key={index}
                            className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg text-center"
                          >
                            <Plane className="w-4 h-4 text-gray-500 mx-auto mb-1" />
                            <p className="text-sm font-medium text-gray-900 dark:text-white">
                              {flight}
                            </p>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              </>
            ) : (
              <div className="text-center py-12">
                <Zap className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                  Ready to Analyze
                </h3>
                <p className="text-gray-500 dark:text-gray-400 mb-6">
                  Configure your scenario parameters and click "Run Analysis" to see the impact.
                </p>
                <Button onClick={runWhatIfAnalysis} className="bg-purple-600 hover:bg-purple-700">
                  <Play className="w-4 h-4 mr-2" />
                  Run What-If Analysis
                </Button>
              </div>
            )}
          </motion.div>
        </main>
      </div>
    </div>
  )
}