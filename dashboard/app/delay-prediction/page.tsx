'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  Brain,
  AlertTriangle,
  TrendingUp,
  Clock,
  Plane,
  RefreshCw,
  Target,
  Activity,
  Zap
} from 'lucide-react'

import { Sidebar } from '@/components/layout/Sidebar'
import { Header } from '@/components/layout/Header'
import { Card, CardContent, CardHeader } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { StatusBadge } from '@/components/ui/StatusBadge'
import { LoadingSpinner } from '@/components/ui/LoadingSpinner'
import { AlgorithmPreview } from '@/components/ui/AlgorithmPreview'
import { formatPercentage } from '@/lib/utils'
import apiClient from '@/lib/api'
import toast from 'react-hot-toast'

interface DelayRisk {
  flight_id: string
  departure_risk: {
    risk_score: number
    expected_delay_minutes: number
    confidence: number
    risk_level: string
    contributing_factors: string[]
    algorithm_factors?: {
      base_probability: number
      weather_factor: number
      traffic_factor: number
      time_factor: number
      aircraft_factor: number
    }
    formula?: string
    calculation?: string
  }
  arrival_risk: {
    risk_score: number
    expected_delay_minutes: number
    confidence: number
    risk_level: string
    contributing_factors: string[]
    formula?: string
    calculation?: string
  }
  model_info?: {
    algorithm: string
    features_used: string[]
    training_accuracy: number
    last_updated: string
  }
}

const fadeIn = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0, transition: { duration: 0.5 } }
}

const staggerContainer = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1
    }
  }
}

export default function DelayPredictionPage() {
  const [delayRisks, setDelayRisks] = useState<DelayRisk[]>([])
  const [loading, setLoading] = useState(false)
  const [selectedAirport, setSelectedAirport] = useState('BOM')
  const [selectedFlight, setSelectedFlight] = useState<DelayRisk | null>(null)
  const [showAlgorithm, setShowAlgorithm] = useState(true)

  useEffect(() => {
    loadDelayRisks()
  }, [selectedAirport])

  const loadDelayRisks = async () => {
    setLoading(true)
    try {
      const risks = await apiClient.getDelayRisks(selectedAirport)
      setDelayRisks(risks)
      if (risks.length > 0) {
        setSelectedFlight(risks[0])
      }
    } catch (error) {
      console.error('Failed to load delay risks:', error)
      toast.error('Failed to load delay predictions')
    } finally {
      setLoading(false)
    }
  }

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'critical':
        return 'text-red-600 bg-red-100 dark:bg-red-900 dark:text-red-200'
      case 'high':
        return 'text-orange-600 bg-orange-100 dark:bg-orange-900 dark:text-orange-200'
      case 'medium':
        return 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900 dark:text-yellow-200'
      case 'low':
        return 'text-green-600 bg-green-100 dark:bg-green-900 dark:text-green-200'
      default:
        return 'text-gray-600 bg-gray-100 dark:bg-gray-900 dark:text-gray-200'
    }
  }

  const getRiskIcon = (riskLevel: string) => {
    switch (riskLevel) {
      case 'critical':
        return '🔴'
      case 'high':
        return '🟠'
      case 'medium':
        return '🟡'
      case 'low':
        return '🟢'
      default:
        return '⚪'
    }
  }

  return (
    <div className="flex h-screen bg-gray-50 dark:bg-gray-950">
      <Sidebar />
      
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header />
        
        <main className="flex-1 overflow-y-auto p-6">
          <motion.div
            variants={staggerContainer}
            initial="hidden"
            animate="show"
            className="space-y-6"
          >
            {/* Header */}
            <motion.div variants={fadeIn} className="mb-8">
              <div className="flex items-center justify-between">
                <div>
                  <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
                    Delay Risk Prediction
                  </h1>
                  <p className="text-gray-600 dark:text-gray-400 mt-2">
                    AI-powered delay prediction with live algorithm preview
                  </p>
                </div>
                
                <div className="flex items-center space-x-4">
                  <select
                    value={selectedAirport}
                    onChange={(e) => setSelectedAirport(e.target.value)}
                    className="px-4 py-2 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 text-gray-900 dark:text-white"
                  >
                    <option value="BOM">Mumbai (BOM)</option>
                    <option value="DEL">Delhi (DEL)</option>
                    <option value="BLR">Bangalore (BLR)</option>
                    <option value="MAA">Chennai (MAA)</option>
                  </select>
                  
                  <Button
                    onClick={loadDelayRisks}
                    loading={loading}
                    size="sm"
                  >
                    <RefreshCw className="w-4 h-4 mr-2" />
                    Refresh
                  </Button>
                </div>
              </div>
            </motion.div>

            {/* Algorithm Preview */}
            {showAlgorithm && (
              <motion.div variants={fadeIn}>
                <AlgorithmPreview 
                  type="delay_prediction" 
                  data={selectedFlight}
                  isVisible={true}
                />
              </motion.div>
            )}

            {/* Main Content */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Flight List */}
              <motion.div variants={fadeIn} className="lg:col-span-1">
                <Card>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                        Flight Risk Analysis
                      </h3>
                      <span className="text-sm text-gray-500 dark:text-gray-400">
                        {delayRisks.length} flights
                      </span>
                    </div>
                  </CardHeader>
                  <CardContent>
                    {loading ? (
                      <div className="flex justify-center py-8">
                        <LoadingSpinner />
                      </div>
                    ) : (
                      <div className="space-y-3">
                        {delayRisks.map((risk, index) => (
                          <motion.div
                            key={risk.flight_id}
                            variants={fadeIn}
                            transition={{ delay: index * 0.05 }}
                            className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
                              selectedFlight?.flight_id === risk.flight_id
                                ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                                : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                            }`}
                            onClick={() => setSelectedFlight(risk)}
                          >
                            <div className="flex items-center justify-between mb-2">
                              <div className="flex items-center space-x-2">
                                <Plane className="w-4 h-4 text-gray-500" />
                                <span className="font-medium text-gray-900 dark:text-white">
                                  {risk.flight_id}
                                </span>
                              </div>
                              <span className="text-xs text-gray-500">
                                {getRiskIcon(risk.departure_risk.risk_level)}
                              </span>
                            </div>
                            
                            <div className="space-y-1">
                              <div className="flex justify-between text-sm">
                                <span className="text-gray-600 dark:text-gray-400">Departure Risk:</span>
                                <span className={`px-2 py-1 rounded text-xs font-medium ${getRiskColor(risk.departure_risk.risk_level)}`}>
                                  {formatPercentage(risk.departure_risk.risk_score, 1)}
                                </span>
                              </div>
                              <div className="flex justify-between text-sm">
                                <span className="text-gray-600 dark:text-gray-400">Expected Delay:</span>
                                <span className="text-gray-900 dark:text-white font-medium">
                                  {risk.departure_risk.expected_delay_minutes.toFixed(1)}m
                                </span>
                              </div>
                            </div>
                          </motion.div>
                        ))}
                      </div>
                    )}
                  </CardContent>
                </Card>
              </motion.div>

              {/* Detailed Analysis */}
              <motion.div variants={fadeIn} className="lg:col-span-2">
                {selectedFlight ? (
                  <div className="space-y-6">
                    {/* Flight Details */}
                    <Card>
                      <CardHeader>
                        <div className="flex items-center justify-between">
                          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                            {selectedFlight.flight_id} - Detailed Analysis
                          </h3>
                          <div className="flex items-center space-x-2">
                            <Brain className="w-5 h-5 text-blue-500" />
                            <span className="text-sm text-gray-500 dark:text-gray-400">
                              AI Prediction
                            </span>
                          </div>
                        </div>
                      </CardHeader>
                      <CardContent>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                          {/* Departure Risk */}
                          <div className="space-y-4">
                            <h4 className="font-medium text-gray-900 dark:text-white flex items-center">
                              <Plane className="w-4 h-4 mr-2 text-blue-500" />
                              Departure Risk Analysis
                            </h4>
                            
                            <div className="space-y-3">
                              <div className="flex justify-between items-center">
                                <span className="text-sm text-gray-600 dark:text-gray-400">Risk Level:</span>
                                <span className={`px-3 py-1 rounded-full text-sm font-medium ${getRiskColor(selectedFlight.departure_risk.risk_level)}`}>
                                  {selectedFlight.departure_risk.risk_level.toUpperCase()}
                                </span>
                              </div>
                              
                              <div className="flex justify-between items-center">
                                <span className="text-sm text-gray-600 dark:text-gray-400">Probability:</span>
                                <span className="text-lg font-bold text-gray-900 dark:text-white">
                                  {formatPercentage(selectedFlight.departure_risk.risk_score, 1)}
                                </span>
                              </div>
                              
                              <div className="flex justify-between items-center">
                                <span className="text-sm text-gray-600 dark:text-gray-400">Expected Delay:</span>
                                <span className="text-lg font-bold text-gray-900 dark:text-white">
                                  {selectedFlight.departure_risk.expected_delay_minutes.toFixed(1)} min
                                </span>
                              </div>
                              
                              <div className="flex justify-between items-center">
                                <span className="text-sm text-gray-600 dark:text-gray-400">Confidence:</span>
                                <span className="text-sm font-medium text-gray-900 dark:text-white">
                                  {formatPercentage(selectedFlight.departure_risk.confidence, 1)}
                                </span>
                              </div>
                            </div>

                            {/* Algorithm Breakdown */}
                            {selectedFlight.departure_risk.algorithm_factors && (
                              <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                                <h5 className="font-medium text-blue-900 dark:text-blue-100 mb-3">
                                  Algorithm Factors
                                </h5>
                                <div className="space-y-2 text-sm">
                                  <div className="flex justify-between">
                                    <span className="text-blue-700 dark:text-blue-300">Base Probability:</span>
                                    <span className="font-mono text-blue-900 dark:text-blue-100">
                                      {selectedFlight.departure_risk.algorithm_factors.base_probability.toFixed(3)}
                                    </span>
                                  </div>
                                  <div className="flex justify-between">
                                    <span className="text-blue-700 dark:text-blue-300">Weather Factor:</span>
                                    <span className="font-mono text-blue-900 dark:text-blue-100">
                                      {selectedFlight.departure_risk.algorithm_factors.weather_factor.toFixed(3)}
                                    </span>
                                  </div>
                                  <div className="flex justify-between">
                                    <span className="text-blue-700 dark:text-blue-300">Traffic Factor:</span>
                                    <span className="font-mono text-blue-900 dark:text-blue-100">
                                      {selectedFlight.departure_risk.algorithm_factors.traffic_factor.toFixed(3)}
                                    </span>
                                  </div>
                                  <div className="flex justify-between">
                                    <span className="text-blue-700 dark:text-blue-300">Time Factor:</span>
                                    <span className="font-mono text-blue-900 dark:text-blue-100">
                                      {selectedFlight.departure_risk.algorithm_factors.time_factor.toFixed(3)}
                                    </span>
                                  </div>
                                  <div className="flex justify-between">
                                    <span className="text-blue-700 dark:text-blue-300">Aircraft Factor:</span>
                                    <span className="font-mono text-blue-900 dark:text-blue-100">
                                      {selectedFlight.departure_risk.algorithm_factors.aircraft_factor.toFixed(3)}
                                    </span>
                                  </div>
                                </div>
                                
                                {selectedFlight.departure_risk.formula && (
                                  <div className="mt-3 p-3 bg-white dark:bg-gray-800 rounded border">
                                    <p className="text-xs font-mono text-gray-700 dark:text-gray-300 mb-1">
                                      <strong>Formula:</strong> {selectedFlight.departure_risk.formula}
                                    </p>
                                    <p className="text-xs font-mono text-gray-700 dark:text-gray-300">
                                      <strong>Calculation:</strong> {selectedFlight.departure_risk.calculation}
                                    </p>
                                  </div>
                                )}
                              </div>
                            )}
                          </div>

                          {/* Arrival Risk */}
                          <div className="space-y-4">
                            <h4 className="font-medium text-gray-900 dark:text-white flex items-center">
                              <Target className="w-4 h-4 mr-2 text-green-500" />
                              Arrival Risk Analysis
                            </h4>
                            
                            <div className="space-y-3">
                              <div className="flex justify-between items-center">
                                <span className="text-sm text-gray-600 dark:text-gray-400">Risk Level:</span>
                                <span className={`px-3 py-1 rounded-full text-sm font-medium ${getRiskColor(selectedFlight.arrival_risk.risk_level)}`}>
                                  {selectedFlight.arrival_risk.risk_level.toUpperCase()}
                                </span>
                              </div>
                              
                              <div className="flex justify-between items-center">
                                <span className="text-sm text-gray-600 dark:text-gray-400">Probability:</span>
                                <span className="text-lg font-bold text-gray-900 dark:text-white">
                                  {formatPercentage(selectedFlight.arrival_risk.risk_score, 1)}
                                </span>
                              </div>
                              
                              <div className="flex justify-between items-center">
                                <span className="text-sm text-gray-600 dark:text-gray-400">Expected Delay:</span>
                                <span className="text-lg font-bold text-gray-900 dark:text-white">
                                  {selectedFlight.arrival_risk.expected_delay_minutes.toFixed(1)} min
                                </span>
                              </div>
                              
                              <div className="flex justify-between items-center">
                                <span className="text-sm text-gray-600 dark:text-gray-400">Confidence:</span>
                                <span className="text-sm font-medium text-gray-900 dark:text-white">
                                  {formatPercentage(selectedFlight.arrival_risk.confidence, 1)}
                                </span>
                              </div>
                            </div>

                            {selectedFlight.arrival_risk.formula && (
                              <div className="mt-4 p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                                <h5 className="font-medium text-green-900 dark:text-green-100 mb-3">
                                  Arrival Formula
                                </h5>
                                <div className="p-3 bg-white dark:bg-gray-800 rounded border">
                                  <p className="text-xs font-mono text-gray-700 dark:text-gray-300 mb-1">
                                    <strong>Formula:</strong> {selectedFlight.arrival_risk.formula}
                                  </p>
                                  <p className="text-xs font-mono text-gray-700 dark:text-gray-300">
                                    <strong>Calculation:</strong> {selectedFlight.arrival_risk.calculation}
                                  </p>
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                      </CardContent>
                    </Card>

                    {/* Model Information */}
                    {selectedFlight.model_info && (
                      <Card>
                        <CardHeader>
                          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                            Model Information
                          </h3>
                        </CardHeader>
                        <CardContent>
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <div>
                              <h4 className="font-medium text-gray-900 dark:text-white mb-3">
                                Algorithm Details
                              </h4>
                              <div className="space-y-2 text-sm">
                                <div className="flex justify-between">
                                  <span className="text-gray-600 dark:text-gray-400">Algorithm:</span>
                                  <span className="text-gray-900 dark:text-white font-medium">
                                    {selectedFlight.model_info.algorithm}
                                  </span>
                                </div>
                                <div className="flex justify-between">
                                  <span className="text-gray-600 dark:text-gray-400">Accuracy:</span>
                                  <span className="text-gray-900 dark:text-white font-medium">
                                    {formatPercentage(selectedFlight.model_info.training_accuracy, 1)}
                                  </span>
                                </div>
                                <div className="flex justify-between">
                                  <span className="text-gray-600 dark:text-gray-400">Last Updated:</span>
                                  <span className="text-gray-900 dark:text-white font-medium">
                                    {new Date(selectedFlight.model_info.last_updated).toLocaleString()}
                                  </span>
                                </div>
                              </div>
                            </div>
                            
                            <div>
                              <h4 className="font-medium text-gray-900 dark:text-white mb-3">
                                Features Used
                              </h4>
                              <div className="flex flex-wrap gap-2">
                                {selectedFlight.model_info.features_used.map((feature, index) => (
                                  <span
                                    key={index}
                                    className="px-2 py-1 bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 text-xs rounded-full"
                                  >
                                    {feature.replace('_', ' ')}
                                  </span>
                                ))}
                              </div>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    )}
                  </div>
                ) : (
                  <Card>
                    <CardContent className="flex items-center justify-center py-12">
                      <div className="text-center">
                        <Brain className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                        <p className="text-gray-500 dark:text-gray-400">
                          Select a flight to view detailed delay risk analysis
                        </p>
                      </div>
                    </CardContent>
                  </Card>
                )}
              </motion.div>
            </div>
          </motion.div>
        </main>
      </div>
    </div>
  )
}