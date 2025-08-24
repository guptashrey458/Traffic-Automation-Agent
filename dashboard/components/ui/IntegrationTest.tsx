'use client'

import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { CheckCircle, XCircle, AlertTriangle, RefreshCw, Zap, Database, Brain } from 'lucide-react'
import { Card, CardContent, CardHeader } from './Card'
import { Button } from './Button'
import { LoadingSpinner } from './LoadingSpinner'
import { StatusBadge } from './StatusBadge'
import apiClient from '@/lib/api'

interface IntegrationTestProps {
  airport: string
}

interface TestResult {
  name: string
  status: 'pass' | 'fail' | 'running' | 'pending'
  message: string
  details?: any
  duration?: number
}

export const IntegrationTest: React.FC<IntegrationTestProps> = ({ airport }) => {
  const [testResults, setTestResults] = useState<TestResult[]>([])
  const [running, setRunning] = useState(false)
  const [overallStatus, setOverallStatus] = useState<'pending' | 'running' | 'pass' | 'fail'>('pending')

  const runIntegrationTests = async () => {
    setRunning(true)
    setOverallStatus('running')
    
    const tests: TestResult[] = [
      { name: 'API Connection', status: 'pending', message: 'Testing API connectivity...' },
      { name: 'Peak Analysis', status: 'pending', message: 'Testing peak analysis endpoint...' },
      { name: 'Alert System', status: 'pending', message: 'Testing alert system...' },
      { name: 'Delay Prediction', status: 'pending', message: 'Testing delay prediction...' },
      { name: 'Schedule Optimization', status: 'pending', message: 'Testing optimization...' },
      { name: 'Weather Integration', status: 'pending', message: 'Testing weather integration...' },
      { name: 'Data Validation', status: 'pending', message: 'Validating data structures...' },
      { name: 'Business Logic', status: 'pending', message: 'Testing business logic...' }
    ]

    setTestResults(tests)

    // Test 1: API Connection
    try {
      const startTime = Date.now()
      const status = await apiClient.getSystemStatus()
      const duration = Date.now() - startTime
      
      tests[0] = {
        name: 'API Connection',
        status: 'pass',
        message: `API connected successfully in ${duration}ms`,
        details: status,
        duration
      }
      setTestResults([...tests])
    } catch (error) {
      tests[0] = {
        name: 'API Connection',
        status: 'fail',
        message: `API connection failed: ${error}`,
        details: error
      }
      setTestResults([...tests])
      setOverallStatus('fail')
      setRunning(false)
      return
    }

    // Test 2: Peak Analysis
    try {
      const startTime = Date.now()
      const peakAnalysis = await apiClient.getPeakAnalysis(airport, 10)
      const duration = Date.now() - startTime
      
      // Validate data structure
      const isValid = peakAnalysis.airport && 
                     peakAnalysis.time_buckets && 
                     peakAnalysis.avg_utilization !== undefined
      
      tests[1] = {
        name: 'Peak Analysis',
        status: isValid ? 'pass' : 'fail',
        message: isValid ? 
          `Peak analysis successful in ${duration}ms` : 
          'Peak analysis data structure invalid',
        details: peakAnalysis,
        duration
      }
      setTestResults([...tests])
    } catch (error) {
      tests[1] = {
        name: 'Peak Analysis',
        status: 'fail',
        message: `Peak analysis failed: ${error}`,
        details: error
      }
      setTestResults([...tests])
    }

    // Test 3: Alert System
    try {
      const startTime = Date.now()
      const alerts = await apiClient.getActiveAlerts(airport)
      const duration = Date.now() - startTime
      
      tests[2] = {
        name: 'Alert System',
        status: 'pass',
        message: `Alert system working in ${duration}ms`,
        details: { alertCount: alerts.length },
        duration
      }
      setTestResults([...tests])
    } catch (error) {
      tests[2] = {
        name: 'Alert System',
        status: 'fail',
        message: `Alert system failed: ${error}`,
        details: error
      }
      setTestResults([...tests])
    }

    // Test 4: Delay Prediction
    try {
      const startTime = Date.now()
      const delayRisks = await apiClient.getDelayRisks(airport, new Date().toISOString().split('T')[0])
      const duration = Date.now() - startTime
      
      tests[3] = {
        name: 'Delay Prediction',
        status: 'pass',
        message: `Delay prediction working in ${duration}ms`,
        details: { riskCount: delayRisks.length },
        duration
      }
      setTestResults([...tests])
    } catch (error) {
      tests[3] = {
        name: 'Delay Prediction',
        status: 'fail',
        message: `Delay prediction failed: ${error}`,
        details: error
      }
      setTestResults([...tests])
    }

    // Test 5: Schedule Optimization
    try {
      const startTime = Date.now()
      // Test with a simple optimization request
      const optimization = await apiClient.optimizeSchedule(
        airport,
        new Date().toISOString().split('T')[0],
        undefined,
        { delay_weight: 1.0, taxi_weight: 0.5, fairness_weight: 0.3 }
      )
      const duration = Date.now() - startTime
      
      tests[4] = {
        name: 'Schedule Optimization',
        status: 'pass',
        message: `Optimization working in ${duration}ms`,
        details: optimization,
        duration
      }
      setTestResults([...tests])
    } catch (error) {
      tests[4] = {
        name: 'Schedule Optimization',
        status: 'fail',
        message: `Optimization failed: ${error}`,
        details: error
      }
      setTestResults([...tests])
    }

    // Test 6: Weather Integration
    try {
      const startTime = Date.now()
      const constraints = await apiClient.getConstraints(airport)
      const duration = Date.now() - startTime
      
      const hasWeatherData = constraints.weather_adjustments && 
                            Object.keys(constraints.weather_adjustments).length > 0
      
      tests[5] = {
        name: 'Weather Integration',
        status: hasWeatherData ? 'pass' : 'fail',
        message: hasWeatherData ? 
          `Weather integration working in ${duration}ms` : 
          'Weather data not available',
        details: constraints,
        duration
      }
      setTestResults([...tests])
    } catch (error) {
      tests[5] = {
        name: 'Weather Integration',
        status: 'fail',
        message: `Weather integration failed: ${error}`,
        details: error
      }
      setTestResults([...tests])
    }

    // Test 7: Data Validation
    try {
      const startTime = Date.now()
      
      // Validate that all required data structures are present
      const [peakAnalysis, alerts, delayRisks] = await Promise.all([
        apiClient.getPeakAnalysis(airport, 10),
        apiClient.getActiveAlerts(airport),
        apiClient.getDelayRisks(airport, new Date().toISOString().split('T')[0])
      ])
      
      const duration = Date.now() - startTime
      
      const validationChecks = [
        peakAnalysis.airport === airport,
        Array.isArray(peakAnalysis.time_buckets),
        Array.isArray(peakAnalysis.overload_windows),
        typeof peakAnalysis.avg_utilization === 'number',
        Array.isArray(alerts),
        Array.isArray(delayRisks)
      ]
      
      const allValid = validationChecks.every(check => check)
      
      tests[6] = {
        name: 'Data Validation',
        status: allValid ? 'pass' : 'fail',
        message: allValid ? 
          `Data validation passed in ${duration}ms` : 
          'Data validation failed - structure mismatch',
        details: { validationChecks },
        duration
      }
      setTestResults([...tests])
    } catch (error) {
      tests[6] = {
        name: 'Data Validation',
        status: 'fail',
        message: `Data validation failed: ${error}`,
        details: error
      }
      setTestResults([...tests])
    }

    // Test 8: Business Logic
    try {
      const startTime = Date.now()
      
      // Test business logic by analyzing peak data
      const peakAnalysis = await apiClient.getPeakAnalysis(airport, 10)
      const duration = Date.now() - startTime
      
      const utilization = peakAnalysis.avg_utilization * 100
      const businessLogicChecks = [
        utilization >= 0 && utilization <= 100,
        peakAnalysis.time_buckets.length > 0,
        peakAnalysis.recommendations.length >= 0
      ]
      
      const allValid = businessLogicChecks.every(check => check)
      
      tests[7] = {
        name: 'Business Logic',
        status: allValid ? 'pass' : 'fail',
        message: allValid ? 
          `Business logic validated in ${duration}ms` : 
          'Business logic validation failed',
        details: { 
          utilization: `${utilization.toFixed(1)}%`,
          businessLogicChecks 
        },
        duration
      }
      setTestResults([...tests])
    } catch (error) {
      tests[7] = {
        name: 'Business Logic',
        status: 'fail',
        message: `Business logic test failed: ${error}`,
        details: error
      }
      setTestResults([...tests])
    }

    // Calculate overall status
    const passedTests = tests.filter(t => t.status === 'pass').length
    const totalTests = tests.length
    
    if (passedTests === totalTests) {
      setOverallStatus('pass')
    } else {
      setOverallStatus('fail')
    }
    
    setRunning(false)
  }

  const getStatusIcon = (status: TestResult['status']) => {
    switch (status) {
      case 'pass': return <CheckCircle className="h-5 w-5 text-green-600" />
      case 'fail': return <XCircle className="h-5 w-5 text-red-600" />
      case 'running': return <RefreshCw className="h-5 w-5 text-blue-600 animate-spin" />
      default: return <AlertTriangle className="h-5 w-5 text-gray-600" />
    }
  }

  const getStatusColor = (status: TestResult['status']) => {
    switch (status) {
      case 'pass': return 'border-green-200 bg-green-50 dark:bg-green-900/20'
      case 'fail': return 'border-red-200 bg-red-50 dark:bg-red-900/20'
      case 'running': return 'border-blue-200 bg-blue-50 dark:bg-blue-900/20'
      default: return 'border-gray-200 bg-gray-50 dark:bg-gray-900/20'
    }
  }

  return (
    <Card>
      <CardHeader className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Zap className="h-5 w-5 text-blue-600" />
          <h3 className="text-lg font-semibold">Integration Test Suite</h3>
        </div>
        <Button 
          onClick={runIntegrationTests} 
          disabled={running}
          className="flex items-center gap-2"
        >
          {running ? <LoadingSpinner /> : <RefreshCw className="h-4 w-4" />}
          {running ? 'Running Tests...' : 'Run Tests'}
        </Button>
      </CardHeader>
      <CardContent>
        {/* Overall Status */}
        <div className={`mb-6 p-4 rounded-lg border ${
          overallStatus === 'pass' ? 'border-green-200 bg-green-50 dark:bg-green-900/20' :
          overallStatus === 'fail' ? 'border-red-200 bg-red-50 dark:bg-red-900/20' :
          overallStatus === 'running' ? 'border-blue-200 bg-blue-50 dark:bg-blue-900/20' :
          'border-gray-200 bg-gray-50 dark:bg-gray-900/20'
        }`}>
          <div className="flex items-center gap-2">
            {overallStatus === 'pass' && <CheckCircle className="h-6 w-6 text-green-600" />}
            {overallStatus === 'fail' && <XCircle className="h-6 w-6 text-red-600" />}
            {overallStatus === 'running' && <RefreshCw className="h-6 w-6 text-blue-600 animate-spin" />}
            {overallStatus === 'pending' && <Database className="h-6 w-6 text-gray-600" />}
            <div>
              <h4 className="font-medium">
                {overallStatus === 'pass' && 'All Tests Passed! 🎉'}
                {overallStatus === 'fail' && 'Some Tests Failed ❌'}
                {overallStatus === 'running' && 'Tests Running... ⏳'}
                {overallStatus === 'pending' && 'Tests Not Run Yet'}
              </h4>
              <p className="text-sm text-gray-600">
                {overallStatus === 'pass' && 'Frontend and backend are fully integrated and working correctly'}
                {overallStatus === 'fail' && 'Some integration issues detected - check details below'}
                {overallStatus === 'running' && 'Validating system integration...'}
                {overallStatus === 'pending' && 'Click "Run Tests" to validate system integration'}
              </p>
            </div>
          </div>
        </div>

        {/* Test Results */}
        <div className="space-y-3">
          {testResults.map((test, index) => (
            <motion.div
              key={test.name}
              className={`p-4 rounded-lg border ${getStatusColor(test.status)}`}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: index * 0.1 }}
            >
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-3">
                  {getStatusIcon(test.status)}
                  <div>
                    <h4 className="font-medium">{test.name}</h4>
                    <p className="text-sm text-gray-600">{test.message}</p>
                    {test.duration && (
                      <p className="text-xs text-gray-500">Duration: {test.duration}ms</p>
                    )}
                  </div>
                </div>
                <StatusBadge status={test.status === 'pass' ? 'success' : test.status === 'fail' ? 'error' : 'warning'} />
              </div>
              
              {test.details && (
                <details className="mt-3">
                  <summary className="text-sm font-medium cursor-pointer text-gray-600">
                    View Details
                  </summary>
                  <pre className="mt-2 text-xs bg-gray-100 dark:bg-gray-800 p-2 rounded overflow-auto">
                    {JSON.stringify(test.details, null, 2)}
                  </pre>
                </details>
              )}
            </motion.div>
          ))}
        </div>

        {/* Test Summary */}
        {testResults.length > 0 && (
          <div className="mt-6 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
            <h4 className="font-medium mb-2">Test Summary</h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <span className="font-medium">Total Tests:</span> {testResults.length}
              </div>
              <div>
                <span className="font-medium">Passed:</span> 
                <span className="text-green-600 ml-1">
                  {testResults.filter(t => t.status === 'pass').length}
                </span>
              </div>
              <div>
                <span className="font-medium">Failed:</span> 
                <span className="text-red-600 ml-1">
                  {testResults.filter(t => t.status === 'fail').length}
                </span>
              </div>
              <div>
                <span className="font-medium">Success Rate:</span> 
                <span className="text-blue-600 ml-1">
                  {testResults.length > 0 ? 
                    Math.round((testResults.filter(t => t.status === 'pass').length / testResults.length) * 100) : 0
                  }%
                </span>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
