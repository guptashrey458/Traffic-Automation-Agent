'use client'

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader } from './Card'
import { Button } from './Button'
import { mockDataService } from '@/lib/mockData'

export const MockDataTest: React.FC = () => {
  const [testResults, setTestResults] = useState<any>({})
  const [loading, setLoading] = useState(false)

  const testMockData = async () => {
    setLoading(true)
    try {
      const results = {
        peakAnalysis: await mockDataService.getPeakAnalysis('BOM', 10),
        alerts: await mockDataService.getActiveAlerts('BOM'),
        delayRisks: await mockDataService.getDelayRisks('BOM', '2025-08-24'),
        optimization: await mockDataService.optimizeSchedule('BOM', '2025-08-24'),
        systemStatus: await mockDataService.getSystemStatus(),
        constraints: await mockDataService.getConstraints('BOM'),
        airports: await mockDataService.getSupportedAirports()
      }
      setTestResults(results)
    } catch (error) {
      console.error('Mock data test failed:', error)
      setTestResults({ error: error.message })
    } finally {
      setLoading(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <h3 className="text-lg font-semibold">Mock Data Test</h3>
      </CardHeader>
      <CardContent>
        <Button onClick={testMockData} disabled={loading}>
          {loading ? 'Testing...' : 'Test Mock Data'}
        </Button>
        
        {Object.keys(testResults).length > 0 && (
          <div className="mt-4">
            <h4 className="font-medium mb-2">Test Results:</h4>
            <div className="space-y-2 text-sm">
              {Object.entries(testResults).map(([key, value]) => (
                <div key={key} className="p-2 bg-gray-50 rounded">
                  <strong>{key}:</strong> {typeof value === 'object' ? JSON.stringify(value).substring(0, 100) + '...' : String(value)}
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
