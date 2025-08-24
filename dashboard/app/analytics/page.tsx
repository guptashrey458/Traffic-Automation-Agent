'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  BarChart3,
  TrendingUp,
  Clock,
  Plane,
  RefreshCw,
  Calendar,
  Filter,
  Download
} from 'lucide-react'
import {
  ResponsiveContainer,
  LineChart,
  Line,
  BarChart,
  Bar,
  AreaChart,
  Area,
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
import { formatNumber, formatPercentage } from '@/lib/utils'
import apiClient from '@/lib/api'
import toast from 'react-hot-toast'

const fadeIn = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0, transition: { duration: 0.5 } }
}

export default function AnalyticsPage() {
  const [peakAnalysis, setPeakAnalysis] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [selectedAirport, setSelectedAirport] = useState('BOM')
  const [selectedDate, setSelectedDate] = useState(new Date().toISOString().split('T')[0])
  const [bucketMinutes, setBucketMinutes] = useState(10)

  useEffect(() => {
    loadPeakAnalysis()
  }, [selectedAirport, selectedDate, bucketMinutes])

  const loadPeakAnalysis = async () => {
    setLoading(true)
    try {
      const analysis = await apiClient.getPeakAnalysis(
        selectedAirport,
        bucketMinutes,
        selectedDate
      )
      setPeakAnalysis(analysis)
    } catch (error) {
      console.error('Failed to load peak analysis:', error)
      toast.error('Failed to load analytics data')
    } finally {
      setLoading(false)
    }
  }

  const timeBucketsData = peakAnalysis?.time_buckets?.map((bucket: any) => ({
    time: new Date(bucket.start_time).toLocaleTimeString('en-US', { 
      hour: '2-digit', 
      minute: '2-digit' 
    }),
    demand: bucket.total_demand,
    capacity: bucket.capacity,
    utilization: bucket.utilization * 100,
    delays: bucket.avg_delay,
    delayed_flights: bucket.delayed_flights
  })) || []

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
                  Flight Analytics
                </h1>
                <p className="text-gray-600 dark:text-gray-400 mt-2">
                  Real-time traffic pattern analysis and capacity optimization
                </p>
              </div>
              
              <div className="flex items-center space-x-3">
                <Button variant="outline" size="sm">
                  <Download className="w-4 h-4 mr-2" />
                  Export
                </Button>
                <Button onClick={loadPeakAnalysis} loading={loading}>
                  <RefreshCw className="w-4 h-4 mr-2" />
                  Refresh
                </Button>
              </div>
            </motion.div>

            {/* Controls */}
            <motion.div variants={fadeIn}>
              <Card>
                <CardContent className="p-4">
                  <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Airport
                      </label>
                      <select
                        value={selectedAirport}
                        onChange={(e) => setSelectedAirport(e.target.value)}
                        className="w-full px-3 py-2 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
                      >
                        <option value="BOM">Mumbai (BOM)</option>
                        <option value="DEL">Delhi (DEL)</option>
                        <option value="BLR">Bangalore (BLR)</option>
                        <option value="MAA">Chennai (MAA)</option>
                      </select>
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Date
                      </label>
                      <input
                        type="date"
                        value={selectedDate}
                        onChange={(e) => setSelectedDate(e.target.value)}
                        className="w-full px-3 py-2 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Time Bucket (minutes)
                      </label>
                      <select
                        value={bucketMinutes}
                        onChange={(e) => setBucketMinutes(Number(e.target.value))}
                        className="w-full px-3 py-2 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
                      >
                        <option value={5}>5 minutes</option>
                        <option value={10}>10 minutes</option>
                        <option value={15}>15 minutes</option>
                        <option value={30}>30 minutes</option>
                        <option value={60}>60 minutes</option>
                      </select>
                    </div>
                    
                    <div className="flex items-end">
                      <Button onClick={loadPeakAnalysis} loading={loading} fullWidth>
                        <BarChart3 className="w-4 h-4 mr-2" />
                        Analyze
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>

            {loading ? (
              <div className="flex items-center justify-center py-12">
                <LoadingSpinner size="lg" />
              </div>
            ) : peakAnalysis ? (
              <>
                {/* Summary Cards */}
                <motion.div variants={fadeIn} className="grid grid-cols-1 md:grid-cols-4 gap-6">
                  <Card>
                    <CardContent className="p-6">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                            Capacity Utilization
                          </p>
                          <p className="text-2xl font-bold text-gray-900 dark:text-white">
                            {formatPercentage(peakAnalysis.capacity_utilization, 1)}
                          </p>
                        </div>
                        <TrendingUp className="w-8 h-8 text-blue-500" />
                      </div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardContent className="p-6">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                            Overload Windows
                          </p>
                          <p className="text-2xl font-bold text-gray-900 dark:text-white">
                            {peakAnalysis.overload_windows?.length || 0}
                          </p>
                        </div>
                        <Clock className="w-8 h-8 text-orange-500" />
                      </div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardContent className="p-6">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                            Peak Demand
                          </p>
                          <p className="text-2xl font-bold text-gray-900 dark:text-white">
                            {Math.max(...timeBucketsData.map((d: any) => d.demand))}
                          </p>
                        </div>
                        <Plane className="w-8 h-8 text-green-500" />
                      </div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardContent className="p-6">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                            Avg Delay
                          </p>
                          <p className="text-2xl font-bold text-gray-900 dark:text-white">
                            {(timeBucketsData.reduce((sum: number, d: any) => sum + d.delays, 0) / timeBucketsData.length).toFixed(1)}m
                          </p>
                        </div>
                        <Clock className="w-8 h-8 text-red-500" />
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>

                {/* Traffic Pattern Chart */}
                <motion.div variants={fadeIn}>
                  <Card>
                    <CardHeader>
                      <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                        Traffic Pattern Analysis
                      </h3>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        Demand vs capacity utilization over time
                      </p>
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
                            <Area 
                              type="monotone" 
                              dataKey="capacity" 
                              stackId="1" 
                              stroke="#82ca9d" 
                              fill="#82ca9d" 
                              name="Capacity"
                            />
                            <Area 
                              type="monotone" 
                              dataKey="demand" 
                              stackId="2" 
                              stroke="#8884d8" 
                              fill="#8884d8" 
                              name="Demand"
                            />
                            <Line 
                              type="monotone" 
                              dataKey="utilization" 
                              stroke="#ff7300" 
                              strokeWidth={2}
                              name="Utilization %"
                            />
                          </AreaChart>
                        </ResponsiveContainer>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>

                {/* Delay Analysis Chart */}
                <motion.div variants={fadeIn}>
                  <Card>
                    <CardHeader>
                      <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                        Delay Analysis
                      </h3>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        Average delays and affected flights by time period
                      </p>
                    </CardHeader>
                    <CardContent>
                      <div className="h-80 w-full">
                        <ResponsiveContainer>
                          <BarChart data={timeBucketsData}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="time" />
                            <YAxis />
                            <Tooltip />
                            <Legend />
                            <Bar dataKey="delays" fill="#ff7300" name="Avg Delay (min)" />
                            <Bar dataKey="delayed_flights" fill="#8884d8" name="Delayed Flights" />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>

                {/* Recommendations */}
                <motion.div variants={fadeIn}>
                  <Card>
                    <CardHeader>
                      <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                        Optimization Recommendations
                      </h3>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-3">
                        {peakAnalysis.recommendations?.map((rec: string, index: number) => (
                          <div key={index} className="flex items-start space-x-3 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                            <div className="w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                              <span className="text-xs font-bold text-white">{index + 1}</span>
                            </div>
                            <p className="text-sm text-gray-700 dark:text-gray-300">{rec}</p>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              </>
            ) : (
              <div className="text-center py-12">
                <BarChart3 className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                  No Data Available
                </h3>
                <p className="text-gray-500 dark:text-gray-400 mb-6">
                  Select an airport and date to view analytics data.
                </p>
                <Button onClick={loadPeakAnalysis}>
                  <BarChart3 className="w-4 h-4 mr-2" />
                  Load Analytics
                </Button>
              </div>
            )}
          </motion.div>
        </main>
      </div>
    </div>
  )
}