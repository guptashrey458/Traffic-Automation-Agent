'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  AlertTriangle,
  Bell,
  CheckCircle,
  Clock,
  TrendingUp,
  Zap,
  X,
  ExternalLink,
  RefreshCw,
  Filter,
  Search
} from 'lucide-react'

import { Sidebar } from '@/components/layout/Sidebar'
import { Header } from '@/components/layout/Header'
import { Card, CardContent, CardHeader } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { StatusBadge } from '@/components/ui/StatusBadge'
import { LoadingSpinner } from '@/components/ui/LoadingSpinner'
import { NotificationDemo } from '@/components/ui/NotificationDemo'
import { formatDateTime, getTimeAgo, getSeverityColor } from '@/lib/utils'
import apiClient, { Alert, AlertSummary } from '@/lib/api'
import toast from 'react-hot-toast'

const fadeIn = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0, transition: { duration: 0.5 } }
}

export default function AlertsPage() {
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [summary, setSummary] = useState<AlertSummary | null>(null)
  const [loading, setLoading] = useState(false)
  const [selectedAirport, setSelectedAirport] = useState<string>('')
  const [selectedSeverity, setSelectedSeverity] = useState<string>('')
  const [searchTerm, setSearchTerm] = useState('')
  const [showNotificationDemo, setShowNotificationDemo] = useState(false)
  const [notificationData, setNotificationData] = useState<any>(null)

  useEffect(() => {
    loadAlerts()
    loadSummary()
  }, [selectedAirport, selectedSeverity])

  const loadAlerts = async () => {
    setLoading(true)
    try {
      const activeAlerts = await apiClient.getActiveAlerts(
        selectedAirport || undefined,
        selectedSeverity || undefined
      )
      setAlerts(activeAlerts)
    } catch (error) {
      console.error('Failed to load alerts:', error)
      toast.error('Failed to load alerts')
    } finally {
      setLoading(false)
    }
  }

  const loadSummary = async () => {
    try {
      const alertSummary = await apiClient.getAlertSummary(selectedAirport || undefined)
      setSummary(alertSummary)
    } catch (error) {
      console.error('Failed to load alert summary:', error)
    }
  }

  const checkForNewAlerts = async () => {
    if (!selectedAirport) {
      toast.error('Please select an airport first')
      return
    }

    setLoading(true)
    try {
      const newAlerts = await apiClient.checkAlerts(selectedAirport, undefined, true)
      if (newAlerts.length > 0) {
        toast.success(`Found ${newAlerts.length} new alerts`)
        loadAlerts()
        loadSummary()
      } else {
        toast.success('No new alerts found')
      }
    } catch (error) {
      console.error('Failed to check for alerts:', error)
      toast.error('Failed to check for new alerts')
    } finally {
      setLoading(false)
    }
  }

  const resolveAlert = async (alertId: string) => {
    try {
      await apiClient.resolveAlert(alertId)
      toast.success('Alert resolved successfully')
      loadAlerts()
      loadSummary()
    } catch (error) {
      console.error('Failed to resolve alert:', error)
      toast.error('Failed to resolve alert')
    }
  }

  const escalateAlert = async (alertId: string) => {
    try {
      await apiClient.escalateAlert(alertId)
      toast.success('Alert escalated successfully')
      loadAlerts()
      loadSummary()
    } catch (error) {
      console.error('Failed to escalate alert:', error)
      toast.error('Failed to escalate alert')
    }
  }

  const testNotification = async () => {
    try {
      // Show loading toast
      const loadingToast = toast.loading('Sending test notification...')
      
      const result = await apiClient.testNotification()
      
      // Dismiss loading toast
      toast.dismiss(loadingToast)
      
      if (result.status === 'success') {
        // Show success toast
        toast.success('Test notification sent successfully!')
        
        // Set notification data and show demo
        setNotificationData(result.notification_details)
        setShowNotificationDemo(true)
        
      } else {
        toast.error('Test notification failed')
      }
    } catch (error) {
      console.error('Failed to send test notification:', error)
      toast.error(`Failed to send test notification: ${error instanceof Error ? error.message : 'Unknown error'}`)
    }
  }

  const filteredAlerts = alerts.filter(alert => {
    if (searchTerm) {
      const searchLower = searchTerm.toLowerCase()
      return (
        alert.title.toLowerCase().includes(searchLower) ||
        alert.description.toLowerCase().includes(searchLower) ||
        alert.affected_flights.some(flight => flight.toLowerCase().includes(searchLower))
      )
    }
    return true
  })

  const getSeverityIcon = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'critical':
        return <AlertTriangle className="w-5 h-5 text-red-500" />
      case 'high':
        return <TrendingUp className="w-5 h-5 text-orange-500" />
      case 'medium':
        return <Clock className="w-5 h-5 text-yellow-500" />
      case 'low':
        return <Bell className="w-5 h-5 text-blue-500" />
      default:
        return <Bell className="w-5 h-5 text-gray-500" />
    }
  }

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
                  Alert Management
                </h1>
                <p className="text-gray-600 dark:text-gray-400 mt-2">
                  Real-time monitoring and alert management system
                </p>
              </div>
              
              <div className="flex items-center space-x-3">
                <Button onClick={testNotification} variant="outline" size="sm">
                  <Bell className="w-4 h-4 mr-2" />
                  Test Notification
                </Button>
                <Button onClick={checkForNewAlerts} loading={loading}>
                  <RefreshCw className="w-4 h-4 mr-2" />
                  Check for Alerts
                </Button>
              </div>
            </motion.div>

            {/* Summary Cards */}
            {summary && (
              <motion.div variants={fadeIn} className="grid grid-cols-1 md:grid-cols-4 gap-6">
                <Card>
                  <CardContent className="p-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                          Total Active
                        </p>
                        <p className="text-2xl font-bold text-gray-900 dark:text-white">
                          {summary.total_active_alerts}
                        </p>
                      </div>
                      <AlertTriangle className="w-8 h-8 text-blue-500" />
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardContent className="p-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                          Critical
                        </p>
                        <p className="text-2xl font-bold text-red-600">
                          {summary.by_severity.critical || 0}
                        </p>
                      </div>
                      <AlertTriangle className="w-8 h-8 text-red-500" />
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardContent className="p-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                          High Priority
                        </p>
                        <p className="text-2xl font-bold text-orange-600">
                          {summary.by_severity.high || 0}
                        </p>
                      </div>
                      <TrendingUp className="w-8 h-8 text-orange-500" />
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardContent className="p-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                          Escalated
                        </p>
                        <p className="text-2xl font-bold text-purple-600">
                          {summary.escalated_alerts}
                        </p>
                      </div>
                      <Zap className="w-8 h-8 text-purple-500" />
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            )}

            {/* Filters */}
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
                        <option value="">All Airports</option>
                        <option value="BOM">Mumbai (BOM)</option>
                        <option value="DEL">Delhi (DEL)</option>
                        <option value="BLR">Bangalore (BLR)</option>
                        <option value="MAA">Chennai (MAA)</option>
                      </select>
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Severity
                      </label>
                      <select
                        value={selectedSeverity}
                        onChange={(e) => setSelectedSeverity(e.target.value)}
                        className="w-full px-3 py-2 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
                      >
                        <option value="">All Severities</option>
                        <option value="critical">Critical</option>
                        <option value="high">High</option>
                        <option value="medium">Medium</option>
                        <option value="low">Low</option>
                      </select>
                    </div>
                    
                    <div className="md:col-span-2">
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Search
                      </label>
                      <div className="relative">
                        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                        <input
                          type="text"
                          placeholder="Search alerts, flights, or descriptions..."
                          value={searchTerm}
                          onChange={(e) => setSearchTerm(e.target.value)}
                          className="w-full pl-10 pr-4 py-2 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
                        />
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>

            {/* Alerts List */}
            <motion.div variants={fadeIn}>
              <Card>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                      Active Alerts ({filteredAlerts.length})
                    </h3>
                    <Button onClick={loadAlerts} variant="outline" size="sm" loading={loading}>
                      <RefreshCw className="w-4 h-4 mr-2" />
                      Refresh
                    </Button>
                  </div>
                </CardHeader>
                <CardContent>
                  {loading ? (
                    <div className="flex items-center justify-center py-8">
                      <LoadingSpinner size="lg" />
                    </div>
                  ) : filteredAlerts.length > 0 ? (
                    <div className="space-y-4">
                      {filteredAlerts.map((alert, index) => (
                        <motion.div
                          key={alert.alert_id}
                          variants={fadeIn}
                          transition={{ delay: index * 0.05 }}
                          className="border border-gray-200 dark:border-gray-700 rounded-lg p-4 hover:shadow-md transition-shadow"
                        >
                          <div className="flex items-start justify-between">
                            <div className="flex items-start space-x-4 flex-1">
                              <div className="flex-shrink-0 mt-1">
                                {getSeverityIcon(alert.severity)}
                              </div>
                              
                              <div className="flex-1">
                                <div className="flex items-center space-x-3 mb-2">
                                  <h4 className="text-lg font-medium text-gray-900 dark:text-white">
                                    {alert.title}
                                  </h4>
                                  <StatusBadge status={alert.severity} size="sm" />
                                  {alert.escalated && (
                                    <StatusBadge status="escalated" size="sm" />
                                  )}
                                </div>
                                
                                <p className="text-gray-600 dark:text-gray-400 mb-3">
                                  {alert.description}
                                </p>
                                
                                <div className="flex items-center space-x-4 text-sm text-gray-500 dark:text-gray-400 mb-3">
                                  <span>Airport: {alert.airport}</span>
                                  <span>•</span>
                                  <span>Type: {alert.alert_type.replace('_', ' ')}</span>
                                  <span>•</span>
                                  <span>{getTimeAgo(alert.timestamp)}</span>
                                </div>
                                
                                {alert.affected_flights.length > 0 && (
                                  <div className="mb-3">
                                    <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                                      Affected Flights ({alert.affected_flights.length}):
                                    </p>
                                    <div className="flex flex-wrap gap-1">
                                      {alert.affected_flights.slice(0, 5).map((flight, idx) => (
                                        <span
                                          key={idx}
                                          className="px-2 py-1 bg-gray-100 dark:bg-gray-800 text-xs rounded"
                                        >
                                          {flight}
                                        </span>
                                      ))}
                                      {alert.affected_flights.length > 5 && (
                                        <span className="px-2 py-1 bg-gray-100 dark:bg-gray-800 text-xs rounded">
                                          +{alert.affected_flights.length - 5} more
                                        </span>
                                      )}
                                    </div>
                                  </div>
                                )}
                                
                                {alert.recommendations.length > 0 && (
                                  <div className="mb-3">
                                    <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                      Recommendations:
                                    </p>
                                    <div className="space-y-2">
                                      {alert.recommendations.slice(0, 2).map((rec, idx) => (
                                        <div key={idx} className="flex items-start space-x-2">
                                          <div className="w-4 h-4 bg-blue-100 dark:bg-blue-900 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                                            <span className="text-xs font-bold text-blue-600 dark:text-blue-400">
                                              {idx + 1}
                                            </span>
                                          </div>
                                          <div>
                                            <p className="text-sm text-gray-700 dark:text-gray-300">
                                              {rec.action}
                                            </p>
                                            <p className="text-xs text-gray-500 dark:text-gray-400">
                                              Impact: {rec.impact} • Priority: {rec.priority}
                                            </p>
                                          </div>
                                        </div>
                                      ))}
                                    </div>
                                  </div>
                                )}
                              </div>
                            </div>
                            
                            <div className="flex items-center space-x-2 flex-shrink-0">
                              {!alert.escalated && alert.severity !== 'critical' && (
                                <Button
                                  onClick={() => escalateAlert(alert.alert_id)}
                                  variant="outline"
                                  size="sm"
                                >
                                  <TrendingUp className="w-4 h-4 mr-1" />
                                  Escalate
                                </Button>
                              )}
                              <Button
                                onClick={() => resolveAlert(alert.alert_id)}
                                variant="outline"
                                size="sm"
                                className="text-green-600 border-green-600 hover:bg-green-50"
                              >
                                <CheckCircle className="w-4 h-4 mr-1" />
                                Resolve
                              </Button>
                            </div>
                          </div>
                        </motion.div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-8">
                      <CheckCircle className="w-16 h-16 text-green-500 mx-auto mb-4" />
                      <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                        No Active Alerts
                      </h3>
                      <p className="text-gray-500 dark:text-gray-400">
                        All systems are operating normally. Check back later or adjust your filters.
                      </p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </motion.div>
          </motion.div>
        </main>
      </div>
      
      {/* Notification Demo Modal */}
      <NotificationDemo
        isVisible={showNotificationDemo}
        onClose={() => setShowNotificationDemo(false)}
        notificationData={notificationData}
      />
    </div>
  )
}