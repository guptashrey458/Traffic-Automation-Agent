'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Bell, 
  X, 
  CheckCircle, 
  AlertTriangle, 
  Info, 
  Zap,
  Clock,
  Mail,
  MessageSquare
} from 'lucide-react'
import { Button } from './Button'
import { Card, CardContent } from './Card'

interface NotificationDemoProps {
  isVisible: boolean
  onClose: () => void
  notificationData?: {
    type: string
    message: string
    severity: string
    timestamp: string
    channels: string[]
    delivery_status: string
  }
}

export function NotificationDemo({ isVisible, onClose, notificationData }: NotificationDemoProps) {
  const [currentStep, setCurrentStep] = useState(0)
  
  const steps = [
    {
      title: "Notification Triggered",
      description: "System detected an event requiring notification",
      icon: <Bell className="w-5 h-5" />,
      color: "bg-blue-500"
    },
    {
      title: "Processing Alert",
      description: "Analyzing severity and determining recipients",
      icon: <Zap className="w-5 h-5" />,
      color: "bg-yellow-500"
    },
    {
      title: "Multi-Channel Delivery",
      description: "Sending to dashboard, email, and Slack",
      icon: <MessageSquare className="w-5 h-5" />,
      color: "bg-purple-500"
    },
    {
      title: "Delivery Confirmed",
      description: "All channels received notification successfully",
      icon: <CheckCircle className="w-5 h-5" />,
      color: "bg-green-500"
    }
  ]

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical':
        return <AlertTriangle className="w-6 h-6 text-red-500" />
      case 'warning':
        return <AlertTriangle className="w-6 h-6 text-yellow-500" />
      case 'info':
        return <Info className="w-6 h-6 text-blue-500" />
      default:
        return <Bell className="w-6 h-6 text-gray-500" />
    }
  }

  const getChannelIcon = (channel: string) => {
    switch (channel) {
      case 'email':
        return <Mail className="w-4 h-4" />
      case 'slack':
        return <MessageSquare className="w-4 h-4" />
      case 'dashboard':
        return <Bell className="w-4 h-4" />
      default:
        return <Bell className="w-4 h-4" />
    }
  }

  return (
    <AnimatePresence>
      {isVisible && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
          onClick={onClose}
        >
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.9, opacity: 0 }}
            className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="p-6">
              {/* Header */}
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center space-x-3">
                  <div className="p-2 bg-blue-100 dark:bg-blue-900 rounded-lg">
                    <Bell className="w-6 h-6 text-blue-600 dark:text-blue-400" />
                  </div>
                  <div>
                    <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                      Notification System Demo
                    </h2>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      Live notification delivery process
                    </p>
                  </div>
                </div>
                <Button variant="ghost" size="sm" onClick={onClose}>
                  <X className="w-5 h-5" />
                </Button>
              </div>

              {/* Notification Details */}
              {notificationData && (
                <Card className="mb-6">
                  <CardContent className="p-4">
                    <div className="flex items-start space-x-4">
                      <div className="flex-shrink-0">
                        {getSeverityIcon(notificationData.severity)}
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center space-x-2 mb-2">
                          <h3 className="font-medium text-gray-900 dark:text-white">
                            {notificationData.type.replace('_', ' ').toUpperCase()}
                          </h3>
                          <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                            notificationData.severity === 'critical' ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200' :
                            notificationData.severity === 'warning' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200' :
                            'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200'
                          }`}>
                            {notificationData.severity}
                          </span>
                        </div>
                        <p className="text-gray-700 dark:text-gray-300 mb-3">
                          {notificationData.message}
                        </p>
                        <div className="flex items-center space-x-4 text-sm text-gray-500 dark:text-gray-400">
                          <div className="flex items-center space-x-1">
                            <Clock className="w-4 h-4" />
                            <span>{new Date(notificationData.timestamp).toLocaleTimeString()}</span>
                          </div>
                          <div className="flex items-center space-x-2">
                            <span>Channels:</span>
                            {notificationData.channels.map((channel, index) => (
                              <div key={index} className="flex items-center space-x-1">
                                {getChannelIcon(channel)}
                                <span className="capitalize">{channel}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Process Steps */}
              <div className="space-y-4">
                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
                  Notification Delivery Process
                </h3>
                
                {steps.map((step, index) => (
                  <motion.div
                    key={index}
                    className={`flex items-center space-x-4 p-4 rounded-lg border-2 transition-all duration-500 ${
                      currentStep >= index 
                        ? 'border-green-500 bg-green-50 dark:bg-green-900/20' 
                        : 'border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800'
                    }`}
                    animate={{
                      scale: currentStep === index ? 1.02 : 1,
                    }}
                  >
                    <div className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center text-white ${
                      currentStep >= index ? 'bg-green-500' : step.color
                    }`}>
                      {currentStep > index ? (
                        <CheckCircle className="w-5 h-5" />
                      ) : (
                        step.icon
                      )}
                    </div>
                    
                    <div className="flex-1">
                      <h4 className="font-medium text-gray-900 dark:text-white">
                        {step.title}
                      </h4>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        {step.description}
                      </p>
                    </div>
                    
                    {currentStep >= index && (
                      <motion.div
                        initial={{ opacity: 0, scale: 0 }}
                        animate={{ opacity: 1, scale: 1 }}
                        className="flex-shrink-0"
                      >
                        <CheckCircle className="w-5 h-5 text-green-500" />
                      </motion.div>
                    )}
                  </motion.div>
                ))}
              </div>

              {/* Action Buttons */}
              <div className="flex items-center justify-between mt-6 pt-6 border-t border-gray-200 dark:border-gray-700">
                <Button
                  onClick={() => {
                    setCurrentStep(0)
                    // Animate through steps
                    steps.forEach((_, index) => {
                      setTimeout(() => {
                        setCurrentStep(index + 1)
                      }, (index + 1) * 800)
                    })
                  }}
                  className="bg-blue-500 hover:bg-blue-600"
                >
                  <Zap className="w-4 h-4 mr-2" />
                  Replay Demo
                </Button>
                
                <div className="flex items-center space-x-3">
                  <span className="text-sm text-gray-500 dark:text-gray-400">
                    Status: {currentStep >= steps.length ? 'Complete' : 'In Progress'}
                  </span>
                  <Button variant="outline" onClick={onClose}>
                    Close
                  </Button>
                </div>
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
