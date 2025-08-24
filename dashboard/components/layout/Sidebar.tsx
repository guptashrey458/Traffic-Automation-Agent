'use client'

import { useState } from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { motion } from 'framer-motion'
import {
  Home,
  BarChart3,
  Target,
  TrendingUp,
  AlertTriangle,
  Clock,
  Activity,
  Settings,
  ChevronLeft,
  ChevronRight
} from 'lucide-react'
import { cn } from '@/lib/utils'

const navigationItems = [
  {
    name: 'Dashboard',
    href: '/',
    icon: Home,
    description: 'Overview and key metrics'
  },
  {
    name: 'Analytics',
    href: '/analytics',
    icon: BarChart3,
    description: 'Traffic analysis and insights'
  },
  {
    name: 'Optimization',
    href: '/optimization',
    icon: Target,
    description: 'Schedule optimization tools'
  },
  {
    name: 'What-If',
    href: '/whatif',
    icon: TrendingUp,
    description: 'Scenario analysis and simulation'
  },
  {
    name: 'Delay Prediction',
    href: '/delay-prediction',
    icon: Clock,
    description: 'ML-powered delay forecasting'
  },
  {
    name: 'Alerts',
    href: '/alerts',
    icon: AlertTriangle,
    description: 'System alerts and notifications'
  },
  {
    name: 'Live Monitor',
    href: '/monitor',
    icon: Activity,
    description: 'Real-time system monitoring'
  },
  {
    name: 'Settings',
    href: '/settings',
    icon: Settings,
    description: 'System configuration'
  }
]

export function Sidebar() {
  const [collapsed, setCollapsed] = useState(false)
  const pathname = usePathname()

  return (
    <motion.aside
      initial={{ width: 280 }}
      animate={{ width: collapsed ? 80 : 280 }}
      className="bg-white dark:bg-gray-900 border-r border-gray-200 dark:border-gray-700 flex flex-col"
    >
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
        {!collapsed && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex items-center space-x-2"
          >
            <div className="w-8 h-8 bg-primary-600 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-lg">🚁</span>
            </div>
            <div>
              <h1 className="text-lg font-bold text-gray-900 dark:text-white">
                Traffic Agent
              </h1>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                AI Flight Scheduler
              </p>
            </div>
          </motion.div>
        )}
        
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
        >
          {collapsed ? (
            <ChevronRight className="w-4 h-4 text-gray-500" />
          ) : (
            <ChevronLeft className="w-4 h-4 text-gray-500" />
          )}
        </button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 space-y-2">
        {navigationItems.map((item) => {
          const Icon = item.icon
          const isActive = pathname === item.href
          
          return (
            <Link
              key={item.name}
              href={item.href}
              className={cn(
                'flex items-center space-x-3 px-3 py-2 rounded-lg transition-colors group',
                isActive
                  ? 'bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-400 border-r-2 border-primary-500'
                  : 'text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-800'
              )}
            >
              <Icon className={cn(
                'w-5 h-5 flex-shrink-0',
                isActive ? 'text-primary-500' : 'text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-300'
              )} />
              
              {!collapsed && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="flex-1 min-w-0"
                >
                  <span className="font-medium">{item.name}</span>
                  <p className="text-xs text-gray-500 dark:text-gray-400 truncate">
                    {item.description}
                  </p>
                </motion.div>
              )}
            </Link>
          )
        })}
      </nav>

      {/* Footer */}
      {!collapsed && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="p-4 border-t border-gray-200 dark:border-gray-700"
        >
          <div className="bg-gradient-to-r from-primary-50 to-blue-50 dark:from-primary-900/20 dark:to-blue-900/20 rounded-lg p-3">
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
              <span className="text-xs font-medium text-gray-700 dark:text-gray-300">
                System Online
              </span>
            </div>
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
              AI Agent Active
            </p>
          </div>
        </motion.div>
      )}
    </motion.aside>
  )
}
