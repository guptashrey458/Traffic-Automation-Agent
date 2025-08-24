'use client'

import { cn } from '@/lib/utils'

interface StatusBadgeProps {
  status: string
  variant?: 'default' | 'outline'
  size?: 'sm' | 'md' | 'lg'
  className?: string
}

export function StatusBadge({ 
  status, 
  variant = 'default', 
  size = 'md', 
  className 
}: StatusBadgeProps) {
  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'on_time':
      case 'scheduled':
      case 'healthy':
      case 'success':
      case 'low':
        return 'bg-green-100 text-green-800 border-green-200 dark:bg-green-900/20 dark:text-green-400 dark:border-green-800'
      case 'delayed':
      case 'warning':
      case 'medium':
        return 'bg-yellow-100 text-yellow-800 border-yellow-200 dark:bg-yellow-900/20 dark:text-yellow-400 dark:border-yellow-800'
      case 'cancelled':
      case 'critical':
      case 'high':
      case 'error':
        return 'bg-red-100 text-red-800 border-red-200 dark:bg-red-900/20 dark:text-red-400 dark:border-red-800'
      case 'boarding':
      case 'info':
        return 'bg-blue-100 text-blue-800 border-blue-200 dark:bg-blue-900/20 dark:text-blue-400 dark:border-blue-800'
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200 dark:bg-gray-900/20 dark:text-gray-400 dark:border-gray-800'
    }
  }

  const sizeClasses = {
    sm: 'px-2 py-0.5 text-xs',
    md: 'px-2.5 py-0.5 text-sm',
    lg: 'px-3 py-1 text-base'
  }

  const baseClasses = 'inline-flex items-center rounded-full font-medium'
  const variantClasses = variant === 'outline' ? 'border bg-transparent' : ''

  return (
    <span
      className={cn(
        baseClasses,
        sizeClasses[size],
        getStatusColor(status),
        variantClasses,
        className
      )}
    >
      {status.replace('_', ' ').toUpperCase()}
    </span>
  )
}
