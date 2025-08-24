'use client'

import { cn } from '@/lib/utils'

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg'
  className?: string
}

export function LoadingSpinner({ size = 'md', className }: LoadingSpinnerProps) {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-8 h-8'
  }

  return (
    <div className={cn('animate-spin rounded-full border-2 border-gray-300 border-t-primary-600', sizeClasses[size], className)}>
      <span className="sr-only">Loading...</span>
    </div>
  )
}

export function LoadingDots({ className }: { className?: string }) {
  return (
    <div className={cn('loading-dots', className)}>
      <div></div>
      <div></div>
      <div></div>
      <div></div>
    </div>
  )
}

export function LoadingCard() {
  return (
    <div className="animate-pulse">
      <div className="bg-gray-200 dark:bg-gray-700 rounded-lg p-6">
        <div className="h-4 bg-gray-300 dark:bg-gray-600 rounded w-3/4 mb-4"></div>
        <div className="h-3 bg-gray-300 dark:bg-gray-600 rounded w-1/2 mb-2"></div>
        <div className="h-3 bg-gray-300 dark:bg-gray-600 rounded w-2/3"></div>
      </div>
    </div>
  )
}
