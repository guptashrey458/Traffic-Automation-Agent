import { type ClassValue, clsx } from 'clsx'

export function cn(...inputs: ClassValue[]) {
  return clsx(inputs)
}

export function formatTime(dateString: string): string {
  try {
    const date = new Date(dateString)
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      hour12: false
    })
  } catch {
    return dateString
  }
}

export function formatDate(dateString: string): string {
  try {
    const date = new Date(dateString)
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    })
  } catch {
    return dateString
  }
}

export function formatDateTime(dateString: string): string {
  try {
    const date = new Date(dateString)
    return date.toLocaleString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      hour12: false
    })
  } catch {
    return dateString
  }
}

export function formatDuration(minutes: number): string {
  if (minutes < 60) {
    return `${minutes}m`
  }
  const hours = Math.floor(minutes / 60)
  const remainingMinutes = minutes % 60
  return remainingMinutes > 0 ? `${hours}h ${remainingMinutes}m` : `${hours}h`
}

export function formatNumber(num: number, decimals: number = 0): string {
  return new Intl.NumberFormat('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals
  }).format(num)
}

export function formatCurrency(amount: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0
  }).format(amount)
}

export function formatPercentage(value: number, decimals: number = 1): string {
  return `${(value * 100).toFixed(decimals)}%`
}

export function getStatusColor(status: string): string {
  switch (status.toLowerCase()) {
    case 'on_time':
    case 'scheduled':
    case 'healthy':
    case 'success':
      return 'text-green-600 bg-green-100 dark:text-green-400 dark:bg-green-900/20'
    case 'delayed':
    case 'warning':
    case 'medium':
      return 'text-yellow-600 bg-yellow-100 dark:text-yellow-400 dark:bg-yellow-900/20'
    case 'cancelled':
    case 'critical':
    case 'high':
    case 'error':
      return 'text-red-600 bg-red-100 dark:text-red-400 dark:bg-red-900/20'
    case 'boarding':
    case 'info':
    case 'low':
      return 'text-blue-600 bg-blue-100 dark:text-blue-400 dark:bg-blue-900/20'
    default:
      return 'text-gray-600 bg-gray-100 dark:text-gray-400 dark:bg-gray-900/20'
  }
}

export function getSeverityColor(severity: string): string {
  switch (severity.toLowerCase()) {
    case 'low':
      return 'text-blue-600 bg-blue-100 dark:text-blue-400 dark:bg-blue-900/20'
    case 'medium':
      return 'text-yellow-600 bg-yellow-100 dark:text-yellow-400 dark:bg-yellow-900/20'
    case 'high':
      return 'text-orange-600 bg-orange-100 dark:text-orange-400 dark:bg-orange-900/20'
    case 'critical':
      return 'text-red-600 bg-red-100 dark:text-red-400 dark:bg-red-900/20'
    default:
      return 'text-gray-600 bg-gray-100 dark:text-gray-400 dark:bg-gray-900/20'
  }
}

export function getTrafficLevelColor(level: string): string {
  switch (level.toLowerCase()) {
    case 'low':
      return 'text-green-600 bg-green-100 dark:text-green-400 dark:bg-green-900/20'
    case 'medium':
      return 'text-yellow-600 bg-yellow-100 dark:text-yellow-400 dark:bg-yellow-900/20'
    case 'high':
      return 'text-orange-600 bg-orange-100 dark:text-orange-400 dark:bg-orange-900/20'
    case 'critical':
      return 'text-red-600 bg-red-100 dark:text-red-400 dark:bg-red-900/20'
    default:
      return 'text-gray-600 bg-gray-100 dark:text-gray-400 dark:bg-gray-900/20'
  }
}

export function getRiskLevelColor(level: string): string {
  switch (level.toLowerCase()) {
    case 'low':
      return 'text-green-600 bg-green-100 dark:text-green-400 dark:bg-green-900/20'
    case 'medium':
      return 'text-yellow-600 bg-yellow-100 dark:text-yellow-400 dark:bg-yellow-900/20'
    case 'high':
      return 'text-red-600 bg-red-100 dark:text-red-400 dark:bg-red-900/20'
    default:
      return 'text-gray-600 bg-gray-100 dark:text-gray-400 dark:bg-gray-900/20'
  }
}

export function calculateDelayImpact(delayMinutes: number): {
  level: string
  color: string
  description: string
} {
  if (delayMinutes <= 5) {
    return {
      level: 'minimal',
      color: 'text-green-600 bg-green-100 dark:text-green-400 dark:bg-green-900/20',
      description: 'Minimal impact'
    }
  } else if (delayMinutes <= 15) {
    return {
      level: 'low',
      color: 'text-yellow-600 bg-yellow-100 dark:text-yellow-400 dark:bg-yellow-900/20',
      description: 'Low impact'
    }
  } else if (delayMinutes <= 30) {
    return {
      level: 'medium',
      color: 'text-orange-600 bg-orange-100 dark:text-orange-400 dark:bg-orange-900/20',
      description: 'Medium impact'
    }
  } else {
    return {
      level: 'high',
      color: 'text-red-600 bg-red-100 dark:text-red-400 dark:bg-red-900/20',
      description: 'High impact'
    }
  }
}

export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout | null = null
  
  return (...args: Parameters<T>) => {
    if (timeout) {
      clearTimeout(timeout)
    }
    
    timeout = setTimeout(() => {
      func(...args)
    }, wait)
  }
}

export function throttle<T extends (...args: any[]) => any>(
  func: T,
  limit: number
): (...args: Parameters<T>) => void {
  let inThrottle: boolean = false
  
  return (...args: Parameters<T>) => {
    if (!inThrottle) {
      func(...args)
      inThrottle = true
      setTimeout(() => {
        inThrottle = false
      }, limit)
    }
  }
}

export function generateId(): string {
  return Math.random().toString(36).substr(2, 9)
}

export function isValidDate(date: string): boolean {
  return !isNaN(Date.parse(date))
}

export function getTodayString(): string {
  return new Date().toISOString().split('T')[0]
}

export function getYesterdayString(): string {
  const yesterday = new Date()
  yesterday.setDate(yesterday.getDate() - 1)
  return yesterday.toISOString().split('T')[0]
}

export function addDays(date: string, days: number): string {
  const d = new Date(date)
  d.setDate(d.getDate() + days)
  return d.toISOString().split('T')[0]
}

export function getTimeAgo(dateString: string): string {
  const now = new Date()
  const date = new Date(dateString)
  const diffInSeconds = Math.floor((now.getTime() - date.getTime()) / 1000)

  if (diffInSeconds < 60) {
    return 'just now'
  } else if (diffInSeconds < 3600) {
    const minutes = Math.floor(diffInSeconds / 60)
    return `${minutes} minute${minutes > 1 ? 's' : ''} ago`
  } else if (diffInSeconds < 86400) {
    const hours = Math.floor(diffInSeconds / 3600)
    return `${hours} hour${hours > 1 ? 's' : ''} ago`
  } else {
    const days = Math.floor(diffInSeconds / 86400)
    return `${days} day${days > 1 ? 's' : ''} ago`
  }
}

export function capitalizeFirst(str: string): string {
  return str.charAt(0).toUpperCase() + str.slice(1)
}

export function truncateText(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text
  return text.substr(0, maxLength) + '...'
}
