import { createContext, useContext, useState, useCallback, ReactNode } from 'react'
import { X, CheckCircle, AlertCircle, AlertTriangle, Info } from 'lucide-react'

type NotificationType = 'success' | 'error' | 'warning' | 'info'

interface Notification {
  id: string
  type: NotificationType
  message: string
  duration?: number
}

interface NotificationContextType {
  notifications: Notification[]
  success: (message: string, duration?: number) => void
  error: (message: string, duration?: number) => void
  warning: (message: string, duration?: number) => void
  info: (message: string, duration?: number) => void
  dismiss: (id: string) => void
  clearAll: () => void
}

const NotificationContext = createContext<NotificationContextType | null>(null)

export function useNotification() {
  const context = useContext(NotificationContext)
  if (!context) {
    throw new Error('useNotification must be used within a NotificationProvider')
  }
  return context
}

interface NotificationProviderProps {
  children: ReactNode
}

export function NotificationProvider({ children }: NotificationProviderProps) {
  const [notifications, setNotifications] = useState<Notification[]>([])

  const addNotification = useCallback((type: NotificationType, message: string, duration = 5000) => {
    const id = `${Date.now()}-${Math.random().toString(36).slice(2)}`
    const notification: Notification = { id, type, message, duration }

    setNotifications(prev => [...prev, notification])

    if (duration > 0) {
      setTimeout(() => {
        setNotifications(prev => prev.filter(n => n.id !== id))
      }, duration)
    }
  }, [])

  const dismiss = useCallback((id: string) => {
    setNotifications(prev => prev.filter(n => n.id !== id))
  }, [])

  const clearAll = useCallback(() => {
    setNotifications([])
  }, [])

  const value: NotificationContextType = {
    notifications,
    success: (message, duration) => addNotification('success', message, duration),
    error: (message, duration) => addNotification('error', message, duration ?? 8000),
    warning: (message, duration) => addNotification('warning', message, duration),
    info: (message, duration) => addNotification('info', message, duration),
    dismiss,
    clearAll,
  }

  return (
    <NotificationContext.Provider value={value}>
      {children}
      <NotificationContainer notifications={notifications} onDismiss={dismiss} />
    </NotificationContext.Provider>
  )
}

interface NotificationContainerProps {
  notifications: Notification[]
  onDismiss: (id: string) => void
}

function NotificationContainer({ notifications, onDismiss }: NotificationContainerProps) {
  if (notifications.length === 0) return null

  return (
    <div className="fixed top-4 right-4 z-50 flex flex-col gap-2 max-w-md">
      {notifications.map(notification => (
        <NotificationItem
          key={notification.id}
          notification={notification}
          onDismiss={onDismiss}
        />
      ))}
    </div>
  )
}

interface NotificationItemProps {
  notification: Notification
  onDismiss: (id: string) => void
}

function NotificationItem({ notification, onDismiss }: NotificationItemProps) {
  const { id, type, message } = notification

  const styles = {
    success: {
      bg: 'bg-green-900/90 border-green-700',
      icon: <CheckCircle className="w-5 h-5 text-green-400" />,
    },
    error: {
      bg: 'bg-red-900/90 border-red-700',
      icon: <AlertCircle className="w-5 h-5 text-red-400" />,
    },
    warning: {
      bg: 'bg-yellow-900/90 border-yellow-700',
      icon: <AlertTriangle className="w-5 h-5 text-yellow-400" />,
    },
    info: {
      bg: 'bg-blue-900/90 border-blue-700',
      icon: <Info className="w-5 h-5 text-blue-400" />,
    },
  }

  const style = styles[type]

  return (
    <div
      className={`${style.bg} border rounded-lg p-4 shadow-lg backdrop-blur-sm flex items-start gap-3 animate-slide-in`}
      role="alert"
    >
      {style.icon}
      <p className="flex-1 text-sm text-gray-100">{message}</p>
      <button
        onClick={() => onDismiss(id)}
        className="text-gray-400 hover:text-gray-200 transition-colors"
        aria-label="Dismiss notification"
      >
        <X className="w-4 h-4" />
      </button>
    </div>
  )
}
