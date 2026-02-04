import { useState, useEffect, useCallback, createContext, useContext, type ReactNode } from 'react'
import type { User, LoginRequest } from '../types'
import { authApi } from '../services/api'

interface AuthContextType {
  user: User | null
  isAuthenticated: boolean
  isLoading: boolean
  login: (data: LoginRequest) => Promise<void>
  logout: () => void
}

const AuthContext = createContext<AuthContextType | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(() => {
    try {
      const stored = localStorage.getItem('user')
      // Check for valid JSON string (not null, undefined, or "undefined")
      if (stored && stored !== 'undefined' && stored !== 'null') {
        return JSON.parse(stored)
      }
    } catch {
      // Invalid JSON, clear it
      localStorage.removeItem('user')
    }
    return null
  })
  const [isLoading, setIsLoading] = useState(true)

  const isAuthenticated = !!user

  // Check if token is still valid on mount
  useEffect(() => {
    const checkAuth = async () => {
      const token = localStorage.getItem('token')
      if (!token) {
        setIsLoading(false)
        return
      }

      try {
        const userData = await authApi.getMe()
        setUser(userData)
        localStorage.setItem('user', JSON.stringify(userData))
      } catch {
        // Token invalid, clear storage
        localStorage.removeItem('token')
        localStorage.removeItem('user')
        setUser(null)
      } finally {
        setIsLoading(false)
      }
    }

    checkAuth()
  }, [])

  const login = useCallback(async (data: LoginRequest) => {
    const response = await authApi.login(data)
    localStorage.setItem('token', response.access_token)
    localStorage.setItem('user', JSON.stringify(response.user))
    setUser(response.user)
  }, [])

  const logout = useCallback(() => {
    localStorage.removeItem('token')
    localStorage.removeItem('user')
    setUser(null)
  }, [])

  return (
    <AuthContext.Provider
      value={{
        user,
        isAuthenticated,
        isLoading,
        login,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}

export { AuthContext }
