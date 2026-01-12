import axios from 'axios'
import type {
  AuthResponse,
  LoginRequest,
  Job,
  JobProgress,
  BatchJob,
  BatchSummary,
  SystemStats,
  User,
} from '../types'

const api = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
})

// Add auth token to requests
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// Handle 401 responses
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token')
      localStorage.removeItem('user')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

// Auth API
export const authApi = {
  login: async (data: LoginRequest): Promise<AuthResponse> => {
    const response = await api.post('/auth/login', data)
    return response.data
  },

  getMe: async (): Promise<User> => {
    const response = await api.get('/auth/me')
    return response.data
  },

  changePassword: async (currentPassword: string, newPassword: string) => {
    const response = await api.post('/auth/change-password', {
      current_password: currentPassword,
      new_password: newPassword,
    })
    return response.data
  },
}

// Jobs API
export const jobsApi = {
  list: async (params?: {
    status?: string
    part_id?: string
    limit?: number
    offset?: number
  }): Promise<{ jobs: Job[]; count: number }> => {
    const response = await api.get('/jobs', { params })
    return response.data
  },

  get: async (jobId: string): Promise<Job> => {
    const response = await api.get(`/jobs/${jobId}`)
    return response.data
  },

  delete: async (jobId: string) => {
    const response = await api.delete(`/jobs/${jobId}`)
    return response.data
  },

  getProgress: async (jobId: string): Promise<JobProgress> => {
    const response = await api.get(`/progress/${jobId}`)
    return response.data
  },

  getResult: async (jobId: string) => {
    const response = await api.get(`/result/${jobId}`)
    return response.data
  },

  startAnalysis: async (formData: FormData): Promise<{ job_id: string }> => {
    const response = await api.post('/analyze', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    return response.data
  },
}

// Batch API
export const batchApi = {
  list: async (params?: {
    status?: string
    limit?: number
  }): Promise<{ batches: BatchJob[] }> => {
    const response = await api.get('/batch', { params })
    return response.data
  },

  get: async (batchId: string): Promise<BatchJob> => {
    const response = await api.get(`/batch/${batchId}`)
    return response.data
  },

  getSummary: async (batchId: string): Promise<BatchSummary> => {
    const response = await api.get(`/batch/${batchId}/summary`)
    return response.data
  },

  delete: async (batchId: string) => {
    const response = await api.delete(`/batch/${batchId}`)
    return response.data
  },

  startAnalysis: async (formData: FormData): Promise<{ batch_id: string }> => {
    const response = await api.post('/batch/analyze', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    return response.data
  },
}

// Stats API
export const statsApi = {
  get: async (): Promise<SystemStats> => {
    const response = await api.get('/stats')
    return response.data
  },
}

// Health API
export const healthApi = {
  check: async () => {
    const response = await api.get('/health')
    return response.data
  },
}

export default api
