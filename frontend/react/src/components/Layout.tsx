import { Outlet, NavLink, useNavigate } from 'react-router-dom'
import {
  LayoutDashboard,
  Upload,
  FileStack,
  Layers,
  LogOut,
  Settings,
  Activity,
} from 'lucide-react'
import { useAuth } from '../hooks/useAuth'
import clsx from 'clsx'

const navItems = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/upload', icon: Upload, label: 'New Analysis' },
  { to: '/jobs', icon: FileStack, label: 'Jobs' },
  { to: '/batch', icon: Layers, label: 'Batch Processing' },
]

export default function Layout() {
  const { user, logout } = useAuth()
  const navigate = useNavigate()

  const handleLogout = () => {
    logout()
    navigate('/login')
  }

  return (
    <div className="min-h-screen flex">
      {/* Sidebar */}
      <aside className="w-64 bg-dark-800 border-r border-dark-700 flex flex-col">
        {/* Logo */}
        <div className="h-16 flex items-center px-6 border-b border-dark-700">
          <Activity className="w-8 h-8 text-primary-500 mr-3" />
          <span className="text-xl font-bold bg-gradient-to-r from-primary-400 to-secondary-400 bg-clip-text text-transparent">
            Sherman QC
          </span>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-4 space-y-1">
          {navItems.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                clsx(
                  'flex items-center px-4 py-2.5 rounded-lg transition-all duration-200',
                  isActive
                    ? 'bg-primary-500/20 text-primary-400'
                    : 'text-dark-300 hover:bg-dark-700 hover:text-dark-100'
                )
              }
            >
              <Icon className="w-5 h-5 mr-3" />
              {label}
            </NavLink>
          ))}
        </nav>

        {/* User section */}
        <div className="p-4 border-t border-dark-700">
          <div className="flex items-center mb-3">
            <div className="w-10 h-10 bg-primary-500/20 rounded-full flex items-center justify-center">
              <span className="text-primary-400 font-semibold">
                {user?.username?.charAt(0).toUpperCase() || 'U'}
              </span>
            </div>
            <div className="ml-3 flex-1 min-w-0">
              <p className="text-sm font-medium text-dark-100 truncate">
                {user?.username || 'User'}
              </p>
              <p className="text-xs text-dark-400 truncate">
                {user?.role || 'operator'}
              </p>
            </div>
          </div>

          <div className="flex gap-2">
            <button className="btn btn-secondary flex-1 flex items-center justify-center text-sm">
              <Settings className="w-4 h-4 mr-1" />
              Settings
            </button>
            <button
              onClick={handleLogout}
              className="btn btn-secondary p-2"
              title="Logout"
            >
              <LogOut className="w-4 h-4" />
            </button>
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 flex flex-col overflow-hidden">
        <div className="flex-1 overflow-auto p-6">
          <Outlet />
        </div>
      </main>
    </div>
  )
}
