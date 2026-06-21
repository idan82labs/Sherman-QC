import { Outlet, NavLink, useNavigate } from 'react-router-dom'
import {
  LayoutDashboard,
  Upload,
  FileStack,
  Layers,
  LogOut,
  Settings,
  Activity,
  Bot,
  Ruler,
  Database,
  Scan,
} from 'lucide-react'
import { useAuth } from '../hooks/useAuth'
import clsx from 'clsx'

const navItems = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/upload', icon: Upload, label: 'New Analysis' },
  { to: '/jobs', icon: FileStack, label: 'Jobs' },
  { to: '/batch', icon: Layers, label: 'Batch Processing' },
  { to: '/bend-inspection', icon: Ruler, label: 'Bend Inspection' },
  { to: '/live-scan', icon: Scan, label: 'Live Scan' },
  { to: '/sherman-chat', icon: Bot, label: 'ShermanAI Chat' },
  { to: '/parts', icon: Database, label: 'Part Catalog' },
]

function NavigationLinks({ compact = false }: { compact?: boolean }) {
  return (
    <>
      {navItems.map(({ to, icon: Icon, label }) => (
        <NavLink
          key={to}
          to={to}
          end={to === '/'}
          className={({ isActive }) =>
            clsx(
              compact
                ? 'inline-flex min-w-max items-center gap-2 rounded-md px-3 py-2 text-sm transition-colors'
                : 'flex items-center rounded-md px-4 py-2.5 transition-colors',
              isActive
                ? 'bg-primary-500/20 text-primary-300'
                : 'text-dark-300 hover:bg-dark-700 hover:text-dark-100'
            )
          }
        >
          <Icon className={clsx('flex-shrink-0', compact ? 'h-4 w-4' : 'mr-3 h-5 w-5')} />
          <span>{label}</span>
        </NavLink>
      ))}
    </>
  )
}

export default function Layout() {
  const { user, logout } = useAuth()
  const navigate = useNavigate()

  const handleLogout = () => {
    logout()
    navigate('/login')
  }

  return (
    <div className="flex min-h-screen flex-col lg:flex-row">
      <header className="border-b border-dark-700 bg-dark-900/95 lg:hidden">
        <div className="flex min-h-16 items-center justify-between gap-3 px-4 py-3">
          <div className="flex min-w-0 items-center">
            <Activity className="mr-3 h-7 w-7 flex-shrink-0 text-primary-500" />
            <span className="truncate text-lg font-bold bg-gradient-to-r from-primary-400 to-secondary-400 bg-clip-text text-transparent">
              Sherman QC
            </span>
          </div>
          <div className="flex items-center gap-2">
            <div className="flex h-9 w-9 items-center justify-center rounded-full bg-primary-500/20 text-sm font-semibold text-primary-300">
              {user?.username?.charAt(0).toUpperCase() || 'U'}
            </div>
            <button
              onClick={handleLogout}
              className="btn btn-secondary flex h-9 w-9 items-center justify-center p-0"
              title="Logout"
              aria-label="Logout"
            >
              <LogOut className="h-4 w-4" />
            </button>
          </div>
        </div>
        <nav className="flex gap-2 overflow-x-auto px-3 pb-3">
          <NavigationLinks compact />
        </nav>
      </header>

      <aside className="hidden w-64 flex-shrink-0 flex-col border-r border-dark-700 bg-dark-800 lg:flex">
        <div className="flex h-16 items-center border-b border-dark-700 px-6">
          <Activity className="mr-3 h-8 w-8 text-primary-500" />
          <span className="text-xl font-bold bg-gradient-to-r from-primary-400 to-secondary-400 bg-clip-text text-transparent">
            Sherman QC
          </span>
        </div>

        <nav className="flex-1 p-4 space-y-1">
          <NavigationLinks />
        </nav>

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
              className="btn btn-secondary flex h-9 w-9 items-center justify-center p-0"
              title="Logout"
              aria-label="Logout"
            >
              <LogOut className="w-4 h-4" />
            </button>
          </div>
        </div>
      </aside>

      <main className="flex min-w-0 flex-1 flex-col overflow-hidden">
        <div className="flex-1 overflow-auto p-3 sm:p-4 lg:p-6">
          <Outlet />
        </div>
      </main>
    </div>
  )
}
