import { Outlet, NavLink, useLocation, useNavigate } from 'react-router-dom'
import { useEffect, useState } from 'react'
import {
  LayoutDashboard,
  Upload,
  FileStack,
  Layers,
  LogOut,
  Settings,
  Bot,
  Ruler,
  Database,
  Scan,
  Factory,
} from 'lucide-react'
import { useAuth } from '../hooks/useAuth'
import { LEGACY_ENGINE_PATHS, useLineFeatures } from '../hooks/useLineFeatures'
import clsx from 'clsx'
import BrandLogo from './BrandLogo'
import { readVendorLanguage, VENDOR_LANGUAGE_EVENT, type VendorLanguage } from '../features/part-catalog/pages/vendorSetupCopy'

const navItems = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/upload', icon: Upload, label: 'New Analysis' },
  { to: '/jobs', icon: FileStack, label: 'Jobs' },
  { to: '/batch', icon: Layers, label: 'Batch Processing' },
  { to: '/bend-inspection', icon: Ruler, label: 'Bend Inspection' },
  { to: '/live-scan', icon: Scan, label: 'Live Scan' },
  { to: '/sherman-chat', icon: Bot, label: 'ShermanAI Chat' },
  { to: '/parts', icon: Database, label: 'Part Catalog' },
  { to: '/production/parts', icon: Factory, label: 'Production Parts' },
]

const hebrewNavLabels: Record<string, string> = {
  '/': 'לוח בקרה', '/upload': 'ניתוח חדש', '/jobs': 'משימות', '/batch': 'עיבוד אצווה',
  '/bend-inspection': 'בדיקת כיפופים', '/live-scan': 'סריקה חיה',
  '/sherman-chat': 'צ׳אט ShermanAI', '/parts': 'קטלוג חלקים',
  '/production/parts': 'חלקי ייצור',
}

function NavigationLinks({ compact = false, rail = false, language = 'en' }: { compact?: boolean; rail?: boolean; language?: VendorLanguage }) {
  // The line refuses the legacy engine uploads (403 ROUTE_DISABLED), so their pages are hidden there.
  const { legacyEngine } = useLineFeatures()
  const items = legacyEngine === true ? navItems : navItems.filter(item => !LEGACY_ENGINE_PATHS.has(item.to))
  return (
    <>
      {items.map(({ to, icon: Icon, label }) => (
        <NavLink
          key={to}
          to={to}
          end={to === '/'}
          aria-label={rail ? language === 'he' ? hebrewNavLabels[to] : label : undefined}
          title={rail ? language === 'he' ? hebrewNavLabels[to] : label : undefined}
          className={({ isActive }) =>
            clsx(
              rail
                ? 'flex h-12 w-12 items-center justify-center rounded-lg transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-primary-400'
                : compact
                ? 'inline-flex min-w-max items-center gap-2 rounded-md px-3 py-2 text-sm transition-colors'
                : 'flex items-center rounded-md px-4 py-2.5 transition-colors',
              isActive
                ? 'bg-primary-500/20 text-primary-300'
                : 'text-dark-300 hover:bg-dark-700 hover:text-dark-100'
            )
          }
        >
          <Icon className={clsx('flex-shrink-0', rail ? 'h-5 w-5' : compact ? 'h-4 w-4' : language === 'he' ? 'ml-3 h-5 w-5' : 'mr-3 h-5 w-5')} />
          <span className={rail ? 'sr-only' : undefined}>{language === 'he' ? hebrewNavLabels[to] : label}</span>
        </NavLink>
      ))}
    </>
  )
}

export default function Layout() {
  const { user, logout } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()
  const vendorSetupRail = location.pathname.startsWith('/parts/vendor-setup')
  const [vendorLanguage, setVendorLanguage] = useState<VendorLanguage>(readVendorLanguage)
  useEffect(() => {
    const onLanguageChange = (event: Event) => setVendorLanguage((event as CustomEvent<VendorLanguage>).detail)
    window.addEventListener(VENDOR_LANGUAGE_EVENT, onLanguageChange)
    return () => window.removeEventListener(VENDOR_LANGUAGE_EVENT, onLanguageChange)
  }, [])
  const navigationLanguage = vendorSetupRail ? vendorLanguage : 'en'

  const handleLogout = () => {
    logout()
    navigate('/login')
  }

  return (
    <div className="flex min-h-screen flex-col lg:flex-row" dir={vendorSetupRail && vendorLanguage === 'he' ? 'rtl' : 'ltr'}>
      <header className="border-b border-dark-700 bg-dark-900/95 lg:hidden">
        <div className="flex min-h-16 items-center justify-between gap-3 px-4 py-3">
          <div className="flex min-w-0 items-center">
            <BrandLogo height={34} alt="Sherman QC" />
          </div>
          <div className="flex items-center gap-2">
            <div className="flex h-9 w-9 items-center justify-center rounded-full bg-primary-500/20 text-sm font-semibold text-primary-300">
              {user?.username?.charAt(0).toUpperCase() || 'U'}
            </div>
            <button
              onClick={handleLogout}
              className="btn btn-secondary flex h-9 w-9 items-center justify-center p-0"
              title={vendorSetupRail && vendorLanguage === 'he' ? 'התנתקו' : 'Logout'}
              aria-label={vendorSetupRail && vendorLanguage === 'he' ? 'התנתקו' : 'Logout'}
            >
              <LogOut className="h-4 w-4" />
            </button>
          </div>
        </div>
        <nav className="flex gap-2 overflow-x-auto px-3 pb-3">
          <NavigationLinks compact language={navigationLanguage} />
        </nav>
      </header>

      <aside className={clsx('hidden flex-shrink-0 flex-col border-r border-dark-700 bg-dark-800 lg:flex', vendorSetupRail ? 'w-20 items-center' : 'w-64')}>
        <div className={clsx('flex h-16 items-center border-b border-dark-700', vendorSetupRail ? 'justify-center px-2' : 'px-6')}>
          {vendorSetupRail ? <BrandLogo height={40} markOnly alt="Sherman QC" /> : <BrandLogo height={40} alt="Sherman QC" />}
        </div>

        <nav className={clsx('flex-1 space-y-1', vendorSetupRail ? 'p-2' : 'p-4')} aria-label="Main navigation">
          <NavigationLinks rail={vendorSetupRail} language={navigationLanguage} />
        </nav>

        <div className={clsx('border-t border-dark-700', vendorSetupRail ? 'p-2' : 'p-4')}>
          {vendorSetupRail ? <button onClick={handleLogout} className="flex h-12 w-12 items-center justify-center rounded-lg bg-dark-700 text-dark-200 hover:bg-dark-600 focus-visible:outline focus-visible:outline-2 focus-visible:outline-primary-400" title={vendorLanguage === 'he' ? 'התנתקו' : 'Logout'} aria-label={vendorLanguage === 'he' ? 'התנתקו' : 'Logout'}><LogOut className="h-5 w-5" /></button> : <>
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
          </>}
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
