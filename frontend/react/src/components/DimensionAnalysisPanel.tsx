import { CheckCircle, XCircle, AlertTriangle, Minus, FileSpreadsheet, Ruler } from 'lucide-react'
import { DimensionAnalysisResult, DimensionComparison, BendMatchAnalysis } from '../types'
import clsx from 'clsx'

interface DimensionAnalysisPanelProps {
  dimensionAnalysis: DimensionAnalysisResult
}

export default function DimensionAnalysisPanel({ dimensionAnalysis }: DimensionAnalysisPanelProps) {
  const { summary, bend_summary, dimensions, bend_analysis, failed_dimensions } = dimensionAnalysis

  return (
    <div className="space-y-6">
      {/* Summary Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <SummaryCard
          label="Total Dimensions"
          value={summary.total_dimensions}
          icon={<Ruler className="w-5 h-5 text-primary-400" />}
        />
        <SummaryCard
          label="Measured"
          value={summary.measured}
          subtext={`of ${summary.total_dimensions}`}
          icon={<FileSpreadsheet className="w-5 h-5 text-secondary-400" />}
        />
        <SummaryCard
          label="Passed"
          value={summary.passed}
          highlight="green"
          icon={<CheckCircle className="w-5 h-5 text-emerald-400" />}
        />
        <SummaryCard
          label="Failed"
          value={summary.failed}
          highlight={summary.failed > 0 ? 'red' : undefined}
          icon={<XCircle className="w-5 h-5 text-red-400" />}
        />
      </div>

      {/* Pass Rate Bar */}
      <div className="bg-dark-700/50 rounded-lg p-4">
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm text-dark-400">Pass Rate</span>
          <span className={clsx(
            'text-lg font-bold',
            summary.pass_rate >= 90 ? 'text-emerald-400' :
            summary.pass_rate >= 70 ? 'text-amber-400' :
            'text-red-400'
          )}>
            {summary.pass_rate.toFixed(1)}%
          </span>
        </div>
        <div className="h-3 bg-dark-600 rounded-full overflow-hidden">
          <div
            className={clsx(
              'h-full rounded-full transition-all',
              summary.pass_rate >= 90 ? 'bg-emerald-500' :
              summary.pass_rate >= 70 ? 'bg-amber-500' :
              'bg-red-500'
            )}
            style={{ width: `${summary.pass_rate}%` }}
          />
        </div>
      </div>

      {/* Dimension Comparison Table */}
      <div>
        <h4 className="font-semibold text-dark-100 mb-3">Dimension Comparison</h4>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-dark-500 border-b border-dark-700">
                <th className="pb-3 pr-4">Dim #</th>
                <th className="pb-3 pr-4">Type</th>
                <th className="pb-3 pr-4">Expected</th>
                <th className="pb-3 pr-4">Measured</th>
                <th className="pb-3 pr-4">Deviation</th>
                <th className="pb-3">Status</th>
              </tr>
            </thead>
            <tbody>
              {dimensions.map((dim) => (
                <DimensionRow key={dim.dim_id} dimension={dim} />
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Bend Analysis (if available) */}
      {bend_analysis && bend_analysis.matches && bend_analysis.matches.length > 0 && (
        <div>
          <h4 className="font-semibold text-dark-100 mb-3">
            Bend-by-Bend Analysis
            <span className="ml-2 text-xs text-dark-400 font-normal">
              {bend_summary.passed} of {bend_summary.total_bends} passed
            </span>
          </h4>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-dark-500 border-b border-dark-700">
                  <th className="pb-3 pr-4">Dim #</th>
                  <th className="pb-3 pr-4">Expected</th>
                  <th className="pb-3 pr-4">Detected</th>
                  <th className="pb-3 pr-4">Measured</th>
                  <th className="pb-3 pr-4">Deviation</th>
                  <th className="pb-3">Status</th>
                </tr>
              </thead>
              <tbody>
                {bend_analysis.matches.map((match) => (
                  <BendRow key={match.dim_id} match={match} />
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Failed Dimensions Detail */}
      {failed_dimensions && failed_dimensions.length > 0 && (
        <div className="bg-red-500/5 border border-red-500/20 rounded-lg p-4">
          <h4 className="font-semibold text-red-400 mb-3 flex items-center">
            <XCircle className="w-4 h-4 mr-2" />
            Failed Dimensions Detail
          </h4>
          <div className="space-y-3">
            {failed_dimensions.map((dim) => (
              <div key={dim.dim_id} className="bg-dark-800/50 rounded p-3">
                <div className="flex justify-between items-start">
                  <div>
                    <span className="font-medium text-dark-200">
                      Dimension {dim.dim_id}
                    </span>
                    <span className="text-dark-500 ml-2">({dim.type})</span>
                  </div>
                  <StatusBadge status={dim.status} />
                </div>
                <div className="mt-2 grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
                  <div>
                    <span className="text-dark-500">Reference ({dim.reference_source ?? 'XLSX'}):</span>
                    <span className="ml-1 text-dark-300">{dim.expected}{getUnit(dim.type)}</span>
                  </div>
                  <div>
                    <span className="text-dark-500">Measured:</span>
                    <span className="ml-1 text-dark-300">
                      {dim.scan_value?.toFixed(2) ?? 'N/A'}{dim.scan_value ? getUnit(dim.type) : ''}
                    </span>
                  </div>
                  <div>
                    <span className="text-dark-500">Deviation:</span>
                    <span className={clsx(
                      'ml-1',
                      dim.scan_deviation && dim.scan_deviation > 0 ? 'text-red-400' : 'text-amber-400'
                    )}>
                      {dim.scan_deviation !== null ? `${dim.scan_deviation > 0 ? '+' : ''}${dim.scan_deviation.toFixed(2)}${getUnit(dim.type)}` : 'N/A'}
                    </span>
                  </div>
                  {dim.scan_deviation_percent !== null && (
                    <div>
                      <span className="text-dark-500">Deviation %:</span>
                      <span className="ml-1 text-red-400">
                        {dim.scan_deviation_percent > 0 ? '+' : ''}{dim.scan_deviation_percent.toFixed(2)}%
                      </span>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// Helper components
function SummaryCard({
  label,
  value,
  subtext,
  highlight,
  icon
}: {
  label: string
  value: number
  subtext?: string
  highlight?: 'green' | 'red'
  icon?: React.ReactNode
}) {
  return (
    <div className="bg-dark-700/50 rounded-lg p-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs text-dark-500">{label}</span>
        {icon}
      </div>
      <p className={clsx(
        'text-2xl font-bold',
        highlight === 'green' ? 'text-emerald-400' :
        highlight === 'red' ? 'text-red-400' :
        'text-dark-100'
      )}>
        {value}
        {subtext && <span className="text-sm text-dark-500 font-normal ml-1">{subtext}</span>}
      </p>
    </div>
  )
}

function DimensionRow({ dimension }: { dimension: DimensionComparison }) {
  const unit = getUnit(dimension.type)

  return (
    <tr className="border-b border-dark-700/50 hover:bg-dark-700/30">
      <td className="py-3 pr-4 text-dark-200 font-medium">{dimension.dim_id}</td>
      <td className="py-3 pr-4 text-dark-400 capitalize">{dimension.type}</td>
      <td className="py-3 pr-4">
        <span className="text-dark-300">{dimension.expected.toFixed(2)}{unit}</span>
        <span className="text-dark-600 text-xs ml-1">({dimension.reference_source ?? 'XLSX'})</span>
      </td>
      <td className="py-3 pr-4 text-dark-300">
        {dimension.scan_value !== null ? `${dimension.scan_value.toFixed(2)}${unit}` : 'N/A'}
      </td>
      <td className="py-3 pr-4">
        {dimension.scan_deviation !== null ? (
          <span className={clsx(
            'font-mono',
            dimension.status === 'pass' ? 'text-emerald-400' :
            dimension.status === 'fail' ? 'text-red-400' :
            dimension.status === 'warning' ? 'text-amber-400' :
            'text-dark-500'
          )}>
            {dimension.scan_deviation > 0 ? '+' : ''}{dimension.scan_deviation.toFixed(2)}{unit}
          </span>
        ) : (
          <span className="text-dark-500">-</span>
        )}
      </td>
      <td className="py-3">
        <StatusBadge status={dimension.status} />
      </td>
    </tr>
  )
}

function BendRow({ match }: { match: BendMatchAnalysis }) {
  const detected = match.cad?.angle
  const measured = match.scan?.angle
  const deviation = match.scan?.deviation

  return (
    <tr className="border-b border-dark-700/50 hover:bg-dark-700/30">
      <td className="py-3 pr-4 text-dark-200 font-medium">{match.dim_id}</td>
      <td className="py-3 pr-4 text-dark-300">
        {match.expected.toFixed(1)}°
        <span className="text-dark-600 text-xs ml-1">(XLSX)</span>
      </td>
      <td className="py-3 pr-4 text-dark-300">
        {detected != null ? `${detected.toFixed(1)}°` : 'N/A'}
        {detected != null && <span className="text-dark-600 text-xs ml-1">(CAD)</span>}
      </td>
      <td className="py-3 pr-4 text-dark-300">
        {measured != null ? `${measured.toFixed(1)}°` : 'N/A'}
      </td>
      <td className="py-3 pr-4">
        {deviation != null ? (
          <span className={clsx(
            'font-mono',
            match.status === 'pass' ? 'text-emerald-400' :
            match.status === 'fail' ? 'text-red-400' :
            match.status === 'warning' ? 'text-amber-400' :
            'text-dark-500'
          )}>
            {deviation > 0 ? '+' : ''}{deviation.toFixed(2)}°
          </span>
        ) : (
          <span className="text-dark-500">-</span>
        )}
      </td>
      <td className="py-3">
        <StatusBadge status={match.status} />
      </td>
    </tr>
  )
}

function StatusBadge({ status }: { status: string }) {
  const config = {
    pass: { icon: CheckCircle, bg: 'bg-emerald-500/20', text: 'text-emerald-400', label: 'PASS' },
    fail: { icon: XCircle, bg: 'bg-red-500/20', text: 'text-red-400', label: 'FAIL' },
    warning: { icon: AlertTriangle, bg: 'bg-amber-500/20', text: 'text-amber-400', label: 'WARN' },
    not_measured: { icon: Minus, bg: 'bg-dark-600', text: 'text-dark-400', label: 'N/A' },
    pending: { icon: Minus, bg: 'bg-dark-600', text: 'text-dark-400', label: 'PENDING' },
  }[status] || { icon: Minus, bg: 'bg-dark-600', text: 'text-dark-400', label: status.toUpperCase() }

  const Icon = config.icon

  return (
    <span className={clsx(
      'inline-flex items-center px-2 py-1 rounded text-xs font-medium',
      config.bg, config.text
    )}>
      <Icon className="w-3 h-3 mr-1" />
      {config.label}
    </span>
  )
}

function getUnit(type: string): string {
  return type === 'angle' ? '°' : 'mm'
}
