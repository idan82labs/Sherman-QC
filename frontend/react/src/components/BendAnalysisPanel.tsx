import React, { useState, useMemo } from 'react'
import { BendResult, BendFeature } from '../types'

interface BendAnalysisPanelProps {
  bends: (BendResult | BendFeature | any)[]
  tolerance: number
}

// Helper to normalize bend data - handles both flat (BendFeature) and nested (BendResult) structures
function normalizeBendResult(item: any): BendResult {
  // If it already has the nested 'bend' property, return as-is
  if (item.bend && typeof item.bend === 'object' && item.bend.bend_id !== undefined) {
    return item as BendResult
  }

  // Otherwise, it's a flat BendFeature - wrap it in BendResult structure
  return {
    bend: {
      bend_id: item.bend_id ?? 0,
      bend_name: item.bend_name ?? `Bend ${item.bend_id ?? 0}`,
      angle_degrees: item.angle_degrees ?? 0,
      radius_mm: item.radius_mm ?? 0,
      bend_line_start: item.bend_line_start ?? [0, 0, 0],
      bend_line_end: item.bend_line_end ?? [0, 0, 0],
      bend_apex: item.bend_apex ?? [0, 0, 0],
      region_point_count: item.region_point_count ?? 0,
      adjacent_surfaces: item.adjacent_surfaces ?? [],
      detection_confidence: item.detection_confidence ?? 0,
      bend_direction: item.bend_direction ?? 'unknown',
    },
    nominal_angle: item.nominal_angle ?? null,
    angle_deviation_deg: item.angle_deviation_deg ?? 0,
    springback_indicator: item.springback_indicator ?? 0,
    over_bend_indicator: item.over_bend_indicator ?? 0,
    apex_deviation_mm: item.apex_deviation_mm ?? 0,
    radius_deviation_mm: item.radius_deviation_mm ?? 0,
    twist_angle_deg: item.twist_angle_deg ?? 0,
    status: item.status ?? 'unknown',
    recommendations: item.recommendations ?? [],
    root_causes: item.root_causes ?? [],
  }
}

const getStatusColor = (status: string): string => {
  switch (status) {
    case 'pass':
      return 'bg-green-100 text-green-800 border-green-200'
    case 'warning':
      return 'bg-yellow-100 text-yellow-800 border-yellow-200'
    case 'fail':
      return 'bg-red-100 text-red-800 border-red-200'
    default:
      return 'bg-gray-100 text-gray-800 border-gray-200'
  }
}

const getStatusIcon = (status: string): string => {
  switch (status) {
    case 'pass':
      return '✓'
    case 'warning':
      return '⚠'
    case 'fail':
      return '✕'
    default:
      return '?'
  }
}

const formatAngle = (angle: number): string => {
  return `${angle.toFixed(1)}°`
}

const formatDeviation = (deviation: number): string => {
  const sign = deviation >= 0 ? '+' : ''
  return `${sign}${deviation.toFixed(2)}°`
}

interface BendCardProps {
  bendResult: BendResult
  isExpanded: boolean
  onToggle: () => void
  tolerance: number
}

const BendCard: React.FC<BendCardProps> = ({
  bendResult,
  isExpanded,
  onToggle,
  tolerance,
}) => {
  const { bend, status, nominal_angle, angle_deviation_deg, springback_indicator, over_bend_indicator } = bendResult

  return (
    <div
      className={`border rounded-lg overflow-hidden ${getStatusColor(status)}`}
    >
      {/* Header - Always visible */}
      <button
        onClick={onToggle}
        className="w-full px-4 py-3 flex items-center justify-between hover:bg-opacity-80 transition-colors"
      >
        <div className="flex items-center gap-3">
          <span className="text-2xl font-mono">{getStatusIcon(status)}</span>
          <div className="text-left">
            <h4 className="font-semibold">{bend.bend_name}</h4>
            <p className="text-sm opacity-75">
              {formatAngle(bend.angle_degrees)}
              {nominal_angle && ` (nominal: ${formatAngle(nominal_angle)})`}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          {angle_deviation_deg !== 0 && (
            <span className={`text-sm font-mono ${Math.abs(angle_deviation_deg) > tolerance ? 'font-bold' : ''}`}>
              {formatDeviation(angle_deviation_deg)}
            </span>
          )}
          <svg
            className={`w-5 h-5 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </button>

      {/* Expanded content */}
      {isExpanded && (
        <div className="px-4 py-3 bg-white bg-opacity-50 border-t">
          {/* Metrics grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
            <div>
              <p className="text-xs uppercase opacity-60">Measured Angle</p>
              <p className="font-mono font-semibold">{formatAngle(bend.angle_degrees)}</p>
            </div>
            <div>
              <p className="text-xs uppercase opacity-60">Nominal Angle</p>
              <p className="font-mono font-semibold">
                {nominal_angle ? formatAngle(nominal_angle) : 'N/A'}
              </p>
            </div>
            <div>
              <p className="text-xs uppercase opacity-60">Deviation</p>
              <p className={`font-mono font-semibold ${Math.abs(angle_deviation_deg) > tolerance ? 'text-red-600' : ''}`}>
                {formatDeviation(angle_deviation_deg)}
              </p>
            </div>
            <div>
              <p className="text-xs uppercase opacity-60">Bend Radius</p>
              <p className="font-mono font-semibold">{bend.radius_mm.toFixed(1)} mm</p>
            </div>
          </div>

          {/* Indicators */}
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div>
              <p className="text-xs uppercase opacity-60 mb-1">Springback Likelihood</p>
              <div className="flex items-center gap-2">
                <div className="flex-1 h-2 bg-gray-200 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-blue-500 rounded-full"
                    style={{ width: `${springback_indicator * 100}%` }}
                  />
                </div>
                <span className="text-xs font-mono w-10 text-right">
                  {(springback_indicator * 100).toFixed(0)}%
                </span>
              </div>
            </div>
            <div>
              <p className="text-xs uppercase opacity-60 mb-1">Overbend Likelihood</p>
              <div className="flex items-center gap-2">
                <div className="flex-1 h-2 bg-gray-200 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-orange-500 rounded-full"
                    style={{ width: `${over_bend_indicator * 100}%` }}
                  />
                </div>
                <span className="text-xs font-mono w-10 text-right">
                  {(over_bend_indicator * 100).toFixed(0)}%
                </span>
              </div>
            </div>
          </div>

          {/* Additional metrics */}
          <div className="grid grid-cols-3 gap-4 mb-4 text-sm">
            <div>
              <p className="text-xs uppercase opacity-60">Position Deviation</p>
              <p className="font-mono">{bendResult.apex_deviation_mm.toFixed(2)} mm</p>
            </div>
            <div>
              <p className="text-xs uppercase opacity-60">Direction</p>
              <p className="capitalize">{bend.bend_direction}</p>
            </div>
            <div>
              <p className="text-xs uppercase opacity-60">Confidence</p>
              <p className="font-mono">{(bend.detection_confidence * 100).toFixed(0)}%</p>
            </div>
          </div>

          {/* Recommendations */}
          {bendResult.recommendations && bendResult.recommendations.length > 0 && (
            <div className="mt-3 pt-3 border-t border-current border-opacity-20">
              <p className="text-xs uppercase opacity-60 mb-2">Recommendations</p>
              <ul className="text-sm space-y-1">
                {bendResult.recommendations.map((rec, idx) => (
                  <li key={idx} className="flex items-start gap-2">
                    <span className="text-blue-600">→</span>
                    <span>{rec}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Root causes */}
          {bendResult.root_causes && bendResult.root_causes.length > 0 && (
            <div className="mt-3 pt-3 border-t border-current border-opacity-20">
              <p className="text-xs uppercase opacity-60 mb-2">Contributing Factors</p>
              <ul className="text-sm space-y-1">
                {bendResult.root_causes.map((cause, idx) => (
                  <li key={idx} className="flex items-start gap-2">
                    <span className="text-orange-600">•</span>
                    <span>{cause}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export const BendAnalysisPanel: React.FC<BendAnalysisPanelProps> = ({
  bends: rawBends,
  tolerance,
}) => {
  // Normalize bend data to handle both flat and nested structures
  const bends = useMemo(() => {
    if (!rawBends || !Array.isArray(rawBends)) return []
    return rawBends.filter(b => b != null).map(normalizeBendResult)
  }, [rawBends])

  const [expandedBends, setExpandedBends] = useState<Set<number>>(new Set())

  const toggleBend = (bendId: number) => {
    setExpandedBends((prev) => {
      const next = new Set(prev)
      if (next.has(bendId)) {
        next.delete(bendId)
      } else {
        next.add(bendId)
      }
      return next
    })
  }

  const expandAll = () => {
    setExpandedBends(new Set(bends.map((b) => b.bend.bend_id)))
  }

  const collapseAll = () => {
    setExpandedBends(new Set())
  }

  // Calculate summary
  const passCount = bends.filter((b) => b.status === 'pass').length
  const warnCount = bends.filter((b) => b.status === 'warning').length
  const failCount = bends.filter((b) => b.status === 'fail').length

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 bg-gray-50 border-b border-gray-200 flex items-center justify-between">
        <div>
          <h3 className="font-semibold text-gray-900">Bend Analysis</h3>
          <p className="text-sm text-gray-500">
            {bends.length} bend{bends.length !== 1 ? 's' : ''} detected
          </p>
        </div>
        <div className="flex items-center gap-4">
          {/* Summary badges */}
          <div className="flex items-center gap-2 text-sm">
            {passCount > 0 && (
              <span className="px-2 py-0.5 rounded-full bg-green-100 text-green-700">
                {passCount} Pass
              </span>
            )}
            {warnCount > 0 && (
              <span className="px-2 py-0.5 rounded-full bg-yellow-100 text-yellow-700">
                {warnCount} Warning
              </span>
            )}
            {failCount > 0 && (
              <span className="px-2 py-0.5 rounded-full bg-red-100 text-red-700">
                {failCount} Fail
              </span>
            )}
          </div>
          {/* Expand/collapse buttons */}
          <div className="flex items-center gap-1">
            <button
              onClick={expandAll}
              className="px-2 py-1 text-xs text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded"
            >
              Expand All
            </button>
            <button
              onClick={collapseAll}
              className="px-2 py-1 text-xs text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded"
            >
              Collapse All
            </button>
          </div>
        </div>
      </div>

      {/* Bend cards */}
      <div className="p-4 space-y-3">
        {bends.length === 0 ? (
          <p className="text-center text-gray-500 py-8">
            No bends detected in this part
          </p>
        ) : (
          bends.map((bendResult) => (
            <BendCard
              key={bendResult.bend.bend_id}
              bendResult={bendResult}
              isExpanded={expandedBends.has(bendResult.bend.bend_id)}
              onToggle={() => toggleBend(bendResult.bend.bend_id)}
              tolerance={tolerance}
            />
          ))
        )}
      </div>
    </div>
  )
}

export default BendAnalysisPanel
