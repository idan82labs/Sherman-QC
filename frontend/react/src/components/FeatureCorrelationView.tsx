import React from 'react'
import { Correlation2D3D, CorrelationMapping, CriticalDeviation } from '../types'

interface FeatureCorrelationViewProps {
  correlation: Correlation2D3D
}

const getConfidenceColor = (confidence: number): string => {
  if (confidence >= 0.8) return 'text-green-600'
  if (confidence >= 0.6) return 'text-yellow-600'
  return 'text-red-600'
}

const getConfidenceBg = (confidence: number): string => {
  if (confidence >= 0.8) return 'bg-green-100'
  if (confidence >= 0.6) return 'bg-yellow-100'
  return 'bg-red-100'
}

const getSeverityColor = (severity: string): string => {
  switch (severity) {
    case 'critical':
      return 'bg-red-100 text-red-800 border-red-200'
    case 'major':
      return 'bg-orange-100 text-orange-800 border-orange-200'
    case 'minor':
      return 'bg-yellow-100 text-yellow-800 border-yellow-200'
    default:
      return 'bg-gray-100 text-gray-800 border-gray-200'
  }
}

interface MappingRowProps {
  mapping: CorrelationMapping
}

const MappingRow: React.FC<MappingRowProps> = ({ mapping }) => {
  return (
    <tr className="hover:bg-gray-50">
      <td className="px-4 py-2 text-sm">
        <div className="font-mono text-gray-900">{mapping.drawing_element_id}</div>
        <div className="text-xs text-gray-500 capitalize">{mapping.drawing_element_type}</div>
      </td>
      <td className="px-4 py-2 text-center">
        <svg className="w-4 h-4 text-gray-400 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
        </svg>
      </td>
      <td className="px-4 py-2 text-sm">
        <div className="font-mono text-gray-900">{mapping.feature_3d_id}</div>
        <div className="text-xs text-gray-500 capitalize">{mapping.feature_3d_type}</div>
      </td>
      <td className="px-4 py-2 text-center">
        <div className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full ${getConfidenceBg(mapping.confidence)}`}>
          <span className={`text-sm font-mono ${getConfidenceColor(mapping.confidence)}`}>
            {(mapping.confidence * 100).toFixed(0)}%
          </span>
        </div>
      </td>
      <td className="px-4 py-2 text-sm font-mono">
        {mapping.nominal_value !== null ? mapping.nominal_value.toFixed(2) : 'N/A'}
      </td>
      <td className="px-4 py-2 text-sm font-mono">
        {mapping.measured_value !== null ? mapping.measured_value.toFixed(2) : 'N/A'}
      </td>
      <td className="px-4 py-2 text-sm font-mono">
        {mapping.deviation !== null ? (
          <span className={mapping.is_in_tolerance === false ? 'text-red-600 font-bold' : ''}>
            {mapping.deviation >= 0 ? '+' : ''}{mapping.deviation.toFixed(3)}
          </span>
        ) : 'N/A'}
      </td>
      <td className="px-4 py-2 text-center">
        {mapping.is_in_tolerance !== null && (
          <span className={`inline-flex items-center justify-center w-6 h-6 rounded-full ${
            mapping.is_in_tolerance ? 'bg-green-100 text-green-600' : 'bg-red-100 text-red-600'
          }`}>
            {mapping.is_in_tolerance ? '✓' : '✕'}
          </span>
        )}
      </td>
    </tr>
  )
}

interface CriticalDeviationCardProps {
  deviation: CriticalDeviation
}

const CriticalDeviationCard: React.FC<CriticalDeviationCardProps> = ({ deviation }) => {
  return (
    <div className={`p-3 rounded-lg border ${getSeverityColor(deviation.severity)}`}>
      <div className="flex items-start justify-between mb-2">
        <div>
          <span className="text-xs uppercase font-semibold">{deviation.severity}</span>
          <h4 className="font-medium">{deviation.feature_type} - {deviation.feature_id}</h4>
        </div>
        <span className="font-mono text-sm font-bold">
          {deviation.deviation_mm.toFixed(3)} mm
        </span>
      </div>
      <p className="text-sm mb-2">{deviation.description}</p>
      <div className="text-xs opacity-75">
        Location: ({deviation.location[0].toFixed(1)}, {deviation.location[1].toFixed(1)}, {deviation.location[2].toFixed(1)})
      </div>
      {deviation.affected_bends.length > 0 && (
        <div className="mt-2 text-xs">
          <span className="opacity-75">Affected bends: </span>
          {deviation.affected_bends.map((b, i) => (
            <span key={b} className="font-mono">
              Bend {b}{i < deviation.affected_bends.length - 1 ? ', ' : ''}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

export const FeatureCorrelationView: React.FC<FeatureCorrelationViewProps> = ({
  correlation,
}) => {
  const criticalCount = correlation.critical_deviations.filter(d => d.severity === 'critical').length
  const majorCount = correlation.critical_deviations.filter(d => d.severity === 'major').length
  const minorCount = correlation.critical_deviations.filter(d => d.severity === 'minor').length

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 bg-gray-50 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="font-semibold text-gray-900">2D-3D Feature Correlation</h3>
            <p className="text-sm text-gray-500">
              {correlation.mappings.length} mappings found
            </p>
          </div>
          <div className="flex items-center gap-4">
            <div className="text-sm">
              <span className="text-gray-500">Correlation Score: </span>
              <span className={`font-mono font-bold ${getConfidenceColor(correlation.overall_correlation_score)}`}>
                {(correlation.overall_correlation_score * 100).toFixed(0)}%
              </span>
            </div>
            {correlation.model_used && (
              <div className="text-xs text-gray-400">
                via {correlation.model_used}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Critical Deviations Alert */}
      {correlation.critical_deviations.length > 0 && (
        <div className="px-4 py-3 bg-red-50 border-b border-red-100">
          <div className="flex items-center justify-between mb-2">
            <h4 className="font-semibold text-red-800 flex items-center gap-2">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
              Critical Deviations ({correlation.critical_deviations.length})
            </h4>
            <div className="flex gap-2 text-xs">
              {criticalCount > 0 && (
                <span className="px-2 py-0.5 rounded bg-red-200 text-red-800">{criticalCount} Critical</span>
              )}
              {majorCount > 0 && (
                <span className="px-2 py-0.5 rounded bg-orange-200 text-orange-800">{majorCount} Major</span>
              )}
              {minorCount > 0 && (
                <span className="px-2 py-0.5 rounded bg-yellow-200 text-yellow-800">{minorCount} Minor</span>
              )}
            </div>
          </div>
          <div className="grid gap-2 md:grid-cols-2">
            {correlation.critical_deviations.slice(0, 4).map((dev) => (
              <CriticalDeviationCard key={dev.deviation_id} deviation={dev} />
            ))}
          </div>
          {correlation.critical_deviations.length > 4 && (
            <p className="mt-2 text-sm text-red-600">
              +{correlation.critical_deviations.length - 4} more deviations...
            </p>
          )}
        </div>
      )}

      {/* Mappings Table */}
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-gray-50 border-b">
            <tr>
              <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Drawing Element</th>
              <th className="px-4 py-2 w-8"></th>
              <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">3D Feature</th>
              <th className="px-4 py-2 text-center text-xs font-medium text-gray-500 uppercase">Confidence</th>
              <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Nominal</th>
              <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Measured</th>
              <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Deviation</th>
              <th className="px-4 py-2 text-center text-xs font-medium text-gray-500 uppercase">In Tol?</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {correlation.mappings.length === 0 ? (
              <tr>
                <td colSpan={8} className="px-4 py-8 text-center text-gray-500">
                  No feature mappings available
                </td>
              </tr>
            ) : (
              correlation.mappings.map((mapping) => (
                <MappingRow key={mapping.mapping_id} mapping={mapping} />
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Unmatched elements warning */}
      {(correlation.unmatched_drawing_elements.length > 0 || correlation.unmatched_3d_features.length > 0) && (
        <div className="px-4 py-3 bg-yellow-50 border-t border-yellow-100">
          <h4 className="text-sm font-medium text-yellow-800 mb-2">Unmatched Elements</h4>
          <div className="grid md:grid-cols-2 gap-4 text-sm">
            {correlation.unmatched_drawing_elements.length > 0 && (
              <div>
                <p className="text-yellow-700 font-medium mb-1">Drawing elements not matched:</p>
                <div className="flex flex-wrap gap-1">
                  {correlation.unmatched_drawing_elements.map((id) => (
                    <span key={id} className="px-2 py-0.5 bg-yellow-100 rounded text-yellow-800 text-xs font-mono">
                      {id}
                    </span>
                  ))}
                </div>
              </div>
            )}
            {correlation.unmatched_3d_features.length > 0 && (
              <div>
                <p className="text-yellow-700 font-medium mb-1">3D features not matched:</p>
                <div className="flex flex-wrap gap-1">
                  {correlation.unmatched_3d_features.map((id) => (
                    <span key={id} className="px-2 py-0.5 bg-yellow-100 rounded text-yellow-800 text-xs font-mono">
                      {id}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

export default FeatureCorrelationView
