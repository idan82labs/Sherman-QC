import React from 'react'
import { PipelineStatus, PipelineStageStatus } from '../types'

interface ModelPipelineStatusProps {
  status: PipelineStatus
  compact?: boolean
}

const stageDisplayNames: Record<string, string> = {
  drawing_analysis: 'Drawing Analysis',
  feature_detection: 'Feature Detection',
  correlation_2d_3d: '2D-3D Correlation',
  root_cause_analysis: 'Root Cause Analysis',
  complete: 'Complete',
}

const stageIcons: Record<string, string> = {
  drawing_analysis: '📐',
  feature_detection: '🔍',
  correlation_2d_3d: '🔗',
  root_cause_analysis: '🎯',
  complete: '✅',
}

const modelBadgeColors: Record<string, string> = {
  anthropic: 'bg-purple-100 text-purple-700',
  google: 'bg-blue-100 text-blue-700',
  openai: 'bg-green-100 text-green-700',
  local: 'bg-gray-100 text-gray-700',
}

const getStatusColor = (status: string): string => {
  switch (status) {
    case 'completed':
      return 'text-green-600'
    case 'running':
      return 'text-blue-600'
    case 'failed':
      return 'text-red-600'
    case 'skipped':
      return 'text-gray-400'
    default:
      return 'text-gray-500'
  }
}

const getStatusBg = (status: string): string => {
  switch (status) {
    case 'completed':
      return 'bg-green-500'
    case 'running':
      return 'bg-blue-500'
    case 'failed':
      return 'bg-red-500'
    case 'skipped':
      return 'bg-gray-300'
    default:
      return 'bg-gray-300'
  }
}

interface StageNodeProps {
  stageName: string
  stageStatus: PipelineStageStatus
  isLast: boolean
}

const StageNode: React.FC<StageNodeProps> = ({ stageName, stageStatus, isLast }) => {
  const displayName = stageDisplayNames[stageName] || stageName
  const icon = stageIcons[stageName] || '⚙️'
  const statusColor = getStatusColor(stageStatus.status)
  const bgColor = getStatusBg(stageStatus.status)

  // Extract provider from model_used
  const getProvider = (modelUsed: string | null): string | null => {
    if (!modelUsed) return null
    if (modelUsed.includes('/')) return modelUsed.split('/')[0]
    if (modelUsed.includes('claude')) return 'anthropic'
    if (modelUsed.includes('gemini')) return 'google'
    if (modelUsed.includes('gpt')) return 'openai'
    return 'local'
  }

  const provider = getProvider(stageStatus.model_used)

  return (
    <div className="flex items-center">
      {/* Node */}
      <div className="flex flex-col items-center">
        <div className={`w-10 h-10 rounded-full flex items-center justify-center ${bgColor} text-white text-lg`}>
          {stageStatus.status === 'running' ? (
            <svg className="w-5 h-5 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
          ) : (
            icon
          )}
        </div>
        <div className="mt-2 text-center max-w-[100px]">
          <p className={`text-xs font-medium ${statusColor}`}>{displayName}</p>
          {stageStatus.duration_ms && (
            <p className="text-xs text-gray-400">{(stageStatus.duration_ms / 1000).toFixed(1)}s</p>
          )}
          {provider && (
            <span className={`inline-block mt-1 px-1.5 py-0.5 text-[10px] rounded ${modelBadgeColors[provider] || modelBadgeColors.local}`}>
              {provider}
            </span>
          )}
          {stageStatus.fallback_used && (
            <span className="inline-block mt-1 px-1.5 py-0.5 text-[10px] rounded bg-yellow-100 text-yellow-700 ml-1">
              fallback
            </span>
          )}
        </div>
      </div>

      {/* Connector line */}
      {!isLast && (
        <div className={`w-16 h-0.5 mx-2 ${stageStatus.status === 'completed' ? 'bg-green-500' : 'bg-gray-300'}`} />
      )}
    </div>
  )
}

const CompactView: React.FC<{ status: PipelineStatus }> = ({ status }) => {
  const stageOrder = ['drawing_analysis', 'feature_detection', 'correlation_2d_3d', 'root_cause_analysis']

  return (
    <div className="flex items-center gap-2">
      {stageOrder.map((stageName, idx) => {
        const stageStatus = status.stages[stageName]
        if (!stageStatus) return null

        return (
          <div key={stageName} className="flex items-center">
            <div
              className={`w-3 h-3 rounded-full ${getStatusBg(stageStatus.status)}`}
              title={`${stageDisplayNames[stageName]}: ${stageStatus.status}`}
            />
            {idx < stageOrder.length - 1 && (
              <div className={`w-4 h-0.5 ${stageStatus.status === 'completed' ? 'bg-green-500' : 'bg-gray-300'}`} />
            )}
          </div>
        )
      })}
      <span className="text-xs text-gray-500 ml-2">
        {status.overall_progress.toFixed(0)}%
      </span>
    </div>
  )
}

export const ModelPipelineStatus: React.FC<ModelPipelineStatusProps> = ({
  status,
  compact = false,
}) => {
  if (compact) {
    return <CompactView status={status} />
  }

  const stageOrder = ['drawing_analysis', 'feature_detection', 'correlation_2d_3d', 'root_cause_analysis']
  const visibleStages = stageOrder.filter((s) => status.stages[s])

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold text-gray-900">AI Analysis Pipeline</h3>
        <div className="flex items-center gap-3">
          {/* Progress bar */}
          <div className="w-32 h-2 bg-gray-200 rounded-full overflow-hidden">
            <div
              className="h-full bg-blue-500 rounded-full transition-all duration-300"
              style={{ width: `${status.overall_progress}%` }}
            />
          </div>
          <span className="text-sm font-mono text-gray-600">
            {status.overall_progress.toFixed(0)}%
          </span>
          {status.total_duration_ms && (
            <span className="text-xs text-gray-400">
              {(status.total_duration_ms / 1000).toFixed(1)}s total
            </span>
          )}
        </div>
      </div>

      {/* Pipeline visualization */}
      <div className="flex items-start justify-center py-4 overflow-x-auto">
        {visibleStages.map((stageName, idx) => (
          <StageNode
            key={stageName}
            stageName={stageName}
            stageStatus={status.stages[stageName]}
            isLast={idx === visibleStages.length - 1}
          />
        ))}
      </div>

      {/* Error message if any stage failed */}
      {Object.values(status.stages).some((s) => s.status === 'failed') && (
        <div className="mt-4 p-3 bg-red-50 rounded-lg border border-red-100">
          <p className="text-sm text-red-700 font-medium">Pipeline errors:</p>
          {Object.entries(status.stages)
            .filter(([, s]) => s.status === 'failed' && s.error_message)
            .map(([name, s]) => (
              <p key={name} className="text-sm text-red-600 mt-1">
                {stageDisplayNames[name] || name}: {s.error_message}
              </p>
            ))}
        </div>
      )}
    </div>
  )
}

export default ModelPipelineStatus
