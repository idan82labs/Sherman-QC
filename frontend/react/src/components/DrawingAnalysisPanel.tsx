import { FileText, Ruler, RotateCcw } from 'lucide-react'

interface DrawingDimension {
  id: string
  dimension_type: string
  nominal: number
  tolerance_plus: number
  tolerance_minus: number
  unit?: string
  location?: string
}

interface DrawingBend {
  id: string
  sequence: number
  angle_nominal: number
  angle_tolerance?: number
  radius?: number
  direction?: string
}

interface DrawingExtraction {
  drawing_id: string
  extraction_timestamp?: string
  dimensions: DrawingDimension[]
  bends: DrawingBend[]
  material?: string
  thickness?: number
  extraction_confidence?: number
}

interface DrawingAnalysisPanelProps {
  drawingExtraction: DrawingExtraction | null
  aiDetailedAnalysis?: string
  drawingContext?: string
}

export default function DrawingAnalysisPanel({
  drawingExtraction,
  aiDetailedAnalysis,
  drawingContext
}: DrawingAnalysisPanelProps) {
  if (!drawingExtraction && !aiDetailedAnalysis && !drawingContext) {
    return null
  }

  return (
    <div className="space-y-4">
      {/* AI Analysis from Drawing */}
      {(aiDetailedAnalysis || drawingContext) && (
        <div className="card p-6">
          <h3 className="font-semibold text-dark-100 mb-4 flex items-center">
            <FileText className="w-5 h-5 mr-2 text-purple-400" />
            AI Drawing Analysis
          </h3>
          <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
            <p className="text-dark-200 text-sm whitespace-pre-wrap">
              {aiDetailedAnalysis || drawingContext}
            </p>
          </div>
        </div>
      )}

      {/* Extracted Dimensions */}
      {drawingExtraction && drawingExtraction.dimensions && drawingExtraction.dimensions.length > 0 && (
        <div className="card p-6">
          <h3 className="font-semibold text-dark-100 mb-4 flex items-center">
            <Ruler className="w-5 h-5 mr-2 text-blue-400" />
            Extracted Dimensions
            {drawingExtraction.extraction_confidence && (
              <span className="ml-2 text-xs bg-blue-500/20 text-blue-300 px-2 py-0.5 rounded">
                {(drawingExtraction.extraction_confidence * 100).toFixed(0)}% confidence
              </span>
            )}
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-dark-500 border-b border-dark-700">
                  <th className="pb-3 pr-4">ID</th>
                  <th className="pb-3 pr-4">Type</th>
                  <th className="pb-3 pr-4">Nominal</th>
                  <th className="pb-3 pr-4">Tolerance</th>
                  <th className="pb-3">Location</th>
                </tr>
              </thead>
              <tbody>
                {drawingExtraction.dimensions.map((dim, index) => (
                  <tr key={index} className="border-b border-dark-700/50">
                    <td className="py-3 pr-4 text-dark-300 font-mono">{dim.id}</td>
                    <td className="py-3 pr-4 text-dark-300 capitalize">{dim.dimension_type}</td>
                    <td className="py-3 pr-4 text-dark-200">{dim.nominal}{dim.unit || 'mm'}</td>
                    <td className="py-3 pr-4 text-dark-300">
                      +{dim.tolerance_plus} / -{dim.tolerance_minus}{dim.unit || 'mm'}
                    </td>
                    <td className="py-3 text-dark-400">{dim.location || '-'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Extracted Bends */}
      {drawingExtraction && drawingExtraction.bends && drawingExtraction.bends.length > 0 && (
        <div className="card p-6">
          <h3 className="font-semibold text-dark-100 mb-4 flex items-center">
            <RotateCcw className="w-5 h-5 mr-2 text-orange-400" />
            Bend Specifications from Drawing
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-dark-500 border-b border-dark-700">
                  <th className="pb-3 pr-4">Bend</th>
                  <th className="pb-3 pr-4">Sequence</th>
                  <th className="pb-3 pr-4">Angle</th>
                  <th className="pb-3 pr-4">Tolerance</th>
                  <th className="pb-3 pr-4">Radius</th>
                  <th className="pb-3">Direction</th>
                </tr>
              </thead>
              <tbody>
                {drawingExtraction.bends.map((bend, index) => (
                  <tr key={index} className="border-b border-dark-700/50">
                    <td className="py-3 pr-4 text-dark-300 font-mono">{bend.id}</td>
                    <td className="py-3 pr-4 text-dark-300">{bend.sequence}</td>
                    <td className="py-3 pr-4 text-dark-200">{bend.angle_nominal}°</td>
                    <td className="py-3 pr-4 text-dark-300">
                      {bend.angle_tolerance ? `±${bend.angle_tolerance}°` : '-'}
                    </td>
                    <td className="py-3 pr-4 text-dark-300">
                      {bend.radius ? `${bend.radius}mm` : '-'}
                    </td>
                    <td className="py-3 text-dark-400">{bend.direction || '-'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Material & Thickness */}
      {drawingExtraction && (drawingExtraction.material || drawingExtraction.thickness) && (
        <div className="card p-6">
          <h3 className="font-semibold text-dark-100 mb-4">Material Specifications</h3>
          <div className="grid grid-cols-2 gap-4">
            {drawingExtraction.material && (
              <div>
                <p className="text-dark-500 text-sm">Material</p>
                <p className="text-dark-200">{drawingExtraction.material}</p>
              </div>
            )}
            {drawingExtraction.thickness && (
              <div>
                <p className="text-dark-500 text-sm">Thickness</p>
                <p className="text-dark-200">{drawingExtraction.thickness}mm</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
