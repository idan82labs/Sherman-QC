export interface BendAnnotation {
  id: number | string
  label?: string
  position: [number, number, number]
  calloutAnchor?: [number, number, number]
  featureType?: string
  expectedAngle: number
  measuredAngle: number | null
  deviation: number | null
  status: 'pass' | 'fail' | 'warning' | 'pending'
  lineStart?: [number, number, number]
  lineEnd?: [number, number, number]
  detectedLineStart?: [number, number, number]
  detectedLineEnd?: [number, number, number]
  active?: boolean
  displayGeometryCanonical?: boolean
  radiusDeviation?: number | null
  lineCenterDeviationMm?: number | null
  toleranceAngle?: number | null
  toleranceRadius?: number | null
}

export interface DeviationStats {
  min: number
  max: number
  mean: number
  minIdx?: number
  maxIdx?: number
  minPosition?: [number, number, number]
  maxPosition?: [number, number, number]
}

export interface ScreenBendCallout {
  id: number | string
  label: string
  status: BendAnnotation['status']
  active: boolean
  side: 'left' | 'right'
  anchorX: number
  anchorY: number
  boxX: number
  boxY: number
  width: number
  height: number
  metricRows: string[]
}

export interface BendCalloutSafeZones {
  leftTop?: number
  rightTop?: number
  leftBottom?: number
  rightBottom?: number
}
