import { Suspense, useState, useRef, useEffect, useMemo } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls, Grid, Environment, Html, Line } from '@react-three/drei'
import * as THREE from 'three'
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js'
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js'
import { Maximize2, Minimize2, RotateCcw, Box, Scan, Palette, Tag, TrendingDown, TrendingUp } from 'lucide-react'
import clsx from 'clsx'

// Types for 3D annotations
export interface BendAnnotation {
  id: number | string
  label?: string
  position: [number, number, number]
  calloutAnchor?: [number, number, number]
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

interface ThreeViewerProps {
  modelUrl?: string
  referenceUrl?: string
  deviations?: number[]
  tolerance?: number
  className?: string
  // New props for annotations
  bendAnnotations?: BendAnnotation[]
  deviationStats?: DeviationStats
  scanPositions?: Float32Array  // Point positions for finding extremes
  focusPoint?: [number, number, number] | null
  focusAnnotationId?: number | string | null
}

interface ScreenBendCallout {
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

export default function ThreeViewer({
  modelUrl,
  referenceUrl,
  deviations,
  tolerance = 0.1,
  className,
  bendAnnotations,
  deviationStats,
  scanPositions,
  focusPoint,
  focusAnnotationId,
}: ThreeViewerProps) {
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [showScan, setShowScan] = useState(true)
  const [showReference, setShowReference] = useState(true)
  const [showHeatmap, setShowHeatmap] = useState(true)
  const [showAnnotations, setShowAnnotations] = useState(true)
  const [showExtremes, setShowExtremes] = useState(true)
  const [isLoading, setIsLoading] = useState(false)
  const [transformData, setTransformData] = useState<{ center: THREE.Vector3; scale: number } | null>(null)
  const [extremePositions, setExtremePositions] = useState<{ minPosition?: [number, number, number]; maxPosition?: [number, number, number] } | null>(null)
  const [referencePositions, setReferencePositions] = useState<Float32Array | null>(null)
  const [screenCallouts, setScreenCallouts] = useState<ScreenBendCallout[]>([])
  const controlsRef = useRef<any>(null)

  const resetCamera = () => {
    if (controlsRef.current) {
      controlsRef.current.reset()
    }
  }

  const hasDeviations = deviations && deviations.length > 0
  const hasBendAnnotations = bendAnnotations && bendAnnotations.length > 0
  const hasExtremes = deviationStats && (deviationStats.minPosition || deviationStats.maxPosition || deviationStats.minIdx !== undefined || deviationStats.maxIdx !== undefined)

  return (
    <div className={clsx('relative bg-dark-900 rounded-lg overflow-hidden', className)}>
      {/* Toggle Controls */}
      <div className="absolute top-4 left-4 z-10 flex flex-col gap-2">
        {modelUrl && (
          <button
            onClick={() => setShowScan(!showScan)}
            className={clsx(
              'flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors',
              showScan ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30' : 'bg-dark-700 text-dark-400'
            )}
            title={showScan ? 'Hide scan' : 'Show scan'}
          >
            <Scan className="w-3.5 h-3.5" />
            Scan
          </button>
        )}
        {referenceUrl && (
          <button
            onClick={() => setShowReference(!showReference)}
            className={clsx(
              'flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors',
              showReference ? 'bg-slate-500/20 text-slate-200 border border-slate-400/30' : 'bg-dark-700 text-dark-400'
            )}
            title={showReference ? 'Hide CAD reference' : 'Show CAD reference'}
          >
            <Box className="w-3.5 h-3.5" />
            CAD
          </button>
        )}
        {hasDeviations && (
          <button
            onClick={() => setShowHeatmap(!showHeatmap)}
            className={clsx(
              'flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors',
              showHeatmap ? 'bg-orange-500/20 text-orange-400 border border-orange-500/30' : 'bg-dark-700 text-dark-400'
            )}
            title={showHeatmap ? 'Hide deviation heatmap' : 'Show deviation heatmap'}
          >
            <Palette className="w-3.5 h-3.5" />
            Heatmap
          </button>
        )}
        {hasBendAnnotations && (
          <button
            onClick={() => setShowAnnotations(!showAnnotations)}
            className={clsx(
              'flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors',
              showAnnotations ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' : 'bg-dark-700 text-dark-400'
            )}
            title={showAnnotations ? 'Hide bend annotations' : 'Show bend annotations'}
          >
            <Tag className="w-3.5 h-3.5" />
            Bends
          </button>
        )}
        {(hasExtremes || hasDeviations) && (
          <button
            onClick={() => setShowExtremes(!showExtremes)}
            className={clsx(
              'flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors',
              showExtremes ? 'bg-red-500/20 text-red-400 border border-red-500/30' : 'bg-dark-700 text-dark-400'
            )}
            title={showExtremes ? 'Hide min/max markers' : 'Show min/max markers'}
          >
            <TrendingUp className="w-3.5 h-3.5" />
            Extremes
          </button>
        )}
      </div>

      {/* Action Controls */}
      <div className="absolute top-4 right-4 z-10 flex gap-2">
        <button
          onClick={resetCamera}
          className="p-2 bg-dark-700 rounded-lg text-dark-400 hover:text-dark-200 transition-colors"
          title="Reset camera"
        >
          <RotateCcw className="w-4 h-4" />
        </button>
        <button
          onClick={() => setIsFullscreen(!isFullscreen)}
          className="p-2 bg-dark-700 rounded-lg text-dark-400 hover:text-dark-200 transition-colors"
          title={isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}
        >
          {isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
        </button>
      </div>

      {/* Color scale legend */}
      {showHeatmap && hasDeviations && (
        <div className="absolute bottom-4 left-4 z-10 bg-dark-800/90 p-3 rounded-lg">
          <p className="text-xs text-dark-400 mb-2">Deviation Scale (mm)</p>
          <div className="flex items-center gap-2">
            <div className="w-32 h-3 rounded" style={{
              background: 'linear-gradient(to right, #22c55e, #eab308, #ef4444)'
            }} />
          </div>
          <div className="flex justify-between text-xs text-dark-400 mt-1">
            <span>-{tolerance}</span>
            <span>0</span>
            <span>+{tolerance}</span>
          </div>
          {deviationStats && (
            <div className="mt-2 pt-2 border-t border-dark-700 text-xs">
              <div className="flex justify-between">
                <span className="text-emerald-400">Min: {deviationStats.min.toFixed(3)}mm</span>
                <span className="text-red-400">Max: {deviationStats.max.toFixed(3)}mm</span>
              </div>
              <div className="text-dark-400 mt-1">
                Mean: {deviationStats.mean.toFixed(3)}mm
              </div>
            </div>
          )}
        </div>
      )}

      {/* Legend */}
      <div className="absolute bottom-4 right-4 z-10 bg-dark-800/90 p-2 rounded-lg text-xs">
        <div className="flex items-center gap-4">
          {modelUrl && showScan && <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-blue-400"></span> Scan</span>}
          {referenceUrl && showReference && <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-slate-300"></span> CAD</span>}
        </div>
      </div>

      {/* Loading indicator */}
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-dark-900/80 z-20">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500" />
        </div>
      )}

      {!isLoading && showAnnotations && screenCallouts.length > 0 && (
        <BendCalloutOverlay callouts={screenCallouts} />
      )}

      {/* 3D Canvas */}
      <Canvas
        camera={{ position: [5, 5, 5], fov: 50 }}
        className={clsx(
          isFullscreen ? 'fixed inset-0 z-50' : 'h-full min-h-[400px]'
        )}
        gl={{ antialias: true, alpha: true }}
      >
        <Suspense fallback={<LoadingFallback />}>
          <ambientLight intensity={0.5} />
          <directionalLight position={[10, 10, 10]} intensity={1} />
          <directionalLight position={[-10, -10, -10]} intensity={0.3} />

          {/* Aligned models - both centered and scaled together */}
          <AlignedModels
            scanUrl={modelUrl}
            referenceUrl={referenceUrl}
            showScan={showScan}
            showReference={showReference}
            deviations={showHeatmap ? deviations : undefined}
            tolerance={tolerance}
            deviationStats={deviationStats}
            onLoadStart={() => setIsLoading(true)}
            onLoadEnd={() => setIsLoading(false)}
            onTransformComputed={setTransformData}
            onExtremePositionsComputed={setExtremePositions}
            onReferencePositionsComputed={setReferencePositions}
          />

          {/* 3D Annotations for bends */}
          {showAnnotations && hasBendAnnotations && transformData && (
            <>
              <BendAnnotations3D
                annotations={bendAnnotations!}
                transform={transformData}
                referencePositions={referencePositions}
                focusAnnotationId={focusAnnotationId}
              />
              <BendCalloutLayoutBridge
                annotations={bendAnnotations!}
                transform={transformData}
                focusAnnotationId={focusAnnotationId}
                onChange={setScreenCallouts}
              />
            </>
          )}

          {transformData && focusPoint && (
            <CameraFocusTarget
              focusPoint={focusPoint}
              focusAnnotation={bendAnnotations?.find((annotation) => annotation.id === focusAnnotationId) ?? null}
              transform={transformData}
              controlsRef={controlsRef}
            />
          )}

          {/* Deviation extremes markers */}
          {showExtremes && deviationStats && transformData && (
            <ExtremeMarkers3D
              stats={{
                ...deviationStats,
                minPosition: deviationStats.minPosition ?? extremePositions?.minPosition,
                maxPosition: deviationStats.maxPosition ?? extremePositions?.maxPosition,
              }}
              transform={transformData}
              deviations={deviations}
              scanPositions={scanPositions}
            />
          )}

          {/* Grid positioned below the model */}
          <Grid
            args={[20, 20]}
            position={[0, -3, 0]}
            cellSize={1}
            cellThickness={0.5}
            cellColor="#1e293b"
            sectionSize={5}
            sectionThickness={1}
            sectionColor="#334155"
            fadeDistance={30}
            fadeStrength={1}
            followCamera={false}
            infiniteGrid
          />

          <OrbitControls
            ref={controlsRef}
            enableDamping
            dampingFactor={0.05}
            minDistance={1}
            maxDistance={50}
          />

          <Environment preset="warehouse" />
        </Suspense>
      </Canvas>

      {/* No model message */}
      {!modelUrl && !referenceUrl && (
        <div className="absolute inset-0 flex items-center justify-center text-dark-400">
          <p>No 3D model available</p>
        </div>
      )}
    </div>
  )
}

function formatBendMetric(label: string, value: number | null | undefined, digits: number, suffix = '') {
  if (value == null || !Number.isFinite(value)) return `${label} -`
  return `${label} ${value >= 0 ? '+' : ''}${value.toFixed(digits)}${suffix}`
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value))
}

function distributeRailCards(
  items: Array<ScreenBendCallout & { preferredTop: number }>,
  minTop: number,
  maxBottom: number,
  gap: number,
) {
  if (!items.length) return items

  const laidOut = items.map((item) => ({ ...item, boxY: clamp(item.preferredTop, minTop, maxBottom - item.height) }))
  for (let index = 1; index < laidOut.length; index += 1) {
    const previous = laidOut[index - 1]
    const current = laidOut[index]
    const minimum = previous.boxY + previous.height + gap
    if (current.boxY < minimum) current.boxY = minimum
  }

  const overflow = laidOut[laidOut.length - 1].boxY + laidOut[laidOut.length - 1].height - maxBottom
  if (overflow > 0) {
    laidOut[laidOut.length - 1].boxY -= overflow
    for (let index = laidOut.length - 2; index >= 0; index -= 1) {
      const next = laidOut[index + 1]
      const current = laidOut[index]
      const maximum = next.boxY - current.height - gap
      if (current.boxY > maximum) current.boxY = maximum
    }
    const topOverflow = minTop - laidOut[0].boxY
    if (topOverflow > 0) {
      for (const item of laidOut) item.boxY += topOverflow
    }
  }

  return laidOut
}

function BendCalloutOverlay({ callouts }: { callouts: ScreenBendCallout[] }) {
  const statusCardClasses: Record<BendAnnotation['status'], string> = {
    pass: 'border-emerald-300/85 bg-emerald-900/92 text-emerald-50',
    fail: 'border-red-300/90 bg-red-950/92 text-red-50',
    warning: 'border-amber-300/90 bg-amber-950/92 text-amber-50',
    pending: 'border-slate-300/70 bg-slate-900/92 text-slate-100',
  }
  const statusStroke: Record<BendAnnotation['status'], string> = {
    pass: '#34d399',
    fail: '#f87171',
    warning: '#fbbf24',
    pending: '#94a3b8',
  }

  return (
    <div className="pointer-events-none absolute inset-0 z-10 overflow-hidden">
      <svg className="absolute inset-0 h-full w-full" aria-hidden="true">
        {callouts.map((callout) => {
          const exitX = callout.side === 'left' ? callout.boxX + callout.width : callout.boxX
          const exitY = callout.boxY + callout.height / 2
          const elbowX = callout.side === 'left' ? exitX + 24 : exitX - 24
          const path = `M ${callout.anchorX.toFixed(1)} ${callout.anchorY.toFixed(1)} L ${elbowX.toFixed(1)} ${callout.anchorY.toFixed(1)} L ${elbowX.toFixed(1)} ${exitY.toFixed(1)} L ${exitX.toFixed(1)} ${exitY.toFixed(1)}`
          return (
            <path
              key={`${callout.id}-connector`}
              d={path}
              fill="none"
              stroke={statusStroke[callout.status]}
              strokeOpacity={callout.active ? 0.9 : 0.68}
              strokeWidth={callout.active ? 2.3 : 1.5}
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          )
        })}
      </svg>

      {callouts.map((callout) => (
        <div
          key={callout.id}
          className={clsx(
            'absolute rounded-xl border shadow-2xl',
            callout.active ? 'ring-2 ring-white/30' : 'ring-1 ring-white/10',
            statusCardClasses[callout.status],
          )}
          style={{
            left: `${callout.boxX}px`,
            top: `${callout.boxY}px`,
            width: `${callout.width}px`,
            minHeight: `${callout.height}px`,
          }}
        >
          <div className="px-3 py-2">
            <div className="flex items-center justify-between gap-2">
              <div className="text-[11px] font-semibold tracking-[0.12em] uppercase text-white/95">
                {callout.label}
              </div>
              {callout.active && (
                <span className="rounded-full border border-white/20 bg-white/10 px-2 py-0.5 text-[9px] font-semibold uppercase tracking-[0.14em] text-white/80">
                  Focus
                </span>
              )}
            </div>
            <div className="mt-1 space-y-0.5 text-[11px] font-mono leading-tight text-white/90">
              {callout.metricRows.map((row) => (
                <div key={`${callout.id}-${row}`}>{row}</div>
              ))}
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}

function LoadingFallback() {
  return (
    <Html center>
      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-500" />
    </Html>
  )
}

// Component that manages loading and aligning both models together
interface AlignedModelsProps {
  scanUrl?: string
  referenceUrl?: string
  showScan: boolean
  showReference: boolean
  deviations?: number[]
  tolerance?: number
  deviationStats?: DeviationStats
  onLoadStart?: () => void
  onLoadEnd?: () => void
  onTransformComputed?: (data: { center: THREE.Vector3; scale: number }) => void
  onExtremePositionsComputed?: (positions: { minPosition?: [number, number, number]; maxPosition?: [number, number, number] }) => void
  onReferencePositionsComputed?: (positions: Float32Array | null) => void
}

function AlignedModels({
  scanUrl,
  referenceUrl,
  showScan,
  showReference,
  deviations,
  tolerance = 0.1,
  deviationStats,
  onLoadStart,
  onLoadEnd,
  onTransformComputed,
  onExtremePositionsComputed,
  onReferencePositionsComputed,
}: AlignedModelsProps) {
  const [scanGeometry, setScanGeometry] = useState<THREE.BufferGeometry | null>(null)
  const [refGeometry, setRefGeometry] = useState<THREE.BufferGeometry | null>(null)
  const [sharedCenter, setSharedCenter] = useState<THREE.Vector3 | null>(null)
  const [sharedScale, setSharedScale] = useState<number>(1)
  const [scanError, setScanError] = useState<string | null>(null)
  const [refError, setRefError] = useState<string | null>(null)
  const [scanIsMesh, setScanIsMesh] = useState<boolean>(false)
  const [refIsMesh, setRefIsMesh] = useState<boolean>(true)

  // Refs to track current geometries for proper disposal
  const scanGeometryRef = useRef<THREE.BufferGeometry | null>(null)
  const refGeometryRef = useRef<THREE.BufferGeometry | null>(null)
  // Store raw (pre-transform) scan positions for extreme marker extraction
  const rawScanPositionsRef = useRef<Float32Array | null>(null)

  // Detect if geometry should be rendered as mesh or point cloud
  // STL files are ALWAYS meshes (triangulated surfaces)
  // PLY files: check if indexed (mesh) or from aligned-scan endpoint (point cloud)
  const detectGeometryType = (url: string, geometry: THREE.BufferGeometry): boolean => {
    const lowerUrl = url.toLowerCase()

    // STL files are always meshes
    if (lowerUrl.endsWith('.stl')) {
      return true
    }

    // Reference mesh endpoint always returns meshes
    if (lowerUrl.includes('/api/reference-mesh/')) {
      return true
    }

    // Aligned scan endpoint returns point clouds
    if (lowerUrl.includes('/api/aligned-scan/')) {
      return false
    }

    // PLY files: check if indexed (mesh) or non-indexed
    // If indexed, it's a mesh
    if (geometry.index !== null) {
      return true
    }

    // For non-indexed PLY, check vertex count
    // If vertex count is divisible by 3 and forms reasonable triangles, it might be a mesh
    // But our aligned scans are point clouds, so default to point cloud for non-indexed PLY
    return false
  }

  // Load geometries
  useEffect(() => {
    const loadGeometry = async (url: string): Promise<{ geometry: THREE.BufferGeometry; isMesh: boolean }> => {
      const isSTL = url.toLowerCase().endsWith('.stl')
      const isPLY = url.toLowerCase().endsWith('.ply')

      let geometry: THREE.BufferGeometry

      if (isSTL) {
        const loader = new STLLoader()
        geometry = await new Promise((resolve, reject) => {
          loader.load(url, resolve, undefined, reject)
        })
      } else if (isPLY) {
        const loader = new PLYLoader()
        geometry = await new Promise((resolve, reject) => {
          loader.load(url, resolve, undefined, reject)
        })
      } else {
        throw new Error('Unsupported file format')
      }

      const isMesh = detectGeometryType(url, geometry)
      return { geometry, isMesh }
    }

    const loadBoth = async () => {
      onLoadStart?.()
      const geometries: { scan?: THREE.BufferGeometry; ref?: THREE.BufferGeometry } = {}

      // Load scan
      if (scanUrl) {
        try {
          const result = await loadGeometry(scanUrl)
          geometries.scan = result.geometry
          // Snapshot raw vertex positions before any transforms
          const posAttr = result.geometry.getAttribute('position')
          if (posAttr) {
            rawScanPositionsRef.current = new Float32Array(posAttr.array)
          }
          setScanIsMesh(result.isMesh)
          setScanError(null)
        } catch (e) {
          console.error('Error loading scan:', e)
          setScanError(e instanceof Error ? e.message : 'Failed to load')
        }
      }

      // Load reference
      if (referenceUrl) {
        try {
          const result = await loadGeometry(referenceUrl)
          geometries.ref = result.geometry
          setRefIsMesh(result.isMesh)
          setRefError(null)
        } catch (e) {
          console.error('Error loading reference:', e)
          setRefError(e instanceof Error ? e.message : 'Failed to load')
        }
      }

      // Compute bounding boxes for centering and scaling
      // Use REFERENCE geometry center for centering (backend already aligned scan to reference)
      // Use COMBINED bounding box for scale (viewport fitting)
      if (geometries.ref) {
        geometries.ref.computeBoundingBox()
      }
      if (geometries.scan) {
        geometries.scan.computeBoundingBox()
      }

      // Center on reference geometry to preserve backend alignment
      const centerBox = geometries.ref?.boundingBox ?? geometries.scan?.boundingBox
      const scaleBox = new THREE.Box3()
      if (geometries.scan?.boundingBox) scaleBox.union(geometries.scan.boundingBox)
      if (geometries.ref?.boundingBox) scaleBox.union(geometries.ref.boundingBox)

      if (centerBox && !centerBox.isEmpty()) {
        const center = new THREE.Vector3()
        centerBox.getCenter(center)
        setSharedCenter(center)

        const size = new THREE.Vector3()
        scaleBox.getSize(size)
        const maxDim = Math.max(size.x, size.y, size.z)
        const targetSize = 4
        const scale = maxDim > 0 ? targetSize / maxDim : 1
        setSharedScale(scale)

        // Notify parent of transform data for annotations
        onTransformComputed?.({ center, scale })
      }

      // Apply transformations and store
      if (geometries.scan && sharedCenter) {
        geometries.scan.translate(-sharedCenter.x, -sharedCenter.y, -sharedCenter.z)
        geometries.scan.scale(sharedScale, sharedScale, sharedScale)
        if (!geometries.scan.attributes.normal) {
          geometries.scan.computeVertexNormals()
        }
        // Apply heatmap colors if deviations provided
        if (deviations && deviations.length > 0) {
          const colors = new Float32Array(geometries.scan.attributes.position.count * 3)
          for (let i = 0; i < geometries.scan.attributes.position.count; i++) {
            const deviation = i < deviations.length ? deviations[i] : 0
            const color = deviationToColor(deviation, tolerance)
            colors[i * 3] = color.r
            colors[i * 3 + 1] = color.g
            colors[i * 3 + 2] = color.b
          }
          geometries.scan.setAttribute('color', new THREE.BufferAttribute(colors, 3))
        }
        // Dispose previous geometry before setting new one
        scanGeometryRef.current?.dispose()
        scanGeometryRef.current = geometries.scan
        setScanGeometry(geometries.scan)
      } else if (geometries.scan) {
        // First load - store raw geometry, will transform on next render
        // Dispose previous geometry before setting new one
        scanGeometryRef.current?.dispose()
        scanGeometryRef.current = geometries.scan
        setScanGeometry(geometries.scan)
      }

      if (geometries.ref && sharedCenter) {
        geometries.ref.translate(-sharedCenter.x, -sharedCenter.y, -sharedCenter.z)
        geometries.ref.scale(sharedScale, sharedScale, sharedScale)
        if (!geometries.ref.attributes.normal) {
          geometries.ref.computeVertexNormals()
        }
        // Dispose previous geometry before setting new one
        refGeometryRef.current?.dispose()
        refGeometryRef.current = geometries.ref
        setRefGeometry(geometries.ref)
      } else if (geometries.ref) {
        // Dispose previous geometry before setting new one
        refGeometryRef.current?.dispose()
        refGeometryRef.current = geometries.ref
        setRefGeometry(geometries.ref)
      }

      onLoadEnd?.()
    }

    loadBoth()

    // Cleanup function uses refs to ensure current geometries are disposed
    return () => {
      scanGeometryRef.current?.dispose()
      scanGeometryRef.current = null
      refGeometryRef.current?.dispose()
      refGeometryRef.current = null
      rawScanPositionsRef.current = null
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scanUrl, referenceUrl])

  // Apply transformations when shared center/scale are computed
  useEffect(() => {
    if (!sharedCenter) return

    if (scanGeometry && !scanGeometry.userData.transformed) {
      scanGeometry.translate(-sharedCenter.x, -sharedCenter.y, -sharedCenter.z)
      scanGeometry.scale(sharedScale, sharedScale, sharedScale)
      if (!scanGeometry.attributes.normal) {
        scanGeometry.computeVertexNormals()
      }
      scanGeometry.userData.transformed = true
    }

    if (refGeometry && !refGeometry.userData.transformed) {
      refGeometry.translate(-sharedCenter.x, -sharedCenter.y, -sharedCenter.z)
      refGeometry.scale(sharedScale, sharedScale, sharedScale)
      if (!refGeometry.attributes.normal) {
        refGeometry.computeVertexNormals()
      }
      refGeometry.userData.transformed = true
    }
  }, [sharedCenter, sharedScale, scanGeometry, refGeometry])

  useEffect(() => {
    if (!onReferencePositionsComputed) return
    if (!refGeometry) {
      onReferencePositionsComputed(null)
      return
    }
    const posAttr = refGeometry.getAttribute('position')
    if (!posAttr) {
      onReferencePositionsComputed(null)
      return
    }
    onReferencePositionsComputed(new Float32Array(posAttr.array))
  }, [onReferencePositionsComputed, refGeometry])

  // Apply/remove heatmap colors reactively based on deviations prop
  useEffect(() => {
    if (!scanGeometry) return

    if (deviations && deviations.length > 0) {
      // Apply heatmap colors
      const colors = new Float32Array(scanGeometry.attributes.position.count * 3)
      for (let i = 0; i < scanGeometry.attributes.position.count; i++) {
        const deviation = i < deviations.length ? deviations[i] : 0
        const color = deviationToColor(deviation, tolerance)
        colors[i * 3] = color.r
        colors[i * 3 + 1] = color.g
        colors[i * 3 + 2] = color.b
      }
      scanGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
    } else {
      // Remove color attribute when heatmap is off
      if (scanGeometry.attributes.color) {
        scanGeometry.deleteAttribute('color')
      }
    }
  }, [scanGeometry, deviations, tolerance])

  // Extract extreme deviation positions from raw (pre-transform) scan vertex data
  useEffect(() => {
    if (!rawScanPositionsRef.current || !deviationStats || !onExtremePositionsComputed) return
    if (deviationStats.minIdx === undefined && deviationStats.maxIdx === undefined) return

    const raw = rawScanPositionsRef.current
    const vertexCount = raw.length / 3
    const result: { minPosition?: [number, number, number]; maxPosition?: [number, number, number] } = {}

    if (deviationStats.minIdx !== undefined && deviationStats.minIdx < vertexCount) {
      const idx = deviationStats.minIdx
      result.minPosition = [raw[idx * 3], raw[idx * 3 + 1], raw[idx * 3 + 2]]
    }
    if (deviationStats.maxIdx !== undefined && deviationStats.maxIdx < vertexCount) {
      const idx = deviationStats.maxIdx
      result.maxPosition = [raw[idx * 3], raw[idx * 3 + 1], raw[idx * 3 + 2]]
    }

    if (result.minPosition || result.maxPosition) {
      onExtremePositionsComputed(result)
    }
  }, [scanGeometry, deviationStats, onExtremePositionsComputed])

  return (
    <>
      {/* Scan - render as mesh if STL/mesh, points if point cloud */}
      {showScan && scanGeometry && !scanError && (
        scanIsMesh ? (
          // Mesh rendering (STL or indexed PLY)
          <mesh geometry={scanGeometry}>
            {deviations && deviations.length > 0 ? (
              <meshStandardMaterial
                vertexColors
                side={THREE.DoubleSide}
                metalness={0.1}
                roughness={0.7}
                transparent
                opacity={0.74}
              />
            ) : (
              <meshStandardMaterial
                color="#38bdf8"
                side={THREE.DoubleSide}
                metalness={0.2}
                roughness={0.6}
                transparent
                opacity={0.34}
              />
            )}
          </mesh>
        ) : (
          // Point cloud rendering
          <points geometry={scanGeometry}>
            {deviations && deviations.length > 0 ? (
              <pointsMaterial
                size={1.65}
                sizeAttenuation={false}
                vertexColors={true}
                transparent
                opacity={0.78}
              />
            ) : (
              <pointsMaterial
                size={1.7}
                sizeAttenuation={false}
                color="#38bdf8"
                transparent
                opacity={0.72}
              />
            )}
          </points>
        )
      )}

      {/* Reference CAD model - render as mesh or points based on type */}
      {showReference && refGeometry && !refError && (
        refIsMesh ? (
          <group>
            <mesh geometry={refGeometry}>
              <meshStandardMaterial
                color="#94a3b8"
                transparent
                opacity={0.18}
                side={THREE.DoubleSide}
                metalness={0.08}
                roughness={0.92}
              />
            </mesh>
            <mesh geometry={refGeometry}>
              <meshBasicMaterial
                color="#cbd5e1"
                transparent
                opacity={0.14}
                wireframe
              />
            </mesh>
          </group>
        ) : (
          // Point cloud rendering for reference
          <points geometry={refGeometry}>
            <pointsMaterial
              size={2}
              sizeAttenuation={false}
              color="#a855f7"
              transparent
              opacity={0.6}
            />
          </points>
        )
      )}

      {/* Error displays */}
      {scanError && showScan && (
        <Html center>
          <div className="bg-red-500/20 text-red-300 px-4 py-2 rounded-lg text-sm">
            Scan: {scanError}
          </div>
        </Html>
      )}
      {refError && showReference && (
        <Html center>
          <div className="bg-red-500/20 text-red-300 px-4 py-2 rounded-lg text-sm">
            CAD: {refError}
          </div>
        </Html>
      )}
    </>
  )
}

// Convert deviation value to color (green -> yellow -> red)
function deviationToColor(deviation: number, tolerance: number): THREE.Color {
  const absDeviation = Math.abs(deviation)
  const normalized = Math.min(absDeviation / tolerance, 1)

  // Green (0) -> Yellow (0.5) -> Red (1)
  if (normalized < 0.5) {
    // Green to Yellow
    const t = normalized * 2
    return new THREE.Color(t, 1, 0) // RGB interpolation
  } else {
    // Yellow to Red
    const t = (normalized - 0.5) * 2
    return new THREE.Color(1, 1 - t, 0)
  }
}

// 3D Bend Annotations Component
interface BendAnnotations3DProps {
  annotations: BendAnnotation[]
  transform: { center: THREE.Vector3; scale: number }
  referencePositions?: Float32Array | null
  focusAnnotationId?: number | string | null
}

function percentile(values: number[], q: number): number {
  if (!values.length) return 0
  const sorted = [...values].sort((a, b) => a - b)
  const idx = Math.min(sorted.length - 1, Math.max(0, (sorted.length - 1) * q))
  const lo = Math.floor(idx)
  const hi = Math.ceil(idx)
  if (lo === hi) return sorted[lo]
  const t = idx - lo
  return sorted[lo] * (1 - t) + sorted[hi] * t
}

function computeModelMaxDim(points?: Float32Array | null): number {
  if (!points || points.length < 3) return 4
  let minX = Infinity
  let minY = Infinity
  let minZ = Infinity
  let maxX = -Infinity
  let maxY = -Infinity
  let maxZ = -Infinity
  for (let i = 0; i < points.length; i += 3) {
    const x = points[i]
    const y = points[i + 1]
    const z = points[i + 2]
    if (x < minX) minX = x
    if (y < minY) minY = y
    if (z < minZ) minZ = z
    if (x > maxX) maxX = x
    if (y > maxY) maxY = y
    if (z > maxZ) maxZ = z
  }
  return Math.max(1, maxX - minX, maxY - minY, maxZ - minZ)
}

function clipLineToReference(
  midpoint: THREE.Vector3,
  start: THREE.Vector3,
  end: THREE.Vector3,
  referencePositions?: Float32Array | null,
): { start: THREE.Vector3; end: THREE.Vector3 } {
  const direction = end.clone().sub(start)
  const originalLength = direction.length()
  if (originalLength < 1e-6) {
    return { start, end }
  }
  const dir = direction.normalize()
  const modelMaxDim = computeModelMaxDim(referencePositions)
  const maxHalfSpan = Math.max(0.25, modelMaxDim * 0.42)

  if (!referencePositions || referencePositions.length < 12) {
    const half = Math.min(originalLength / 2, maxHalfSpan)
    return {
      start: midpoint.clone().addScaledVector(dir, -half),
      end: midpoint.clone().addScaledVector(dir, half),
    }
  }

  const nearDistance = Math.max(0.08, modelMaxDim * 0.045)
  const samples: number[] = []
  const offset = new THREE.Vector3()
  const perp = new THREE.Vector3()

  for (let i = 0; i < referencePositions.length; i += 3) {
    offset.set(
      referencePositions[i] - midpoint.x,
      referencePositions[i + 1] - midpoint.y,
      referencePositions[i + 2] - midpoint.z
    )
    const axial = offset.dot(dir)
    if (Math.abs(axial) > maxHalfSpan * 1.35) continue
    perp.copy(offset).addScaledVector(dir, -axial)
    if (perp.length() <= nearDistance) {
      samples.push(axial)
    }
  }

  if (samples.length < 12) {
    const half = Math.min(originalLength / 2, maxHalfSpan)
    return {
      start: midpoint.clone().addScaledVector(dir, -half),
      end: midpoint.clone().addScaledVector(dir, half),
    }
  }

  const lo = percentile(samples, 0.08)
  const hi = percentile(samples, 0.92)
  const pad = Math.max(0.04, modelMaxDim * 0.02)
  const half = Math.min(maxHalfSpan, Math.max(0.18, ((hi - lo) / 2) + pad))
  return {
    start: midpoint.clone().addScaledVector(dir, -half),
    end: midpoint.clone().addScaledVector(dir, half),
  }
}

function BendAnnotations3D({ annotations, transform, referencePositions, focusAnnotationId }: BendAnnotations3DProps) {
  return (
    <>
      {annotations.map((annotation, index) => {
        if (!annotation.position) return null

        const toViewerPoint = (source: [number, number, number]) => new THREE.Vector3(
          (source[0] - transform.center.x) * transform.scale,
          (source[1] - transform.center.y) * transform.scale,
          (source[2] - transform.center.z) * transform.scale
        )

        const anchorSource = annotation.calloutAnchor ?? annotation.position
        const pos = toViewerPoint(anchorSource)
        const rawCadLineStart = annotation.lineStart ? toViewerPoint(annotation.lineStart) : null
        const rawCadLineEnd = annotation.lineEnd ? toViewerPoint(annotation.lineEnd) : null
        const clippedCadLine = rawCadLineStart && rawCadLineEnd && !annotation.displayGeometryCanonical
          ? clipLineToReference(pos, rawCadLineStart, rawCadLineEnd, referencePositions)
          : null
        const cadLineStart = clippedCadLine?.start ?? rawCadLineStart
        const cadLineEnd = clippedCadLine?.end ?? rawCadLineEnd
        const rawDetectedLineStart = annotation.detectedLineStart ? toViewerPoint(annotation.detectedLineStart) : null
        const rawDetectedLineEnd = annotation.detectedLineEnd ? toViewerPoint(annotation.detectedLineEnd) : null
        const clippedDetectedLine = rawDetectedLineStart && rawDetectedLineEnd && !annotation.displayGeometryCanonical
          ? clipLineToReference(pos, rawDetectedLineStart, rawDetectedLineEnd, referencePositions)
          : null
        const detectedLineStart = clippedDetectedLine?.start ?? rawDetectedLineStart
        const detectedLineEnd = clippedDetectedLine?.end ?? rawDetectedLineEnd
        const label = annotation.label ?? `Bend ${annotation.id}`

        const statusColors = {
          pass: { bg: 'bg-emerald-500/90', border: 'border-emerald-400', text: 'text-emerald-100' },
          fail: { bg: 'bg-red-500/90', border: 'border-red-400', text: 'text-red-100' },
          warning: { bg: 'bg-amber-500/90', border: 'border-amber-400', text: 'text-amber-100' },
          pending: { bg: 'bg-gray-500/90', border: 'border-gray-400', text: 'text-gray-100' },
        }

        const colors = statusColors[annotation.status] || statusColors.pending
        const statusColor = {
          pass: '#22c55e',
          fail: '#ef4444',
          warning: '#f59e0b',
          pending: '#94a3b8',
        }[annotation.status] || '#94a3b8'
        const isActive = !!annotation.active || focusAnnotationId === annotation.id
        const lineWidth = isActive ? 5.4 : 2.2
        const lineOpacity = isActive ? 1 : 0.82
        const isIssue = annotation.status === 'fail' || annotation.status === 'warning'
        const showReferenceLine = isIssue || isActive || index === 0
        const showDetectedLine = isIssue || isActive
        const showLabelBubble = isActive

        return (
          <group key={annotation.id} position={[pos.x, pos.y, pos.z]}>
            {cadLineStart && cadLineEnd && showReferenceLine && (
              <>
                <Line
                  points={[
                    [cadLineStart.x - pos.x, cadLineStart.y - pos.y, cadLineStart.z - pos.z],
                    [cadLineEnd.x - pos.x, cadLineEnd.y - pos.y, cadLineEnd.z - pos.z],
                  ]}
                  color="#cbd5e1"
                  lineWidth={isActive ? 4.2 : isIssue ? 2.4 : 1.2}
                  transparent
                  opacity={isActive ? 0.98 : isIssue ? 0.72 : 0.18}
                />
                {showDetectedLine && detectedLineStart && detectedLineEnd && (
                  <Line
                    points={[
                      [detectedLineStart.x - pos.x, detectedLineStart.y - pos.y, detectedLineStart.z - pos.z],
                      [detectedLineEnd.x - pos.x, detectedLineEnd.y - pos.y, detectedLineEnd.z - pos.z],
                    ]}
                    color={statusColor}
                    lineWidth={lineWidth}
                    transparent
                    opacity={lineOpacity}
                  />
                )}
                {isActive && (
                  <Line
                    points={[
                      [
                        (detectedLineStart ?? cadLineStart).x - pos.x,
                        (detectedLineStart ?? cadLineStart).y - pos.y,
                        (detectedLineStart ?? cadLineStart).z - pos.z,
                      ],
                      [
                        (detectedLineEnd ?? cadLineEnd).x - pos.x,
                        (detectedLineEnd ?? cadLineEnd).y - pos.y,
                        (detectedLineEnd ?? cadLineEnd).z - pos.z,
                      ],
                    ]}
                    color={annotation.status === 'pass' ? '#86efac' : annotation.status === 'warning' ? '#fde68a' : '#fca5a5'}
                    lineWidth={7.5}
                    transparent
                    opacity={0.24}
                  />
                )}
              </>
            )}

            {(isIssue || isActive) && (
              <mesh scale={isActive ? 1.35 : 1}>
                <sphereGeometry args={[0.08, 16, 16]} />
                <meshStandardMaterial
                  color={statusColor}
                  emissive={statusColor}
                  emissiveIntensity={isActive ? 0.5 : 0.32}
                />
              </mesh>
            )}

            {showLabelBubble && (
              <line>
                <bufferGeometry>
                  <bufferAttribute
                    attach="attributes-position"
                    count={2}
                    array={new Float32Array([0, 0, 0, 0, 0.32, 0])}
                    itemSize={3}
                  />
                </bufferGeometry>
                <lineBasicMaterial color="#ffffff" opacity={0.34} transparent />
              </line>
            )}

            {showLabelBubble && (
              <Html
                position={[0, 0.42, 0]}
                center
                distanceFactor={9}
                style={{ pointerEvents: 'none' }}
              >
                <div className={clsx(
                  'px-2 py-1 rounded-full border shadow-lg backdrop-blur-sm whitespace-nowrap',
                  colors.bg, colors.border
                )} title={
                  annotation.measuredAngle != null
                    ? `${label} | expected ${annotation.expectedAngle.toFixed(1)}°, measured ${annotation.measuredAngle.toFixed(1)}°, delta ${(annotation.deviation ?? 0) > 0 ? '+' : ''}${(annotation.deviation ?? 0).toFixed(1)}°`
                    : `${label} | expected ${annotation.expectedAngle.toFixed(1)}°`
                }>
                  <div className="text-[11px] font-semibold text-white">
                    {label}
                  </div>
                </div>
              </Html>
            )}
          </group>
        )
      })}
    </>
  )
}

function BendCalloutLayoutBridge({
  annotations,
  transform,
  focusAnnotationId,
  onChange,
}: {
  annotations: BendAnnotation[]
  transform: { center: THREE.Vector3; scale: number }
  focusAnnotationId?: number | string | null
  onChange: (callouts: ScreenBendCallout[]) => void
}) {
  const { camera, size } = useThree()
  const lastSerializedRef = useRef('')

  useEffect(() => () => onChange([]), [onChange])

  useFrame(() => {
    const issueAnnotations = annotations.filter((annotation) => annotation.status === 'fail' || annotation.status === 'warning')
    const activeAnnotation = annotations.find((annotation) => annotation.id === focusAnnotationId)
    const candidateAnnotations = [
      ...issueAnnotations,
      ...(activeAnnotation && !issueAnnotations.some((annotation) => annotation.id === activeAnnotation.id)
        ? [activeAnnotation]
        : []),
    ]

    if (!candidateAnnotations.length) {
      if (lastSerializedRef.current !== '[]') {
        lastSerializedRef.current = '[]'
        onChange([])
      }
      return
    }

    const toViewerPoint = (source: [number, number, number]) => new THREE.Vector3(
      (source[0] - transform.center.x) * transform.scale,
      (source[1] - transform.center.y) * transform.scale,
      (source[2] - transform.center.z) * transform.scale,
    )

    const projected = candidateAnnotations
      .map((annotation) => {
        const anchorSource = annotation.calloutAnchor ?? annotation.position
        const anchor = toViewerPoint(anchorSource)
        const normalized = anchor.clone().project(camera)
        if (!Number.isFinite(normalized.x) || !Number.isFinite(normalized.y) || normalized.z > 1.05) {
          return null
        }
        const anchorX = ((normalized.x + 1) / 2) * size.width
        const anchorY = ((1 - normalized.y) / 2) * size.height
        const active = annotation.id === focusAnnotationId || !!annotation.active
        return {
          id: annotation.id,
          label: annotation.label ?? `Bend ${annotation.id}`,
          status: annotation.status,
          active,
          side: (normalized.x >= 0 ? 'left' : 'right') as 'left' | 'right',
          anchorX,
          anchorY,
          width: active ? 208 : 176,
          height: active ? 108 : 94,
          metricRows: annotation.measuredAngle == null
            ? ['Pending measurement']
            : [
                formatBendMetric('ΔA', annotation.deviation, 1, '°'),
                formatBendMetric('ΔR', annotation.radiusDeviation, 2, 'mm'),
                formatBendMetric('ΔC', annotation.lineCenterDeviationMm, 2, 'mm'),
              ],
        }
      })
      .filter((value): value is Omit<ScreenBendCallout, 'boxX' | 'boxY'> => value !== null)

    const marginLeft = 22
    const marginRight = 22
    const marginTop = 76
    const marginBottom = 28
    const cardGap = 10
    const rails = {
      left: projected.filter((item) => item.side === 'left').sort((a, b) => a.anchorY - b.anchorY),
      right: projected.filter((item) => item.side === 'right').sort((a, b) => a.anchorY - b.anchorY),
    }

    const layoutSide = (items: Array<Omit<ScreenBendCallout, 'boxX' | 'boxY'>>) => {
      const preferred = items.map((item) => ({
        ...item,
        preferredTop: clamp(item.anchorY - item.height / 2, marginTop, size.height - item.height - marginBottom),
        boxX: item.side === 'left' ? marginLeft : size.width - item.width - marginRight,
        boxY: 0,
      }))
      return distributeRailCards(preferred, marginTop, size.height - marginBottom, cardGap)
        .map(({ preferredTop, ...item }) => item)
    }

    const callouts = [...layoutSide(rails.left), ...layoutSide(rails.right)]
      .sort((a, b) => (a.active === b.active ? 0 : a.active ? -1 : 1))
      .map((item) => ({
        ...item,
        anchorX: Math.round(item.anchorX),
        anchorY: Math.round(item.anchorY),
        boxX: Math.round(item.boxX),
        boxY: Math.round(item.boxY),
      }))

    const serialized = JSON.stringify(callouts)
    if (serialized !== lastSerializedRef.current) {
      lastSerializedRef.current = serialized
      onChange(callouts)
    }
  })

  return null
}

function CameraFocusTarget({
  focusPoint,
  focusAnnotation,
  transform,
  controlsRef,
}: {
  focusPoint: [number, number, number]
  focusAnnotation?: BendAnnotation | null
  transform: { center: THREE.Vector3; scale: number }
  controlsRef: any
}) {
  const { camera } = useThree()
  const focusKey = focusPoint.join('|')

  useEffect(() => {
    const controls = controlsRef.current
    if (!controls) return

    const target = new THREE.Vector3(
      (focusPoint[0] - transform.center.x) * transform.scale,
      (focusPoint[1] - transform.center.y) * transform.scale,
      (focusPoint[2] - transform.center.z) * transform.scale,
    )

    const worldUp = new THREE.Vector3(0, 1, 0)
    let inspectionDirection = new THREE.Vector3(1, 0.55, 1).normalize()
    if (focusAnnotation?.lineStart && focusAnnotation?.lineEnd) {
      const start = new THREE.Vector3(
        (focusAnnotation.lineStart[0] - transform.center.x) * transform.scale,
        (focusAnnotation.lineStart[1] - transform.center.y) * transform.scale,
        (focusAnnotation.lineStart[2] - transform.center.z) * transform.scale,
      )
      const end = new THREE.Vector3(
        (focusAnnotation.lineEnd[0] - transform.center.x) * transform.scale,
        (focusAnnotation.lineEnd[1] - transform.center.y) * transform.scale,
        (focusAnnotation.lineEnd[2] - transform.center.z) * transform.scale,
      )
      const lineDirection = end.clone().sub(start).normalize()
      const lateral = new THREE.Vector3().crossVectors(lineDirection, worldUp)
      if (lateral.lengthSq() < 1e-4) {
        lateral.set(1, 0, 0)
      } else {
        lateral.normalize()
      }
      inspectionDirection = lateral.multiplyScalar(0.88).add(worldUp.clone().multiplyScalar(0.58)).normalize()
    }

    const lineLength = focusAnnotation?.lineStart && focusAnnotation?.lineEnd
      ? new THREE.Vector3(
          (focusAnnotation.lineStart[0] - focusAnnotation.lineEnd[0]) * transform.scale,
          (focusAnnotation.lineStart[1] - focusAnnotation.lineEnd[1]) * transform.scale,
          (focusAnnotation.lineStart[2] - focusAnnotation.lineEnd[2]) * transform.scale,
        ).length()
      : 0
    const desiredDistance = clamp(Math.max(2.3, lineLength * 3.8), 2.3, 7.4)
    const nextPosition = target.clone().addScaledVector(inspectionDirection, desiredDistance)
    controls.target.copy(target)
    camera.position.copy(nextPosition)
    camera.updateProjectionMatrix()
    controls.update()
  }, [camera, controlsRef, focusAnnotation, focusKey, focusPoint, transform.center.x, transform.center.y, transform.center.z, transform.scale])

  return null
}

// Extreme deviation markers component
interface ExtremeMarkers3DProps {
  stats: DeviationStats
  transform: { center: THREE.Vector3; scale: number }
  deviations?: number[]
  scanPositions?: Float32Array
}

function ExtremeMarkers3D({ stats, transform, deviations, scanPositions }: ExtremeMarkers3DProps) {
  // Find positions of min/max deviations if not provided but we have deviation data
  const extremePositions = useMemo(() => {
    const result: { min?: THREE.Vector3; max?: THREE.Vector3 } = {}

    // Use provided positions if available
    if (stats.minPosition) {
      result.min = new THREE.Vector3(
        (stats.minPosition[0] - transform.center.x) * transform.scale,
        (stats.minPosition[1] - transform.center.y) * transform.scale,
        (stats.minPosition[2] - transform.center.z) * transform.scale
      )
    }

    if (stats.maxPosition) {
      result.max = new THREE.Vector3(
        (stats.maxPosition[0] - transform.center.x) * transform.scale,
        (stats.maxPosition[1] - transform.center.y) * transform.scale,
        (stats.maxPosition[2] - transform.center.z) * transform.scale
      )
    }

    // If positions not provided, find them from scan data
    if ((!result.min || !result.max) && deviations && scanPositions && scanPositions.length >= deviations.length * 3) {
      let minIdx = 0
      let maxIdx = 0
      let minVal = Infinity
      let maxVal = -Infinity

      for (let i = 0; i < deviations.length; i++) {
        const dev = deviations[i]
        if (dev < minVal) {
          minVal = dev
          minIdx = i
        }
        if (dev > maxVal) {
          maxVal = dev
          maxIdx = i
        }
      }

      if (!result.min && minIdx * 3 + 2 < scanPositions.length) {
        result.min = new THREE.Vector3(
          (scanPositions[minIdx * 3] - transform.center.x) * transform.scale,
          (scanPositions[minIdx * 3 + 1] - transform.center.y) * transform.scale,
          (scanPositions[minIdx * 3 + 2] - transform.center.z) * transform.scale
        )
      }

      if (!result.max && maxIdx * 3 + 2 < scanPositions.length) {
        result.max = new THREE.Vector3(
          (scanPositions[maxIdx * 3] - transform.center.x) * transform.scale,
          (scanPositions[maxIdx * 3 + 1] - transform.center.y) * transform.scale,
          (scanPositions[maxIdx * 3 + 2] - transform.center.z) * transform.scale
        )
      }
    }

    return result
  }, [stats, transform, deviations, scanPositions])

  return (
    <>
      {/* Min deviation marker */}
      {extremePositions.min && (
        <group position={[extremePositions.min.x, extremePositions.min.y, extremePositions.min.z]}>
          {/* Pulsing ring */}
          <mesh rotation={[Math.PI / 2, 0, 0]}>
            <ringGeometry args={[0.1, 0.15, 32]} />
            <meshBasicMaterial color="#22c55e" transparent opacity={0.8} side={THREE.DoubleSide} />
          </mesh>

          {/* Label */}
          <Html
            position={[0, 0.4, 0]}
            center
            distanceFactor={8}
            style={{ pointerEvents: 'none' }}
          >
            <div className="px-2 py-1 rounded-lg bg-emerald-600/90 border border-emerald-400 shadow-lg whitespace-nowrap">
              <div className="flex items-center gap-1 text-xs font-bold text-white">
                <TrendingDown className="w-3 h-3" />
                MIN
              </div>
              <div className="text-xs text-emerald-100 font-mono">
                {stats.min.toFixed(3)}mm
              </div>
            </div>
          </Html>
        </group>
      )}

      {/* Max deviation marker */}
      {extremePositions.max && (
        <group position={[extremePositions.max.x, extremePositions.max.y, extremePositions.max.z]}>
          {/* Pulsing ring */}
          <mesh rotation={[Math.PI / 2, 0, 0]}>
            <ringGeometry args={[0.1, 0.15, 32]} />
            <meshBasicMaterial color="#ef4444" transparent opacity={0.8} side={THREE.DoubleSide} />
          </mesh>

          {/* Label */}
          <Html
            position={[0, 0.4, 0]}
            center
            distanceFactor={8}
            style={{ pointerEvents: 'none' }}
          >
            <div className="px-2 py-1 rounded-lg bg-red-600/90 border border-red-400 shadow-lg whitespace-nowrap">
              <div className="flex items-center gap-1 text-xs font-bold text-white">
                <TrendingUp className="w-3 h-3" />
                MAX
              </div>
              <div className="text-xs text-red-100 font-mono">
                {stats.max.toFixed(3)}mm
              </div>
            </div>
          </Html>
        </group>
      )}
    </>
  )
}
