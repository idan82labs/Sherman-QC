import { Suspense, useState, useRef, useEffect, useMemo } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Grid, Environment, Html } from '@react-three/drei'
import * as THREE from 'three'
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js'
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js'
import { Maximize2, Minimize2, RotateCcw, Box, Scan, Palette, Tag, TrendingDown, TrendingUp } from 'lucide-react'
import clsx from 'clsx'

// Types for 3D annotations
interface BendAnnotation {
  id: number
  position: [number, number, number]
  expectedAngle: number
  measuredAngle: number | null
  deviation: number | null
  status: 'pass' | 'fail' | 'warning' | 'pending'
}

interface DeviationStats {
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
              showReference ? 'bg-purple-500/20 text-purple-400 border border-purple-500/30' : 'bg-dark-700 text-dark-400'
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
          {showScan && <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-blue-400"></span> Scan</span>}
          {showReference && <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-purple-400"></span> CAD</span>}
        </div>
      </div>

      {/* Loading indicator */}
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-dark-900/80 z-20">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500" />
        </div>
      )}

      {/* 3D Canvas */}
      <Canvas
        camera={{ position: [5, 5, 5], fov: 50 }}
        className={clsx(
          isFullscreen ? 'fixed inset-0 z-50' : 'h-[400px]'
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
          />

          {/* 3D Annotations for bends */}
          {showAnnotations && hasBendAnnotations && transformData && (
            <BendAnnotations3D
              annotations={bendAnnotations!}
              transform={transformData}
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
              />
            ) : (
              <meshStandardMaterial
                color="#3b82f6"
                side={THREE.DoubleSide}
                metalness={0.2}
                roughness={0.6}
              />
            )}
          </mesh>
        ) : (
          // Point cloud rendering
          <points geometry={scanGeometry}>
            {deviations && deviations.length > 0 ? (
              <pointsMaterial
                size={2}
                sizeAttenuation={false}
                vertexColors={true}
              />
            ) : (
              <pointsMaterial
                size={2}
                sizeAttenuation={false}
                color="#3b82f6"
              />
            )}
          </points>
        )
      )}

      {/* Reference CAD model - render as mesh or points based on type */}
      {showReference && refGeometry && !refError && (
        refIsMesh ? (
          // Mesh rendering with wireframe overlay
          <mesh geometry={refGeometry}>
            <meshStandardMaterial
              color="#a855f7"
              transparent
              opacity={0.4}
              side={THREE.DoubleSide}
              wireframe
            />
          </mesh>
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
}

function BendAnnotations3D({ annotations, transform }: BendAnnotations3DProps) {
  return (
    <>
      {annotations.map((annotation) => {
        if (!annotation.position) return null

        // Transform the position to match the model
        const pos = new THREE.Vector3(
          (annotation.position[0] - transform.center.x) * transform.scale,
          (annotation.position[1] - transform.center.y) * transform.scale,
          (annotation.position[2] - transform.center.z) * transform.scale
        )

        const statusColors = {
          pass: { bg: 'bg-emerald-500/90', border: 'border-emerald-400', text: 'text-emerald-100' },
          fail: { bg: 'bg-red-500/90', border: 'border-red-400', text: 'text-red-100' },
          warning: { bg: 'bg-amber-500/90', border: 'border-amber-400', text: 'text-amber-100' },
          pending: { bg: 'bg-gray-500/90', border: 'border-gray-400', text: 'text-gray-100' },
        }

        const colors = statusColors[annotation.status] || statusColors.pending

        return (
          <group key={annotation.id} position={[pos.x, pos.y, pos.z]}>
            {/* Marker sphere */}
            <mesh>
              <sphereGeometry args={[0.08, 16, 16]} />
              <meshStandardMaterial
                color={annotation.status === 'pass' ? '#22c55e' : annotation.status === 'fail' ? '#ef4444' : '#f59e0b'}
                emissive={annotation.status === 'pass' ? '#22c55e' : annotation.status === 'fail' ? '#ef4444' : '#f59e0b'}
                emissiveIntensity={0.3}
              />
            </mesh>

            {/* Connecting line going up */}
            <line>
              <bufferGeometry>
                <bufferAttribute
                  attach="attributes-position"
                  count={2}
                  array={new Float32Array([0, 0, 0, 0, 0.5, 0])}
                  itemSize={3}
                />
              </bufferGeometry>
              <lineBasicMaterial color="#ffffff" opacity={0.5} transparent />
            </line>

            {/* HTML Label */}
            <Html
              position={[0, 0.6, 0]}
              center
              distanceFactor={8}
              style={{ pointerEvents: 'none' }}
            >
              <div className={clsx(
                'px-2 py-1 rounded-lg border shadow-lg backdrop-blur-sm whitespace-nowrap',
                colors.bg, colors.border
              )}>
                <div className="text-xs font-bold text-white">
                  Bend {annotation.id}
                </div>
                <div className={clsx('text-xs', colors.text)}>
                  {annotation.measuredAngle != null ? (
                    <>
                      <span className="font-mono">{annotation.measuredAngle.toFixed(1)}°</span>
                      {annotation.deviation != null && (
                        <span className="ml-1 opacity-80">
                          ({annotation.deviation > 0 ? '+' : ''}{annotation.deviation.toFixed(1)}°)
                        </span>
                      )}
                    </>
                  ) : (
                    <span className="opacity-60">Expected: {annotation.expectedAngle.toFixed(1)}°</span>
                  )}
                </div>
              </div>
            </Html>
          </group>
        )
      })}
    </>
  )
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

// Export types for use in other components
export type { BendAnnotation, DeviationStats }
