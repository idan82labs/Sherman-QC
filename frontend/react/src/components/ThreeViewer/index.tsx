import { Suspense, useState, useRef, useEffect } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Grid, Environment, Html } from '@react-three/drei'
import * as THREE from 'three'
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js'
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js'
import { Maximize2, Minimize2, RotateCcw, Eye, EyeOff, Palette } from 'lucide-react'
import clsx from 'clsx'

interface ThreeViewerProps {
  modelUrl?: string
  referenceUrl?: string
  deviations?: number[]
  tolerance?: number
  className?: string
}

export default function ThreeViewer({
  modelUrl,
  referenceUrl,
  deviations,
  tolerance = 0.1,
  className,
}: ThreeViewerProps) {
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [showReference, setShowReference] = useState(true)
  const [showHeatmap, setShowHeatmap] = useState(true)
  const [isLoading, setIsLoading] = useState(false)
  const controlsRef = useRef<any>(null)

  const resetCamera = () => {
    if (controlsRef.current) {
      controlsRef.current.reset()
    }
  }

  return (
    <div className={clsx('relative bg-dark-900 rounded-lg overflow-hidden', className)}>
      {/* Controls */}
      <div className="absolute top-4 right-4 z-10 flex gap-2">
        {referenceUrl && (
          <button
            onClick={() => setShowReference(!showReference)}
            className={clsx(
              'p-2 rounded-lg transition-colors',
              showReference ? 'bg-primary-500/20 text-primary-400' : 'bg-dark-700 text-dark-400'
            )}
            title={showReference ? 'Hide reference' : 'Show reference'}
          >
            {showReference ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
          </button>
        )}
        {deviations && (
          <button
            onClick={() => setShowHeatmap(!showHeatmap)}
            className={clsx(
              'p-2 rounded-lg transition-colors',
              showHeatmap ? 'bg-primary-500/20 text-primary-400' : 'bg-dark-700 text-dark-400'
            )}
            title={showHeatmap ? 'Hide heatmap' : 'Show heatmap'}
          >
            <Palette className="w-4 h-4" />
          </button>
        )}
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
      {showHeatmap && deviations && (
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
        </div>
      )}

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

          {modelUrl && (
            <Model
              url={modelUrl}
              deviations={showHeatmap ? deviations : undefined}
              tolerance={tolerance}
              onLoadStart={() => setIsLoading(true)}
              onLoadEnd={() => setIsLoading(false)}
            />
          )}

          {referenceUrl && showReference && (
            <Model
              url={referenceUrl}
              isReference
              onLoadStart={() => setIsLoading(true)}
              onLoadEnd={() => setIsLoading(false)}
            />
          )}

          <Grid
            args={[20, 20]}
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

interface ModelProps {
  url: string
  deviations?: number[]
  tolerance?: number
  isReference?: boolean
  onLoadStart?: () => void
  onLoadEnd?: () => void
}

function Model({
  url,
  deviations,
  tolerance = 0.1,
  isReference = false,
  onLoadStart,
  onLoadEnd,
}: ModelProps) {
  const meshRef = useRef<THREE.Mesh>(null)
  const [geometry, setGeometry] = useState<THREE.BufferGeometry | null>(null)

  useEffect(() => {
    const loadModel = async () => {
      onLoadStart?.()

      try {
        const isSTL = url.toLowerCase().endsWith('.stl')
        const isPLY = url.toLowerCase().endsWith('.ply')

        let loadedGeometry: THREE.BufferGeometry

        if (isSTL) {
          const loader = new STLLoader()
          loadedGeometry = await new Promise((resolve, reject) => {
            loader.load(url, resolve, undefined, reject)
          })
        } else if (isPLY) {
          const loader = new PLYLoader()
          loadedGeometry = await new Promise((resolve, reject) => {
            loader.load(url, resolve, undefined, reject)
          })
        } else {
          throw new Error('Unsupported file format')
        }

        // Center the geometry
        loadedGeometry.computeBoundingBox()
        const center = new THREE.Vector3()
        loadedGeometry.boundingBox?.getCenter(center)
        loadedGeometry.translate(-center.x, -center.y, -center.z)

        // Compute normals if not present
        if (!loadedGeometry.attributes.normal) {
          loadedGeometry.computeVertexNormals()
        }

        // Apply deviation colors if provided
        if (deviations && deviations.length > 0) {
          const colors = new Float32Array(loadedGeometry.attributes.position.count * 3)

          for (let i = 0; i < deviations.length && i < colors.length / 3; i++) {
            const deviation = deviations[i]
            const color = deviationToColor(deviation, tolerance)
            colors[i * 3] = color.r
            colors[i * 3 + 1] = color.g
            colors[i * 3 + 2] = color.b
          }

          loadedGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
        }

        setGeometry(loadedGeometry)
      } catch (error) {
        console.error('Error loading model:', error)
      } finally {
        onLoadEnd?.()
      }
    }

    loadModel()
  }, [url, deviations, tolerance, onLoadStart, onLoadEnd])

  if (!geometry) return null

  return (
    <mesh ref={meshRef} geometry={geometry}>
      {isReference ? (
        <meshStandardMaterial
          color="#3b82f6"
          transparent
          opacity={0.3}
          side={THREE.DoubleSide}
          wireframe
        />
      ) : deviations ? (
        <meshStandardMaterial
          vertexColors
          side={THREE.DoubleSide}
          metalness={0.1}
          roughness={0.8}
        />
      ) : (
        <meshStandardMaterial
          color="#94a3b8"
          side={THREE.DoubleSide}
          metalness={0.2}
          roughness={0.6}
        />
      )}
    </mesh>
  )
}

function deviationToColor(deviation: number, tolerance: number): THREE.Color {
  const normalizedDeviation = Math.max(-1, Math.min(1, deviation / tolerance))

  // Green (0,1,0) -> Yellow (1,1,0) -> Red (1,0,0)
  if (normalizedDeviation <= 0) {
    // Green to Yellow transition for negative deviations
    const t = Math.abs(normalizedDeviation)
    return new THREE.Color(t, 1, 0)
  } else {
    // Yellow to Red transition for positive deviations
    return new THREE.Color(1, 1 - normalizedDeviation, 0)
  }
}
