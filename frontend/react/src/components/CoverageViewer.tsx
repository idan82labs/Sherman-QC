import { useMemo, useRef } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Text, Grid } from '@react-three/drei'
import * as THREE from 'three'
import type { GapCluster } from '../types'

interface CoverageViewerProps {
  coverage: number
  gaps: GapCluster[]
  totalPoints: number
  state: string
}

// Gap cluster visualization
function GapMarker({ gap, index }: { gap: GapCluster; index: number }) {
  const meshRef = useRef<THREE.Mesh>(null)

  // Animate pulsing
  useFrame(({ clock }) => {
    if (meshRef.current) {
      const scale = 1 + Math.sin(clock.elapsedTime * 3 + index) * 0.2
      meshRef.current.scale.setScalar(scale)
    }
  })

  // Scale diameter to reasonable size for visualization
  const radius = Math.min(gap.diameter_mm / 20, 3)

  return (
    <group position={[gap.center[0] / 10, gap.center[2] / 10, gap.center[1] / 10]}>
      <mesh ref={meshRef}>
        <sphereGeometry args={[radius, 16, 16]} />
        <meshStandardMaterial
          color="#ef4444"
          transparent
          opacity={0.6}
          emissive="#ef4444"
          emissiveIntensity={0.3}
        />
      </mesh>
      <Text
        position={[0, radius + 1, 0]}
        fontSize={0.8}
        color="#fca5a5"
        anchorX="center"
        anchorY="bottom"
      >
        {gap.location_hint}
      </Text>
    </group>
  )
}

// Coverage ring indicator
function CoverageRing({ coverage }: { coverage: number }) {
  // Coverage angle (0-100% maps to 0-2π)
  const coverageAngle = (coverage / 100) * Math.PI * 2

  const geometry = useMemo(() => {
    const shape = new THREE.Shape()
    const outerRadius = 12
    const innerRadius = 10

    // Draw arc
    shape.absarc(0, 0, outerRadius, 0, coverageAngle, false)
    shape.lineTo(Math.cos(coverageAngle) * innerRadius, Math.sin(coverageAngle) * innerRadius)
    shape.absarc(0, 0, innerRadius, coverageAngle, 0, true)
    shape.lineTo(outerRadius, 0)

    return new THREE.ShapeGeometry(shape, 64)
  }, [coverageAngle])

  // Color based on coverage
  const color = useMemo(() => {
    if (coverage >= 95) return '#22c55e' // green
    if (coverage >= 80) return '#3b82f6' // blue
    if (coverage >= 50) return '#eab308' // yellow
    return '#ef4444' // red
  }, [coverage])

  return (
    <group rotation={[-Math.PI / 2, 0, 0]} position={[0, -5, 0]}>
      {/* Background ring */}
      <mesh>
        <ringGeometry args={[10, 12, 64]} />
        <meshStandardMaterial color="#374151" transparent opacity={0.5} />
      </mesh>
      {/* Coverage arc */}
      <mesh geometry={geometry}>
        <meshStandardMaterial color={color} />
      </mesh>
      {/* Center text */}
      <Text
        position={[0, 0, 0.1]}
        rotation={[Math.PI / 2, 0, 0]}
        fontSize={3}
        color="#ffffff"
        anchorX="center"
        anchorY="middle"
      >
        {coverage.toFixed(0)}%
      </Text>
    </group>
  )
}

// Scan progress particles
function ScanParticles({ totalPoints }: { totalPoints: number }) {
  const pointsRef = useRef<THREE.Points>(null)

  // Generate random points within a bounding box
  const positions = useMemo(() => {
    const count = Math.min(totalPoints / 100, 2000) // Cap at 2000 particles
    const pos = new Float32Array(count * 3)

    for (let i = 0; i < count; i++) {
      pos[i * 3] = (Math.random() - 0.5) * 15
      pos[i * 3 + 1] = (Math.random() - 0.5) * 10
      pos[i * 3 + 2] = (Math.random() - 0.5) * 15
    }

    return pos
  }, [totalPoints])

  useFrame(() => {
    if (pointsRef.current) {
      pointsRef.current.rotation.y += 0.001
    }
  })

  if (totalPoints === 0) return null

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={positions.length / 3}
          array={positions}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial size={0.1} color="#60a5fa" transparent opacity={0.8} />
    </points>
  )
}

// Idle state animation
function IdleIndicator() {
  const meshRef = useRef<THREE.Mesh>(null)

  useFrame(({ clock }) => {
    if (meshRef.current) {
      meshRef.current.rotation.y = clock.elapsedTime * 0.5
      meshRef.current.position.y = Math.sin(clock.elapsedTime) * 0.5
    }
  })

  return (
    <mesh ref={meshRef}>
      <torusGeometry args={[3, 0.3, 16, 32]} />
      <meshStandardMaterial color="#6b7280" wireframe />
    </mesh>
  )
}

// Scene content
function Scene({ coverage, gaps, totalPoints, state }: CoverageViewerProps) {
  return (
    <>
      {/* Lights */}
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      <pointLight position={[-10, -10, -10]} intensity={0.5} />

      {/* Grid */}
      <Grid
        args={[30, 30]}
        position={[0, -5, 0]}
        cellColor="#374151"
        sectionColor="#4b5563"
        fadeDistance={50}
        infiniteGrid
      />

      {/* State-dependent content */}
      {state === 'idle' && <IdleIndicator />}

      {(state === 'scanning' || state === 'identifying' || state === 'analyzing') && (
        <>
          {/* Scan particles */}
          <ScanParticles totalPoints={totalPoints} />

          {/* Coverage ring */}
          <CoverageRing coverage={coverage} />

          {/* Gap markers */}
          {gaps.map((gap, i) => (
            <GapMarker key={i} gap={gap} index={i} />
          ))}
        </>
      )}

      {state === 'complete' && (
        <>
          <ScanParticles totalPoints={totalPoints} />
          <CoverageRing coverage={coverage} />
          <Text
            position={[0, 3, 0]}
            fontSize={2}
            color="#22c55e"
            anchorX="center"
            anchorY="middle"
          >
            COMPLETE
          </Text>
        </>
      )}

      {state === 'error' && (
        <Text
          position={[0, 0, 0]}
          fontSize={2}
          color="#ef4444"
          anchorX="center"
          anchorY="middle"
        >
          ERROR
        </Text>
      )}

      {/* Camera controls */}
      <OrbitControls
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
        minDistance={10}
        maxDistance={50}
      />
    </>
  )
}

export default function CoverageViewer(props: CoverageViewerProps) {
  return (
    <div className="w-full h-full bg-dark-900">
      <Canvas
        camera={{ position: [15, 15, 15], fov: 50 }}
        gl={{ antialias: true }}
      >
        <Scene {...props} />
      </Canvas>
    </div>
  )
}
