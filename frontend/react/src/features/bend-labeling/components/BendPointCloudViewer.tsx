import { Html, Line, OrbitControls, GizmoHelper, GizmoViewport } from '@react-three/drei'
import { Canvas, type ThreeEvent, useThree } from '@react-three/fiber'
import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js'

import type { BendLineLabel, CameraPreset, Point3 } from '../types'

interface ViewerProps {
  assetUrl: string
  faceCount: number | null
  bends: BendLineLabel[]
  draftPoint: Point3 | null
  selectedBendId: string | null
  cameraPreset: CameraPreset
  pointSize: number
  pickingEnabled: boolean
  onPick: (point: Point3) => void
  onSelectBend: (bendId: string) => void
  onLoadingChange: (loading: boolean) => void
  onError: (message: string | null) => void
}

interface Transform {
  center: THREE.Vector3
  scale: number
}

function CameraRig({ preset, resetKey }: { preset: CameraPreset; resetKey: string }) {
  const { camera, invalidate } = useThree()
  const controls = useRef<any>(null)

  useEffect(() => {
    const positions: Record<CameraPreset, [number, number, number]> = {
      iso: [5.2, 4.2, 5.2],
      front: [0, 0, 7],
      right: [7, 0, 0],
      top: [0, 7, 0.001],
    }
    camera.position.set(...positions[preset])
    camera.up.set(0, 1, 0)
    if (preset === 'top') camera.up.set(0, 0, -1)
    camera.lookAt(0, 0, 0)
    controls.current?.target.set(0, 0, 0)
    controls.current?.update()
    invalidate()
  }, [camera, invalidate, preset, resetKey])

  return (
    <OrbitControls
      ref={controls}
      makeDefault
      enableDamping
      dampingFactor={0.08}
      minDistance={0.5}
      maxDistance={40}
      mouseButtons={{
        LEFT: THREE.MOUSE.ROTATE,
        MIDDLE: THREE.MOUSE.DOLLY,
        RIGHT: THREE.MOUSE.PAN,
      }}
    />
  )
}

function ModelScene({
  assetUrl,
  faceCount,
  bends,
  draftPoint,
  selectedBendId,
  pointSize,
  pickingEnabled,
  onPick,
  onSelectBend,
  onLoadingChange,
  onError,
}: Omit<ViewerProps, 'cameraPreset'>) {
  const [geometry, setGeometry] = useState<THREE.BufferGeometry | null>(null)
  const [transform, setTransform] = useState<Transform | null>(null)
  const groupRef = useRef<THREE.Group>(null)

  useEffect(() => {
    let active = true
    onLoadingChange(true)
    onError(null)
    const loader = new PLYLoader()
    const token = localStorage.getItem('token')
    if (token) loader.setRequestHeader({ Authorization: `Bearer ${token}` })
    loader.load(
      assetUrl,
      (loaded) => {
        if (!active) {
          loaded.dispose()
          return
        }
        loaded.computeBoundingBox()
        const box = loaded.boundingBox
        if (!box || box.isEmpty()) {
          loaded.dispose()
          onError('This PLY has no displayable geometry.')
          onLoadingChange(false)
          return
        }
        const center = box.getCenter(new THREE.Vector3())
        const size = box.getSize(new THREE.Vector3())
        const maxDimension = Math.max(size.x, size.y, size.z)
        setTransform({ center, scale: maxDimension > 0 ? 4.8 / maxDimension : 1 })
        setGeometry((current) => {
          current?.dispose()
          return loaded
        })
        onLoadingChange(false)
      },
      undefined,
      (reason) => {
        if (!active) return
        onError(reason instanceof Error ? reason.message : 'Could not load this PLY.')
        onLoadingChange(false)
      },
    )
    return () => {
      active = false
      setGeometry((current) => {
        current?.dispose()
        return null
      })
    }
  }, [assetUrl, onError, onLoadingChange])

  const handlePick = useCallback((event: ThreeEvent<MouseEvent>) => {
    if (!pickingEnabled || !geometry || !groupRef.current || event.delta > 3) return
    event.stopPropagation()
    if (event.index != null && faceCount === 0) {
      const positions = geometry.getAttribute('position')
      if (event.index < positions.count) {
        onPick([
          positions.getX(event.index),
          positions.getY(event.index),
          positions.getZ(event.index),
        ])
        return
      }
    }
    const raw = groupRef.current.worldToLocal(event.point.clone())
    onPick([raw.x, raw.y, raw.z])
  }, [faceCount, geometry, onPick, pickingEnabled])

  const groupTransform = useMemo(() => transform ? {
    position: transform.center.clone().multiplyScalar(-transform.scale),
    scale: transform.scale,
  } : null, [transform])

  if (!geometry || !groupTransform || !transform) return null
  const isMesh = (faceCount ?? 0) > 0
  const markerRadius = 0.055 / transform.scale

  return (
    <group ref={groupRef} position={groupTransform.position} scale={groupTransform.scale}>
      {isMesh ? (
        <mesh geometry={geometry} onPointerUp={handlePick}>
          <meshStandardMaterial color="#8ca3b8" side={THREE.DoubleSide} roughness={0.86} metalness={0.04} />
        </mesh>
      ) : (
        <points geometry={geometry} onPointerUp={handlePick}>
          <pointsMaterial
            color="#9fb4c6"
            size={pointSize}
            sizeAttenuation={false}
            vertexColors={geometry.hasAttribute('color')}
          />
        </points>
      )}

      {bends.map((bend) => {
        const selected = bend.bend_id === selectedBendId
        return (
          <group key={bend.bend_id}>
            <Line
              points={[bend.line_start, bend.line_end]}
              color="#111821"
              lineWidth={selected ? 10 : 8}
              depthTest={false}
              renderOrder={20}
              onClick={(event) => {
                event.stopPropagation()
                onSelectBend(bend.bend_id)
              }}
            />
            <Line
              points={[bend.line_start, bend.line_end]}
              color={selected ? '#ffd166' : '#e6a23c'}
              lineWidth={selected ? 6 : 4}
              depthTest={false}
              renderOrder={21}
              onClick={(event) => {
                event.stopPropagation()
                onSelectBend(bend.bend_id)
              }}
            />
            <mesh position={bend.line_start} renderOrder={22}>
              <sphereGeometry args={[markerRadius, 16, 16]} />
              <meshBasicMaterial color="#ffd166" depthTest={false} />
            </mesh>
            <mesh position={bend.line_end} renderOrder={22}>
              <boxGeometry args={[markerRadius * 1.7, markerRadius * 1.7, markerRadius * 1.7]} />
              <meshBasicMaterial color="#ffd166" depthTest={false} />
            </mesh>
          </group>
        )
      })}
      {draftPoint && (
        <mesh position={draftPoint} renderOrder={23}>
          <sphereGeometry args={[markerRadius * 1.2, 18, 18]} />
          <meshBasicMaterial color="#f97316" depthTest={false} />
        </mesh>
      )}
    </group>
  )
}

function CanvasFallback() {
  return <Html center><span className="text-sm text-slate-300">Preparing 3D viewer</span></Html>
}

export default function BendPointCloudViewer(props: ViewerProps) {
  return (
    <Canvas
      camera={{ position: [5.2, 4.2, 5.2], fov: 48, near: 0.01, far: 200 }}
      gl={{ antialias: true, alpha: false, preserveDrawingBuffer: true }}
      onCreated={({ gl, raycaster }) => {
        gl.setClearColor('#111821')
        raycaster.params.Points.threshold = 0.035
      }}
    >
      <Suspense fallback={<CanvasFallback />}>
        <ambientLight intensity={1.2} />
        <directionalLight position={[6, 8, 5]} intensity={1.5} />
        <directionalLight position={[-5, -3, -6]} intensity={0.45} />
        <ModelScene {...props} />
        <CameraRig preset={props.cameraPreset} resetKey={props.assetUrl} />
        <GizmoHelper alignment="bottom-right" margin={[74, 74]}>
          <GizmoViewport axisColors={['#d97757', '#7faa72', '#668fc2']} labelColor="#e7edf3" />
        </GizmoHelper>
      </Suspense>
    </Canvas>
  )
}
