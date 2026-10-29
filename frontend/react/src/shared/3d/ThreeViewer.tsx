import { Suspense, useState, useRef, useEffect, useMemo, useCallback, type PointerEvent as ReactPointerEvent } from 'react'
import { Canvas, useFrame, useThree, type ThreeEvent } from '@react-three/fiber'
import { OrbitControls, Grid, Html, Line } from '@react-three/drei'
import * as THREE from 'three'
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js'
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js'
import { Maximize2, Minimize2, RotateCcw, Box, Scan, Palette, Tag, TrendingDown, TrendingUp } from 'lucide-react'
import clsx from 'clsx'

import type { BendAnnotation, BendCalloutSafeZones, DeviationStats, ManualBendSegment, ScreenBendCallout } from './bend-annotations/types'
import { BendAnnotations3D } from './bend-annotations/BendAnnotations3D'
import { BendCalloutLayoutBridge } from './bend-annotations/BendCalloutLayoutBridge'
import { ViewerCameraController } from './scene/ViewerCameraController'
import BendCalloutOverlay from './overlays/BendCalloutOverlay'
import { fetchAuthenticatedArtifact } from '../../services/authenticatedArtifact'
import { layoutManualMarkers, manualMarkerIntrinsicWidth, manualMarkerLeaderEnd, type MarkerRect } from './bend-annotations/manualMarkerLayout'

/** A small, repeatable display texture. It carries no claim about CAD material. */
function makeBrushedAluminumTexture(): THREE.CanvasTexture {
  const canvas = document.createElement('canvas')
  canvas.width = 128
  canvas.height = 128
  const context = canvas.getContext('2d')!
  const pixels = context.createImageData(128, 128)
  for (let y = 0; y < 128; y++) {
    // Long, subtle streaks along the sheet; deterministic across reloads.
    const streak = Math.sin(y * 0.53) * 13 + Math.sin(y * 2.47) * 3
    for (let x = 0; x < 128; x++) {
      const noise = ((x * 37 + y * 101) % 17) - 8
      const shade = Math.round(221 + streak + noise * 0.6)
      const index = (y * 128 + x) * 4
      pixels.data[index] = shade
      pixels.data[index + 1] = shade
      pixels.data[index + 2] = shade
      pixels.data[index + 3] = 255
    }
  }
  context.putImageData(pixels, 0, 0)
  const texture = new THREE.CanvasTexture(canvas)
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping
  texture.colorSpace = THREE.SRGBColorSpace
  texture.anisotropy = 4
  return texture
}

export interface ViewerDimensionSegment {
  id: string
  lineStart: [number, number, number]
  lineEnd: [number, number, number]
  label: string
  color?: string
  selected?: boolean
}

export interface ViewerSnapshot {
  dataUrl: string
  width: number
  height: number
  labels: Array<{ x: number; y: number; text: string; color: string }>
}

export interface ViewerMeshData {
  vertices: number[]
  triangles: number[]
}

interface ThreeViewerProps {
  modelUrl?: string
  referenceUrl?: string
  /** Feature-owned inline geometry, under the same centering/scale contract. */
  referenceMesh?: ViewerMeshData
  viewPreset?: 'default' | 'front'
  deviations?: number[]
  tolerance?: number
  className?: string
  // New props for annotations
  bendAnnotations?: BendAnnotation[]
  deviationStats?: DeviationStats
  scanPositions?: Float32Array  // Point positions for finding extremes
  focusPoint?: [number, number, number] | null
  focusAnnotationId?: number | string | null
  labelingMode?: boolean
  manualBendSegments?: ManualBendSegment[]
  draftLabelPoint?: [number, number, number] | null
  onLabelPointPick?: (point: [number, number, number]) => void
  /** Opt-in enlarged controls for operator marking surfaces. */
  touchFriendly?: boolean
  referencePresentation?: 'default' | 'solid' | 'aluminum'
  /** Optional CAD-frame triangles. The feature layer owns surface association. */
  selectedSurface?: { vertices: number[]; triangles: number[]; color: string; opacity?: number } | null
  /** Multiple CAD-frame flanges, for a selected operator machine move. */
  selectedSurfaces?: { vertices: number[]; triangles: number[]; color: string; opacity?: number }[]
  dimensionSegments?: ViewerDimensionSegment[]
  onSnapshotReady?: (capture: (() => ViewerSnapshot) | null) => void
  onManualSegmentSelect?: (id: string) => void
  selectedManualSegmentId?: string | null
  /** Feature-owned operator copy; other inspection views retain English defaults. */
  operatorLabels?: {
    zoomIn: string; zoomOut: string; reset: string; rotateControls: string
    rotate: [string, string, string, string]; noModel: string
    showCad: string; hideCad: string; selectSegment: string; onModel: string
  }
}


/**
 * Accept only annotations whose backend provenance places them in the raw
 * reference-mesh frame. The viewer applies the same center/scale transform as
 * the mesh; it must never infer a transform from annotation bounds.
 */
function useRemappedAnnotations(
  annotations: BendAnnotation[],
  _referencePositions: Float32Array | null | undefined,
  _transform: { center: THREE.Vector3; scale: number },
): BendAnnotation[] {
  return useMemo(
    () => annotations.filter(annotation => annotation.coordinateFrame === 'reference_mesh_raw'),
    [annotations],
  )
}

/**
 * Wrapper that remaps annotations to the reference mesh coordinate space
 * and passes corrected data to all bend annotation sub-components.
 */
function BendAnnotationGroup({
  annotations,
  transform,
  referencePositions,
  focusAnnotationId,
  focusPoint,
  controlsRef,
  safeZones,
  onCalloutsChange,
}: {
  annotations: BendAnnotation[]
  transform: { center: THREE.Vector3; scale: number }
  referencePositions?: Float32Array | null
  focusAnnotationId?: number | string | null
  focusPoint?: [number, number, number] | null
  controlsRef: React.RefObject<any>
  safeZones: BendCalloutSafeZones
  onCalloutsChange: (callouts: ScreenBendCallout[]) => void
}) {
  const remappedAnnotations = useRemappedAnnotations(annotations, referencePositions, transform)

  // Recompute focusPoint from remapped annotations
  const remappedFocusPoint = useMemo<[number, number, number] | null>(() => {
    if (!referencePositions || referencePositions.length < 9) return null
    if (!focusPoint) return null
    const focused = remappedAnnotations.find(a => a.id === focusAnnotationId)
    if (focused) return focused.calloutAnchor ?? focused.position
    return focusPoint
  }, [focusPoint, focusAnnotationId, remappedAnnotations, referencePositions])

  const focusAnnotation = remappedAnnotations.find((a) => a.id === focusAnnotationId) ?? null

  // Don't render annotations until reference mesh positions are loaded —
  // without them, clipLineToReference can't clip lines to the mesh edges
  if (!referencePositions || referencePositions.length < 9) {
    return null
  }

  return (
    <>
      <BendAnnotations3D
        annotations={remappedAnnotations}
        transform={transform}
        referencePositions={referencePositions}
        focusAnnotationId={focusAnnotationId}
      />
      <BendCalloutLayoutBridge
        annotations={remappedAnnotations}
        transform={transform}
        focusAnnotationId={focusAnnotationId}
        safeZones={safeZones}
        onChange={onCalloutsChange}
      />
      {remappedFocusPoint && (
        <ViewerCameraController
          focusPoint={remappedFocusPoint}
          focusAnnotation={focusAnnotation}
          transform={transform}
          controlsRef={controlsRef}
        />
      )}
    </>
  )
}

export default function ThreeViewer({
  modelUrl,
  referenceUrl,
  referenceMesh,
  viewPreset = 'default',
  deviations,
  tolerance = 0.1,
  className,
  bendAnnotations,
  deviationStats,
  scanPositions,
  focusPoint,
  focusAnnotationId,
  labelingMode = false,
  manualBendSegments = [],
  draftLabelPoint = null,
  onLabelPointPick,
  touchFriendly = false,
  referencePresentation = 'default',
  selectedSurface = null,
  selectedSurfaces = [],
  dimensionSegments = [],
  onSnapshotReady,
  onManualSegmentSelect,
  selectedManualSegmentId,
  operatorLabels,
}: ThreeViewerProps) {
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [showScan, setShowScan] = useState(true)
  const [showReference, setShowReference] = useState(true)
  const [showHeatmap, setShowHeatmap] = useState(true)
  const [showAnnotations, setShowAnnotations] = useState(true)
  const [showExtremes, setShowExtremes] = useState(true)
  const [isLoading, setIsLoading] = useState(false)
  const [transformData, setTransformData] = useState<{ center: THREE.Vector3; scale: number; normalizedRadius?: number } | null>(null)
  const [extremePositions, setExtremePositions] = useState<{ minPosition?: [number, number, number]; maxPosition?: [number, number, number] } | null>(null)
  const [referencePositions, setReferencePositions] = useState<Float32Array | null>(null)
  const [screenCallouts, setScreenCallouts] = useState<ScreenBendCallout[]>([])
  const controlsRef = useRef<any>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const touchPointersRef = useRef(new Map<number, { x: number; y: number }>())
  const touchTravelRef = useRef(0)
  const touchMovedRef = useRef(false)
  const suppressMarkerClickRef = useRef(false)
  const markerTrayGestureRef = useRef<{ pointerId: number; tray: HTMLElement; capture: HTMLElement;
    x: number; y: number; travel: number; moved: boolean } | null>(null)

  // OrbitControls receives touches on its canvas, but Drei's HTML bend markers
  // sit above that canvas. Handle both surfaces as one continuous touch area.
  const handleTouchDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!touchFriendly || event.pointerType !== 'touch') return
    const target = event.target as Element
    const tray = target.closest<HTMLElement>('[data-manual-marker-tray="true"]')
    if (tray || markerTrayGestureRef.current) {
      event.stopPropagation()
      if (!markerTrayGestureRef.current && tray) {
        const capture = target.closest<HTMLElement>('[data-bend-marker]') || tray
        markerTrayGestureRef.current = { pointerId: event.pointerId, tray, capture,
          x: event.clientX, y: event.clientY, travel: 0, moved: false }
        touchPointersRef.current.clear()
        suppressMarkerClickRef.current = false
        if (controlsRef.current) controlsRef.current.enabled = false
        try { capture.setPointerCapture(event.pointerId) } catch { /* Pointer may already be released. */ }
      }
      return
    }
    if (target.tagName !== 'CANVAS' && !target.closest('[data-bend-marker]')) return
    if (touchPointersRef.current.size === 0) {
      touchTravelRef.current = 0
      touchMovedRef.current = false
      suppressMarkerClickRef.current = false
      if (controlsRef.current) controlsRef.current.enabled = false
    }
    touchPointersRef.current.set(event.pointerId, { x: event.clientX, y: event.clientY })
    if (touchPointersRef.current.size > 1) touchMovedRef.current = true
  }

  const handleTouchMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    const gesture = markerTrayGestureRef.current
    if (gesture) {
      event.stopPropagation()
      if (event.pointerId === gesture.pointerId) {
        const dx = event.clientX - gesture.x, dy = event.clientY - gesture.y
        gesture.travel += Math.hypot(dx, dy)
        gesture.x = event.clientX; gesture.y = event.clientY
        if (gesture.travel > 6) {
          gesture.moved = true
          suppressMarkerClickRef.current = true
          gesture.tray.scrollTop -= dy
          event.preventDefault()
        }
      }
      return
    }
    const pointers = touchPointersRef.current
    const before = pointers.get(event.pointerId)
    if (!before || !containerRef.current || !controlsRef.current) return
    const previous = new Map(pointers)
    pointers.set(event.pointerId, { x: event.clientX, y: event.clientY })
    const dx = event.clientX - before.x
    const dy = event.clientY - before.y
    touchTravelRef.current += Math.hypot(dx, dy)
    if (touchTravelRef.current > 6) touchMovedRef.current = true
    const controls = controlsRef.current
    const camera = controls.object as THREE.PerspectiveCamera
    const height = Math.max(1, containerRef.current.clientHeight)
    if (pointers.size === 1) {
      const spherical = new THREE.Spherical().setFromVector3(camera.position.clone().sub(controls.target))
      spherical.theta -= 2 * Math.PI * dx / height
      spherical.phi = THREE.MathUtils.clamp(spherical.phi - 2 * Math.PI * dy / height, 0.08, Math.PI - 0.08)
      camera.position.copy(controls.target).add(new THREE.Vector3().setFromSpherical(spherical))
    } else if (pointers.size === 2) {
      const ids = [...pointers.keys()]
      const oldA = previous.get(ids[0])!
      const oldB = previous.get(ids[1])!
      const newA = pointers.get(ids[0])!
      const newB = pointers.get(ids[1])!
      const oldSpan = Math.hypot(oldA.x - oldB.x, oldA.y - oldB.y)
      const newSpan = Math.hypot(newA.x - newB.x, newA.y - newB.y)
      const currentDistance = camera.position.distanceTo(controls.target)
      if (oldSpan > 1 && newSpan > 1) {
        const nextDistance = THREE.MathUtils.clamp(currentDistance * oldSpan / newSpan, controls.minDistance, controls.maxDistance)
        camera.position.sub(controls.target).multiplyScalar(nextDistance / currentDistance).add(controls.target)
      }
      const panX = ((newA.x + newB.x) - (oldA.x + oldB.x)) / 2
      const panY = ((newA.y + newB.y) - (oldA.y + oldB.y)) / 2
      const worldPerPixel = 2 * camera.position.distanceTo(controls.target) * Math.tan(camera.fov * Math.PI / 360) / height
      const pan = new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion).multiplyScalar(-panX * worldPerPixel)
        .add(new THREE.Vector3(0, 1, 0).applyQuaternion(camera.quaternion).multiplyScalar(panY * worldPerPixel))
      camera.position.add(pan)
      controls.target.add(pan)
    }
    controls.update()
  }

  const handleTouchEnd = (event: ReactPointerEvent<HTMLDivElement>) => {
    const gesture = markerTrayGestureRef.current
    if (gesture) {
      event.stopPropagation()
      if (event.pointerId === gesture.pointerId) {
        if (gesture.moved || event.type === 'pointercancel') suppressMarkerClickRef.current = true
        try { gesture.capture.releasePointerCapture(event.pointerId) } catch { /* Already released on cancel. */ }
        markerTrayGestureRef.current = null
        if (controlsRef.current) controlsRef.current.enabled = true
      }
      return
    }
    if (!touchPointersRef.current.delete(event.pointerId)) return
    if (touchMovedRef.current) suppressMarkerClickRef.current = true
    if (touchPointersRef.current.size === 0 && controlsRef.current) controlsRef.current.enabled = true
  }

  const fitTouchCamera = useCallback((force: boolean) => {
    if (!touchFriendly || !transformData?.normalizedRadius || !controlsRef.current || !containerRef.current) return
    const controls = controlsRef.current
    const camera = controls.object as THREE.PerspectiveCamera
    const width = containerRef.current.clientWidth
    const height = containerRef.current.clientHeight
    if (width <= 0 || height <= 0) return
    const verticalHalfFov = (camera.fov * Math.PI / 180) / 2
    const horizontalHalfFov = Math.atan(Math.tan(verticalHalfFov) * width / height)
    const limitingHalfFov = Math.min(verticalHalfFov, horizontalHalfFov)
    const requiredDistance = transformData.normalizedRadius * 1.16 / Math.sin(limitingHalfFov)
    const direction = camera.position.clone().sub(controls.target)
    if (force || direction.length() < requiredDistance) {
      direction.setLength(requiredDistance)
      camera.position.copy(controls.target).add(direction)
      controls.update()
    }
  }, [touchFriendly, transformData])

  useEffect(() => {
    if (!touchFriendly || !transformData?.normalizedRadius) return
    fitTouchCamera(true)
    const element = containerRef.current
    if (!element) return
    const observer = new ResizeObserver(() => fitTouchCamera(false))
    observer.observe(element)
    return () => observer.disconnect()
  }, [touchFriendly, transformData, fitTouchCamera])

  const resetCamera = () => {
    if (controlsRef.current) {
      controlsRef.current.reset()
      if (touchFriendly) fitTouchCamera(true)
    }
  }

  const zoomCamera = (factor: number) => {
    const controls = controlsRef.current
    if (!controls) return
    const camera = controls.object as THREE.PerspectiveCamera
    camera.position.sub(controls.target).multiplyScalar(factor).add(controls.target)
    controls.update()
  }

  const rotateCamera = (axis: 'left' | 'right' | 'up' | 'down') => {
    const controls = controlsRef.current
    if (!controls) return
    const camera = controls.object as THREE.PerspectiveCamera
    const spherical = new THREE.Spherical().setFromVector3(camera.position.clone().sub(controls.target))
    const step = Math.PI / 12
    if (axis === 'left') spherical.theta -= step
    if (axis === 'right') spherical.theta += step
    if (axis === 'up') spherical.phi = Math.max(0.08, spherical.phi - step)
    if (axis === 'down') spherical.phi = Math.min(Math.PI - 0.08, spherical.phi + step)
    camera.position.copy(controls.target).add(new THREE.Vector3().setFromSpherical(spherical))
    controls.update()
  }

  const handleSurfacePick = useCallback((viewerPoint: [number, number, number]) => {
    if (!labelingMode || !transformData || !onLabelPointPick) return
    onLabelPointPick([
      viewerPoint[0] / transformData.scale + transformData.center.x,
      viewerPoint[1] / transformData.scale + transformData.center.y,
      viewerPoint[2] / transformData.scale + transformData.center.z,
    ])
  }, [labelingMode, onLabelPointPick, transformData])

  const hasDeviations = deviations && deviations.length > 0
  const hasBendAnnotations = bendAnnotations && bendAnnotations.length > 0
  const hasExtremes = deviationStats && (deviationStats.minPosition || deviationStats.maxPosition || deviationStats.minIdx !== undefined || deviationStats.maxIdx !== undefined)
  const toggleButtonCount = [
    !!modelUrl,
    !!(referenceUrl || referenceMesh),
    !!hasDeviations,
    !!hasBendAnnotations,
    !!hasExtremes,
  ].filter(Boolean).length
  const calloutSafeZones: BendCalloutSafeZones = {
    leftTop: 18 + (toggleButtonCount * 40),
    rightTop: 68,
    leftBottom: showHeatmap && hasDeviations ? 148 : 0,
    rightBottom: (modelUrl || referenceUrl || referenceMesh) ? 72 : 0,
  }

  return (
    <div ref={containerRef} className={clsx('relative bg-dark-900 rounded-lg overflow-hidden', touchFriendly && 'touch-none', className)}
      onPointerDownCapture={handleTouchDown} onPointerMoveCapture={handleTouchMove}
      onPointerUpCapture={handleTouchEnd} onPointerCancelCapture={handleTouchEnd}
      onClickCapture={event => {
        if (suppressMarkerClickRef.current && (event.target as Element).closest('[data-bend-marker]')) {
          event.preventDefault()
          event.stopPropagation()
          suppressMarkerClickRef.current = false
        }
      }}>
      {/* Toggle Controls */}
      <div className="absolute top-4 left-4 z-10 flex flex-col gap-2">
        {modelUrl && (
          <button
            onClick={() => setShowScan(!showScan)}
            className={clsx(
              'flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors',
              touchFriendly && 'min-h-12 min-w-12 text-sm',
              showScan ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30' : 'bg-dark-700 text-dark-400'
            )}
            title={showScan ? 'Hide scan' : 'Show scan'}
          >
            <Scan className="w-3.5 h-3.5" />
            Scan
          </button>
        )}
        {(referenceUrl || referenceMesh) && (
          <button
            onClick={() => setShowReference(!showReference)}
            className={clsx(
              'flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors',
              showReference ? 'bg-slate-500/20 text-slate-200 border border-slate-400/30' : 'bg-dark-700 text-dark-400'
            )}
            title={showReference ? operatorLabels?.hideCad || 'Hide CAD reference' : operatorLabels?.showCad || 'Show CAD reference'}
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
        {touchFriendly && <>
          <button onClick={() => zoomCamera(0.78)} className="flex h-12 w-12 items-center justify-center rounded-lg bg-dark-700 text-xl text-dark-100 hover:bg-dark-600 focus-visible:outline focus-visible:outline-2 focus-visible:outline-primary-400" aria-label={operatorLabels?.zoomIn || 'Zoom in'}>+</button>
          <button onClick={() => zoomCamera(1.28)} className="flex h-12 w-12 items-center justify-center rounded-lg bg-dark-700 text-xl text-dark-100 hover:bg-dark-600 focus-visible:outline focus-visible:outline-2 focus-visible:outline-primary-400" aria-label={operatorLabels?.zoomOut || 'Zoom out'}>−</button>
        </>}
        <button
          onClick={resetCamera}
          className={clsx('p-2 bg-dark-700 rounded-lg text-dark-400 hover:text-dark-200 transition-colors', touchFriendly && 'min-h-12 min-w-12')}
          title={operatorLabels?.reset || 'Fit and reset camera'}
          aria-label={operatorLabels?.reset || 'Fit and reset camera'}
        >
          <RotateCcw className="w-4 h-4" />
        </button>
        {!touchFriendly && <button
          onClick={() => setIsFullscreen(!isFullscreen)}
          className="p-2 bg-dark-700 rounded-lg text-dark-400 hover:text-dark-200 transition-colors"
          title={isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}
        >
          {isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
        </button>}
      </div>

      {touchFriendly && <div className="absolute bottom-4 left-4 z-10 grid grid-cols-4 gap-2 rounded-lg bg-dark-900/85 p-2" aria-label={operatorLabels?.rotateControls || 'Rotate model controls'}>
        {(['left', 'right', 'up', 'down'] as const).map((axis, index) => <button key={axis} type="button" aria-label={operatorLabels?.rotate[index] || `Rotate ${axis}`} onClick={() => rotateCamera(axis)} className="flex h-12 w-12 items-center justify-center rounded-lg bg-dark-700 text-lg text-dark-100 hover:bg-dark-600 focus-visible:outline focus-visible:outline-2 focus-visible:outline-primary-400">{{ left: '←', right: '→', up: '↑', down: '↓' }[axis]}</button>)}
      </div>}

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
          {(referenceUrl || referenceMesh) && showReference && <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-slate-300"></span> CAD</span>}
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
        camera={{ position: viewPreset === 'front' ? [0, 0, 6] : touchFriendly ? [3.6, 3.6, 3.6] : [5, 5, 5], fov: 50 }}
        className={clsx(
          isFullscreen ? 'fixed inset-0 z-50' : 'h-full min-h-[400px]'
        )}
        gl={{ antialias: true, alpha: true }}
      >
        <Suspense fallback={<LoadingFallback />}>
          <ambientLight intensity={referencePresentation === 'default' ? 0.5 : referencePresentation === 'aluminum' ? 0.42 : 0.64} />
          <directionalLight position={[10, 10, 10]} intensity={referencePresentation === 'default' ? 1 : referencePresentation === 'aluminum' ? 1.15 : 0.9} />
          <directionalLight position={[-10, -10, -10]} intensity={referencePresentation === 'default' ? 0.3 : referencePresentation === 'aluminum' ? 0.28 : 0.44} />
          {referencePresentation !== 'default' && <>
            <directionalLight position={[-3, 11, 2]} intensity={referencePresentation === 'aluminum' ? 0.55 : 0.32} />
            <directionalLight position={[3, 6, -12]} intensity={referencePresentation === 'aluminum' ? 1.0 : 0.38} />
          </>}

          {/* Aligned models - both centered and scaled together */}
          <AlignedModels
            key={`${modelUrl ?? ''}:${referenceUrl ?? ''}`}
            scanUrl={modelUrl}
            referenceUrl={referenceUrl}
            referenceMesh={referenceMesh}
            referencePresentation={referencePresentation}
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
            labelingMode={labelingMode}
            onSurfacePick={handleSurfacePick}
          />

          {showReference && transformData && (selectedSurfaces.length ? selectedSurfaces : selectedSurface ? [selectedSurface] : []).map((surface, index) => (
            <SelectedSurfaceHighlight key={index} surface={surface} transform={transformData} />
          ))}

          {transformData && (manualBendSegments.length > 0 || draftLabelPoint) && (
            <ManualBendLabelOverlay
              segments={manualBendSegments}
              draftPoint={draftLabelPoint}
              transform={transformData}
              touchFriendly={touchFriendly}
              onSelect={labelingMode ? undefined : onManualSegmentSelect}
              selectedId={selectedManualSegmentId}
              selectLabel={operatorLabels ? label => `${operatorLabels.selectSegment} ${label} ${operatorLabels.onModel}` : undefined}
            />
          )}
          {transformData && <>
            <ReferenceDimensionOverlay segments={dimensionSegments} transform={transformData} />
            <ViewerSnapshotBridge transform={transformData} segments={manualBendSegments} dimensions={dimensionSegments} onReady={onSnapshotReady} />
          </>}

          {/* 3D Annotations for bends */}
          {showAnnotations && hasBendAnnotations && transformData && (
            <BendAnnotationGroup
              annotations={bendAnnotations!}
              transform={transformData}
              referencePositions={referencePositions}
              focusAnnotationId={focusAnnotationId}
              focusPoint={focusPoint}
              controlsRef={controlsRef}
              safeZones={calloutSafeZones}
              onCalloutsChange={setScreenCallouts}
            />
          )}

          {!hasBendAnnotations && transformData && focusPoint && (
            <ViewerCameraController
              focusPoint={focusPoint}
              focusAnnotation={null}
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
          <group name="viewer-grid"><Grid
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
          /></group>

          <OrbitControls
            ref={controlsRef}
            enableDamping
            dampingFactor={0.05}
            minDistance={1}
            maxDistance={50}
          />

          {/* Local lighting above is sufficient for inspection. No remote HDR
              dependency may erase the viewer when floor connectivity is absent. */}
        </Suspense>
      </Canvas>

      {/* No model message */}
      {!modelUrl && !referenceUrl && !referenceMesh && (
        <div className="absolute inset-0 flex items-center justify-center text-dark-400">
          <p>{operatorLabels?.noModel || 'No 3D model available'}</p>
        </div>
      )}
    </div>
  )
}

function SelectedSurfaceHighlight({ surface, transform }: {
  surface: { vertices: number[]; triangles: number[]; color: string; opacity?: number }
  transform: { center: THREE.Vector3; scale: number }
}) {
  const geometry = useMemo(() => {
    const { vertices, triangles } = surface
    if (vertices.length < 9 || vertices.length % 3 !== 0 || triangles.length < 3
        || triangles.length % 3 !== 0 || vertices.length > 90_000 || triangles.length > 90_000
        || !vertices.every(Number.isFinite)
        || !triangles.every(index => Number.isInteger(index) && index >= 0 && index < vertices.length / 3)) {
      return null
    }
    const result = new THREE.BufferGeometry()
    result.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3))
    result.setIndex(triangles)
    result.translate(-transform.center.x, -transform.center.y, -transform.center.z)
    result.scale(transform.scale, transform.scale, transform.scale)
    result.computeVertexNormals()
    return result
  }, [surface, transform])
  useEffect(() => () => geometry?.dispose(), [geometry])
  if (!geometry) return null
  return <mesh geometry={geometry} renderOrder={8}>
    <meshBasicMaterial color={surface.color} side={THREE.DoubleSide} transparent opacity={surface.opacity ?? 0.65}
      depthTest depthWrite={false} polygonOffset polygonOffsetFactor={-3} polygonOffsetUnits={-3} />
  </mesh>
}

function ReferenceDimensionOverlay({ segments, transform }: {
  segments: ViewerDimensionSegment[]
  transform: { center: THREE.Vector3; scale: number }
}) {
  const remap = (point: [number, number, number]): [number, number, number] => [
    (point[0] - transform.center.x) * transform.scale,
    (point[1] - transform.center.y) * transform.scale,
    (point[2] - transform.center.z) * transform.scale,
  ]
  return <group>{segments.map(segment => {
    const start = remap(segment.lineStart), end = remap(segment.lineEnd)
    const midpoint: [number, number, number] = start.map((value, index) => (value + end[index]) / 2) as [number, number, number]
    return <group key={segment.id}>
      <Line points={[start, end]} color={segment.color || '#f59e0b'} lineWidth={segment.selected ? 4 : 2} dashed dashSize={0.08} gapSize={0.05} depthTest={false} depthWrite={false} renderOrder={25} />
      {[start, end].map((point, index) => <mesh key={index} position={point} renderOrder={26}><sphereGeometry args={[0.045, 12, 12]} /><meshBasicMaterial color={segment.color || '#f59e0b'} depthTest={false} /></mesh>)}
      <Html position={midpoint} center style={{ pointerEvents: 'none' }}><span data-reference-dimension={segment.id} className="whitespace-nowrap rounded-md border border-amber-300 bg-slate-950/95 px-2 py-1 text-sm font-semibold text-amber-100" dir="ltr">{segment.label}</span></Html>
    </group>
  })}</group>
}

function ViewerSnapshotBridge({ transform, segments, dimensions, onReady }: {
  transform: { center: THREE.Vector3; scale: number }
  segments: ManualBendSegment[]
  dimensions: ViewerDimensionSegment[]
  onReady?: (capture: (() => ViewerSnapshot) | null) => void
}) {
  const { gl, scene, camera } = useThree()
  useEffect(() => {
    if (!onReady) return
    onReady(() => {
      const background = scene.background, grid = scene.getObjectByName('viewer-grid')
      const gridVisible = grid?.visible, pixelRatio = gl.getPixelRatio()
      try {
        if (grid) grid.visible = false
        scene.background = new THREE.Color('#ffffff')
        gl.setPixelRatio(Math.max(pixelRatio, 2))
        gl.render(scene, camera)
        const width = gl.domElement.width, height = gl.domElement.height
        const project = (a: [number, number, number], b: [number, number, number]) => {
          const point = new THREE.Vector3((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2)
            .sub(transform.center).multiplyScalar(transform.scale).project(camera)
          return { x: (point.x + 1) / 2 * width, y: (1 - point.y) / 2 * height, visible: point.z >= -1 && point.z <= 1 }
        }
        const labels = [...segments.map(segment => ({ ...project(segment.lineStart, segment.lineEnd),
          text: [segment.groupLabel, segment.accessibleLabel, segment.label && `#${segment.label}`].filter(Boolean).join(' · '), color: segment.color || '#334155' })),
          ...dimensions.map(segment => ({ ...project(segment.lineStart, segment.lineEnd), text: segment.label, color: segment.color || '#b45309' }))]
          .filter(label => label.visible && label.x >= 0 && label.x <= width && label.y >= 0 && label.y <= height)
          .map(({ x, y, text, color }) => ({ x, y, text, color }))
        return { dataUrl: gl.domElement.toDataURL('image/png'), width, height, labels }
      } finally {
        scene.background = background
        if (grid) grid.visible = gridVisible ?? true
        gl.setPixelRatio(pixelRatio)
        gl.render(scene, camera)
      }
    })
    return () => onReady(null)
  }, [gl, scene, camera, transform, segments, dimensions, onReady])
  return null
}

function LoadingFallback() {
  return (
    <Html center>
      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-500" />
    </Html>
  )
}

// Convert deviation value to color (green -> yellow -> red)
function deviationToColor(deviation: number, tolerance: number): THREE.Color {
  const absDeviation = Math.abs(deviation)
  const normalized = Math.min(absDeviation / tolerance, 1)

  if (normalized < 0.5) {
    const t = normalized * 2
    return new THREE.Color(t, 1, 0)
  }

  const t = (normalized - 0.5) * 2
  return new THREE.Color(1, 1 - t, 0)
}

// Component that manages loading and aligning both models together
interface AlignedModelsProps {
  scanUrl?: string
  referenceUrl?: string
  referenceMesh?: ViewerMeshData
  referencePresentation?: 'default' | 'solid' | 'aluminum'
  showScan: boolean
  showReference: boolean
  deviations?: number[]
  tolerance?: number
  deviationStats?: DeviationStats
  onLoadStart?: () => void
  onLoadEnd?: () => void
  onTransformComputed?: (data: { center: THREE.Vector3; scale: number; normalizedRadius?: number }) => void
  onExtremePositionsComputed?: (positions: { minPosition?: [number, number, number]; maxPosition?: [number, number, number] }) => void
  onReferencePositionsComputed?: (positions: Float32Array | null) => void
  labelingMode?: boolean
  onSurfacePick?: (point: [number, number, number]) => void
}

function AlignedModels({
  scanUrl,
  referenceUrl,
  referenceMesh,
  referencePresentation = 'default',
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
  labelingMode = false,
  onSurfacePick,
}: AlignedModelsProps) {
  const [scanGeometry, setScanGeometry] = useState<THREE.BufferGeometry | null>(null)
  const [refGeometry, setRefGeometry] = useState<THREE.BufferGeometry | null>(null)
  const [scanError, setScanError] = useState<string | null>(null)
  const [refError, setRefError] = useState<string | null>(null)
  const [scanIsMesh, setScanIsMesh] = useState<boolean>(false)
  const [refIsMesh, setRefIsMesh] = useState<boolean>(true)
  const aluminumTexture = useMemo(
    () => referencePresentation === 'aluminum' ? makeBrushedAluminumTexture() : null,
    [referencePresentation],
  )
  useEffect(() => () => aluminumTexture?.dispose(), [aluminumTexture])
  const contourGeometry = useMemo(() => {
    if (referencePresentation === 'default' || !refGeometry || !refIsMesh) return null
    const faceCount = (refGeometry.index?.count ?? refGeometry.getAttribute('position')?.count ?? 0) / 3
    // CAD tessellations can be huge. Keep edge construction bounded.
    if (faceCount > 120_000) return null
    return new THREE.EdgesGeometry(refGeometry, 30)
  }, [refGeometry, refIsMesh, referencePresentation])
  useEffect(() => () => contourGeometry?.dispose(), [contourGeometry])

  // Refs to track current geometries for proper disposal
  const scanGeometryRef = useRef<THREE.BufferGeometry | null>(null)
  const refGeometryRef = useRef<THREE.BufferGeometry | null>(null)
  // Store raw (pre-transform) scan positions for extreme marker extraction
  const rawScanPositionsRef = useRef<Float32Array | null>(null)

  const handleSurfaceClick = useCallback((event: ThreeEvent<MouseEvent>) => {
    if (!labelingMode || !onSurfacePick) return
    // OrbitControls also sees pointer gestures; only a stationary tap/click
    // may create a bend endpoint.
    if (event.delta > 5) return
    event.stopPropagation()
    onSurfacePick([event.point.x, event.point.y, event.point.z])
  }, [labelingMode, onSurfacePick])

  // Detect if geometry should be rendered as mesh or point cloud
  // STL files are ALWAYS meshes (triangulated surfaces)
  // PLY files: check if indexed (mesh) or from aligned-scan endpoint (point cloud)
  const detectGeometryType = (url: string, geometry: THREE.BufferGeometry): boolean => {
    const lowerPath = new URL(url, window.location.origin).pathname.toLowerCase()

    // STL files are always meshes
    if (lowerPath.endsWith('.stl')) {
      return true
    }

    // Reference mesh endpoint always returns meshes
    if (lowerPath.includes('/api/reference-mesh/')) {
      return true
    }

    // Aligned scan endpoint returns point clouds
    if (lowerPath.includes('/api/aligned-scan/')) {
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
    const controller = new AbortController()
    const loadGeometry = async (url: string): Promise<{ geometry: THREE.BufferGeometry; isMesh: boolean }> => {
      const parsedUrl = new URL(url, window.location.origin)
      if (parsedUrl.origin !== window.location.origin || url.startsWith('//') || url.includes('\\')) {
        throw new Error('Inspection geometry must be served by this local application.')
      }
      const isSTL = parsedUrl.pathname.toLowerCase().endsWith('.stl')
      const isPLY = parsedUrl.pathname.toLowerCase().endsWith('.ply')

      let geometry: THREE.BufferGeometry
      // Browser/Three loaders do not use the application's Axios interceptor.
      // Fetch protected local binaries explicitly; parse without another request.
      const localApi = parsedUrl.pathname.startsWith('/api/')
      const binary = localApi ? await (await fetchAuthenticatedArtifact(url, controller.signal)).arrayBuffer() : null

      if (isSTL) {
        const loader = new STLLoader()
        geometry = binary ? loader.parse(binary) : await new Promise((resolve, reject) => {
          loader.load(url, resolve, undefined, reject)
        })
      } else if (isPLY) {
        const loader = new PLYLoader()
        geometry = binary ? loader.parse(binary) : await new Promise((resolve, reject) => {
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
          if (controller.signal.aborted) { result.geometry.dispose(); return }
          geometries.scan = result.geometry
          // Snapshot raw vertex positions before any transforms
          const posAttr = result.geometry.getAttribute('position')
          if (posAttr) {
            rawScanPositionsRef.current = new Float32Array(posAttr.array)
          }
          setScanIsMesh(result.isMesh)
          setScanError(null)
        } catch (e) {
          if (controller.signal.aborted) return
          console.error('Error loading scan:', e)
          setScanError(e instanceof Error ? e.message : 'Failed to load')
        }
      }

      // Load reference
      if (referenceUrl || referenceMesh) {
        try {
          let result: { geometry: THREE.BufferGeometry; isMesh: boolean }
          if (referenceMesh) {
            const { vertices, triangles } = referenceMesh
            if (vertices.length < 9 || vertices.length > 1_800_000 || vertices.length % 3
                || triangles.length < 3 || triangles.length > 600_000 || triangles.length % 3
                || !vertices.every(value => Number.isFinite(value) && Math.abs(value) <= 3.4028234e38) || !triangles.every(index => Number.isInteger(index) && index >= 0 && index < vertices.length / 3)) {
              throw new Error('Invalid inline reference geometry')
            }
            const geometry = new THREE.BufferGeometry()
            geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3))
            geometry.setIndex(triangles)
            result = { geometry, isMesh: true }
          } else result = await loadGeometry(referenceUrl!)
          if (controller.signal.aborted) { result.geometry.dispose(); geometries.scan?.dispose(); return }
          geometries.ref = result.geometry
          setRefIsMesh(result.isMesh)
          setRefError(null)
        } catch (e) {
          if (controller.signal.aborted) { geometries.scan?.dispose(); return }
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

        const size = new THREE.Vector3()
        scaleBox.getSize(size)
        const maxDim = Math.max(size.x, size.y, size.z)
        const targetSize = 4
        const scale = maxDim > 0 ? targetSize / maxDim : 1

        // Apply transforms immediately to fresh geometries (no stale state)
        if (geometries.scan) {
          geometries.scan.translate(-center.x, -center.y, -center.z)
          geometries.scan.scale(scale, scale, scale)
          if (!geometries.scan.attributes.normal) {
            geometries.scan.computeVertexNormals()
          }
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
          geometries.scan.userData.transformed = true
        }
        if (geometries.ref) {
          geometries.ref.translate(-center.x, -center.y, -center.z)
          geometries.ref.scale(scale, scale, scale)
          if (!geometries.ref.attributes.normal) {
            geometries.ref.computeVertexNormals()
          }
          if (!geometries.ref.attributes.uv) {
            const positions = geometries.ref.getAttribute('position')
            const uv = new Float32Array(positions.count * 2)
            for (let i = 0; i < positions.count; i++) {
              uv[i * 2] = positions.getX(i) * 2
              uv[i * 2 + 1] = (positions.getY(i) + positions.getZ(i)) * 2
            }
            geometries.ref.setAttribute('uv', new THREE.BufferAttribute(uv, 2))
          }
          geometries.ref.userData.transformed = true
        }

        onTransformComputed?.({ center, scale, normalizedRadius: size.length() * scale / 2 })
      }

      // Store geometries (already transformed if center was computed)
      if (geometries.scan) {
        scanGeometryRef.current?.dispose()
        scanGeometryRef.current = geometries.scan
        setScanGeometry(geometries.scan)
      }
      if (geometries.ref) {
        refGeometryRef.current?.dispose()
        refGeometryRef.current = geometries.ref
        setRefGeometry(geometries.ref)
      }

      onLoadEnd?.()
    }

    loadBoth()

    // Cleanup function uses refs to ensure current geometries are disposed
    return () => {
      controller.abort()
      scanGeometryRef.current?.dispose()
      scanGeometryRef.current = null
      refGeometryRef.current?.dispose()
      refGeometryRef.current = null
      rawScanPositionsRef.current = null
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scanUrl, referenceUrl, referenceMesh])

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
          <mesh geometry={scanGeometry} onClick={handleSurfaceClick}>
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
          <points geometry={scanGeometry} onClick={handleSurfaceClick}>
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
            <mesh geometry={refGeometry} onClick={handleSurfaceClick}>
              {referencePresentation === 'solid' ? <meshBasicMaterial color="#8397aa" side={THREE.DoubleSide} /> :
                <meshStandardMaterial
                  color={referencePresentation === 'aluminum' ? '#d9e1e8' : '#94a3b8'}
                  map={aluminumTexture}
                  transparent={referencePresentation === 'default'}
                  opacity={referencePresentation === 'default' ? 0.18 : 1}
                  side={THREE.DoubleSide}
                  metalness={referencePresentation === 'aluminum' ? 0.36 : 0.08}
                  roughness={referencePresentation === 'aluminum' ? 0.34 : 0.92}
                />}
            </mesh>
            {referencePresentation === 'default' && <mesh geometry={refGeometry}>
              <meshBasicMaterial
                color="#cbd5e1"
                transparent
                opacity={0.14}
                wireframe
              />
            </mesh>}
            {contourGeometry && <lineSegments geometry={contourGeometry} renderOrder={2}>
              <lineBasicMaterial color={referencePresentation === 'aluminum' ? '#152432' : '#5e7184'} transparent opacity={referencePresentation === 'aluminum' ? 0.94 : 0.46} depthWrite={false} />
            </lineSegments>}
          </group>
        ) : (
          // Point cloud rendering for reference
          <points geometry={refGeometry} onClick={handleSurfaceClick}>
            <pointsMaterial
              size={2}
              sizeAttenuation={false}
              color="#a855f7"
              transparent
              opacity={referencePresentation === 'solid' ? 0.9 : 0.6}
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

function ManualBendLabelOverlay({
  segments,
  draftPoint,
  transform,
  touchFriendly = false,
  onSelect,
  selectedId,
  selectLabel,
}: {
  segments: ManualBendSegment[]
  draftPoint?: [number, number, number] | null
  transform: { center: THREE.Vector3; scale: number }
  touchFriendly?: boolean
  onSelect?: (id: string) => void
  selectedId?: string | null
  selectLabel?: (label: string) => string
}) {
  const { camera, gl, size } = useThree()
  const markerViewportRef = useRef<HTMLDivElement>(null)
  const markerContentRef = useRef<HTMLDivElement>(null)
  const buttonsRef = useRef(new Map<string, HTMLButtonElement | HTMLSpanElement>())
  const leadersRef = useRef(new Map<string, SVGPathElement[]>())
  const dimensionsRef = useRef(new Map<string, { width: number; height: number }>())
  const reservedRef = useRef<MarkerRect[]>([])
  const layoutCacheRef = useRef<{ key: string; value: ReturnType<typeof layoutManualMarkers> } | null>(null)
  const labelled = useMemo(() => segments.filter(segment => segment.label), [segments])
  const toViewerPoint = useCallback((point: [number, number, number]): [number, number, number] => [
    (point[0] - transform.center.x) * transform.scale,
    (point[1] - transform.center.y) * transform.scale,
    (point[2] - transform.center.z) * transform.scale,
  ], [transform])

  const draftViewerPoint = draftPoint ? toViewerPoint(draftPoint) : null

  useEffect(() => {
    const liveIds = new Set(labelled.map(segment => segment.id))
    for (const id of leadersRef.current.keys()) if (!liveIds.has(id)) leadersRef.current.delete(id)
    for (const id of dimensionsRef.current.keys()) if (!liveIds.has(id)) dimensionsRef.current.delete(id)
    const controls = gl.domElement.closest('div.relative.bg-dark-900')
    const controlElements = controls ? [...controls.querySelectorAll<HTMLElement>(':scope > .absolute')]
      .filter(element => element.classList.contains('top-4') || element.classList.contains('bottom-4')) : []
    for (const element of buttonsRef.current.values()) {
      element.style.width = 'max-content'
      element.style.height = ''
      element.style.maxWidth = '180px'
    }
    const textContext = document.createElement('canvas').getContext('2d')
    const measure = () => {
      // Only resize/content changes read DOM geometry; never one read per frame.
      for (const segment of labelled) {
        const element = buttonsRef.current.get(segment.id)
        if (!element) continue
        const rect = element.getBoundingClientRect()
        const buttonStyle = window.getComputedStyle(element)
        const inset = ['paddingLeft', 'paddingRight', 'borderLeftWidth', 'borderRightWidth']
          .reduce((total, key) => total + (Number.parseFloat(buttonStyle[key as keyof CSSStyleDeclaration] as string) || 0), 0)
        const widths = textContext ? [...element.children].slice(0, 3).map(child => {
          const style = window.getComputedStyle(child)
          textContext.font = style.font || `${style.fontWeight} ${style.fontSize} ${style.fontFamily}`
          const text = child.textContent || '', spacing = Number.parseFloat(style.letterSpacing) || 0
          return Math.max(0, textContext.measureText(text).width + Math.max(0, text.length - 1) * spacing)
        }) : []
        const intrinsicWidth = manualMarkerIntrinsicWidth(widths, inset)
        dimensionsRef.current.set(segment.id, { width: Math.max(intrinsicWidth, Math.ceil(rect.width)), height: Math.max(48, Math.ceil(rect.height)) })
      }
      const canvas = gl.domElement.getBoundingClientRect()
      reservedRef.current = controlElements.map(element => {
        const rect = element.getBoundingClientRect()
        return { x: rect.left - canvas.left, y: rect.top - canvas.top, width: rect.width, height: rect.height }
      }).filter(rect => rect.width > 0 && rect.height > 0)
    }
    const observer = new ResizeObserver(measure)
    for (const element of controlElements) observer.observe(element)
    for (const element of buttonsRef.current.values()) observer.observe(element)
    measure()
    return () => observer.disconnect()
  }, [labelled, gl, size.width, size.height])

  useFrame(() => {
    const host = markerViewportRef.current, content = markerContentRef.current
    if (!host || !content || !labelled.length || size.width < 64 || size.height < 64) return
    const projected = labelled.slice(0, 32).map(segment => {
      const start = toViewerPoint(segment.lineStart), end = toViewerPoint(segment.lineEnd)
      const point = new THREE.Vector3((start[0] + end[0]) / 2, (start[1] + end[1]) / 2, (start[2] + end[2]) / 2).project(camera)
      const dimensions = dimensionsRef.current.get(segment.id) || { width: 48, height: segment.groupLabel || segment.label === '?' && segment.accessibleLabel ? 64 : 48 }
      return { id: segment.id, anchorX: Number.isFinite(point.x) ? (point.x + 1) / 2 * size.width : size.width / 2,
        anchorY: Number.isFinite(point.y) ? (1 - point.y) / 2 * size.height : size.height / 2,
        ...dimensions, inFront: Number.isFinite(point.z) && point.z >= -1 && point.z <= 1 }
    })
    const reserved = reservedRef.current.length ? reservedRef.current : [
      { x: 8, y: 8, width: 112, height: 56 }, { x: size.width - 184, y: 8, width: 176, height: 56 },
      ...(touchFriendly ? [{ x: 8, y: size.height - 88, width: 256, height: 80 }] : []),
    ]
    // Existing callers with >32 labels retain every button in a CSS tray;
    // projection/packing work is still strictly limited to 32.
    const forceTray = labelled.length > 32
    const anchors = projected.map(point => ({ ...point, anchorX: Math.round(point.anchorX), anchorY: Math.round(point.anchorY) }))
    const key = JSON.stringify([anchors, size.width, size.height, reserved, forceTray])
    let layout
    try {
      layout = layoutCacheRef.current?.key === key ? layoutCacheRef.current.value : layoutManualMarkers(anchors, size, reserved, forceTray)
      layoutCacheRef.current = { key, value: layout }
    } catch { return }
    const tray = layout.tray
    const isTray = layout.mode === 'tray' || forceTray
    host.dataset.markerLayout = isTray ? 'tray' : 'callouts'
    host.dataset.manualMarkerTray = isTray ? 'true' : 'false'
    Object.assign(host.style, { left: `${tray?.x || 0}px`, top: `${tray?.y || 0}px`, right: 'auto', bottom: 'auto', width: `${tray?.width || size.width}px`,
      height: `${tray?.height || size.height}px`, overflowY: isTray ? 'auto' : 'hidden', overflowX: 'hidden',
      pointerEvents: isTray ? 'auto' : 'none', background: isTray ? 'rgba(2,6,23,.82)' : 'transparent' })
    Object.assign(content.style, { height: forceTray ? 'auto' : `${layout.contentHeight}px`, display: forceTray ? 'grid' : 'block',
      gridTemplateColumns: forceTray ? 'repeat(auto-fill,minmax(80px,1fr))' : '', gap: forceTray ? '8px' : '' })
    if (forceTray) {
      // Reset EVERY rendered label, including old absolute positions for IDs
      // reordered beyond index31 and new labels never projected by this frame.
      // CSS grid lays out the entire original ordered collection without
      // shrinking touch targets or overflowing a cell with max-content width.
      for (const segment of labelled) {
        const button = buttonsRef.current.get(segment.id)
        if (button) Object.assign(button.style, { position: 'relative', left: '', top: '', width: '100%', maxWidth: '100%',
          height: '', minHeight: segment.groupLabel || segment.label === '?' && segment.accessibleLabel ? '64px' : '48px',
          pointerEvents: onSelect ? 'auto' : 'none' })
      }
      for (const paths of leadersRef.current.values()) for (const leader of paths) leader.style.display = 'none'
      return
    }
    for (const placement of layout.placements) {
      const button = buttonsRef.current.get(placement.id)
      if (!button) continue
      Object.assign(button.style, { position: forceTray ? 'relative' : 'absolute', left: forceTray ? '' : `${placement.x}px`, top: forceTray ? '' : `${placement.y}px`,
        width: `${placement.width}px`, height: `${placement.height}px`, maxWidth: '100%', pointerEvents: onSelect ? 'auto' : 'none' })
      const point = projected.find(item => item.id === placement.id)!
      const rect = { ...placement, x: placement.x + (tray?.x || 0), y: placement.y + (tray?.y || 0) - (isTray ? host.scrollTop : 0) }
      const visible = point.inFront && !forceTray && (!isTray || rect.y >= tray!.y && rect.y + rect.height <= tray!.y + tray!.height)
      const end = manualMarkerLeaderEnd(rect, point.anchorX, point.anchorY)
      const moved = Math.hypot(end.x - point.anchorX, end.y - point.anchorY) > 2
      for (const leader of leadersRef.current.get(placement.id) || []) {
        leader.style.display = visible && moved ? '' : 'none'
        leader.setAttribute('d', `M ${point.anchorX} ${point.anchorY} L ${end.x} ${end.y}`)
      }
    }
  })

  return (
    <group>
      {segments.map((segment) => {
        const start = toViewerPoint(segment.lineStart)
        const end = toViewerPoint(segment.lineEnd)
        return (
          <group key={segment.id}>
            <Line
              points={[start, end]}
              color="#020617"
              lineWidth={touchFriendly ? segment.selected ? 18 : 13 : 9}
              depthTest={false}
              transparent={touchFriendly}
              opacity={1}
              depthWrite={!touchFriendly}
              renderOrder={20}
              onClick={onSelect ? (event) => { if (event.delta <= 5) { event.stopPropagation(); onSelect(segment.id) } } : undefined}
            />
            {touchFriendly && segment.selected && <Line points={[start, end]} color="#ffffff" lineWidth={13} depthTest={false} depthWrite={false} transparent opacity={1} renderOrder={21} onClick={onSelect ? (event) => { if (event.delta <= 5) { event.stopPropagation(); onSelect(segment.id) } } : undefined} />}
            <Line
              points={[start, end]}
              color={segment.color || '#facc15'}
              lineWidth={touchFriendly ? segment.selected ? 8 : 7 : 5}
              depthTest={false}
              transparent={touchFriendly}
              opacity={1}
              depthWrite={!touchFriendly}
              renderOrder={22}
              onClick={onSelect ? (event) => { if (event.delta <= 5) { event.stopPropagation(); onSelect(segment.id) } } : undefined}
            />
            <mesh position={start} renderOrder={touchFriendly ? 23 : 0}>
              <sphereGeometry args={[0.07, 16, 16]} />
              <meshBasicMaterial color={segment.color || '#facc15'} depthTest={false} transparent={touchFriendly} opacity={1} depthWrite={!touchFriendly} />
            </mesh>
            <mesh position={end} renderOrder={touchFriendly ? 23 : 0}>
              <sphereGeometry args={[0.07, 16, 16]} />
              <meshBasicMaterial color={segment.color || '#facc15'} depthTest={false} transparent={touchFriendly} opacity={1} depthWrite={!touchFriendly} />
            </mesh>
          </group>
        )
      })}
      <Html calculatePosition={() => [0, 0]} zIndexRange={[9, 0]} style={{ pointerEvents: 'none' }}>
        <div style={{ position: 'relative', width: size.width, height: size.height, pointerEvents: 'none' }}>
          <svg width={size.width} height={size.height} style={{ position: 'absolute', inset: 0, overflow: 'hidden', pointerEvents: 'none' }} aria-hidden="true">
            {labelled.map(segment => <g key={segment.id}>
              {[['#020617', 5], [segment.color || '#facc15', segment.selected ? 3 : 2]].map(([color, width], index) => <path key={index}
                ref={element => { const paths = leadersRef.current.get(segment.id) || []; if (element) { paths[index] = element; leadersRef.current.set(segment.id, paths) } }}
                fill="none" stroke={String(color)} strokeWidth={width} style={{ display: 'none' }} />)}
            </g>)}
          </svg>
          <div ref={markerViewportRef} style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}>
            <div ref={markerContentRef} style={{ position: 'relative', width: '100%', height: '100%' }}>
              {labelled.map(segment => onSelect ? <button key={segment.id}
                ref={element => { if (element) buttonsRef.current.set(segment.id, element); else buttonsRef.current.delete(segment.id) }}
                type="button"
                data-bend-marker
                data-bend-group={segment.groupLabel || undefined}
                onClick={(event) => { event.stopPropagation(); onSelect(segment.id) }}
                onPointerDown={(event) => event.stopPropagation()}
                aria-label={selectLabel ? selectLabel(segment.accessibleLabel || segment.label || '') : `Select bend ${segment.accessibleLabel || segment.label} on model`}
                aria-pressed={selectedId === segment.id || segment.selected === true}
                title={[segment.accessibleLabel, segment.groupLabel].filter(Boolean).join(' · ') || undefined}
                style={{ pointerEvents: 'auto', width: 'max-content', maxWidth: 180, minHeight: segment.groupLabel || segment.label === '?' && segment.accessibleLabel ? 64 : 48,
                  borderColor: segment.selected ? '#ffffff' : segment.color || '#facc15', color: segment.color || '#facc15' }}
                className="flex min-h-12 min-w-12 flex-col items-center justify-center rounded-2xl border-[3px] bg-slate-950/95 px-2 py-1 text-base font-bold shadow-lg hover:bg-slate-800 focus-visible:outline focus-visible:outline-2 focus-visible:outline-sky-400 aria-pressed:ring-2 aria-pressed:ring-white"
              ><span>{segment.label}</span>{segment.groupLabel ? <span className="text-sm font-semibold" style={{ whiteSpace: 'nowrap' }}>{segment.groupLabel}</span>
                : segment.label === '?' && segment.accessibleLabel && <span className="text-xs font-semibold">{segment.accessibleLabel}</span>}</button>
                : <span key={segment.id} ref={element => { if (element) buttonsRef.current.set(segment.id, element); else buttonsRef.current.delete(segment.id) }}
                  style={{ width: 'max-content', minWidth: 48, minHeight: 48, maxWidth: 180 }} className="flex items-center justify-center rounded bg-slate-950/90 px-2 py-1 text-xs font-semibold text-white">{segment.groupLabel ? `${segment.groupLabel} · ` : ''}{segment.label}</span>)}
            </div>
          </div>
        </div>
      </Html>
      {draftViewerPoint && (
        <mesh position={draftViewerPoint}>
          <sphereGeometry args={[0.075, 20, 20]} />
          <meshBasicMaterial color="#f59e0b" depthTest={false} />
        </mesh>
      )}
    </group>
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
