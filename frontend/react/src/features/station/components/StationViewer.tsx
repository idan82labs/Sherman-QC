/**
 * Station 3D viewer (react-three-fiber).
 *
 * Layers, all in the one declared frame `cad_model_mm` and all inside a
 * single group offset (the part centre); no layer has its own transform:
 * - CAD: the stage-model display mesh (`stage_mesh.ply`), semi-transparent
 *   aluminium with crisp sheet edges; when no stage mesh is published (the
 *   stage was not decoded, e.g. an invalid inspection) the finished part
 *   (`final_mesh.ply`) is drawn instead, opaque when there are no scan points,
 *   so the operator always sees the part (`data-cad-mesh=stage|final|none|off`);
 * - stage ghost: the finished part (`final_mesh.ply`) as a faint outline,
 *   shown only when some bend is NOT_YET_FORMED;
 * - scan points (`viewer_scan.ply`), neutral aluminium, or coloured by the
 *   signed heatmap (`viewer_deviations.f32`) through `heatmapScale`;
 * - bend lines: nominal OML dashed grey, measured OML solid in the row's
 *   decision colour (only when BEND_LINE_POSITION was measured);
 * - bend tags: the studio bend name (`bend_display_labels`) with the verdict
 *   icon (OK / CHECK / FIX), anchored at the middle of the ACTUAL (measured)
 *   bend line, else at the engine's label anchor; NOT_YET_FORMED tags are
 *   greyed. "Problems only" hides the rest;
 * - callout: the selected bend's card (`BendCallout`: expected -> actual,
 *   flange first, verdict, what to do) floats beside the part at the end of a
 *   leader line that rises from the actual bend line. It is placed every
 *   frame before the tags (up and away from the anchor, mirrored in RTL,
 *   clear of the toolbar and legend), and the tags avoid it.
 *   Labels are DOM buttons laid out in screen space every frame
 *   (`model/labelLayout.ts`): no two overlap, none covers a viewer control
 *   (elements marked `data-viewer-reserved`), an offset label gets a leader
 *   line to its anchor, and the selected label is placed first and drawn on
 *   top, so the highlighted pill is always the selected bend's. The container
 *   reports `data-label-overlaps` (labels that found no free spot).
 *
 * Orbit / zoom / pan with mouse and touch (one finger rotates, two fingers
 * pinch-zoom and pan). Zoom stops at 15 % of the part radius, and scan
 * points are capped at MAX_POINT_PX on screen, so a close zoom never turns
 * the cloud into squares that hide the CAD and the labels. `focusBend(uid)` animates the camera to the bend's
 * label anchor; `fitAll()` frames the part. The container exposes
 * `data-camera-target`, `data-scan-colour` and `data-scan-sample` so
 * browser checks can assert what is on screen.
 */
import { forwardRef, memo, Suspense, useCallback, useEffect, useImperativeHandle, useMemo, useRef, useState } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { Line, OrbitControls } from '@react-three/drei'
import * as THREE from 'three'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'

import { t } from '../copy'
import { fillPointColors, rgbToHex } from '../model/heatmapScale'
import { layoutLabels, overlappingPairs, type LabelInput, type Rect } from '../model/labelLayout'
import type { LayerState } from '../model/layers'
import { bendTagAnchor, bendVerdict, type BendViewRow } from '../model/stationView'
import type { Lang, Vec3 } from '../model/types'
import type { ViewerAssets } from '../hooks/useViewerAssets'
import { BendCallout } from './BendCallout'
import { VERDICT_ICON } from './BendBits'

export interface StationViewerHandle {
  focusBend: (uid: string) => void
  fitAll: () => void
}

export interface StationViewerProps {
  lang: Lang
  /** Changes when a different result is shown (re-fits the camera). */
  viewKey: string
  rows: BendViewRow[]
  assets: ViewerAssets
  layers: LayerState
  heatmapBand: number
  selectedUid: string | null
  onSelectBend: (uid: string) => void
  /** Close the callout (clears the selection). */
  onClearSelection?: () => void
}

const FOV = 35
const CAD_COLOR = '#C5CDD6'
const EDGE_COLOR = '#E6EBF0'
const GHOST_COLOR = '#8394A8'
const NOMINAL_LINE = '#94A3B8'
/** On-screen cap of a scan point, CSS px (scaled by the device pixel ratio). */
export const MAX_POINT_PX = 8
/** Closest zoom, as a fraction of the part radius. */
export const MIN_DISTANCE_RADIUS = 0.15

/**
 * PointsMaterial with size attenuation whose on-screen size is clamped
 * (`gl_PointSize <= maxPointPx`) and drawn as discs: world-size points read
 * as a surface at normal zoom, and stay small dots when the camera is close.
 */
function makeClampedPointsMaterial(maxPx: number, size: number): THREE.PointsMaterial {
  const material = new THREE.PointsMaterial({ vertexColors: true, size, sizeAttenuation: true, toneMapped: false })
  material.onBeforeCompile = shader => {
    shader.uniforms.maxPointPx = { value: maxPx }
    shader.vertexShader = shader.vertexShader
      .replace('void main() {', 'uniform float maxPointPx;\nvoid main() {')
      .replace('#include <fog_vertex>', '#include <fog_vertex>\n\tgl_PointSize = min( gl_PointSize, maxPointPx );')
    // Round points: corners are discarded (no effect below ~3 px, where every
    // covered pixel centre lies inside the disc).
    shader.fragmentShader = shader.fragmentShader.replace(
      '#include <clipping_planes_fragment>',
      '#include <clipping_planes_fragment>\n\tif ( length( gl_PointCoord - vec2( 0.5 ) ) > 0.5 ) discard;',
    )
  }
  material.customProgramCacheKey = () => `st-clamped-points-${maxPx}`
  return material
}

function hasWebGL(): boolean {
  try {
    const c = document.createElement('canvas')
    return !!(c.getContext('webgl2') || c.getContext('webgl'))
  } catch {
    return false
  }
}

function prefersReducedMotion(): boolean {
  try {
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches
  } catch {
    return false
  }
}

function srgbToLinear(c: number): number {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)
}

/** Half the triangle area of a closed sheet mesh ~ one skin's area (mm²). */
function sheetArea(geometry: THREE.BufferGeometry | null): number | null {
  if (!geometry) return null
  const pos = geometry.getAttribute('position')
  const index = geometry.getIndex()
  if (!index || index.count < 3) return null
  const a = new THREE.Vector3()
  const b = new THREE.Vector3()
  const c = new THREE.Vector3()
  let total = 0
  for (let i = 0; i + 2 < index.count; i += 3) {
    a.fromBufferAttribute(pos, index.getX(i))
    b.fromBufferAttribute(pos, index.getX(i + 1))
    c.fromBufferAttribute(pos, index.getX(i + 2))
    total += b.sub(a).cross(c.sub(a)).length() / 2
  }
  return total > 0 ? total / 2 : null
}

function polylineLength(points: Vec3[]): number {
  let len = 0
  for (let i = 1; i < points.length; i++) {
    const [a, b] = [points[i - 1], points[i]]
    len += Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2])
  }
  return len
}

/** Bounding box (frame coordinates) of the first available geometry, else of the bend geometry. */
function sceneBox(assets: ViewerAssets, rows: BendViewRow[]): THREE.Box3 {
  for (const g of [assets.stageMesh, assets.scan, assets.finalMesh]) {
    if (g?.boundingBox && !g.boundingBox.isEmpty()) return g.boundingBox.clone()
  }
  const box = new THREE.Box3()
  for (const r of rows) {
    for (const p of r.bend.display.nominal_polyline_mm) box.expandByPoint(new THREE.Vector3(...p))
    box.expandByPoint(new THREE.Vector3(...r.bend.display.label_anchor_mm))
  }
  if (box.isEmpty()) box.set(new THREE.Vector3(-50, -50, -50), new THREE.Vector3(50, 50, 50))
  return box
}

interface Flight {
  fromPos: THREE.Vector3
  toPos: THREE.Vector3
  fromTarget: THREE.Vector3
  toTarget: THREE.Vector3
  start: number
  duration: number
}

interface RigApi {
  /** `lift` (fraction of the half view height) moves the target below the screen centre, leaving room above it for the callout. */
  flyTo: (target: THREE.Vector3, distance: number, keepDirection: boolean, lift?: number) => void
}

/** Camera controller: owns OrbitControls, animates focus, reports the target to the DOM. */
function CameraRig({
  apiRef,
  report,
}: {
  apiRef: React.MutableRefObject<RigApi | null>
  report: (target: THREE.Vector3, position: THREE.Vector3) => void
}) {
  const camera = useThree(s => s.camera) as THREE.PerspectiveCamera
  const controls = useThree(s => s.controls) as unknown as OrbitControlsImpl | null
  const flight = useRef<Flight | null>(null)

  useEffect(() => {
    apiRef.current = {
      flyTo(anchor, distance, keepDirection, lift = 0) {
        const fromTarget = controls ? controls.target.clone() : new THREE.Vector3()
        const dir = keepDirection
          ? camera.position.clone().sub(fromTarget).normalize()
          : new THREE.Vector3(0.85, -1.15, 0.95).normalize()
        if (!Number.isFinite(dir.x) || dir.lengthSq() < 1e-9) dir.set(0.85, -1.15, 0.95).normalize()
        const target = anchor.clone()
        if (lift) {
          // Screen-up in world space: the camera's up with the view direction removed.
          const up = camera.up.clone().sub(dir.clone().multiplyScalar(camera.up.dot(dir)))
          if (up.lengthSq() > 1e-9) target.add(up.normalize().multiplyScalar(lift * distance * Math.tan((camera.fov * Math.PI) / 360)))
        }
        const toPos = target.clone().add(dir.multiplyScalar(distance))
        if (prefersReducedMotion()) {
          camera.position.copy(toPos)
          controls?.target.copy(target)
          controls?.update()
          report(target, camera.position)
          return
        }
        flight.current = {
          fromPos: camera.position.clone(),
          toPos,
          fromTarget,
          toTarget: target.clone(),
          start: performance.now(),
          duration: 520,
        }
      },
    }
  }, [apiRef, camera, controls, report])

  useEffect(() => {
    if (!controls) return
    const onChange = () => report(controls.target, camera.position)
    controls.addEventListener('change', onChange)
    onChange()
    return () => controls.removeEventListener('change', onChange)
  }, [camera, controls, report])

  useFrame(() => {
    const f = flight.current
    if (!f) return
    const k = Math.min(1, (performance.now() - f.start) / f.duration)
    const e = k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2
    camera.position.lerpVectors(f.fromPos, f.toPos, e)
    controls?.target.lerpVectors(f.fromTarget, f.toTarget, e)
    controls?.update()
    if (k >= 1) {
      flight.current = null
      if (controls) report(controls.target, camera.position)
    }
  })
  return null
}

function Edges({ geometry, color, opacity }: { geometry: THREE.BufferGeometry; color: string; opacity: number }) {
  const edges = useMemo(() => (geometry.getIndex() ? new THREE.EdgesGeometry(geometry, 28) : null), [geometry])
  useEffect(() => () => edges?.dispose(), [edges])
  if (!edges) return null
  return (
    <lineSegments geometry={edges} renderOrder={3}>
      <lineBasicMaterial color={color} transparent opacity={opacity} depthWrite={false} />
    </lineSegments>
  )
}

function useDisposeOnChange(geometry: THREE.BufferGeometry | null) {
  useEffect(() => () => geometry?.dispose(), [geometry])
}

function ScanPoints({
  geometry,
  deviations,
  heatmap,
  band,
  surfaceArea,
  onColoured,
}: {
  geometry: THREE.BufferGeometry
  deviations: Float32Array | null
  heatmap: boolean
  band: number
  /** One-sided surface area of the part (mm²), when a mesh is known. */
  surfaceArea: number | null
  onColoured: (mode: 'heatmap' | 'neutral', sample: string) => void
}) {
  const count = geometry.getAttribute('position').count
  const useHeat = heatmap && deviations !== null
  // Colours are computed in sRGB (exactly the legend's values), then stored
  // linear for WebGL. The canvas renders without tone mapping (`flat`), so the
  // displayed colour equals the legend swatch.
  const sample = useMemo(() => {
    const srgb = new Float32Array(count * 3)
    fillPointColors(srgb, count, useHeat ? deviations : null, band)
    let attr = geometry.getAttribute('color') as THREE.BufferAttribute | undefined
    if (!attr || attr.count !== count || attr.itemSize !== 3 || !(attr.array instanceof Float32Array)) {
      attr = new THREE.BufferAttribute(new Float32Array(count * 3), 3)
      geometry.setAttribute('color', attr)
    }
    const linear = attr.array as Float32Array
    for (let i = 0; i < srgb.length; i++) linear[i] = srgbToLinear(srgb[i])
    attr.needsUpdate = true
    return count ? rgbToHex({ r: srgb[0] * 255, g: srgb[1] * 255, b: srgb[2] * 255 }) : ''
  }, [geometry, count, useHeat, deviations, band])
  useEffect(() => {
    onColoured(useHeat ? 'heatmap' : 'neutral', sample)
  }, [useHeat, sample, onColoured])
  // Point size from the mean point spacing (surface area / point count), so
  // the cloud reads as a surface without hiding the mesh-free gaps.
  const size = useMemo(() => {
    const radius = geometry.boundingSphere?.radius ?? 100
    const area = surfaceArea ?? 2 * radius * radius
    const spacing = Math.sqrt(area / Math.max(1, count))
    return Math.min(4, Math.max(0.3, spacing * 1.5))
  }, [geometry, count, surfaceArea])
  const dpr = useThree(st => st.viewport.dpr)
  const material = useMemo(() => makeClampedPointsMaterial(MAX_POINT_PX * dpr, size), [dpr, size])
  useEffect(() => () => material.dispose(), [material])
  return <points geometry={geometry} material={material} renderOrder={2} />
}

/** Scale of the selected label (applied in its inline transform); its layout box is scaled too. */
const SELECTED_SCALE = 1.18

/**
 * The label layer's DOM nodes and per-frame layout state. The layout runs in
 * the render loop and writes positions straight to the DOM (no React render per
 * frame), and only when a value changed.
 */
class LabelLayoutEngine {
  private buttons = new Map<string, HTMLButtonElement>()
  private leaders = new Map<string, SVGLineElement>()
  private dots = new Map<string, SVGCircleElement>()
  private callout: HTMLDivElement | null = null
  private calloutLine: SVGLineElement | null = null
  private calloutDot: SVGCircleElement | null = null
  private calloutKey = ''
  /** The card sits below its anchor (the selected tag then moves above the bend line). */
  private calloutBelow = false
  private anchors = new WeakMap<BendViewRow, Vec3>()
  /** Candidate each label used last frame (stability while orbiting). */
  private previous = new Map<string, number>()
  /** Last values written per label, with the element they went to (a remounted button is rewritten). */
  private written = new Map<string, { el: HTMLButtonElement; key: string }>()
  private reserved: { at: number; rects: Rect[] } = { at: -1, rects: [] }
  private frame = 0
  private v = new THREE.Vector3()

  setButton(id: string, el: HTMLButtonElement | null) {
    if (el) this.buttons.set(id, el)
    else this.buttons.delete(id)
  }

  setLeader(id: string, el: SVGLineElement | null) {
    if (el) this.leaders.set(id, el)
    else this.leaders.delete(id)
  }

  setDot(id: string, el: SVGCircleElement | null) {
    if (el) this.dots.set(id, el)
    else this.dots.delete(id)
  }

  setCallout(el: HTMLDivElement | null) {
    this.callout = el
    this.calloutKey = ''
  }

  setCalloutLine(el: SVGLineElement | null) {
    this.calloutLine = el
    this.calloutKey = ''
  }

  setCalloutDot(el: SVGCircleElement | null) {
    this.calloutDot = el
    this.calloutKey = ''
  }

  /** Frame position of a bend's tag (actual bend line middle, else the label anchor), cached per row. */
  anchorOf(row: BendViewRow): Vec3 {
    let a = this.anchors.get(row)
    if (!a) {
      a = bendTagAnchor(row.bend)
      this.anchors.set(row, a)
    }
    return a
  }

  /**
   * Place the selected bend's callout card: up and to the reading-end side of
   * the anchor first (mirrored in RTL), then the other side, straight up,
   * then below; the first spot inside the viewer and clear of the controls
   * wins. Returns the card's rect (reserved for the tags) or null.
   */
  private placeCallout(ax: number, ay: number, visible: boolean, size: { width: number; height: number }, reserved: Rect[], rtl: boolean): Rect | null {
    const card = this.callout
    const line = this.calloutLine
    const dot = this.calloutDot
    if (!card) return null
    const hide = () => {
      if (this.calloutKey !== 'hidden') {
        card.style.visibility = 'hidden'
        line?.setAttribute('visibility', 'hidden')
        dot?.setAttribute('visibility', 'hidden')
        this.calloutKey = 'hidden'
      }
      return null
    }
    if (!visible) return hide()
    const w = card.offsetWidth
    const h = card.offsetHeight
    const m = 12
    const dx = 80
    const dy = 110
    const side = rtl ? -1 : 1
    const right = { x: ax + dx, y: ay - dy - h }
    const left = { x: ax - dx - w, y: ay - dy - h }
    const cands = [
      { name: rtl ? 'up-left' : 'up-right', ...(side > 0 ? right : left) },
      { name: rtl ? 'up-right' : 'up-left', ...(side > 0 ? left : right) },
      { name: 'up', x: ax - w / 2, y: ay - dy - h },
      { name: rtl ? 'down-left' : 'down-right', x: side > 0 ? ax + dx : ax - dx - w, y: ay + dy },
      { name: rtl ? 'down-right' : 'down-left', x: side > 0 ? ax - dx - w : ax + dx, y: ay + dy },
    ]
    const inter = (a: Rect, b: Rect) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h
    const clampX = (x: number) => Math.min(Math.max(m, x), Math.max(m, size.width - m - w))
    const clampY = (y: number) => Math.min(Math.max(m, y), Math.max(m, size.height - m - h))
    // Cost of a spot, after clamping it into the viewer: how far it had to move, controls it covers,
    // and (worst) covering the bend line it points at.
    let best: { name: string; x: number; y: number; cost: number } | null = null
    for (const c of cands) {
      const x = clampX(c.x)
      const y = clampY(c.y)
      const box = { x, y, w, h }
      const moved = Math.abs(x - c.x) + Math.abs(y - c.y)
      const hits = reserved.filter(r => inter(box, r)).length
      const coversAnchor = inter(box, { x: ax - 36, y: ay - 36, w: 72, h: 72 }) ? 1 : 0
      const cost = moved + hits * 400 + coversAnchor * 2000
      if (!best || cost < best.cost) best = { name: c.name, x, y, cost }
      if (cost === 0) break
    }
    if (!best) return hide()
    this.calloutBelow = best.name.startsWith('down')
    const { x, y } = best
    const lx = Math.min(Math.max(ax, x), x + w)
    const ly = Math.min(Math.max(ay, y), y + h)
    const key = `${x.toFixed(1)},${y.toFixed(1)},${ax.toFixed(1)},${ay.toFixed(1)},${best.name}`
    if (key !== this.calloutKey) {
      this.calloutKey = key
      card.style.visibility = 'visible'
      card.style.transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px)`
      card.dataset.placement = best.name
      if (line) {
        line.setAttribute('x1', ax.toFixed(1))
        line.setAttribute('y1', ay.toFixed(1))
        line.setAttribute('x2', lx.toFixed(1))
        line.setAttribute('y2', ly.toFixed(1))
        line.setAttribute('visibility', 'visible')
      }
      if (dot) {
        dot.setAttribute('cx', ax.toFixed(1))
        dot.setAttribute('cy', ay.toFixed(1))
        dot.setAttribute('visibility', 'visible')
      }
    }
    return { x, y, w, h }
  }

  /** Viewer controls (elements marked `data-viewer-reserved` in the stage area), in viewer coordinates. */
  private reservedRects(container: HTMLElement): Rect[] {
    if (this.reserved.at >= 0 && this.frame - this.reserved.at <= 30) return this.reserved.rects
    const box = container.getBoundingClientRect()
    const rects: Rect[] = []
    // The callout is placed by this engine each frame and reserved separately.
    container.parentElement?.querySelectorAll<HTMLElement>('[data-viewer-reserved]:not([data-callout])').forEach(node => {
      const b = node.getBoundingClientRect()
      if (b.width > 0 && b.height > 0) rects.push({ x: b.left - box.left, y: b.top - box.top, w: b.width, h: b.height })
    })
    this.reserved = { at: this.frame, rects }
    return rects
  }

  update(
    camera: THREE.Camera,
    size: { width: number; height: number },
    rows: BendViewRow[],
    offset: Vec3,
    selectedUid: string | null,
    container: HTMLElement | null,
    rtl: boolean,
    calloutRow: BendViewRow | null,
  ) {
    if (!container) return
    this.frame += 1
    let reserved = this.reservedRects(container)
    let hasCallout = false
    if (calloutRow) {
      const a = this.anchorOf(calloutRow)
      this.v.set(a[0] + offset[0], a[1] + offset[1], a[2] + offset[2]).project(camera)
      const ax = ((this.v.x + 1) / 2) * size.width
      const ay = ((1 - this.v.y) / 2) * size.height
      const visible = this.v.z > -1 && this.v.z < 1 && ax >= 0 && ay >= 0 && ax <= size.width && ay <= size.height
      const rect = this.placeCallout(ax, ay, visible, size, reserved, rtl)
      if (rect) reserved = [...reserved, rect]
      hasCallout = !!rect
    }
    if (this.buttons.size === 0) return
    const inputs: LabelInput[] = []
    for (const r of rows) {
      const btn = this.buttons.get(r.bendUid)
      if (!btn) continue
      const a = this.anchorOf(r)
      this.v.set(a[0] + offset[0], a[1] + offset[1], a[2] + offset[2]).project(camera)
      const ax = ((this.v.x + 1) / 2) * size.width
      const selected = r.bendUid === selectedUid
      // With a callout open, the selected tag steps off the bend line (to the side away from the
      // card) so the leader visibly starts on the line itself.
      const ay = ((1 - this.v.y) / 2) * size.height + (selected && hasCallout ? (this.calloutBelow ? -46 : 46) : 0)
      const k = selected ? SELECTED_SCALE : 1
      const hidden = !(this.v.z > -1 && this.v.z < 1) || ax < 0 || ay < 0 || ax > size.width || ay > size.height
      inputs.push({ id: r.bendUid, ax, ay, w: btn.offsetWidth * k, h: btn.offsetHeight * k, hidden, priority: selected ? 0 : r.attention ? 1 : 2 })
    }
    // More rings while a callout takes part of the view, so crowded tags still find free spots.
    const placements = layoutLabels(inputs, { width: size.width, height: size.height, reserved, previous: this.previous, rings: hasCallout ? 7 : 4 })
    let overlaps = 0
    for (const p of placements) {
      this.previous.set(p.id, p.candidate)
      const btn = this.buttons.get(p.id)
      if (!btn) continue
      const selected = p.id === selectedUid
      // The layout box is the scaled size; the button is placed unscaled at its centre and scaled by CSS.
      const left = p.x + p.w / 2 - btn.offsetWidth / 2
      const top = p.y + p.h / 2 - btn.offsetHeight / 2
      if (p.overlapping && !p.hidden) overlaps += 1
      const key = p.hidden ? 'hidden' : `${left.toFixed(1)},${top.toFixed(1)},${selected ? 1 : 0},${p.candidate},${p.overlapping ? 1 : 0}`
      const last = this.written.get(p.id)
      if (last && last.el === btn && last.key === key) continue
      this.written.set(p.id, { el: btn, key })
      if (p.hidden) {
        btn.style.visibility = 'hidden'
      } else {
        btn.style.visibility = 'visible'
        btn.style.transform = `translate(${left.toFixed(1)}px, ${top.toFixed(1)}px)${selected ? ` scale(${SELECTED_SCALE})` : ''}`
        btn.style.zIndex = selected ? '3' : '1'
        btn.dataset.candidate = String(p.candidate)
        btn.dataset.overlapping = p.overlapping ? 'true' : 'false'
      }
      const line = this.leaders.get(p.id)
      const dot = this.dots.get(p.id)
      if (!line || !dot) continue
      if (p.hidden || !p.leader) {
        line.setAttribute('visibility', 'hidden')
        dot.setAttribute('visibility', 'hidden')
      } else {
        line.setAttribute('x1', p.leader.x1.toFixed(1))
        line.setAttribute('y1', p.leader.y1.toFixed(1))
        line.setAttribute('x2', p.leader.x2.toFixed(1))
        line.setAttribute('y2', p.leader.y2.toFixed(1))
        line.setAttribute('visibility', 'visible')
        dot.setAttribute('cx', p.leader.x1.toFixed(1))
        dot.setAttribute('cy', p.leader.y1.toFixed(1))
        dot.setAttribute('visibility', 'visible')
      }
    }
    const pairs = String(overlappingPairs(placements).length)
    if (container.dataset.labelOverlaps !== String(overlaps) || container.dataset.labelPairs !== pairs) {
      container.dataset.labelOverlaps = String(overlaps)
      container.dataset.labelPairs = pairs
    }
  }
}

/** Tag tone: the bend's verdict colour (OK / CHECK / FIX), grey when not bent or undecided. */
function tagTone(r: BendViewRow): string {
  const v = bendVerdict(r)
  return v === 'FIX' ? 'fail' : v === 'CHECK' ? 'review' : v === 'OK' ? 'pass' : v === 'UNSURE' ? 'unknown' : 'inactive'
}

function TagIcon({ row }: { row: BendViewRow }) {
  const Icon = VERDICT_ICON[bendVerdict(row)]
  return <Icon className="st-label-icon" size={18} strokeWidth={3} aria-hidden />
}

function BendLabelLayer({
  rows,
  selectedUid,
  lang,
  onSelect,
  engine,
}: {
  rows: BendViewRow[]
  selectedUid: string | null
  lang: Lang
  onSelect: (uid: string) => void
  engine: LabelLayoutEngine
}) {
  return (
    <div className="st-labels" data-testid="station-labels">
      <svg className="st-leaders" aria-hidden>
        {rows.map(r => (
          <g key={r.bendUid}>
            <line ref={el => engine.setLeader(r.bendUid, el)} stroke={r.color} strokeWidth={2} visibility="hidden" />
            <circle ref={el => engine.setDot(r.bendUid, el)} r={4} fill={r.color} stroke="#fff" strokeWidth={1.5} visibility="hidden" />
          </g>
        ))}
      </svg>
      {rows.map(r => (
        <button
          key={r.bendUid}
          ref={el => engine.setButton(r.bendUid, el)}
          type="button"
          className={`st-label st-tone-${tagTone(r)}${r.bendUid === selectedUid ? ' is-selected' : ''}${r.state === 'NOT_YET_FORMED' ? ' is-unformed' : ''}`}
          style={{ visibility: 'hidden' }}
          data-bend-uid={r.bendUid}
          data-label={r.label}
          data-verdict={bendVerdict(r)}
          data-selected={r.bendUid === selectedUid ? 'true' : undefined}
          dir="ltr"
          aria-label={`${t('colBend', lang)} ${r.label}, ${r.stateLabel}`}
          onPointerDown={e => e.stopPropagation()}
          onClick={() => onSelect(r.bendUid)}
        >
          <TagIcon row={r} />
          <span className="st-label-name">{r.label}</span>
        </button>
      ))}
    </div>
  )
}

/** Runs the label layout once per rendered frame. */
function LabelLayoutDriver({
  engine,
  rows,
  offset,
  selectedUid,
  getContainer,
  rtl,
  calloutRow,
}: {
  engine: LabelLayoutEngine
  rows: BendViewRow[]
  offset: Vec3
  selectedUid: string | null
  getContainer: () => HTMLElement | null
  rtl: boolean
  calloutRow: BendViewRow | null
}) {
  const camera = useThree(s => s.camera)
  const size = useThree(s => s.size)
  useFrame(() => engine.update(camera, size, rows, offset, selectedUid, getContainer(), rtl, calloutRow))
  return null
}

function BendLines({ row, selected }: { row: BendViewRow; selected: boolean }) {
  const d = row.bend.display
  const measured = d.measured_polyline_mm
  return (
    <group>
      <Line
        points={d.nominal_polyline_mm}
        color={NOMINAL_LINE}
        lineWidth={selected ? 2.5 : 1.5}
        dashed
        dashSize={3}
        gapSize={2}
        transparent
        opacity={0.9}
      />
      {measured && <Line points={measured} color={row.color} lineWidth={selected ? 6 : 4} />}
    </group>
  )
}

export const StationViewer = memo(forwardRef<StationViewerHandle, StationViewerProps>(function StationViewer(props, ref) {
  const { lang, viewKey, rows, assets, layers, heatmapBand, selectedUid, onSelectBend, onClearSelection } = props
  const containerRef = useRef<HTMLDivElement>(null)
  const rigApi = useRef<RigApi | null>(null)
  const [webgl] = useState(hasWebGL)

  useDisposeOnChange(assets.stageMesh)
  useDisposeOnChange(assets.finalMesh)
  useDisposeOnChange(assets.scan)

  const box = useMemo(() => sceneBox(assets, rows), [assets, rows])
  const center = useMemo(() => box.getCenter(new THREE.Vector3()), [box])
  const radius = useMemo(() => Math.max(10, box.getSize(new THREE.Vector3()).length() / 2), [box])
  const offset = useMemo<Vec3>(() => [-center.x, -center.y, -center.z], [center])
  const surfaceArea = useMemo(() => sheetArea(assets.stageMesh ?? assets.finalMesh), [assets.stageMesh, assets.finalMesh])
  // CAD layer: the stage mesh, else the finished part (stage not decoded, e.g. an invalid inspection).
  const cadMesh = assets.stageMesh ?? assets.finalMesh
  const cadSource: 'stage' | 'final' | 'none' = assets.stageMesh ? 'stage' : assets.finalMesh ? 'final' : 'none'
  const scanShown = layers.scan && !!assets.scan
  const [labelEngine] = useState(() => new LabelLayoutEngine())
  const getContainer = useCallback(() => containerRef.current, [])

  const report = useCallback((target: THREE.Vector3, position: THREE.Vector3) => {
    const el = containerRef.current
    if (!el) return
    el.dataset.cameraTarget = [target.x, target.y, target.z].map(v => v.toFixed(2)).join(',')
    el.dataset.cameraDistance = position.distanceTo(target).toFixed(2)
    el.dataset.cameraPosition = [position.x, position.y, position.z].map(v => v.toFixed(1)).join(',')
  }, [])
  const onColoured = useCallback((mode: 'heatmap' | 'neutral', sample: string) => {
    const el = containerRef.current
    if (!el) return
    el.dataset.scanColour = mode
    el.dataset.scanSample = sample
  }, [])

  const fitAll = useCallback(() => {
    const distance = (radius * 1.08) / Math.sin((FOV * Math.PI) / 360)
    rigApi.current?.flyTo(new THREE.Vector3(0, 0, 0), distance, false)
  }, [radius])

  const focusBend = useCallback(
    (uid: string) => {
      const row = rows.find(r => r.bendUid === uid)
      if (!row) return
      // World position = frame position minus the shared group offset (the part centre).
      const target = new THREE.Vector3(...labelEngine.anchorOf(row)).sub(center)
      const span = Math.max(60, polylineLength(row.bend.display.nominal_polyline_mm))
      const distance = (span * 1.15) / (2 * Math.tan((FOV * Math.PI) / 360))
      rigApi.current?.flyTo(target, Math.min(distance, radius * 4), true, 0.4)
    },
    [rows, center, radius, labelEngine],
  )

  useImperativeHandle(ref, () => ({ focusBend, fitAll }), [focusBend, fitAll])

  // Frame the part whenever a different result (or its geometry) arrives.
  const fitKey = `${viewKey}|${box.min.toArray().map(v => v.toFixed(1)).join(',')}|${box.max.toArray().map(v => v.toFixed(1)).join(',')}`
  useEffect(() => {
    const id = window.setTimeout(fitAll, 30)
    return () => window.clearTimeout(id)
  }, [fitKey, fitAll])

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    if (!assets.scan || !layers.scan) {
      el.dataset.scanColour = 'none'
      delete el.dataset.scanSample
    }
  }, [assets.scan, layers.scan])

  const incomplete = rows.some(r => r.state === 'NOT_YET_FORMED')
  const visibleRows = layers.problemsOnly ? rows.filter(r => r.attention || r.bendUid === selectedUid) : rows
  const calloutRow = selectedUid ? (rows.find(r => r.bendUid === selectedUid) ?? null) : null
  const rtl = lang === 'he'
  const initialDistance = (radius * 1.08) / Math.sin((FOV * Math.PI) / 360)

  if (!webgl) {
    return (
      <div className="st-viewer st-viewer-empty" ref={containerRef} data-webgl="unavailable">
        <p>{t('noWebgl', lang)}</p>
      </div>
    )
  }

  return (
    <div
      className="st-viewer"
      ref={containerRef}
      data-webgl="ok"
      data-testid="station-viewer"
      data-selected-bend={selectedUid ?? ''}
      data-cad-mesh={layers.cad ? cadSource : 'off'}
    >
      <Canvas
        flat
        dpr={[1, 2]}
        gl={{ antialias: true, alpha: true, preserveDrawingBuffer: true }}
        camera={{ fov: FOV, near: 0.5, far: 50_000, up: [0, 0, 1], position: [initialDistance * 0.5, -initialDistance * 0.68, initialDistance * 0.56] }}
        onCreated={({ gl }) => gl.setClearColor(0x000000, 0)}
        style={{ touchAction: 'none' }}
      >
        <ambientLight intensity={0.55} />
        <hemisphereLight args={['#E8EEF5', '#1B232D', 0.65]} />
        <directionalLight position={[1, -1.5, 2]} intensity={1.1} />
        <directionalLight position={[-1.5, 1, 0.6]} intensity={0.35} />
        <Suspense fallback={null}>
          <group position={offset}>
            {layers.cad && cadMesh && (
              <group>
                <mesh geometry={cadMesh} renderOrder={1}>
                  <meshStandardMaterial
                    color={CAD_COLOR}
                    metalness={0.35}
                    roughness={0.5}
                    transparent
                    opacity={scanShown ? 0.16 : 0.88}
                    depthWrite={!scanShown}
                    side={THREE.DoubleSide}
                  />
                </mesh>
                <Edges geometry={cadMesh} color={EDGE_COLOR} opacity={0.55} />
              </group>
            )}
            {incomplete && layers.ghost && assets.stageMesh && assets.finalMesh && (
              <group>
                <mesh geometry={assets.finalMesh} renderOrder={0}>
                  <meshBasicMaterial color={GHOST_COLOR} transparent opacity={0.07} depthWrite={false} side={THREE.DoubleSide} />
                </mesh>
                <Edges geometry={assets.finalMesh} color={GHOST_COLOR} opacity={0.55} />
              </group>
            )}
            {layers.scan && assets.scan && (
              <ScanPoints
                geometry={assets.scan}
                deviations={assets.deviations}
                heatmap={layers.heatmap}
                band={heatmapBand}
                surfaceArea={surfaceArea}
                onColoured={onColoured}
              />
            )}
            {layers.labels && visibleRows.map(r => <BendLines key={r.bendUid} row={r} selected={r.bendUid === selectedUid} />)}
          </group>
        </Suspense>
        <LabelLayoutDriver
          engine={labelEngine}
          rows={layers.labels ? visibleRows : []}
          offset={offset}
          selectedUid={selectedUid}
          getContainer={getContainer}
          rtl={rtl}
          calloutRow={calloutRow}
        />
        <OrbitControls
          makeDefault
          enableDamping
          dampingFactor={0.12}
          rotateSpeed={0.8}
          zoomSpeed={0.9}
          minDistance={radius * MIN_DISTANCE_RADIUS}
          maxDistance={radius * 20}
          touches={{ ONE: THREE.TOUCH.ROTATE, TWO: THREE.TOUCH.DOLLY_PAN }}
        />
        <CameraRig apiRef={rigApi} report={report} />
      </Canvas>
      {layers.labels && (
        <BendLabelLayer rows={visibleRows} selectedUid={selectedUid} lang={lang} onSelect={onSelectBend} engine={labelEngine} />
      )}
      {calloutRow && (
        <>
          <svg className="st-leaders st-callout-leader" aria-hidden>
            <line ref={el => labelEngine.setCalloutLine(el)} className={`st-co-leader st-tone-${tagTone(calloutRow)}`} visibility="hidden" />
            <circle ref={el => labelEngine.setCalloutDot(el)} className={`st-co-anchor st-tone-${tagTone(calloutRow)}`} r={7} visibility="hidden" />
          </svg>
          <BendCallout
            key={calloutRow.bendUid}
            ref={el => labelEngine.setCallout(el)}
            row={calloutRow}
            lang={lang}
            onClose={() => onClearSelection?.()}
          />
        </>
      )}
      {assets.loading && <div className="st-viewer-note">{t('loading3d', lang)}</div>}
      {!assets.loading && assets.errors.length > 0 && !assets.stageMesh && !assets.scan && (
        <div className="st-viewer-note is-error">{t('viewerError', lang)}</div>
      )}
    </div>
  )
}))
