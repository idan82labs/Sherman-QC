import { useEffect, useMemo, useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'

import type { BendAnnotation, BendCalloutSafeZones, ScreenBendCallout } from './types'
import { clamp, distributeRailCards, formatBendMetric } from './layout'

function calloutsEqual(previous: ScreenBendCallout[], next: ScreenBendCallout[]) {
  if (previous.length !== next.length) return false
  for (let index = 0; index < previous.length; index += 1) {
    const a = previous[index]
    const b = next[index]
    if (
      a.id !== b.id
      || a.label !== b.label
      || a.status !== b.status
      || a.active !== b.active
      || a.side !== b.side
      || a.anchorX !== b.anchorX
      || a.anchorY !== b.anchorY
      || a.boxX !== b.boxX
      || a.boxY !== b.boxY
      || a.width !== b.width
      || a.height !== b.height
      || a.metricRows.length !== b.metricRows.length
    ) {
      return false
    }
    for (let rowIndex = 0; rowIndex < a.metricRows.length; rowIndex += 1) {
      if (a.metricRows[rowIndex] !== b.metricRows[rowIndex]) return false
    }
  }
  return true
}

export function BendCalloutLayoutBridge({
  annotations,
  transform,
  focusAnnotationId,
  safeZones,
  onChange,
}: {
  annotations: BendAnnotation[]
  transform: { center: THREE.Vector3; scale: number }
  focusAnnotationId?: number | string | null
  safeZones?: BendCalloutSafeZones
  onChange: (callouts: ScreenBendCallout[]) => void
}) {
  const { camera, size } = useThree()
  const lastCalloutsRef = useRef<ScreenBendCallout[]>([])
  const candidateAnnotations = useMemo(
    () => annotations.filter((annotation) => annotation.lineStart && annotation.lineEnd),
    [annotations],
  )

  useEffect(() => () => onChange([]), [onChange])

  useFrame(() => {
    const renderAnnotations = candidateAnnotations

    if (!renderAnnotations.length) {
      if (lastCalloutsRef.current.length) {
        lastCalloutsRef.current = []
        onChange([])
      }
      return
    }

    const toViewerPoint = (source: [number, number, number]) => new THREE.Vector3(
      (source[0] - transform.center.x) * transform.scale,
      (source[1] - transform.center.y) * transform.scale,
      (source[2] - transform.center.z) * transform.scale,
    )

    const projected = renderAnnotations
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
    const marginTop = 24
    const marginBottom = 28
    const cardGap = 10
    const leftTopInset = safeZones?.leftTop ?? 0
    const rightTopInset = safeZones?.rightTop ?? 0
    const leftBottomInset = safeZones?.leftBottom ?? 0
    const rightBottomInset = safeZones?.rightBottom ?? 0
    const rails = {
      left: projected.filter((item) => item.side === 'left').sort((a, b) => a.anchorY - b.anchorY),
      right: projected.filter((item) => item.side === 'right').sort((a, b) => a.anchorY - b.anchorY),
    }

    const layoutSide = (
      items: Array<Omit<ScreenBendCallout, 'boxX' | 'boxY'>>,
      side: 'left' | 'right',
    ) => {
      const sideTopInset = side === 'left' ? leftTopInset : rightTopInset
      const sideBottomInset = side === 'left' ? leftBottomInset : rightBottomInset
      const minTop = marginTop + sideTopInset
      const maxBottom = size.height - (marginBottom + sideBottomInset)
      const preferred = items.map((item) => ({
        ...item,
        preferredTop: clamp(item.anchorY - item.height / 2, minTop, maxBottom - item.height),
        boxX: item.side === 'left' ? marginLeft : size.width - item.width - marginRight,
        boxY: 0,
      }))
      return distributeRailCards(preferred, minTop, maxBottom, cardGap)
        .map(({ preferredTop, ...item }) => item)
    }

    const callouts = [...layoutSide(rails.left, 'left'), ...layoutSide(rails.right, 'right')]
      .sort((a, b) => (a.active === b.active ? 0 : a.active ? -1 : 1))
      .map((item) => ({
        ...item,
        anchorX: Math.round(item.anchorX),
        anchorY: Math.round(item.anchorY),
        boxX: Math.round(item.boxX),
        boxY: Math.round(item.boxY),
      }))

    if (!calloutsEqual(lastCalloutsRef.current, callouts)) {
      lastCalloutsRef.current = callouts
      onChange(callouts)
    }
  })

  return null
}

export default BendCalloutLayoutBridge
