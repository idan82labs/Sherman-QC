import { Line } from '@react-three/drei'
import * as THREE from 'three'

import type { BendAnnotation } from './types'
import { clipLineToReference } from './layout'

export interface BendAnnotations3DProps {
  annotations: BendAnnotation[]
  transform: { center: THREE.Vector3; scale: number }
  referencePositions?: Float32Array | null
  focusAnnotationId?: number | string | null
}

export function BendAnnotations3D({ annotations, transform, referencePositions, focusAnnotationId }: BendAnnotations3DProps) {
  const renderAnnotations = annotations

  return (
    <>
      {renderAnnotations.map((annotation) => {
        if (!annotation.position) return null
        const isRolledSection = annotation.featureType === 'ROLLED_SECTION'

        const toViewerPoint = (source: [number, number, number]) => new THREE.Vector3(
          (source[0] - transform.center.x) * transform.scale,
          (source[1] - transform.center.y) * transform.scale,
          (source[2] - transform.center.z) * transform.scale
        )

        const rawCadLineStart = annotation.lineStart ? toViewerPoint(annotation.lineStart) : null
        const rawCadLineEnd = annotation.lineEnd ? toViewerPoint(annotation.lineEnd) : null
        const rawCadMid = rawCadLineStart && rawCadLineEnd
          ? new THREE.Vector3().addVectors(rawCadLineStart, rawCadLineEnd).multiplyScalar(0.5)
          : null
        const clippedCadLine = rawCadLineStart && rawCadLineEnd && !annotation.displayGeometryCanonical
          ? clipLineToReference(rawCadMid ?? toViewerPoint(annotation.calloutAnchor ?? annotation.position), rawCadLineStart, rawCadLineEnd, referencePositions)
          : null
        const cadLineStart = clippedCadLine?.start ?? rawCadLineStart
        const cadLineEnd = clippedCadLine?.end ?? rawCadLineEnd
        const rawDetectedLineStart = annotation.detectedLineStart ? toViewerPoint(annotation.detectedLineStart) : null
        const rawDetectedLineEnd = annotation.detectedLineEnd ? toViewerPoint(annotation.detectedLineEnd) : null
        const rawDetectedMid = rawDetectedLineStart && rawDetectedLineEnd
          ? new THREE.Vector3().addVectors(rawDetectedLineStart, rawDetectedLineEnd).multiplyScalar(0.5)
          : null
        const detectedLineLooksAligned = !isRolledSection && rawDetectedMid && rawCadMid && rawCadLineStart && rawCadLineEnd
          ? rawDetectedMid.distanceTo(rawCadMid) <= Math.max(rawCadLineStart.distanceTo(rawCadLineEnd), 0.12) * 2.5
          : false
        const baseMid = !isRolledSection && rawDetectedMid
          && detectedLineLooksAligned
          ? rawDetectedMid
          : rawCadMid ?? toViewerPoint(annotation.calloutAnchor ?? annotation.position)
        const clippedDetectedLine = rawDetectedLineStart && rawDetectedLineEnd && !annotation.displayGeometryCanonical
          && detectedLineLooksAligned
          ? clipLineToReference(baseMid, rawDetectedLineStart, rawDetectedLineEnd, referencePositions)
          : null
        const detectedLineStart = detectedLineLooksAligned ? (clippedDetectedLine?.start ?? rawDetectedLineStart) : null
        const detectedLineEnd = detectedLineLooksAligned ? (clippedDetectedLine?.end ?? rawDetectedLineEnd) : null
        const pos = !isRolledSection && detectedLineStart && detectedLineEnd
          ? new THREE.Vector3().addVectors(detectedLineStart, detectedLineEnd).multiplyScalar(0.5)
          : cadLineStart && cadLineEnd
            ? new THREE.Vector3().addVectors(cadLineStart, cadLineEnd).multiplyScalar(0.5)
            : baseMid
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
        const isPending = annotation.status === 'pending'
        const hasDetectedGeometry = !!(detectedLineStart && detectedLineEnd)
        const showReferenceLine = true
        const showDetectedLine = hasDetectedGeometry
        const markerRadius = isRolledSection ? (isActive ? 0.07 : 0.055) : 0.08
        const referenceColor = isPending ? '#94a3b8' : isIssue ? '#cbd5e1' : '#22c55e'
        const referenceOpacity = isActive ? 0.98 : isPending ? 0.42 : isIssue ? 0.72 : 0.55

        return (
          <group key={annotation.id} position={[pos.x, pos.y, pos.z]}>
            {cadLineStart && cadLineEnd && showReferenceLine && (
              <>
                <Line
                  points={[
                    [cadLineStart.x - pos.x, cadLineStart.y - pos.y, cadLineStart.z - pos.z],
                    [cadLineEnd.x - pos.x, cadLineEnd.y - pos.y, cadLineEnd.z - pos.z],
                  ]}
                  color={referenceColor}
                  lineWidth={isActive ? 4.2 : isIssue ? 2.8 : 2.0}
                  transparent
                  opacity={referenceOpacity}
                />
                {showDetectedLine && detectedLineStart && detectedLineEnd && (
                  <Line
                    points={[
                      [detectedLineStart.x - pos.x, detectedLineStart.y - pos.y, detectedLineStart.z - pos.z],
                      [detectedLineEnd.x - pos.x, detectedLineEnd.y - pos.y, detectedLineEnd.z - pos.z],
                    ]}
                    color={statusColor}
                    lineWidth={isActive ? lineWidth : isIssue ? 3.0 : 2.4}
                    transparent
                    opacity={isActive ? lineOpacity : isIssue ? 0.92 : 0.72}
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
                    color={
                      annotation.status === 'pass'
                        ? '#86efac'
                        : annotation.status === 'warning'
                          ? '#fde68a'
                          : annotation.status === 'pending'
                            ? '#cbd5e1'
                            : '#fca5a5'
                    }
                    lineWidth={7.5}
                    transparent
                    opacity={0.24}
                  />
                )}
              </>
            )}

            <mesh scale={isActive ? 1.2 : 1}>
              <sphereGeometry args={[markerRadius, 16, 16]} />
              <meshStandardMaterial
                color={statusColor}
                emissive={statusColor}
                emissiveIntensity={isActive ? 0.5 : 0.32}
              />
            </mesh>
          </group>
        )
      })}
    </>
  )
}

export default BendAnnotations3D
