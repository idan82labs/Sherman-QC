import { useEffect } from 'react'
import { useThree } from '@react-three/fiber'
import * as THREE from 'three'

import type { BendAnnotation } from '../bend-annotations/types'
import { clamp } from '../bend-annotations/layout'

export function ViewerCameraController({
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
    const rolledFallbackDistance = focusAnnotation?.featureType === 'ROLLED_SECTION' ? 5.4 : 2.3
    const desiredDistance = clamp(Math.max(rolledFallbackDistance, lineLength * 3.8), rolledFallbackDistance, 7.4)
    const nextPosition = target.clone().addScaledVector(inspectionDirection, desiredDistance)
    controls.target.copy(target)
    camera.position.copy(nextPosition)
    camera.updateProjectionMatrix()
    controls.update()
  }, [camera, controlsRef, focusAnnotation, focusKey, focusPoint, transform.center.x, transform.center.y, transform.center.z, transform.scale])

  return null
}

export default ViewerCameraController
