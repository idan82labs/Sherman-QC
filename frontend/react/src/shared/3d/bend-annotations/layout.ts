import * as THREE from 'three'

export function formatBendMetric(label: string, value: number | null | undefined, digits: number, suffix = '') {
  if (value == null || !Number.isFinite(value)) return `${label} -`
  return `${label} ${value >= 0 ? '+' : ''}${value.toFixed(digits)}${suffix}`
}

export function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value))
}

export function distributeRailCards<T extends { preferredTop: number; height: number; boxY: number }>(
  items: T[],
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

export function percentile(values: number[], q: number): number {
  if (!values.length) return 0
  const sorted = [...values].sort((a, b) => a - b)
  const idx = Math.min(sorted.length - 1, Math.max(0, (sorted.length - 1) * q))
  const lo = Math.floor(idx)
  const hi = Math.ceil(idx)
  if (lo === hi) return sorted[lo]
  const t = idx - lo
  return sorted[lo] * (1 - t) + sorted[hi] * t
}

export function computeModelMaxDim(points?: Float32Array | null): number {
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

export function clipLineToReference(
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
    const half = Math.min(originalLength / 2, maxHalfSpan * 0.5)
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
    const half = Math.min(originalLength / 2, maxHalfSpan * 0.5)
    return {
      start: midpoint.clone().addScaledVector(dir, -half),
      end: midpoint.clone().addScaledVector(dir, half),
    }
  }

  const lo = percentile(samples, 0.02)
  const hi = percentile(samples, 0.98)
  const pad = Math.max(0.02, modelMaxDim * 0.01)
  const half = Math.min(maxHalfSpan * 0.5, Math.max(0.12, ((hi - lo) / 2) + pad))
  return {
    start: midpoint.clone().addScaledVector(dir, -half),
    end: midpoint.clone().addScaledVector(dir, half),
  }
}
