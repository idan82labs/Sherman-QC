/**
 * Loads the decimated viewer artifacts of one job (C06 `result/`):
 * `stage_mesh.ply` (the stage model the scan was decoded against),
 * `final_mesh.ply` (finished part, used as the stage ghost, and as the CAD
 * layer when no stage mesh is published),
 * `viewer_scan.ply` (points) and `viewer_deviations.f32` (float32 LE, NaN =
 * no value, index-aligned with the scan points).
 *
 * Each artifact loads independently: a missing one disables its layer with
 * a reason instead of failing the whole view. The heatmap is published only
 * when the deviation count equals the scan point count; otherwise it is
 * reported unavailable (never re-indexed or guessed). While the deviations
 * (or the points they index) are still loading, `heatmapLoading` is true and
 * the legend says so rather than claiming the heatmap is on.
 *
 * The returned object is memoised on its parts, so the memoised viewer does
 * not re-render while nothing changed.
 */
import { useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import * as THREE from 'three'
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js'

import { fetchArtifact } from '../api/stationApi'
import type { ShopfloorInspectionResult } from '../model/types'

export interface ViewerAssetSources {
  jobId: string
  stageMesh: string | null
  finalMesh: string | null
  scan: string | null
  deviations: string | null
}

export interface ViewerAssets {
  stageMesh: THREE.BufferGeometry | null
  finalMesh: THREE.BufferGeometry | null
  scan: THREE.BufferGeometry | null
  deviations: Float32Array | null
  /** Why the heatmap cannot be shown (contract reason or a local integrity reason). */
  heatmapIssue: string | null
  /** The heatmap is published but its data is not loaded yet. */
  heatmapLoading: boolean
  loading: boolean
  errors: string[]
}

function parsePly(data: ArrayBuffer): THREE.BufferGeometry {
  const geometry = new PLYLoader().parse(data)
  const position = geometry.getAttribute('position')
  if (!position || position.count === 0) throw new Error('PLY has no vertices')
  geometry.computeBoundingBox()
  geometry.computeBoundingSphere()
  if (geometry.getIndex() && !geometry.getAttribute('normal')) geometry.computeVertexNormals()
  return geometry
}

function parseDeviations(data: ArrayBuffer): Float32Array {
  if (data.byteLength % 4 !== 0) throw new Error('deviation file length is not a multiple of 4 bytes')
  const view = new DataView(data)
  const out = new Float32Array(data.byteLength / 4)
  for (let i = 0; i < out.length; i++) out[i] = view.getFloat32(i * 4, true)
  return out
}

function useArtifact<T>(stationId: string, url: string | null, parse: (data: ArrayBuffer) => T) {
  return useQuery({
    queryKey: ['station-artifact', stationId, url],
    enabled: !!url,
    staleTime: Infinity,
    gcTime: 10 * 60 * 1000,
    retry: 1,
    refetchOnWindowFocus: false,
    queryFn: async ({ signal }) => parse((await fetchArtifact(stationId, url as string, signal)).data),
  })
}

export function useViewerAssets(
  stationId: string,
  result: ShopfloorInspectionResult | null,
  sources: ViewerAssetSources | null,
  wantGhost: boolean,
): ViewerAssets {
  const stage = useArtifact(stationId, sources?.stageMesh ?? null, parsePly)
  // The finished part is needed as the ghost of an incomplete part, and as the
  // CAD layer itself when no stage mesh is published (e.g. an invalid inspection
  // whose stage was never decoded): the operator must always see the part.
  const needFinal = wantGhost || !sources?.stageMesh
  const final = useArtifact(stationId, needFinal ? (sources?.finalMesh ?? null) : null, parsePly)
  const scan = useArtifact(stationId, sources?.scan ?? null, parsePly)
  const publishHeatmap = !!result && result.viewer_assets.deviations != null
  const devs = useArtifact(stationId, publishHeatmap ? (sources?.deviations ?? null) : null, parseDeviations)

  const named: [string, { isError: boolean; error: unknown }][] = [
    ['stage_mesh', stage],
    ['final_mesh', final],
    ['viewer_scan', scan],
    ['viewer_deviations', devs],
  ]
  // A string, so the memo below changes only when the error set does.
  const errorText = named
    .filter(([, q]) => q.isError)
    .map(([name, q]) => `${name}: ${q.error instanceof Error ? q.error.message : String(q.error)}`)
    .join('\n')
  const unavailableReason = result ? (result.viewer_assets.heatmap_unavailable_reason ?? 'UNAVAILABLE') : null
  const loading = stage.isLoading || scan.isLoading || final.isLoading || devs.isLoading
  return useMemo(() => {
    let heatmapIssue: string | null = null
    let deviations: Float32Array | null = null
    let heatmapLoading = false
    if (!unavailableReason) heatmapIssue = null
    else if (!publishHeatmap) heatmapIssue = unavailableReason
    else if (devs.isError) heatmapIssue = 'DEVIATIONS_NOT_LOADED'
    else if (scan.isError) heatmapIssue = 'SCAN_NOT_LOADED'
    else if (devs.data && scan.data) {
      const count = scan.data.getAttribute('position').count
      if (devs.data.length !== count) heatmapIssue = 'COUNT_MISMATCH'
      else deviations = devs.data
    } else heatmapLoading = true
    return {
      stageMesh: stage.data ?? null,
      finalMesh: final.data ?? null,
      scan: scan.data ?? null,
      deviations,
      heatmapIssue,
      heatmapLoading,
      loading,
      errors: errorText ? errorText.split('\n') : [],
    }
  }, [stage.data, final.data, scan.data, scan.isError, devs.data, devs.isError, publishHeatmap, unavailableReason, loading, errorText])
}
