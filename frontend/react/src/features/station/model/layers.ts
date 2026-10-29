/**
 * Viewer layer state, persisted per station in localStorage (per-viewer
 * convenience only; every read and write is wrapped so a blocked storage
 * just falls back to the defaults).
 */
export interface LayerState {
  /** Stage model the scan was decoded against (CAD at the scanned stage). */
  cad: boolean
  /** Finished part as a ghost; meaningful only while bends are not yet formed. */
  ghost: boolean
  scan: boolean
  heatmap: boolean
  labels: boolean
  /** Show labels/lines only for bends that need attention. */
  problemsOnly: boolean
}

export const DEFAULT_LAYERS: LayerState = {
  cad: true,
  ghost: true,
  scan: true,
  // The heatmap is engineer tooling: off by default on the bender view (station v2), one tap away.
  heatmap: false,
  labels: true,
  problemsOnly: false,
}

// v2: a new key so stations that saved the old default (heatmap on) start from the bender default.
const key = (stationId: string) => `sherman.station.layers.v2.${stationId}`

export function loadLayers(stationId: string): LayerState {
  try {
    const raw = window.localStorage.getItem(key(stationId))
    if (!raw) return { ...DEFAULT_LAYERS }
    const parsed = JSON.parse(raw) as Partial<Record<keyof LayerState, unknown>>
    const out = { ...DEFAULT_LAYERS }
    for (const k of Object.keys(DEFAULT_LAYERS) as (keyof LayerState)[]) {
      if (typeof parsed[k] === 'boolean') out[k] = parsed[k] as boolean
    }
    return out
  } catch {
    return { ...DEFAULT_LAYERS }
  }
}

export function saveLayers(stationId: string, layers: LayerState): void {
  try {
    window.localStorage.setItem(key(stationId), JSON.stringify(layers))
  } catch {
    // storage blocked: keep the in-memory state only
  }
}
