/**
 * Signed, diverging deviation colour scale for the station heatmap.
 *
 * Deviations are signed along the stage model's outward normal
 * (`SIGNED_ALONG_STAGE_MODEL_OUTWARD_NORMAL`): negative = material inside
 * the nominal surface (under), positive = outside (over).
 *
 * The scale is defined by five stops at `[-2b, -b, 0, +b, +2b]` where `b` is
 * `viewer_assets.heatmap_band_mm`. Colours are interpolated linearly in sRGB
 * between neighbouring stops and saturate (clamp) beyond `±2b`. A NaN (no
 * value) is a fixed dark grey that is not on the scale.
 *
 * The legend is drawn from the very same stops (`legendTicks` and
 * `legendGradientCss`), so `heatmapColor(tick)` equals the legend swatch at
 * every tick by construction; `scripts/check-station-view-model.mjs` asserts it.
 */

export interface Rgb {
  r: number
  g: number
  b: number
}

/** Stop colours from -2b to +2b (blue, light blue, neutral, light red, red). */
export const HEATMAP_STOP_HEX = ['#1D4ED8', '#60A5FA', '#E7E9EC', '#F87171', '#B91C1C'] as const
/** Stop positions in units of the band. */
export const HEATMAP_STOP_POSITIONS = [-2, -1, 0, 1, 2] as const
/** Colour of a point with no deviation value (NaN). Not on the scale. */
export const HEATMAP_NAN_HEX = '#4B5563'
/** Scan-point colour when the heatmap layer is off (brushed aluminium). */
export const SCAN_NEUTRAL_HEX = '#9FB0C0'

export function hexToRgb(hex: string): Rgb {
  const m = /^#([0-9a-f]{6})$/i.exec(hex)
  if (!m) throw new Error(`bad hex colour ${hex}`)
  const n = parseInt(m[1], 16)
  return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 }
}

export function rgbToHex(c: Rgb): string {
  const h = (v: number) => Math.round(v).toString(16).padStart(2, '0')
  return `#${h(c.r)}${h(c.g)}${h(c.b)}`.toUpperCase()
}

const STOPS: Rgb[] = HEATMAP_STOP_HEX.map(hexToRgb)
const NAN_RGB = hexToRgb(HEATMAP_NAN_HEX)

/**
 * Colour (0-255 integer channels) for a signed deviation in mm.
 * `band` must be > 0 (contract: `heatmap_band_mm` is strictly positive).
 */
export function heatmapColor(deviation: number, band: number): Rgb {
  if (!(band > 0) || !Number.isFinite(band)) throw new RangeError('heatmap band must be a positive number')
  if (Number.isNaN(deviation)) return { ...NAN_RGB }
  const x = Math.max(-2, Math.min(2, deviation / band))
  // segment index 0..3 between stops at -2,-1,0,1,2
  const seg = Math.min(3, Math.floor(x + 2))
  const t = x + 2 - seg
  const a = STOPS[seg]
  const b = STOPS[seg + 1]
  return {
    r: Math.round(a.r + (b.r - a.r) * t),
    g: Math.round(a.g + (b.g - a.g) * t),
    b: Math.round(a.b + (b.b - a.b) * t),
  }
}

export function heatmapHex(deviation: number, band: number): string {
  return rgbToHex(heatmapColor(deviation, band))
}

/** Legend ticks in mm: `[-2b, -b, 0, +b, +2b]`. */
export function legendTicks(band: number): number[] {
  if (!(band > 0)) throw new RangeError('heatmap band must be a positive number')
  return HEATMAP_STOP_POSITIONS.map(p => (p === 0 ? 0 : p * band))
}

/** Legend swatch colour for each tick (identical to `heatmapHex(tick, band)`). */
export function legendSwatches(band: number): { value: number; hex: string }[] {
  return legendTicks(band).map((value, i) => ({ value, hex: HEATMAP_STOP_HEX[i] }))
}

/**
 * CSS gradient matching the scale exactly between `-2b` and `+2b`
 * (linear sRGB interpolation between evenly spaced stops).
 * `direction` is a CSS gradient direction such as `to right` or `to top`.
 */
export function legendGradientCss(direction: string): string {
  const stops = HEATMAP_STOP_HEX.map((hex, i) => `${hex} ${(i * 100) / (HEATMAP_STOP_HEX.length - 1)}%`)
  return `linear-gradient(${direction}, ${stops.join(', ')})`
}

/**
 * Fill a Float32 RGB buffer (0..1 channels) for `count` points.
 * When `deviations` is null, every point gets the neutral scan colour.
 */
export function fillPointColors(out: Float32Array, count: number, deviations: Float32Array | null, band: number): void {
  if (out.length < count * 3) throw new RangeError('colour buffer too small')
  if (deviations === null) {
    const c = hexToRgb(SCAN_NEUTRAL_HEX)
    for (let i = 0; i < count; i++) {
      out[i * 3] = c.r / 255
      out[i * 3 + 1] = c.g / 255
      out[i * 3 + 2] = c.b / 255
    }
    return
  }
  if (deviations.length !== count) throw new RangeError('deviations must be index-aligned with the scan points')
  for (let i = 0; i < count; i++) {
    const c = heatmapColor(deviations[i], band)
    out[i * 3] = c.r / 255
    out[i * 3 + 1] = c.g / 255
    out[i * 3 + 2] = c.b / 255
  }
}
