/**
 * Viewer toolbar: layer toggles (CAD, finished-part ghost, scan, heatmap,
 * bend labels, problems only) plus the camera actions "Worst first" and
 * "Fit part". Every control is a >= 56 px touch target with aria-pressed.
 * Layer state is owned by the page (persisted per station).
 */
import { AlertTriangle, Box, BoxSelect, Crosshair, Maximize, Palette, ScanLine, Tag, type LucideIcon } from 'lucide-react'

import { t, type UiKey } from '../copy'
import type { LayerState } from '../model/layers'
import type { Lang } from '../model/types'

interface ToggleDef {
  key: keyof LayerState
  label: UiKey
  Icon: LucideIcon
}

const TOGGLES: ToggleDef[] = [
  { key: 'cad', label: 'layerCad', Icon: Box },
  { key: 'ghost', label: 'layerGhost', Icon: BoxSelect },
  { key: 'scan', label: 'layerScan', Icon: ScanLine },
  { key: 'heatmap', label: 'layerHeatmap', Icon: Palette },
  { key: 'labels', label: 'layerLabels', Icon: Tag },
  { key: 'problemsOnly', label: 'layerOutOnly', Icon: AlertTriangle },
]

export interface LayerTogglesProps {
  lang: Lang
  layers: LayerState
  onChange: (next: LayerState) => void
  /** Layers that cannot be used right now (e.g. ghost on a finished part). */
  disabled: Partial<Record<keyof LayerState, boolean>>
  onWorst: () => void
  worstLabel: string
  worstDisabled: boolean
  onFit: () => void
}

export function LayerToggles({ lang, layers, onChange, disabled, onWorst, worstLabel, worstDisabled, onFit }: LayerTogglesProps) {
  return (
    <div className="st-toolbar" role="toolbar" aria-label={t('layers', lang)}>
      <div className="st-toolbar-group" data-viewer-reserved>
        {TOGGLES.map(({ key, label, Icon }) => {
          const off = !!disabled[key]
          return (
            <button
              key={key}
              type="button"
              className="st-tool"
              aria-pressed={layers[key] && !off}
              disabled={off}
              data-layer={key}
              onClick={() => onChange({ ...layers, [key]: !layers[key] })}
            >
              <Icon size={22} strokeWidth={2} aria-hidden />
              <span>{t(label, lang)}</span>
            </button>
          )
        })}
      </div>
      <div className="st-toolbar-group" data-viewer-reserved>
        <button type="button" className="st-tool st-tool-action" onClick={onWorst} disabled={worstDisabled} data-action="worst">
          <Crosshair size={22} strokeWidth={2} aria-hidden />
          <span>{worstLabel}</span>
        </button>
        <button type="button" className="st-tool st-tool-action" onClick={onFit} data-action="fit">
          <Maximize size={22} strokeWidth={2} aria-hidden />
          <span>{t('fitAll', lang)}</span>
        </button>
      </div>
    </div>
  )
}
