export type Point3 = [number, number, number]

export type AssetKind = 'scan' | 'aligned_scan' | 'live_scan' | 'reference_mesh'
export type LabelStatus = 'not_started' | 'in_progress' | 'complete' | 'needs_review'

export interface PlyAsset {
  asset_id: string
  name: string
  source: 'source-assets' | 'data' | 'uploads' | 'live_scans' | 'output'
  relative_path: string
  asset_kind: AssetKind
  size_bytes: number
  modified_at: string
  ply_format: string | null
  vertex_count: number | null
  face_count: number | null
  label_status: LabelStatus
  bend_count: number
  annotator: string | null
  revision: number
}

export interface BendLineLabel {
  bend_id: string
  line_start: Point3
  line_end: Point3
  confidence: 'certain' | 'uncertain'
  notes: string
}

export interface HumanBendLabels {
  schema_version: 1
  kind: 'sherman_qc_human_bend_labels'
  asset_id: string
  source: PlyAsset['source']
  source_relative_path: string
  coordinate_frame: 'source_ply_raw'
  review_status: Exclude<LabelStatus, 'not_started'>
  annotator: string
  revision: number
  no_bends_present: boolean
  bends: BendLineLabel[]
  notes: string
  created_at?: string
  updated_at?: string
}

export interface AssetInventoryResponse {
  assets: PlyAsset[]
  total: number
  inventory_total: number
  labels_root: string
}

export type CameraPreset = 'iso' | 'front' | 'right' | 'top'
export type EndpointEdit = { bendId: string; endpoint: 'start' | 'end' } | null
