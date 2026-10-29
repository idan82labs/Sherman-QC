import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { ArrowLeft, Download, FileUp, Maximize2, Minimize2, Minus, Plus, RotateCcw, Ruler, Save, Trash2 } from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import ThreeViewer, { type ViewerSnapshot } from '../../../shared/3d/ThreeViewer'
import { getErrorMessage, partCatalogApi } from '../../../services/api'
import type { Point3, VendorBendMark, VendorGeneralMeasurement, VendorSetupDraft, VendorSetupResponse } from '../../../types'
import { exportVendorSetupPdf } from './vendorSetupPdf'
import { captureUnfoldPreview, validatedUnfoldPreview } from './vendorUnfoldPreview'
import { canMapUnfoldDraft } from './vendorUnfoldDraft'
import { localizeVendorError, persistVendorLanguage, readVendorLanguage, vendorCopy, type VendorLanguage } from './vendorSetupCopy'
// Production onboarding (C09): uploaded bound flat as the unfolded source, and the wizard context
// (embedded tagging studio with its guide, or the full page opened from the wizard with a return link).
import { boundFlatCopy, useBoundFlatUnfold, useChiralityOverlay, WizardStandaloneBar, WizardStudioBar, type StudioProgress, type VendorWizardEmbed } from '../../part-onboarding/vendorBridge'
import './VendorSetupPage.css'

const demoCandidates: VendorBendMark[] = [{
  feature_id: 'demo-cad-1', label: 'B1', source: 'cad_candidate', status: 'active',
  issue: null, note: '', line_start: [-2, -1, 0], line_end: [2, -1, 0], target_angle: 90, target_radius: null,
}, {
  feature_id: 'demo-cad-2', label: 'B2', source: 'cad_candidate', status: 'active',
  issue: null, note: '', line_start: [-2, -1, 1], line_end: [2, -1, 1], target_angle: 90, target_radius: null,
}, {
  feature_id: 'demo-cad-3', label: 'B3', source: 'cad_candidate', status: 'active',
  issue: null, note: '', line_start: [-2, 0, 1], line_end: [2, 0, 1], target_angle: 90, target_radius: null,
}]

const BEND_COLORS = ['#22d3ee', '#fb7185', '#a3e635', '#fbbf24', '#c084fc', '#60a5fa', '#f472b6', '#34d399', '#fb923c', '#e879f9', '#facc15', '#2dd4bf']

function bendColor(index: number): string {
  return BEND_COLORS[index] || `hsl(${Math.round((index * 137.508) % 360)} 95% 68%)`
}

function machineMovesFor(sequence: string[], moves?: string[][]): string[][] {
  let cursor = 0
  if (!moves?.every(move => {
    const start = sequence.indexOf(move[0], cursor)
    if (!move.length || start < 0 || start + move.length > sequence.length || sequence.slice(start, start + move.length).some((id, index) => id !== move[index])) return false
    cursor = start + move.length
    return true
  })) return []
  return moves || []
}

function LanguageSwitcher({ language, onChange }: { language: VendorLanguage; onChange: (next: VendorLanguage) => void }) {
  const t = vendorCopy[language]
  return <div className="inline-flex items-center gap-2 rounded-lg border border-dark-600 bg-dark-800 p-1" role="group" aria-label={t.language}>
    {(['he', 'en'] as const).map(option => <button key={option} type="button" lang={option} dir={option === 'he' ? 'rtl' : 'ltr'}
      className={`rounded-md px-3 py-2 text-sm font-semibold ${language === option ? 'bg-primary-500 text-white' : 'text-dark-200 hover:bg-dark-700'}`}
      aria-pressed={language === option} onClick={() => onChange(option)}>{option === 'he' ? t.hebrew : t.english}</button>)}
  </div>
}

function freshDraft(data: VendorSetupResponse): VendorSetupDraft {
  return {
    schema_version: 1, cad_file_hash: data.part.cad_file_hash || '',
    cad_uploaded_at: data.part.cad_uploaded_at,
    drawing_file_hash: data.part.drawing_file_hash,
    candidate_generation: data.candidate_generation,
    part_revision: data.part.revision, expected_draft_revision: 0,
    marks: data.candidates.map(mark => ({ ...mark, review_status: 'pending' as const })),
    sequence: data.candidates.map(mark => mark.feature_id),
    machine_moves: [], machine_moves_explicit: true,
    general_measurements: [],
    operator_sequence_confirmed: false, completed_steps: 0,
    matching_search: { search_radius_mm: null, search_angle_deg: null },
  }
}

function demoData(): VendorSetupResponse {
  return {
    part: {
      id: 'demo', part_number: 'SYNTHETIC-DEMO', part_name: 'Multi-bend sample',
      customer: null, revision: 'demo', cad_file_path: null, cad_file_hash: 'synthetic-only',
      cad_uploaded_at: null, preview_mesh_path: '/vendor-demo.ply',
      drawing_file_path: null, drawing_file_hash: null, drawing_uploaded_at: null,
      material: null, default_tolerance_angle: 1, default_tolerance_linear: 0.5,
      bend_count: 3, has_embedding: false, embedding_version: null,
      embedding_computed_at: null, created_at: '', updated_at: '', erp_import_id: null, notes: null,
    },
    candidates: demoCandidates, draft: null, candidate_generation: 'synthetic-only',
  }
}

/** `wizard`: set only when the onboarding wizard embeds this page as its tagging studio. Without it nothing changes. */
export default function VendorSetupPage({ wizard }: { wizard?: VendorWizardEmbed } = {}) {
  const { partId = '' } = useParams()
  const navigate = useNavigate()
  const demo = partId === 'demo'
  const corpusDemo = (import.meta.env.DEV || import.meta.env.VITE_VENDOR_CORPUS_DEMO === 'true') && /^corpus-\d{8}$/.test(partId)
  const draftBaseline = useRef('')
  const [selectedPartId, setSelectedPartId] = useState('')
  const [newNumber, setNewNumber] = useState('')
  const [newName, setNewName] = useState('')
  const [busy, setBusy] = useState(false)
  const [exportingPdf, setExportingPdf] = useState(false)
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')
  const [draft, setDraft] = useState<VendorSetupDraft | null>(null)
  const [firstPoint, setFirstPoint] = useState<Point3 | null>(null)
  const [marking, setMarking] = useState(false)
  const [correctionForId, setCorrectionForId] = useState<string | null>(null)
  const [activeId, setActiveId] = useState<string | null>(null)
  const [allowStaleReset, setAllowStaleReset] = useState(false)
  const [workspaceExpanded, setWorkspaceExpanded] = useState(() => (import.meta.env.DEV || import.meta.env.VITE_VENDOR_CORPUS_DEMO === 'true') && /^corpus-\d{8}$/.test(partId))
  const [aluminumAppearance, setAluminumAppearance] = useState(true)
  const [requestedViewMode, setViewMode] = useState<'folded' | 'unfolded_visual'>('folded')
  const [ownLanguage, setLanguage] = useState<VendorLanguage>(() => readVendorLanguage(corpusDemo ? 'he' : undefined))
  // Embedded in the wizard, the wizard's language switch is the only one.
  const language: VendorLanguage = wizard?.language ?? ownLanguage
  const [orderMode, setOrderMode] = useState(false)
  const [orderAssigned, setOrderAssigned] = useState<string[]>([])
  const [orderMoves, setOrderMoves] = useState<string[][]>([])
  const [orderMoveCursor, setOrderMoveCursor] = useState(0)
  const [selectedMoveStartId, setSelectedMoveStartId] = useState<string | null>(null)
  const snapshotRef = useRef<(() => ViewerSnapshot) | null>(null)
  const onSnapshotReady = useCallback((capture: (() => ViewerSnapshot) | null) => { snapshotRef.current = capture }, [])
  const [measurementMode, setMeasurementMode] = useState<'bend_pair' | 'points' | null>(null)
  const [measurementBends, setMeasurementBends] = useState<string[]>([])
  const [measurementPoints, setMeasurementPoints] = useState<Point3[]>([])
  const [measurementLabel, setMeasurementLabel] = useState('')
  const [measurementTarget, setMeasurementTarget] = useState('')
  const [measurementNote, setMeasurementNote] = useState('')
  const [selectedMeasurementId, setSelectedMeasurementId] = useState<string | null>(null)
  const t = vendorCopy[language]

  useEffect(() => {
    persistVendorLanguage(language)
    const originalLanguage = document.documentElement.lang
    const originalDirection = document.documentElement.dir
    document.documentElement.lang = language
    document.documentElement.dir = language === 'he' ? 'rtl' : 'ltr'
    return () => {
      document.documentElement.lang = originalLanguage
      document.documentElement.dir = originalDirection
    }
  }, [language])

  const switchLanguage = (next: VendorLanguage) => {
    setLanguage(next)
    setMessage('')
    setError('')
  }

  useEffect(() => {
    if (!workspaceExpanded) return
    const oldOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    const onEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setWorkspaceExpanded(false)
    }
    window.addEventListener('keydown', onEscape)
    return () => {
      document.body.style.overflow = oldOverflow
      window.removeEventListener('keydown', onEscape)
    }
  }, [workspaceExpanded])

  const { data: parts } = useQuery({ queryKey: ['vendor-setup-parts'], queryFn: () => partCatalogApi.list({ limit: 100 }), enabled: !partId || corpusDemo })
  const setupQuery = useQuery({
    queryKey: ['vendor-setup', partId],
    queryFn: () => partCatalogApi.getVendorSetup(partId),
    enabled: !!partId && !demo,
    refetchOnWindowFocus: false,
  })
  const data = useMemo(() => demo ? demoData() : setupQuery.data, [demo, setupQuery.data])
  useEffect(() => {
    if (data) {
      const saved = data.draft && !data.draft.is_stale ? data.draft : null
      // A new server snapshot is an explicit reload boundary for the editable draft.
      const initialDraft = saved ? { ...saved,
        machine_moves: machineMovesFor(saved.sequence, saved.machine_moves_explicit
          ? saved.machine_moves : saved.machine_moves?.filter(move => move.length > 1)),
        machine_moves_explicit: true,
        general_measurements: saved.general_measurements || [],
        operator_sequence_confirmed: saved.operator_sequence_confirmed === true, expected_draft_revision: saved.draft_revision || 0 }
        : { ...freshDraft(data), expected_draft_revision: data.draft?.draft_revision || 0 }
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setDraft(initialDraft)
      draftBaseline.current = JSON.stringify(initialDraft)
      setAllowStaleReset(false)
      setError('')
      setOrderMode(false)
      setOrderAssigned([])
      setOrderMoves([])
      setOrderMoveCursor(0)
      setSelectedMoveStartId(null)
      setMeasurementMode(null)
      setMeasurementBends([])
      setMeasurementPoints([])
      setSelectedMeasurementId(null)
      setViewMode('folded')
    }
  }, [data])

  const sequence = useMemo(() => draft?.sequence || [], [draft])
  const machineMoves = useMemo(() => machineMovesFor(sequence, draft?.machine_moves), [sequence, draft])
  const visibleMoves = orderMode ? orderMoves : machineMoves
  const selectedMoveIndex = visibleMoves.findIndex(move => move[0] === selectedMoveStartId)
  const selectedMoveIds = useMemo(() => selectedMoveIndex >= 0 ? visibleMoves[selectedMoveIndex] : [], [visibleMoves, selectedMoveIndex])
  const moveById = useMemo(() => new Map(visibleMoves.flatMap((move, index) => move.map(id => [id, index] as const))), [visibleMoves])
  const generalMeasurements = useMemo(() => draft?.general_measurements || [], [draft])
  const selectedMeasurement = generalMeasurements.find(item => item.id === selectedMeasurementId)
  const measurementPickReady = measurementMode === 'points' || measurementMode === 'bend_pair' && measurementBends.length === 2
  const measurementDistance = measurementPoints.length === 2 ? Math.hypot(...measurementPoints[0].map((value, index) => value - measurementPoints[1][index])) : null
  const marksById = useMemo(() => new Map(draft?.marks.map(mark => [mark.feature_id, mark]) || []), [draft])
  const selectedId = selectedMoveIndex >= 0 || selectedMeasurementId || measurementMode ? null : activeId && marksById.has(activeId) ? activeId : orderMode ? null : sequence[0] || null
  const confirmedCount = sequence.filter(id => marksById.get(id)?.review_status === 'confirmed').length
  // Wizard guide (onboarding context only): live sub-step state and unsaved changes.
  const [dirty, setDirty] = useState(false)
  const onDirtyChange = wizard?.onDirtyChange
  useEffect(() => {
    const next = !!draft && (orderMode || !!measurementMode || JSON.stringify(draft) !== draftBaseline.current)
    setDirty(next)
    onDirtyChange?.(next)
  }, [draft, orderMode, measurementMode, onDirtyChange])
  const studioProgress: StudioProgress = { total: sequence.length, confirmed: confirmedCount, orderConfirmed: !!draft?.operator_sequence_confirmed,
    orderMode, allAssigned: orderAssigned.length === sequence.length, moves: (orderMode ? orderMoves : machineMoves).length, dirty }
  const bendStepById = useMemo(() => {
    const ordered = orderMode ? orderAssigned : draft?.operator_sequence_confirmed
      ? sequence : sequence.filter(id => marksById.get(id)?.review_status === 'confirmed')
    return new Map(ordered.map((id, index) => [id, index + 1]))
  }, [orderMode, orderAssigned, draft, sequence, marksById])
  const selectedMark = selectedId ? marksById.get(selectedId) : undefined
  const bendColors = useMemo(() => new Map(draft?.marks.map((mark, index) => [mark.feature_id, bendColor(moveById.get(mark.feature_id) ?? visibleMoves.length + index)]) || []), [draft, moveById, visibleMoves.length])
  const selectedFlange = selectedMark?.status === 'active'
    ? data?.flange_surfaces?.find(surface => surface.feature_id === selectedId)
    : undefined
  const flangeDepthById = useMemo(() => new Map(data?.flange_surfaces?.map(surface => [surface.feature_id, surface.nominal_flange_depth_mm]) || []), [data])
  const highlightedBendIds = useMemo(() => new Set([...selectedMoveIds, ...measurementBends, ...(selectedMeasurement?.bend_ids || [])]), [selectedMoveIds, measurementBends, selectedMeasurement])
  const cadUnfoldPreview = useMemo(() => validatedUnfoldPreview(data), [data])
  // A FRESH operator-confirmed uploaded flat replaces the CAD development; otherwise nothing changes.
  const boundFlat = useBoundFlatUnfold(partId, !demo && !corpusDemo, data, draft)
  // Left/right-sensitive part (display only): a warning and numbered markers on the folded view.
  const chirality = useChiralityOverlay(partId, !demo && !corpusDemo, language)
  const unfoldPreview = boundFlat?.preview ?? cadUnfoldPreview
  const unfoldLines = useMemo(() => new Map(unfoldPreview?.bend_lines.map(line => [line.feature_id, line]) || []), [unfoldPreview])
  const unfoldMesh = useMemo(() => unfoldPreview ? { vertices: unfoldPreview.vertices, triangles: unfoldPreview.triangles } : undefined, [unfoldPreview])
  const canUnfold = boundFlat ? true : canMapUnfoldDraft(data, draft, unfoldPreview)
  // Restoring a manual correction must immediately invalidate an open flat view,
  // not merely disable its toggle while silently dropping the mark.
  const viewMode = canUnfold ? requestedViewMode : 'folded'
  useEffect(() => {
    if (!canUnfold && requestedViewMode !== 'folded') {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setViewMode('folded')
    }
  }, [canUnfold, requestedViewMode])
  const viewerSurfaces = useMemo(() => (viewMode === 'unfolded_visual' ? unfoldPreview?.flange_surfaces || [] : data?.flange_surfaces || []).flatMap(surface => {
    const selected = selectedId === surface.feature_id || highlightedBendIds.has(surface.feature_id)
    return selected || moveById.has(surface.feature_id) ? [{ vertices: surface.vertices, triangles: surface.triangles,
      color: bendColors.get(surface.feature_id) || '#fbbf24', opacity: selected ? 0.65 : 0.28 }] : []
  }), [data, unfoldPreview, viewMode, selectedId, highlightedBendIds, moveById, bendColors])
  const activeSegments = useMemo(() => draft?.marks.filter(mark => mark.status === 'active' && mark.line_start && mark.line_end)
    .map(mark => ({ id: mark.feature_id, lineStart: mark.line_start!, lineEnd: mark.line_end!,
      label: String(bendStepById.get(mark.feature_id) ?? '?'), accessibleLabel: mark.label,
      color: bendColors.get(mark.feature_id) || '#facc15', selected: selectedId === mark.feature_id || highlightedBendIds.has(mark.feature_id),
      groupLabel: moveById.has(mark.feature_id) ? `${t.machineMove} ${moveById.get(mark.feature_id)! + 1}` : undefined,
    })) || [], [draft, bendColors, selectedId, highlightedBendIds, bendStepById, moveById, t.machineMove])
  const flatSegments = useMemo(() => activeSegments.flatMap(segment => {
    const line = unfoldLines.get(segment.id)
    return line ? [{ ...segment, lineStart: line.line_start, lineEnd: line.line_end }] : []
  }), [activeSegments, unfoldLines])
  const dimensionSegments = useMemo(() => [...generalMeasurements.map((item, index) => ({ id: item.id, lineStart: item.point_start,
    lineEnd: item.point_end, label: `R${index + 1} · ${item.cad_distance.toFixed(1)} mm*`, selected: item.id === selectedMeasurementId })),
    ...(measurementPoints.length === 2 && measurementDistance != null ? [{ id: 'pending', lineStart: measurementPoints[0], lineEnd: measurementPoints[1], label: `${measurementDistance.toFixed(1)} mm*`, selected: true }] : [])], [generalMeasurements, selectedMeasurementId, measurementPoints, measurementDistance])
  const previewUrl = demo ? '/vendor-demo.ply' : data?.part.preview_mesh_path
    ? import.meta.env.VITE_VENDOR_CORPUS_DEMO === 'true' && corpusDemo
      ? `/vendor-corpus/${data.part.part_number}.ply`
      : `/api/parts/${encodeURIComponent(partId)}/preview.ply?cad_sha256=${encodeURIComponent(data.part.cad_file_hash || '')}`
    : undefined

  const selectBend = (id: string) => {
    if (!marksById.has(id)) return
    setSelectedMeasurementId(null)
    setSelectedMoveStartId(null)
    setActiveId(id)
    if (measurementMode === 'bend_pair' && measurementBends.length < 2 && marksById.get(id)?.status === 'active'
        && marksById.get(id)?.line_start && marksById.get(id)?.line_end) {
      setMeasurementBends(previous => previous.includes(id) ? previous : [...previous, id])
      return
    }
    if (orderMode && marksById.get(id)?.status === 'active') {
      setOrderAssigned(previous => previous.includes(id) ? previous : [...previous, id])
    }
  }

  const beginOrderTagging = () => {
    setMeasurementMode(null)
    setMeasurementBends([])
    setMeasurementPoints([])
    setSelectedMeasurementId(null)
    setOrderAssigned([])
    setOrderMoves([])
    setOrderMoveCursor(0)
    setOrderMode(true)
    setActiveId(null)
    setSelectedMoveStartId(null)
    setMarking(false)
    setFirstPoint(null)
    setCorrectionForId(null)
    setMessage('')
  }

  const applyOrderTagging = () => {
    if (!draft || orderAssigned.length !== sequence.length || new Set(orderAssigned).size !== sequence.length) return
    const changed = orderAssigned.some((id, index) => draft.sequence[index] !== id)
    const resetStage = changed && draft.completed_steps > 0
    setDraft({ ...draft, sequence: [...orderAssigned], machine_moves: orderMoves.map(move => [...move]), machine_moves_explicit: true,
      operator_sequence_confirmed: true, completed_steps: changed ? 0 : draft.completed_steps })
    setOrderMode(false)
    setOrderAssigned([])
    setOrderMoves([])
    setOrderMoveCursor(0)
    setMessage(resetStage ? t.orderAppliedReset : t.orderApplied)
  }

  const changeMark = (id: string, patch: Partial<VendorBendMark>) => {
    setDraft(previous => {
      if (!previous) return previous
      const clearsReview = patch.status !== undefined || patch.line_start !== undefined || patch.line_end !== undefined
      const marks = previous.marks.map(mark => mark.feature_id === id ? { ...mark, ...patch, ...(clearsReview ? { review_status: 'pending' as const } : {}) } : mark)
      const sequence = marks.filter(mark => mark.status === 'active').map(mark => mark.feature_id)
      const kept = previous.sequence.filter(key => sequence.includes(key))
      const nextSequence = [...kept, ...sequence.filter(key => !kept.includes(key))]
      const invalidatesOrder = clearsReview || nextSequence.length !== previous.sequence.length || nextSequence.some((id, index) => id !== previous.sequence[index])
      return { ...previous, marks, sequence: nextSequence,
        machine_moves: invalidatesOrder ? [] : previous.machine_moves,
        machine_moves_explicit: true,
        general_measurements: invalidatesOrder ? (previous.general_measurements || []).filter(item => !item.bend_ids.includes(id)) : previous.general_measurements,
        operator_sequence_confirmed: previous.operator_sequence_confirmed && !invalidatesOrder,
        completed_steps: invalidatesOrder ? 0 : previous.completed_steps }
    })
  }

  const handlePick = (point: Point3) => {
    if (viewMode !== 'folded') return
    if (measurementMode && measurementPickReady && measurementPoints.length < 2) {
      if (!point.every(Number.isFinite)) return
      if (measurementPoints[0]?.every((value, index) => value === point[index])) { setError(t.pickDifferent); return }
      setMeasurementPoints(previous => [...previous, point])
      return
    }
    if (!marking || !draft) return
    if (!firstPoint) { setFirstPoint(point); return }
    if (firstPoint.every((value, index) => value === point[index])) {
      setError(t.pickDifferent)
      return
    }
    const feature_id = `manual:${crypto.randomUUID()}`
    const mark: VendorBendMark = {
      feature_id, label: `M${draft.marks.filter(item => item.source === 'manual').length + 1}`,
      source: 'manual', status: 'active', issue: correctionForId ? 'wrong_location' : 'missing',
      review_status: 'pending',
      corrects_feature_id: correctionForId, note: '',
      line_start: firstPoint, line_end: point,
    }
    const nextSequence = correctionForId ? draft.sequence.map(id => id === correctionForId ? feature_id : id)
      : [...draft.sequence, feature_id]
    setDraft({ ...draft, marks: [...draft.marks.map(item => item.feature_id === correctionForId ?
      { ...item, status: 'removed' as const, issue: 'wrong_location' as const, review_status: 'pending' as const } : item), mark],
      sequence: nextSequence, machine_moves: [], machine_moves_explicit: true,
      general_measurements: (draft.general_measurements || []).filter(item => !correctionForId || !item.bend_ids.includes(correctionForId)),
      operator_sequence_confirmed: false,
      completed_steps: 0 })
    setActiveId(feature_id)
    setFirstPoint(null)
    setMarking(false)
    setCorrectionForId(null)
    setMessage(t.manualAdded)
  }

  const confirmAndNext = () => {
    if (!draft || !selectedId || marking) return
    const current = marksById.get(selectedId)
    if (!current || current.status !== 'active' || !current.line_start || !current.line_end) return
    const nextMarks = draft.marks.map(mark => mark.feature_id === selectedId ? { ...mark, review_status: 'confirmed' as const } : mark)
    // Until an explicit order is applied, the worker's confirmation sequence is
    // a second, equally explicit way to assign steps. Never infer it from CAD order.
    const nextSequence = draft.operator_sequence_confirmed ? draft.sequence : [
      ...draft.sequence.filter(id => id !== selectedId && marksById.get(id)?.review_status === 'confirmed'),
      selectedId,
      ...draft.sequence.filter(id => id !== selectedId && marksById.get(id)?.review_status !== 'confirmed'),
    ]
    const allReviewed = nextMarks.filter(mark => mark.status === 'active').every(mark => mark.review_status === 'confirmed')
    setDraft({ ...draft, marks: nextMarks, sequence: nextSequence,
      machine_moves: draft.operator_sequence_confirmed ? machineMoves : [], machine_moves_explicit: true,
      operator_sequence_confirmed: draft.operator_sequence_confirmed || allReviewed })
    const from = nextSequence.indexOf(selectedId)
    const orderedNext = [...nextSequence.slice(from + 1), ...nextSequence.slice(0, from)]
    const nextId = orderedNext.find(id => nextMarks.find(mark => mark.feature_id === id)?.review_status !== 'confirmed')
    if (nextId) setActiveId(nextId)
  }

  const undoConfirmation = () => {
    if (!draft || !selectedId) return
    setDraft({ ...draft, marks: draft.marks.map(mark => mark.feature_id === selectedId ? { ...mark, review_status: 'pending' as const } : mark) })
  }

  const selectMove = (index: number) => {
    if (measurementMode) return
    const move = visibleMoves[index]
    if (!move) return
    setSelectedMeasurementId(null)
    setSelectedMoveStartId(move[0])
    setActiveId(null)
  }

  const tagMachineMove = () => {
    const pending = orderAssigned.slice(orderMoveCursor)
    if (!orderMode || !pending.length) return
    setOrderMoves(previous => [...previous, pending])
    setOrderMoveCursor(orderAssigned.length)
    setSelectedMoveStartId(pending[0])
    setActiveId(null)
  }

  const leaveBendsWithoutMove = () => {
    setOrderMoveCursor(orderAssigned.length)
    setSelectedMoveStartId(null)
  }

  const undoLastOrderTag = () => {
    const removedId = orderAssigned[orderAssigned.length - 1]
    if (!removedId) return
    const remaining = orderAssigned.slice(0, -1)
    setOrderAssigned(remaining)
    setOrderMoves(previous => previous.map(move => move.filter(id => id !== removedId)).filter(move => move.length))
    setOrderMoveCursor(previous => Math.min(previous, remaining.length))
    setSelectedMoveStartId(null)
    setActiveId(null)
  }

  const cancelMeasurement = () => {
    setMeasurementMode(null)
    setMeasurementBends([])
    setMeasurementPoints([])
  }

  const beginMeasurement = (kind: 'bend_pair' | 'points') => {
    if (viewMode !== 'folded') return
    setMeasurementMode(kind)
    setMeasurementBends([])
    setMeasurementPoints([])
    setMeasurementLabel(`${t.generalMeasurement} ${generalMeasurements.length + 1}`)
    setMeasurementTarget('')
    setMeasurementNote('')
    setSelectedMeasurementId(null)
    setSelectedMoveStartId(null)
    setActiveId(null)
    setMarking(false)
    setFirstPoint(null)
    setCorrectionForId(null)
    setError('')
  }

  const addMeasurement = () => {
    if (!draft || !measurementMode || measurementPoints.length !== 2 || !measurementDistance || !Number.isFinite(measurementDistance) || !measurementLabel.trim()) return
    const target = measurementTarget.trim() ? Number(measurementTarget) : null
    if (target != null && (!Number.isFinite(target) || target <= 0)) { setError(t.invalidReferenceTarget); return }
    const item: VendorGeneralMeasurement = { id: `reference:${crypto.randomUUID()}`, label: measurementLabel.trim(),
      kind: measurementMode, bend_ids: measurementMode === 'bend_pair' ? [...measurementBends] : [],
      point_start: measurementPoints[0], point_end: measurementPoints[1], cad_distance: measurementDistance,
      target_mm: target, note: measurementNote.trim() }
    setDraft({ ...draft, general_measurements: [...generalMeasurements, item] })
    setSelectedMeasurementId(item.id)
    setActiveId(null)
    cancelMeasurement()
    setMessage(t.referenceAdded)
  }

  const downloadPdf = async () => {
    if (!data || !draft) return
    setBusy(true); setExportingPdf(true); setError('')
    try {
      // The PDF's unfolded page describes a CAD development, so an uploaded flat is never captured into it;
      // with a bound flat the CAD development is still captured exactly as before.
      const cadFlatSegments = boundFlat ? activeSegments.flatMap(segment => {
        const line = cadUnfoldPreview?.bend_lines.find(item => item.feature_id === segment.id)
        return line ? [{ ...segment, lineStart: line.line_start, lineEnd: line.line_end }] : []
      }) : flatSegments
      const cadCanUnfold = boundFlat ? canMapUnfoldDraft(data, draft, cadUnfoldPreview) : canUnfold
      const flatView = cadCanUnfold && cadUnfoldPreview ? captureUnfoldPreview(cadUnfoldPreview, cadFlatSegments, bendColors) : null
      const result = await exportVendorSetupPdf({ data, draft, language, viewMode,
        view: viewMode === 'unfolded_visual' ? flatView : snapshotRef.current?.() || null,
        additionalView: viewMode === 'folded' && flatView ? { mode: 'unfolded_visual', view: flatView } : undefined })
      const url = URL.createObjectURL(result.blob)
      const anchor = document.createElement('a')
      anchor.href = url; anchor.download = result.filename; anchor.click()
      window.setTimeout(() => URL.revokeObjectURL(url), 30_000)
      setMessage(t.pdfExported)
    } catch { setError(t.pdfExportFailed) }
    finally { setBusy(false); setExportingPdf(false) }
  }

  const save = async () => {
    if (!draft || demo) return
    setBusy(true); setError(''); setMessage('')
    try {
      const result = await partCatalogApi.saveVendorSetup(partId, draft)
      const savedDraft = { ...result.draft, expected_draft_revision: result.draft.draft_revision || 0 }
      setDraft(savedDraft)
      draftBaseline.current = JSON.stringify(savedDraft)
      setMessage(corpusDemo ? t.localDemoSaved : t.saved)
    } catch (err) { setError(localizeVendorError(getErrorMessage(err), language)) }
    finally { setBusy(false) }
  }

  const switchDemoPart = (nextId: string) => {
    if (!/^corpus-\d{8}$/.test(nextId) || nextId === partId) return
    if ((orderMode || measurementMode || draft && JSON.stringify(draft) !== draftBaseline.current) && !window.confirm(t.unsavedPartSwitch)) return
    setActiveId(null)
    setSelectedMoveStartId(null)
    navigate(`/parts/vendor-setup/${nextId}`)
  }

  const upload = async (kind: 'cad' | 'drawing', file?: File) => {
    if (!file || demo) return
    setBusy(true); setError(''); setMessage('')
    try {
      if (kind === 'cad') {
        const result = await partCatalogApi.uploadCad(partId, file)
        if (result.status === 'CAD_PROCESSED') setMessage(t.cadProcessed)
        else if (result.status === 'CAD_UPLOADED_UNANALYZED') setMessage(t.cadSavedWithoutAnalysis)
        else if (result.status === 'CAD_UPLOAD_SUPERSEDED_NOT_PUBLISHED') setError(t.cadSuperseded)
        else setError(language === 'he' ? t.cadUnavailable
          : `${t.cadUnavailable}${result.bend_analysis_error ? ` (${result.bend_analysis_error})` : ''}`)
      } else {
        await partCatalogApi.uploadDrawing(partId, file)
        setMessage(t.drawingAttached)
      }
      await setupQuery.refetch()
    } catch (err) { setError(localizeVendorError(getErrorMessage(err), language)) }
    finally { setBusy(false) }
  }

  if (!partId) return <div className="vendor-setup-root space-y-5" dir={language === 'he' ? 'rtl' : 'ltr'} lang={language}>
    <div className="flex flex-wrap items-center justify-between gap-3"><Link to="/parts" className="inline-flex items-center gap-2 text-dark-300 hover:text-white"><ArrowLeft size={16} className={language === 'he' ? 'rotate-180' : ''} /> {t.partCatalog}</Link><LanguageSwitcher language={language} onChange={switchLanguage} /></div>
    <div><h1 className="text-2xl font-bold text-dark-100">{t.title}</h1><p className="text-dark-400">{t.chooseSku}</p></div>
    <div className="grid gap-4 lg:grid-cols-2">
      <section className="card p-5 space-y-3"><h2 className="text-lg font-semibold">{t.existingSku}</h2>
        <label className="label" htmlFor="setup-part">{t.part}</label>
        <select id="setup-part" className="input" value={selectedPartId} onChange={e => setSelectedPartId(e.target.value)}>
          <option value="">{t.selectPart}</option>{parts?.parts.map(part => <option key={part.id} value={part.id}>{part.part_number} — {part.part_name}</option>)}
        </select><button className="btn btn-primary" disabled={!selectedPartId} onClick={() => navigate(`/parts/vendor-setup/${selectedPartId}`)}>{t.openSetup}</button>
      </section>
      <section className="card p-5 space-y-3"><h2 className="text-lg font-semibold">{t.newSku}</h2>
        <label className="label" htmlFor="new-part-number">{t.partNumber}</label><input id="new-part-number" className="input" dir="ltr" value={newNumber} onChange={e => setNewNumber(e.target.value)} />
        <label className="label" htmlFor="new-part-name">{t.partName}</label><input id="new-part-name" className="input" value={newName} onChange={e => setNewName(e.target.value)} />
        <button className="btn btn-primary" disabled={busy || !newNumber.trim() || !newName.trim()} onClick={async () => {
          setBusy(true); setError(''); try { const result = await partCatalogApi.create({ part_number: newNumber.trim(), part_name: newName.trim() }); navigate(`/parts/vendor-setup/${result.part.id}`) } catch (err) { setError(localizeVendorError(getErrorMessage(err), language)) } finally { setBusy(false) }
        }}>{t.createContinue}</button>
      </section>
    </div>
    {error && <p role="alert" className="text-red-300">{error}</p>}
    <Link to="/parts/vendor-setup/demo" className="inline-flex rounded-lg border border-amber-400/50 px-4 py-2 text-amber-200 hover:bg-amber-400/10">{t.syntheticDemo}</Link>
  </div>

  if (!data || !draft) return <div className="vendor-setup-root" dir={language === 'he' ? 'rtl' : 'ltr'} lang={language}>{!wizard && <div className="mb-3"><LanguageSwitcher language={language} onChange={switchLanguage} /></div>}<div className="card p-6">{setupQuery.isError ? localizeVendorError(getErrorMessage(setupQuery.error), language) : t.loading}</div></div>

  const saveButton = <button className="btn btn-primary inline-flex items-center gap-2 disabled:opacity-50" data-save-draft disabled={busy || demo || orderMode || !!measurementMode || sequence.length > 0 && !draft.operator_sequence_confirmed || !data.part.cad_file_hash || !!data.draft?.is_stale && !allowStaleReset} onClick={save}><Save size={16} /> {orderMode ? t.finishOrderFirst : sequence.length > 0 && !draft.operator_sequence_confirmed ? t.setOrderFirst : t.saveDraft}</button>

  return <div className={`vendor-setup-root space-y-5 pb-8${wizard ? ' vendor-setup-root--wizard' : ''}`} dir={language === 'he' ? 'rtl' : 'ltr'} lang={language} data-embedded={wizard ? 'wizard' : undefined}>
    {wizard ? <WizardStudioBar language={language} progress={studioProgress} save={saveButton} onDone={wizard.onDone} /> : <>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>{!corpusDemo && <Link to="/parts/vendor-setup" className="inline-flex items-center gap-2 text-sm text-dark-300 hover:text-white"><ArrowLeft size={15} className={language === 'he' ? 'rotate-180' : ''} /> {t.allSetups}</Link>}
          <h1 className="mt-2 text-2xl font-bold text-dark-100"><bdi dir="ltr">{data.part.part_number}</bdi> · {t.title}</h1>
          <p className="text-dark-300">{t.setupSubtitle}</p></div>
        <div className="flex flex-wrap items-center gap-2"><LanguageSwitcher language={language} onChange={switchLanguage} />{saveButton}</div>
      </div>
      {!demo && !corpusDemo && <WizardStandaloneBar partId={partId} language={language} progress={studioProgress} />}
    </>}
    {demo && <div className="rounded-lg border border-amber-400/60 bg-amber-400/10 p-3 text-amber-100">{t.syntheticNotice}</div>}
    {corpusDemo && <div className="rounded-lg border border-sky-400/50 bg-sky-400/10 p-3 text-sm text-sky-100">{t.corpusDemoNotice}</div>}
    {data.draft?.is_stale && <div role="alert" className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-amber-400/60 p-3 text-amber-100"><span>{t.staleNotice}</span><button className="btn btn-secondary" onClick={() => setAllowStaleReset(true)}>{allowStaleReset ? t.staleAck : t.staleReview}</button></div>}
    {error && <div role="alert" className="rounded-lg border border-red-400/60 p-3 text-red-200">{error}</div>}
    {message && <div role="status" className="rounded-lg border border-emerald-400/50 p-3 text-emerald-200">{message}</div>}
    {chirality.notice}
    {!demo && !wizard && <div className="card flex flex-wrap items-center gap-4 p-4">
      <label className="btn btn-secondary inline-flex cursor-pointer items-center gap-2"><FileUp size={16} /> {data.part.cad_file_path ? t.replaceCad : t.uploadCad}<input className="sr-only" type="file" accept=".stl,.ply,.obj,.step,.stp" disabled={busy} onChange={e => { void upload('cad', e.target.files?.[0]); e.target.value = '' }} /></label>
      <label className="btn btn-secondary inline-flex cursor-pointer items-center gap-2"><FileUp size={16} /> {data.part.drawing_file_path ? t.replaceDrawing : t.attachDrawing}<input className="sr-only" type="file" accept=".pdf" disabled={busy} onChange={e => { void upload('drawing', e.target.files?.[0]); e.target.value = '' }} /></label>
      <span className="text-sm text-dark-300"><bdi dir="ltr">CAD</bdi> {data.part.cad_file_hash ? t.uploaded : t.required} · {t.drawing} {data.part.drawing_file_hash ? t.attached : t.optional}</span>
      <span className="text-sm text-dark-300">{t.cadFormatSupport}</span>
    </div>}
    <div className={`vendor-setup-workspace ${workspaceExpanded ? 'vendor-setup-workspace--expanded' : ''}`}>
      <section className="card vendor-setup-model overflow-hidden"><div className="flex flex-wrap items-center justify-between gap-3 border-b border-dark-700 p-4"><div><h2 className="text-lg font-semibold">{t.cadMap}</h2><p className="text-sm text-dark-300">{t.cadFrameNote}</p></div>
        <div className="flex flex-wrap items-center gap-2">{corpusDemo && <><span className="rounded-full border border-sky-400/50 px-3 py-2 text-xs font-semibold text-sky-100">{t.corpusDemoShort}</span><label className="vendor-setup-demo-switch inline-flex items-center gap-2 rounded-lg border border-sky-400/50 bg-dark-800 px-2 text-sm text-sky-100"><span>{t.switchPart}</span><select className="min-w-0 bg-transparent px-2 text-sm text-white focus-visible:outline focus-visible:outline-2 focus-visible:outline-sky-400" aria-label={t.switchPart} value={partId} onChange={event => switchDemoPart(event.target.value)}>{(parts?.parts || [data.part]).map(part => <option key={part.id} value={part.id} className="bg-dark-800 text-white">{part.part_number} · {part.bend_count} {t.cadCandidates}</option>)}</select></label></>}
        <div role="group" aria-label={t.modelView} className="inline-flex gap-1 rounded-lg border border-dark-500 p-1">
          <button type="button" data-view-mode="folded" className={`min-h-11 rounded-md px-4 font-semibold ${viewMode === 'folded' ? 'bg-primary-500 text-white' : 'text-dark-100'}`} aria-pressed={viewMode === 'folded'} disabled={busy || marking || !!measurementMode} onClick={() => { setViewMode('folded'); setSelectedMeasurementId(null) }}>{t.foldedView}</button>
          <button type="button" data-view-mode="unfolded_visual" className={`min-h-11 rounded-md px-4 font-semibold disabled:opacity-40 ${viewMode === 'unfolded_visual' ? 'bg-primary-500 text-white' : 'text-dark-100'}`} aria-pressed={viewMode === 'unfolded_visual'} disabled={busy || !canUnfold || marking || !!measurementMode} onClick={() => { setViewMode('unfolded_visual'); setSelectedMeasurementId(null) }}>{t.unfoldedView}</button>
        </div>
        <button className="btn btn-secondary inline-flex items-center gap-2" disabled={viewMode !== 'folded' || orderMode || !!measurementMode || !previewUrl || !data.part.cad_file_hash && !demo} onClick={() => { setMarking(!marking); setFirstPoint(null); setCorrectionForId(null) }}><Plus size={18} /> {marking ? t.cancelMarking : t.addMissedBend}</button>
        <button className="btn btn-secondary inline-flex min-h-11 items-center gap-2" disabled={busy || orderMode || marking || !!measurementMode || !!boundFlat && viewMode === 'unfolded_visual'} title={boundFlat && viewMode === 'unfolded_visual' ? boundFlatCopy(language).pdfFoldedOnly : undefined} onClick={() => void downloadPdf()}><Download size={18} /> {exportingPdf ? t.exportingPdf : t.exportPdf}</button>
        <button className="btn btn-secondary inline-flex items-center gap-2" aria-label={t.aluminumToggle} aria-pressed={aluminumAppearance} title={t.aluminumTitle} onClick={() => setAluminumAppearance(value => !value)}>{aluminumAppearance ? t.aluminumOn : t.aluminumOff}</button>
        {workspaceExpanded && !wizard && <LanguageSwitcher language={language} onChange={switchLanguage} />}
        <button className="btn btn-secondary inline-flex items-center gap-2" aria-label={workspaceExpanded ? t.exitWorkspace : t.expandWorkspace} onClick={() => setWorkspaceExpanded(!workspaceExpanded)}>{workspaceExpanded ? <Minimize2 size={18} /> : <Maximize2 size={18} />}{workspaceExpanded ? t.exitWorkspace : t.expandWorkspace}</button></div></div>
        {previewUrl || viewMode === 'unfolded_visual' && unfoldMesh ? <div className="vendor-setup-canvas bg-dark-900" dir="ltr" data-current-view={viewMode}><ThreeViewer key={`${partId}:${viewMode}`} referenceUrl={viewMode === 'folded' ? previewUrl : undefined} referenceMesh={viewMode === 'unfolded_visual' ? unfoldMesh : undefined} viewPreset={viewMode === 'unfolded_visual' ? 'front' : 'default'} referencePresentation={aluminumAppearance ? 'aluminum' : 'solid'} selectedSurfaces={viewerSurfaces} dimensionSegments={viewMode === 'folded' ? [...dimensionSegments, ...chirality.markers] : []} onSnapshotReady={onSnapshotReady} className="h-full" labelingMode={viewMode === 'folded' && (marking || !!measurementMode && measurementPickReady && measurementPoints.length < 2)} manualBendSegments={viewMode === 'folded' ? activeSegments : flatSegments} draftLabelPoint={viewMode === 'folded' ? measurementMode && measurementPoints.length === 1 ? measurementPoints[0] : firstPoint : null} onLabelPointPick={handlePick} touchFriendly onManualSegmentSelect={selectBend} selectedManualSegmentId={selectedId} operatorLabels={{ zoomIn: t.viewerZoomIn, zoomOut: t.viewerZoomOut, reset: t.viewerReset, rotateControls: t.viewerRotateControls, rotate: [t.viewerRotateLeft, t.viewerRotateRight, t.viewerRotateUp, t.viewerRotateDown], noModel: t.viewerNoModel, showCad: t.viewerShowCad, hideCad: t.viewerHideCad, selectSegment: orderMode ? t.tagBend : t.modelSelect, onModel: t.modelSelectSuffix }} /></div>
          : <div className="flex h-[320px] items-center justify-center p-6 text-center text-dark-300">{t.noPreview}</div>}
        <p className={`px-3 pt-3 text-sm ${canUnfold ? 'text-amber-100' : 'text-dark-300'}`} data-unfold-status={canUnfold ? 'available' : 'unavailable'} data-unfold-source={boundFlat ? 'uploaded_flat' : 'cad_development'}>{boundFlat ? boundFlatCopy(language).notice : viewMode === 'unfolded_visual' ? t.unfoldNotice : !unfoldPreview ? t.unfoldUnavailable : !canUnfold ? t.unfoldManualUnavailable : t.unfoldNotice}</p>
        <p className="p-3 text-sm text-dark-200">{measurementMode ? !measurementPickReady ? t.chooseMeasurementBends : measurementPoints.length === 2 ? t.measurementPointsReady : measurementPoints.length === 1 ? t.secondMeasurementPoint : t.chooseMeasurementPoints : orderMode ? selectedMoveIndex >= 0 ? t.moveModelHint : t.orderModelHint : marking ? firstPoint ? t.pickSecond : t.pickFirst : selectedMoveIndex >= 0 ? t.moveModelHint : selectedFlange ? t.selectedSurfaceHint : t.selectedLineHint}</p>
      </section>
      <section className="card vendor-setup-bend-panel p-4 space-y-3">
        <div className="vendor-setup-order rounded-xl border border-sky-400/50 bg-sky-400/10 p-3" data-order-mode={orderMode} data-order-confirmed={!!draft.operator_sequence_confirmed}>
          <h2 className="text-lg font-semibold text-sky-100">{t.orderTitle}</h2>
          <p className="mt-1 text-sm text-dark-200">{t.orderLead}</p>
          {!orderMode && <><p className={`mt-2 text-sm font-semibold ${draft.operator_sequence_confirmed ? 'text-emerald-300' : 'text-amber-200'}`} role="status">{draft.operator_sequence_confirmed ? t.orderConfirmed : t.orderNotConfirmed}</p>{!draft.operator_sequence_confirmed && <p className="mt-1 text-sm text-dark-200">{t.confirmSetsOrder}</p>}</>}
          {orderMode ? <>
            <div className="mt-3 rounded-lg bg-dark-900/80 p-3" role="status" aria-live="polite">
              <strong>{orderAssigned.length === sequence.length ? t.orderComplete : `${t.nextStep} ${orderAssigned.length + 1} ${t.of} ${sequence.length}`}</strong>
              {orderAssigned.length < sequence.length && <p className="mt-1 text-sm text-dark-200">{t.tapNext}</p>}
            </div>
            <div className="mt-3 flex flex-wrap gap-2" aria-label={t.orderMode}>{orderAssigned.map((id, index) => <span key={id} data-order-assigned={id} className="inline-flex items-center gap-2 rounded-lg border border-dark-500 bg-dark-800 px-3 py-2 text-sm"><bdi dir="ltr">{index + 1} · {marksById.get(id)?.label}</bdi></span>)}</div>
            <div className="mt-3 rounded-xl border border-cyan-400/50 bg-cyan-400/10 p-3">
              <p className="text-sm text-sky-100">{t.pendingMoveBends}: <bdi dir="ltr">{orderAssigned.slice(orderMoveCursor).map(id => marksById.get(id)?.label).join(' · ') || '—'}</bdi></p>
              <div className="mt-2 flex flex-wrap gap-2">
                <button type="button" data-tag-move className="btn btn-primary min-h-11" disabled={orderAssigned.length === orderMoveCursor} onClick={tagMachineMove}>{t.tagMachineMove}</button>
                <button type="button" data-leave-without-move className="btn btn-secondary min-h-11" disabled={orderAssigned.length === orderMoveCursor} onClick={leaveBendsWithoutMove}>{t.leaveWithoutMove}</button>
              </div>
              <p className="mt-2 text-sm text-dark-200">{t.machineMoveOptionalHint}</p>
            </div>
            {orderMoves.length > 0 && <div className="mt-3 space-y-2" aria-label={t.machineMoves}>{orderMoves.map((move, index) => <button key={move[0]} type="button" data-machine-move={index + 1} data-select-move={index + 1} aria-label={`${t.selectMove} ${index + 1}`} aria-pressed={selectedMoveIndex === index} className={`flex min-h-11 w-full items-center gap-3 rounded-xl border p-3 text-start ${selectedMoveIndex === index ? 'border-cyan-300 bg-cyan-400/15' : 'border-dark-600 bg-dark-800'}`} onClick={() => selectMove(index)}><span className="font-bold text-sky-100">{t.machineMove} {index + 1}</span><bdi dir="ltr">{move.map(id => marksById.get(id)?.label).join(' · ')}</bdi></button>)}</div>}
            {draft.completed_steps > 0 && <p className="mt-2 text-sm text-amber-200">{t.orderResetWarning}</p>}
            {machineMoves.length > 0 && <p className="mt-2 text-sm text-amber-200">{t.orderResetsMoves}</p>}
            <div className="mt-3 flex flex-wrap gap-2">
              <button className="btn btn-secondary min-h-11" disabled={orderAssigned.length === 0} onClick={undoLastOrderTag}>{t.undoLast}</button>
              <button className="btn btn-secondary min-h-11" disabled={orderAssigned.length === 0} onClick={() => { setOrderAssigned([]); setOrderMoves([]); setOrderMoveCursor(0); setActiveId(null); setSelectedMoveStartId(null) }}>{t.restartOrder}</button>
              <button className="btn btn-secondary min-h-11" onClick={() => { setOrderMode(false); setOrderAssigned([]); setOrderMoves([]); setOrderMoveCursor(0); setSelectedMoveStartId(null) }}>{t.cancelOrder}</button>
              <button className="btn btn-primary" data-apply-order disabled={orderAssigned.length !== sequence.length} onClick={applyOrderTagging}>{t.applyOrder}</button>
            </div>
          </> : <>
            <button className="btn btn-primary mt-3 w-full min-h-11" data-tag-order disabled={sequence.length === 0 || !!measurementMode} onClick={beginOrderTagging}>{draft.operator_sequence_confirmed ? t.editOrderAndMoves : t.tagOrder}</button>
            <h3 className="mt-4 font-semibold text-sky-100">{t.machineMoves}{machineMoves.length ? ` · ${machineMoves.length}` : ''}</h3>
            <p className="mt-1 text-sm text-dark-200">{machineMoves.length ? `${t.machineMovesHint} ${t.groupColorHint}` : t.noMachineMoves}</p>
            <div className="mt-3 space-y-2" aria-label={t.machineMoves}>{machineMoves.map((move, index) => <div key={move[0]} data-machine-move={index + 1} data-group-color={bendColor(index)} style={{ borderInlineStart: `5px solid ${bendColor(index)}` }} className={`rounded-xl border p-2 ${selectedMoveIndex === index ? 'border-cyan-300 bg-cyan-400/15' : 'border-dark-600 bg-dark-800'}`}>
              <button type="button" data-select-move={index + 1} aria-label={`${t.selectMove} ${index + 1}`} aria-pressed={selectedMoveIndex === index} className="flex w-full items-center gap-3 rounded-lg px-2 py-2 text-start focus-visible:outline focus-visible:outline-2 focus-visible:outline-sky-400" onClick={() => selectMove(index)}>
                <span className="flex h-11 min-w-11 items-center justify-center rounded-full bg-sky-500/20 font-bold text-sky-100">{index + 1}</span>
                <span className="min-w-0 flex-1"><span className="block font-semibold">{t.machineMove} {index + 1}</span><span className="block text-sm text-dark-200" dir="ltr">{move.map(id => marksById.get(id)?.label).join(' · ')}</span></span>
                <span className="text-sm text-dark-200">{move.length} {move.length === 1 ? t.bendShort : t.bendsShort}</span>
              </button>
            </div>)}</div>
          </>}
        </div>
        <section className="rounded-xl border border-amber-400/50 bg-amber-400/5 p-3 space-y-3" data-measurement-mode={measurementMode || ''}>
          <h2 className="flex items-center gap-2 text-lg font-semibold text-amber-100"><Ruler size={20} />{t.generalMeasurements}</h2>
          <p className="text-sm text-dark-200">{t.measurementIntro}</p>
          {viewMode === 'unfolded_visual' && <p className="text-sm text-amber-100">{t.unfoldMeasurementNotice}</p>}
          {!measurementMode ? <div className="flex flex-wrap gap-2"><button className="btn btn-secondary min-h-11" disabled={viewMode !== 'folded' || orderMode || marking || generalMeasurements.length >= 100 || sequence.length < 2 || !previewUrl} onClick={() => beginMeasurement('bend_pair')}>{t.measureBends}</button><button className="btn btn-secondary min-h-11" disabled={viewMode !== 'folded' || orderMode || marking || generalMeasurements.length >= 100 || !previewUrl} onClick={() => beginMeasurement('points')}>{t.measurePart}</button></div> : <>
            <p role="status" className="text-sm font-semibold text-amber-100">{!measurementPickReady ? `${t.chooseMeasurementBends} (${measurementBends.length}/2)` : measurementPoints.length === 2 ? t.measurementPointsReady : measurementPoints.length === 1 ? t.secondMeasurementPoint : t.chooseMeasurementPoints}</p>
            {measurementBends.length > 0 && <p dir="ltr" className="text-sm">{measurementBends.map(id => marksById.get(id)?.label).join(' · ')}</p>}
            {measurementDistance != null && <p className="text-sm">{t.cadReferenceDistance}: <strong dir="ltr">{measurementDistance.toFixed(2)}</strong></p>}
            <label className="block text-sm" htmlFor="reference-label">{t.measurementLabel}<input id="reference-label" className="input mt-1" maxLength={100} value={measurementLabel} onChange={event => setMeasurementLabel(event.target.value)} /></label>
            <label className="block text-sm" htmlFor="reference-target">{t.requiredDimension}<input id="reference-target" className="input mt-1" type="number" dir="ltr" min="0.001" step="0.1" value={measurementTarget} onChange={event => setMeasurementTarget(event.target.value)} /></label>
            <label className="block text-sm" htmlFor="reference-note">{t.measurementNote}<textarea id="reference-note" className="input mt-1" rows={2} maxLength={500} value={measurementNote} onChange={event => setMeasurementNote(event.target.value)} /></label>
            <div className="flex flex-wrap gap-2"><button className="btn btn-primary min-h-11" disabled={!measurementDistance || !measurementLabel.trim()} onClick={addMeasurement}>{t.addReference}</button><button className="btn btn-secondary min-h-11" onClick={cancelMeasurement}>{t.cancelReference}</button></div>
          </>}
          {!generalMeasurements.length && <p className="text-sm text-dark-300">{t.noReferences}</p>}
          {generalMeasurements.map((item, index) => <div key={item.id} data-general-measurement={item.id} className={`rounded-lg border p-3 ${item.id === selectedMeasurementId ? 'border-amber-300 bg-amber-400/10' : 'border-dark-600 bg-dark-800'}`}><div className="flex items-start gap-2"><button type="button" className="min-h-11 min-w-0 flex-1 text-start" aria-label={`${t.generalMeasurement} R${index + 1}`} aria-pressed={item.id === selectedMeasurementId} onClick={() => { setSelectedMeasurementId(item.id); setSelectedMoveStartId(null); setActiveId(null) }}><strong dir="ltr">R{index + 1}</strong> · {item.label}<span className="block text-sm text-dark-200" dir="ltr">{item.bend_ids.map(id => marksById.get(id)?.label).join(' ↔ ')}</span></button><button type="button" className="min-h-11 min-w-11 rounded-lg hover:bg-dark-600" aria-label={`${t.deleteReference} R${index + 1}`} disabled={!!measurementMode} onClick={() => { setDraft({ ...draft, general_measurements: generalMeasurements.filter(row => row.id !== item.id) }); setSelectedMeasurementId(null) }}><Trash2 size={18} className="mx-auto" /></button></div><p className="text-sm">{t.cadReferenceDistance}: <bdi dir="ltr">{item.cad_distance.toFixed(2)}</bdi></p>{item.target_mm != null && <p className="text-sm">{t.requiredDimension}: <bdi dir="ltr">{item.target_mm}</bdi></p>}{item.note && <p className="mt-1 whitespace-pre-wrap text-sm text-dark-200">{item.note}</p>}</div>)}
        </section>
        <h2 className="text-lg font-semibold">{t.bendMarkings} · {confirmedCount} {t.of} {sequence.length} {t.reviewed}</h2>
        <div className="h-2 overflow-hidden rounded-full bg-dark-600" role="progressbar" aria-label={t.bendMarkings} aria-valuenow={confirmedCount} aria-valuemin={0} aria-valuemax={sequence.length}><div className="h-full bg-emerald-400 transition-all" style={{ width: `${sequence.length ? confirmedCount / sequence.length * 100 : 0}%` }} /></div>
        <p className="text-sm text-dark-300">{selectedMoveIndex >= 0
          ? `${t.machineMove} ${selectedMoveIndex + 1}: ${selectedMoveIds.map(id => marksById.get(id)?.label).join(' · ')}.`
          : selectedMark ? `${t.selected} ${selectedMark.label}: ${selectedMark.review_status === 'confirmed' ? t.confirmedReview : t.pendingReview}.` : t.selectReview} {t.notQcApproval}</p>
        {draft.marks.length === 0 && <p className="text-dark-400">{t.noCandidates}</p>}
        <div className="vendor-setup-bend-list space-y-2 overflow-y-auto">{[...sequence.map(id => marksById.get(id)!).filter(Boolean), ...draft.marks.filter(mark => mark.status === 'removed')].map(mark => <div key={mark.feature_id} data-bend-id={mark.feature_id} data-bend-label={mark.label} data-review-status={mark.review_status || 'pending'} data-mark-status={mark.status} data-vendor-step={mark.status === 'active' ? bendStepById.get(mark.feature_id) ?? '' : ''} className={`rounded-lg border p-3 ${selectedId === mark.feature_id ? 'border-primary-400 bg-dark-700' : 'border-dark-600 bg-dark-800'}`} style={{ borderInlineStartWidth: 5, borderInlineStartColor: bendColors.get(mark.feature_id) || '#facc15' }}>
          <div className="flex items-start justify-between gap-2"><button className="vendor-setup-select-bend flex min-w-0 flex-1 items-center gap-3 rounded text-start font-medium focus-visible:outline focus-visible:outline-2 focus-visible:outline-primary-400" aria-label={`${orderMode ? t.tagBend : t.selectBend} ${mark.label}`} aria-pressed={selectedId === mark.feature_id} onClick={() => selectBend(mark.feature_id)}><span className="flex h-9 min-w-9 items-center justify-center rounded-full border-2 bg-dark-900 text-base" style={{ borderColor: bendColors.get(mark.feature_id) || '#facc15', color: bendColors.get(mark.feature_id) || '#facc15' }}>{mark.status === 'active' ? bendStepById.get(mark.feature_id) ?? '?' : '×'}</span><span className="min-w-0 text-base"><bdi dir="ltr">{mark.label}</bdi><span className="block text-sm font-normal text-dark-200">{mark.status === 'removed' ? t.removed : mark.review_status === 'confirmed' ? t.confirmed : t.pending}</span></span></button>
            {selectedId === mark.feature_id && !orderMode && <button className="rounded p-2 hover:bg-dark-600 disabled:opacity-30" aria-label={`${mark.status === 'active' ? t.remove : t.restore} ${mark.label}`} disabled={mark.status === 'removed' && draft.marks.some(item => item.status === 'active' && item.corrects_feature_id === mark.feature_id)} onClick={() => changeMark(mark.feature_id, mark.status === 'active' ? { status: 'removed', issue: 'not_a_bend' } : { status: 'active', issue: mark.source === 'manual' ? mark.issue : null })}>{mark.status === 'active' ? <Trash2 size={16} /> : <RotateCcw size={16} />}</button>}</div>
          {moveById.has(mark.feature_id) && <p className="mt-2 text-sm font-semibold" style={{ color: bendColors.get(mark.feature_id) }}>{t.machineMove} {moveById.get(mark.feature_id)! + 1}</p>}
          <div className="vendor-setup-measures mt-2 grid grid-cols-3 gap-2 text-sm"><div><span>{t.angle}</span><strong><bdi dir="ltr">{mark.target_angle == null ? '—' : `${mark.target_angle}°`}</bdi></strong></div><div title={mark.source === 'manual' ? t.manualWidthTitle : t.cadWidthTitle}><span>{t.width}</span><strong><bdi dir="ltr">{mark.line_start && mark.line_end ? `${Math.hypot(...mark.line_start.map((v, i) => v - mark.line_end![i])).toFixed(1)} ${language === 'he' ? 'מ״מ' : 'mm'}*` : '—'}</bdi></strong></div><div title={t.depthTitle}><span>{t.flangeDepth}</span><strong><bdi dir="ltr">{mark.status === 'active' && flangeDepthById.get(mark.feature_id) != null ? `${flangeDepthById.get(mark.feature_id)!.toFixed(1)} ${language === 'he' ? 'מ״מ' : 'mm'}*` : t.unavailable}</bdi></strong></div></div>
          {selectedId === mark.feature_id && !orderMode && <><p className="mt-2 text-sm text-dark-300">{mark.line_start && mark.line_end ? t.locatedCad : t.locationUnavailable} · {mark.source === 'manual' ? t.manualHypothesis : t.cadCandidate} · {mark.status === 'active' ? t.active : t.removed}</p>
          {mark.status === 'active' && <p className="mt-1 text-sm text-dark-200">{selectedFlange ? t.surfaceAvailable : t.surfaceUnavailable}</p>}
          {mark.target_radius != null && <p className="mt-1 text-sm text-dark-300">{t.nominalRadius} <bdi dir="ltr">{mark.target_radius}</bdi> {t.assumedMm}</p>}
          {mark.status === 'removed' && <select aria-label={`${t.removeReason} ${mark.label}`} className="input mt-2 text-sm" disabled={draft.marks.some(item => item.status === 'active' && item.corrects_feature_id === mark.feature_id)} value={mark.issue || 'not_a_bend'} onChange={e => changeMark(mark.feature_id, { issue: e.target.value as VendorBendMark['issue'] })}><option value="not_a_bend">{t.notABend}</option><option value="wrong_location">{t.wrongLocation}</option></select>}
          {viewMode === 'unfolded_visual' && unfoldLines.has(mark.feature_id) && <p className="mt-1 text-sm text-dark-200">{t.unfoldDirection}: {unfoldLines.get(mark.feature_id)!.bend_direction === 'up' ? t.bendUp : unfoldLines.get(mark.feature_id)!.bend_direction === 'down' ? t.bendDown : t.directionUnknown}</p>}
          {viewMode === 'folded' && mark.source === 'cad_candidate' && mark.status === 'active' && previewUrl && <button className="mt-2 rounded border border-dark-500 px-3 py-2 text-sm text-dark-200 hover:bg-dark-600" onClick={() => { setActiveId(mark.feature_id); setCorrectionForId(mark.feature_id); setFirstPoint(null); setMarking(true) }}>{t.correctLocation}</button>}
          {mark.source === 'manual' && <input className="input mt-2 text-sm" aria-label={`${t.noteFor} ${mark.label}`} placeholder={t.correctionNote} value={mark.note} onChange={e => changeMark(mark.feature_id, { note: e.target.value })} />}
          {mark.status === 'active' && <div className="mt-2 flex flex-wrap gap-2">{mark.review_status === 'confirmed' ? <button className="rounded-lg border border-dark-500 px-3 py-2 text-sm text-dark-100 hover:bg-dark-600" aria-label={`${t.undoConfirmation} ${mark.label}`} onClick={undoConfirmation}>{t.undoReview}</button> : <button className="rounded-lg bg-emerald-500 px-3 py-2 text-sm font-semibold text-dark-900 hover:bg-emerald-400 disabled:opacity-40" data-confirm-bend={mark.feature_id} aria-label={`${t.confirmBend} ${mark.label}${draft.operator_sequence_confirmed ? '' : ` · ${t.confirmAsStep} ${confirmedCount + 1}`}`} disabled={marking || !mark.line_start || !mark.line_end} onClick={confirmAndNext}>{draft.operator_sequence_confirmed ? t.confirmNext : `${t.confirmAsStep} ${confirmedCount + 1}`}</button>}</div>}</>}
        </div>)}</div>
        <p className="text-xs text-dark-300">{t.dataFootnote}</p>
        <div className="border-t border-dark-700 pt-3 space-y-2"><h3 className="font-semibold">{t.plannedStage} · {draft.completed_steps} {t.of} {sequence.length}</h3><p className="text-sm text-dark-300">{t.completedBeforeScan}</p>
          <div className="flex items-center gap-2"><button className="rounded-lg bg-dark-700 p-2 disabled:opacity-30" aria-label={t.decreaseSteps} disabled={orderMode || draft.completed_steps === 0} onClick={() => setDraft({ ...draft, completed_steps: draft.completed_steps - 1 })}><Minus size={20} /></button><span className="flex-1 text-center text-sm font-medium" dir="ltr">{draft.completed_steps} / {sequence.length}</span><button className="rounded-lg bg-dark-700 p-2 disabled:opacity-30" aria-label={t.increaseSteps} disabled={orderMode || draft.completed_steps >= sequence.length} onClick={() => setDraft({ ...draft, completed_steps: draft.completed_steps + 1 })}><Plus size={20} /></button></div>
        </div>
      </section>
    </div>
    <div className="card p-4 space-y-3"><details><summary className="cursor-pointer font-semibold">{t.advanced}</summary><p className="mt-3 text-sm text-dark-300">{t.advancedNote}</p>
      <label className="label" htmlFor="search-radius">{t.positionSearch}</label><input id="search-radius" className="input" type="number" dir="ltr" min="0.01" max="1000" step="0.1" value={draft.matching_search.search_radius_mm ?? ''} placeholder={t.notSet} onChange={e => setDraft({ ...draft, matching_search: { ...draft.matching_search, search_radius_mm: e.target.value ? Number(e.target.value) : null } })} />
      <label className="label" htmlFor="search-angle">{t.orientationSearch}</label><input id="search-angle" className="input" type="number" dir="ltr" min="0.01" max="180" step="0.1" value={draft.matching_search.search_angle_deg ?? ''} placeholder={t.notSet} onChange={e => setDraft({ ...draft, matching_search: { ...draft.matching_search, search_angle_deg: e.target.value ? Number(e.target.value) : null } })} />
      <p className="text-xs text-dark-400">{t.catalogTolerance} <bdi dir="ltr">±{data.part.default_tolerance_angle}°</bdi> {t.angle}, <bdi dir="ltr">±{data.part.default_tolerance_linear} mm</bdi> {t.width}.</p></details>
    </div>
    <p className="text-xs text-dark-400">{t.finalDisclaimer}</p>
  </div>
}
