/**
 * Every machine code the onboarding wizard can be shown, grouped by producer.
 *
 * The lists mirror the Python sources of truth; `scripts/check-onboarding-model.mjs`
 * re-reads those files and fails when a code is added there without being added
 * here (and given en + he copy in `copy.ts`).
 *
 * - RECIPE_BUILD_CODES: `contracts/production/part_recipe.py` RecipeBuildCode
 *   (C01 builder and the C05 release gate).
 * - COMPILE_ISSUE_CODES / COMPILE_WARNING_CODES: `compiled_part.py` CompileIssueCode
 *   (E1 reconciliation and compile blockers, ENGINE_IO §2.1).
 * - COMPILE_ERROR_CODES: `domains/production/engine/compile/compiler.py` (invalid
 *   compile input; `compile_error.json`), plus the CLI-only error codes.
 * - COMPILE_CROSSCHECK_CODES: `compiled_part.check_compiled_against_recipe`.
 * - FLAT_PARSE_CODES: `domains/production/engine/compile/flat.py` FlatParseError.
 * - C05_CODES: the C05 release gate (`release_service.CODE_MESSAGES`), flat intake
 *   (`flat_intake.FLAT_INTAKE_CODES`) and API service (`production_recipe_service`).
 * - WIZARD_CODES: produced by this UI or by the transport layer.
 */

export const RECIPE_BUILD_CODES = [
  'DRAFT_MISSING', 'DRAFT_STALE', 'DRAFT_INVALID', 'DRAFT_REVISION_CHANGED', 'SEQUENCE_NOT_CONFIRMED',
  'NO_ACTIVE_BENDS', 'UNCONFIRMED_ACTIVE_MARKS', 'MARK_WITHOUT_LINE', 'MANUAL_MARK_MISSING_NOMINAL_ANGLE',
  'DUPLICATE_BEND_IDENTITY', 'CAD_MISSING', 'CAD_HASH_MISMATCH', 'DRAWING_MISSING', 'DRAWING_HASH_MISMATCH',
  'FLAT_HASH_MISMATCH', 'FLAT_BINDING_UNKNOWN_BEND', 'SKU_INVALID', 'SKU_COLLISION', 'TOLERANCES_MISSING',
  'TOLERANCE_INVALID', 'OVERRIDE_UNKNOWN_BEND', 'OVERRIDE_UNKNOWN_MEASUREMENT', 'MEASUREMENT_INVALID',
  'CONCURRENT_RELEASE', 'RELEASE_DRAFT_REVISION_CHANGED',
] as const

export const COMPILE_WARNING_CODES = [
  'UNSUPPORTED_FEATURE', 'NOMINAL_UNAVAILABLE', 'K_FACTOR_DEFAULTED', 'ANGLE_CONVENTION_DISAGREES_WITH_RECIPE',
  'FLAT_NOT_BOUND',
] as const

export const COMPILE_BLOCKER_CODES = [
  'MARK_NOT_IN_FLANGE_GRAPH', 'UNMARKED_BEND', 'MARK_AMBIGUOUS', 'DATUM_PANEL_UNDETERMINED',
  'FLANGE_GRAPH_NOT_A_TREE', 'CAD_READ_FAILED', 'CAD_HASH_MISMATCH', 'RECIPE_INVALID', 'THICKNESS_INCONSISTENT',
  'FLAT_PARSE_FAILED', 'FLAT_BINDING_INCONSISTENT', 'FLAT_DEVELOPED_LENGTH_MISMATCH', 'STAGE_MODEL_FAILED',
] as const

export const COMPILE_ISSUE_CODES = [...COMPILE_BLOCKER_CODES, ...COMPILE_WARNING_CODES] as const

export const COMPILE_ERROR_CODES = [
  'RECIPE_INVALID', 'CAD_MISSING', 'CAD_HASH_MISMATCH', 'FLAT_MISSING', 'FLAT_HASH_MISMATCH', 'FLAT_UNEXPECTED',
  'CONFIG_INVALID', 'SHOP_K_INVALID', 'OUT_DIR_NOT_EMPTY', 'COMPILE_TIMEOUT', 'INTERNAL_ERROR',
] as const

export const COMPILE_CROSSCHECK_CODES = [
  'RECIPE_REF_MISMATCH', 'CAD_SHA_MISMATCH', 'FLAT_SHA_MISMATCH', 'BEND_INVENTORY_MISMATCH',
  'GENERAL_MEASUREMENT_MISMATCH', 'NOMINAL_UNKNOWN_REFERENCE', 'TOLERANCE_MISMATCH', 'REQUIREMENT_MISMATCH',
] as const

export const FLAT_PARSE_CODES = [
  'FLAT_FORMAT_UNSUPPORTED', 'EZDXF_UNAVAILABLE', 'DXF_UNREADABLE', 'NO_OUTLINE', 'NO_CLOSED_OUTLINE',
  'FLAT_STEP_UNREADABLE', 'FLAT_STEP_NOT_PLANAR',
] as const

export const C05_CODES = [
  // release gate (release_service.CODE_MESSAGES beyond the builder / E1 codes)
  'PART_NOT_FOUND', 'CAD_NOT_STEP', 'DRAWING_FORMAT_UNSUPPORTED', 'FLAT_MISSING', 'FLAT_PARSE_FAILED',
  'FLAT_BINDING_UNCONFIRMED', 'FLAT_BINDING_STALE', 'COMPILE_BLOCKED', 'COMPILE_FAILED', 'COMPILE_TIMEOUT',
  'COMPILE_INCONSISTENT', 'CONFIRMATION_NOT_ALLOWED',
  // flat intake
  'FLAT_NOT_FOUND', 'FLAT_EMPTY', 'FLAT_TOO_LARGE', 'FLAT_FORMAT_UNSUPPORTED', 'FLAT_TOOL_FAILED', 'FLAT_TOOL_TIMEOUT',
  'FLAT_REVISION_CHANGED', 'FLAT_REVISION_REQUIRED', 'FLAT_BINDING_INVALID', 'FLAT_NOT_PARSED', 'RECONCILIATION_BLOCKED',
  // API service
  'RECIPE_NOT_FOUND', 'PART_EXISTS', 'COMPILE_UNAVAILABLE', 'FLAT_TOOL_UNAVAILABLE', 'RECIPE_RETIRED', 'LINE_STORE_UNAVAILABLE',
] as const

/** Codes this UI derives itself, or that the transport adds around the server codes. */
export const WIZARD_CODES = [
  'CAD_NOT_STEP', 'CAD_NO_CANDIDATES', 'FLAT_BINDING_UNCONFIRMED', 'FLAT_BINDING_STALE', 'FLAT_BINDING_AMBIGUOUS',
  'FLAT_RECONCILIATION_BLOCKED', 'COMPILE_PREVIEW_STALE', 'COMPILE_PREVIEW_MISSING', 'COMPILE_BLOCKED',
  'EXPECTATIONS_FAILED', 'LINE_NOT_CONFIGURED', 'FORBIDDEN', 'NOT_FOUND', 'NETWORK_ERROR', 'SERVER_ERROR',
  'READY_FLAG_DISAGREES', 'COMPILE_PREVIEW_UNAVAILABLE', 'RESPONSE_INVALID',
] as const

export type ReasonCode =
  | typeof RECIPE_BUILD_CODES[number]
  | typeof COMPILE_ISSUE_CODES[number]
  | typeof COMPILE_ERROR_CODES[number]
  | typeof COMPILE_CROSSCHECK_CODES[number]
  | typeof FLAT_PARSE_CODES[number]
  | typeof C05_CODES[number]
  | typeof WIZARD_CODES[number]

/** Every code that must have plain-language copy in both languages. */
export const ALL_REASON_CODES: readonly ReasonCode[] = Array.from(new Set<ReasonCode>([
  ...RECIPE_BUILD_CODES, ...COMPILE_ISSUE_CODES, ...COMPILE_ERROR_CODES, ...COMPILE_CROSSCHECK_CODES,
  ...FLAT_PARSE_CODES, ...C05_CODES, ...WIZARD_CODES,
]))

const WARNING_SET: ReadonlySet<string> = new Set(COMPILE_WARNING_CODES)

/** Compile issues are BLOCKER unless listed as WARNING (ENGINE_IO §2.1); unknown codes are treated as blockers. */
export function compileSeverity(code: string): 'BLOCKER' | 'WARNING' {
  return WARNING_SET.has(code) ? 'WARNING' : 'BLOCKER'
}
