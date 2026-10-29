/**
 * Station rule for the left/right-sensitive part badge (display only, until engine enforcement ships).
 *
 * The body is `sherman_chirality_status_v0` from `GET /api/production/recipes/{recipe_id}/chirality`
 * (`apps/production_line/recipes/chirality.py`). The badge shows only when that map is READY for the
 * recipe on screen and its final-stage class is CHIRAL_BY_FEATURES: such a part and its mirror image
 * differ only in a few features, so a mirrored part can still register and measure as VALID. Anything
 * else (achiral, gross, pending, not computed, another recipe, a malformed body) shows no badge.
 */
export function needsEyeCheck(body: unknown, recipeId: string): boolean {
  if (!body || typeof body !== 'object') return false
  const b = body as { schema?: unknown; status?: unknown; recipe_id?: unknown; summary?: { part_class?: unknown } | null }
  return (
    b.schema === 'sherman_chirality_status_v0' &&
    b.status === 'READY' &&
    b.recipe_id === recipeId &&
    !!b.summary &&
    b.summary.part_class === 'CHIRAL_BY_FEATURES'
  )
}
