/** Same-origin binary transport. Tokens never enter URLs or follow redirects. */
export function artifactRequestUrl(value: string, origin: string): string {
  if (value !== value.trim() || value.startsWith('//') || value.includes('\\')) {
    throw new Error('Artifact address must be an explicit local inspection resource.')
  }
  const url = new URL(value, origin)
  const localCorpusDemo = import.meta.env.DEV && /^\/api\/parts\/corpus-\d{8}\/preview\.ply$/.test(url.pathname)
  const catalogPreview = /^\/api\/parts\/[0-9a-fA-F-]{36}\/preview\.ply$/.test(url.pathname)
  // The viewer binds a folded artifact to the exact CAD snapshot returned by
  // vendor setup. Only this endpoint may carry this one non-secret query key.
  const queryAllowed = !url.search || catalogPreview && /^\?cad_sha256=[0-9a-f]{64}$/.test(url.search)
  const allowed = /^\/api\/(?:bend-inspection|aligned-scan|reference-mesh|heatmap|heatmaps|files)\//.test(url.pathname)
    || catalogPreview
    || localCorpusDemo
    || url.pathname === '/vendor-demo.ply'
  if (url.origin !== origin || url.username || url.password || !queryAllowed || url.hash || !allowed) {
    throw new Error('Artifact address is not an allowed local inspection resource.')
  }
  return url.href
}

export async function fetchAuthenticatedArtifact(value: string, signal?: AbortSignal): Promise<Response> {
  const url = artifactRequestUrl(value, window.location.origin)
  const token = localStorage.getItem('token')
  const syntheticDemo = new URL(url).pathname === '/vendor-demo.ply'
  const localCorpusDemo = import.meta.env.DEV && /^\/api\/parts\/corpus-\d{8}\/preview\.ply$/.test(new URL(url).pathname)
  if (!token && !syntheticDemo && !localCorpusDemo) throw new Error('Sign in again to load this inspection artifact.')
  const response = await fetch(url, {
    method: 'GET', headers: syntheticDemo || localCorpusDemo ? {} : { Authorization: `Bearer ${token}` },
    credentials: 'same-origin', redirect: 'error', cache: 'no-store', signal,
  })
  if (!response.ok) {
    if (response.status === 401) throw new Error('Your session expired. Sign in again to load this artifact.')
    if (response.status === 403 || response.status === 404) throw new Error('Artifact unavailable for this account or inspection.')
    throw new Error('Artifact could not be loaded. Retry or reopen the inspection.')
  }
  return response
}
