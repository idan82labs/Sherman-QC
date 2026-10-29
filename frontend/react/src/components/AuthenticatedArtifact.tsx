import { useEffect, useState, type ImgHTMLAttributes } from 'react'
import { fetchAuthenticatedArtifact } from '../services/authenticatedArtifact'

export function AuthenticatedArtifactImage({ src, alt, ...props }: ImgHTMLAttributes<HTMLImageElement> & { src: string }) {
  const [retry, setRetry] = useState(0)
  const [loaded, setLoaded] = useState<{ source: string; url?: string; error?: string } | null>(null)
  useEffect(() => {
    const controller = new AbortController()
    let objectUrl: string | undefined
    void (async () => {
      try {
        const response = await fetchAuthenticatedArtifact(src, controller.signal)
        const blob = await response.blob()
        if (controller.signal.aborted) return
        if (!['image/png', 'image/jpeg', 'image/webp'].includes(blob.type.split(';')[0])) {
          throw new Error('Artifact is not a supported inspection image.')
        }
        objectUrl = URL.createObjectURL(blob)
        setLoaded({ source: src, url: objectUrl })
      } catch (error) {
        if (!controller.signal.aborted) setLoaded({ source: src, error: error instanceof Error ? error.message : 'Image unavailable.' })
      }
    })()
    return () => {
      controller.abort()
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
  }, [src, retry])
  const current = loaded?.source === src ? loaded : null
  if (current?.error) return (
    <div className="p-4 text-sm text-dark-200">
      <p role="alert">{current.error} This does not establish a part defect.</p>
      <button type="button" className="mt-2 min-h-[44px] rounded border border-dark-500 px-3 hover:bg-dark-700 focus-visible:outline focus-visible:outline-2"
        onClick={() => { setLoaded(null); setRetry((value) => value + 1) }}>Retry image</button>
    </div>
  )
  if (!current?.url) return <div role="status" className="min-h-[160px] p-4 text-sm text-dark-300">Loading inspection image…</div>
  return <img {...props} src={current.url} alt={alt} onError={() => setLoaded({ source: src, error: 'Image could not be decoded.' })} />
}

export function AuthenticatedArtifactDownload({ url, filename }: { url: string; filename: string }) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const download = async () => {
    setBusy(true)
    setError(null)
    try {
      const response = await fetchAuthenticatedArtifact(url)
      const blob = await response.blob()
      const objectUrl = URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = objectUrl
      anchor.download = filename
      document.body.appendChild(anchor)
      anchor.click()
      anchor.remove()
      setTimeout(() => URL.revokeObjectURL(objectUrl), 1000)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Download unavailable. Please retry.')
    } finally {
      setBusy(false)
    }
  }
  return (
    <div>
      <button type="button" disabled={busy} onClick={() => void download()}
        className="min-h-[44px] rounded-lg border border-primary-500/40 bg-primary-500/10 px-3 py-2 text-xs font-medium text-primary-300 hover:bg-primary-500/15 disabled:opacity-50 focus-visible:outline focus-visible:outline-2">
        {busy ? 'Preparing download…' : 'Download PDF'}
      </button>
      {error && <p role="alert" className="mt-2 max-w-xs text-xs text-dark-200">{error}</p>}
    </div>
  )
}
