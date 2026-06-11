import type { ProgressEvent, RecentQuery, Source } from './types'

export interface SearchHandlers {
  onProgress: (ev: ProgressEvent) => void
  onSources: (sources: Source[]) => void
  onToken: (text: string) => void
  onAnswer: (text: string) => void
  onError: (message: string) => void
  onDone: () => void
}

// EventSource auto-reconnects (and would re-run the search), so every terminal
// event must close it.
export function streamSearch(query: string, h: SearchHandlers): () => void {
  const es = new EventSource(`/api/search?q=${encodeURIComponent(query)}`)

  es.addEventListener('progress', (e) =>
    h.onProgress(JSON.parse((e as MessageEvent).data)),
  )
  es.addEventListener('sources', (e) =>
    h.onSources(JSON.parse((e as MessageEvent).data).sources),
  )
  es.addEventListener('token', (e) =>
    h.onToken(JSON.parse((e as MessageEvent).data).text),
  )
  es.addEventListener('answer', (e) =>
    h.onAnswer(JSON.parse((e as MessageEvent).data).text),
  )
  es.addEventListener('done', () => {
    es.close()
    h.onDone()
  })
  // Fires both for server-sent "error" events (with data) and transport
  // failures (without).
  es.addEventListener('error', (e) => {
    es.close()
    const data = (e as MessageEvent).data
    h.onError(data ? JSON.parse(data).message : 'Connection lost')
  })

  return () => es.close()
}

export async function fetchRecent(limit = 25): Promise<RecentQuery[]> {
  const res = await fetch(`/api/recent?limit=${limit}`)
  if (!res.ok) throw new Error(`recent queries failed: ${res.status}`)
  const body = await res.json()
  return body.queries
}
