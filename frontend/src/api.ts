import type { Me, ProgressEvent, RecentQuery, Source, Stats } from './types'

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

export async function fetchMe(): Promise<Me> {
  const res = await fetch('/auth/me')
  if (!res.ok) throw new Error(`me failed: ${res.status}`)
  return res.json()
}

export async function logout(): Promise<void> {
  const res = await fetch('/auth/logout', { method: 'POST' })
  if (!res.ok) throw new Error(`logout failed: ${res.status}`)
}

/** Forget a recent query server-side. Idempotent: a row that has already been
 *  trimmed succeeds rather than 404ing. */
export async function deleteRecent(query: string): Promise<void> {
  const res = await fetch(`/api/recent?q=${encodeURIComponent(query)}`, {
    method: 'DELETE',
  })
  if (!res.ok) throw new Error(`delete failed: ${res.status}`)
}

export async function fetchStats(): Promise<Stats> {
  const res = await fetch('/api/stats')
  if (!res.ok) throw new Error(`stats failed: ${res.status}`)
  return res.json()
}
