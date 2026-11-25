import { useCallback, useRef, useState } from 'react'
import { fetchSaved, streamSearch } from '../api'
import type { ProgressEvent, SavedAnswer, SearchStatus, Source } from '../types'

export interface SearchState {
  status: SearchStatus
  steps: ProgressEvent[]
  answer: string
  sources: Source[]
  error: string | null
  shareId: string | null
  /** Set when showing a saved answer rather than a live one. */
  saved: SavedAnswer | null
}

export function useSearch() {
  const [status, setStatus] = useState<SearchStatus>('idle')
  const [steps, setSteps] = useState<ProgressEvent[]>([])
  const [answer, setAnswer] = useState('')
  const [sources, setSources] = useState<Source[]>([])
  const [error, setError] = useState<string | null>(null)
  const [shareId, setShareId] = useState<string | null>(null)
  const [saved, setSaved] = useState<SavedAnswer | null>(null)

  const cancelRef = useRef<(() => void) | null>(null)
  // Tokens accumulate in a ref and flush once per frame so react-markdown
  // isn't re-parsed on every SSE chunk.
  const bufRef = useRef('')
  const rafRef = useRef(0)

  // Bumped by every run/load, so a saved answer that arrives after the
  // visitor has moved on is dropped instead of replacing what they see.
  const seqRef = useRef(0)

  const reset = useCallback(() => {
    seqRef.current += 1
    cancelRef.current?.()
    cancelRef.current = null
    if (rafRef.current) cancelAnimationFrame(rafRef.current)
    bufRef.current = ''
    rafRef.current = 0
    setStatus('running')
    setSteps([])
    setAnswer('')
    setSources([])
    setError(null)
    setShareId(null)
    setSaved(null)
  }, [])

  const run = useCallback(
    (query: string) => {
      reset()

      cancelRef.current = streamSearch(query, {
        onProgress: (ev) =>
          setSteps((prev) => {
            if (ev.status === 'done') {
              const i = prev.findIndex(
                (s) => s.step === ev.step && s.status === 'start',
              )
              if (i >= 0) {
                const next = [...prev]
                next[i] = ev
                return next
              }
            }
            return [...prev, ev]
          }),
        onSources: setSources,
        onToken: (text) => {
          bufRef.current += text
          if (!rafRef.current) {
            rafRef.current = requestAnimationFrame(() => {
              rafRef.current = 0
              setAnswer(bufRef.current)
            })
          }
        },
        onAnswer: (text, id) => {
          bufRef.current = text
          setAnswer(text)
          setShareId(id)
        },
        onError: (message) => {
          setError(message)
          setStatus('error')
        },
        onDone: () => setStatus('done'),
      })
    },
    [reset],
  )

  /** Show a saved answer as it was, without running anything. Resolves to
   *  the saved answer, or null when the id is unknown. */
  const load = useCallback(
    async (id: string): Promise<SavedAnswer | null> => {
      reset()
      const seq = seqRef.current
      let snap: SavedAnswer | null
      try {
        snap = await fetchSaved(id)
      } catch (e) {
        if (seq !== seqRef.current) return null
        setError(e instanceof Error ? e.message : String(e))
        setStatus('error')
        return null
      }
      if (seq !== seqRef.current) return null
      if (!snap) {
        setError('This saved answer has expired, was deleted, or never existed.')
        setStatus('error')
        return null
      }
      setSaved(snap)
      setAnswer(snap.answer)
      setSources(snap.sources)
      setShareId(snap.id)
      setStatus('done')
      return snap
    },
    [reset],
  )

  return { status, steps, answer, sources, error, shareId, saved, run, load }
}
