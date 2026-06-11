import { useCallback, useRef, useState } from 'react'
import { streamSearch } from '../api'
import type { ProgressEvent, SearchStatus, Source } from '../types'

export interface SearchState {
  status: SearchStatus
  steps: ProgressEvent[]
  answer: string
  sources: Source[]
  error: string | null
}

export function useSearch() {
  const [status, setStatus] = useState<SearchStatus>('idle')
  const [steps, setSteps] = useState<ProgressEvent[]>([])
  const [answer, setAnswer] = useState('')
  const [sources, setSources] = useState<Source[]>([])
  const [error, setError] = useState<string | null>(null)

  const cancelRef = useRef<(() => void) | null>(null)
  // Tokens accumulate in a ref and flush once per frame so react-markdown
  // isn't re-parsed on every SSE chunk.
  const bufRef = useRef('')
  const rafRef = useRef(0)

  const run = useCallback((query: string) => {
    cancelRef.current?.()
    if (rafRef.current) cancelAnimationFrame(rafRef.current)
    bufRef.current = ''
    rafRef.current = 0
    setStatus('running')
    setSteps([])
    setAnswer('')
    setSources([])
    setError(null)

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
      onAnswer: (text) => {
        bufRef.current = text
        setAnswer(text)
      },
      onError: (message) => {
        setError(message)
        setStatus('error')
      },
      onDone: () => setStatus('done'),
    })
  }, [])

  return { status, steps, answer, sources, error, run }
}
