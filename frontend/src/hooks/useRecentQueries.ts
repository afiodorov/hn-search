import { useCallback, useEffect, useRef, useState } from 'react'
import { deleteRecent, fetchRecent } from '../api'
import type { RecentQuery } from '../types'

export function useRecentQueries(limit = 25) {
  const [queries, setQueries] = useState<RecentQuery[]>([])
  // Skip setState when the payload hasn't changed, so polling never causes
  // a re-render (the old Gradio UI re-rendered every tick and blinked).
  const lastRef = useRef('')

  const refresh = useCallback(async () => {
    try {
      const next = await fetchRecent(limit)
      const json = JSON.stringify(next)
      if (json !== lastRef.current) {
        lastRef.current = json
        setQueries(next)
      }
    } catch {
      // Polling failure is non-fatal; keep the last good list.
    }
  }, [limit])

  // Drop the row at once rather than waiting for the next poll, then refresh
  // so the list reflects what the server actually kept.
  const remove = useCallback(
    async (query: string) => {
      setQueries((prev) => prev.filter((q) => q.query !== query))
      try {
        await deleteRecent(query)
      } finally {
        lastRef.current = ''
        refresh()
      }
    },
    [refresh],
  )

  useEffect(() => {
    refresh()
    const id = setInterval(refresh, 5000)
    return () => clearInterval(id)
  }, [refresh])

  return { queries, refresh, remove }
}
