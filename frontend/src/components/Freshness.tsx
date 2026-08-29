import { useEffect, useState } from 'react'
import { fetchStats } from '../api'
import type { Stats } from '../types'

// Short, relative-when-recent rendering of the newest indexed comment: the
// point is to answer "is the index current?" at a glance.
function describe(iso: string): string {
  const t = new Date(iso)
  if (Number.isNaN(t.getTime())) return iso
  const hours = (Date.now() - t.getTime()) / 36e5
  if (hours < 1) return `${Math.max(1, Math.round(hours * 60))} min ago`
  if (hours < 48) return `${Math.round(hours)} h ago`
  return t.toLocaleDateString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  })
}

export function Freshness() {
  const [stats, setStats] = useState<Stats | null>(null)

  useEffect(() => {
    let cancelled = false
    fetchStats()
      .then((s) => !cancelled && setStats(s))
      .catch(() => {}) // purely informational; stay silent if unavailable
    return () => {
      cancelled = true
    }
  }, [])

  if (!stats || !stats.latest_timestamp) return null
  return (
    <p className="freshness" title={`newest comment: ${stats.latest_timestamp}`}>
      {stats.count.toLocaleString()} comments · latest{' '}
      {describe(stats.latest_timestamp)}
    </p>
  )
}
