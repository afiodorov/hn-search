import type { RecentQuery } from '../types'

interface Props {
  queries: RecentQuery[]
  onSelect: (query: string) => void
}

export function RecentQueries({ queries, onSelect }: Props) {
  if (queries.length === 0) return null

  return (
    <aside className="recent">
      <h2>Recent searches</h2>
      <ul>
        {queries.map((q) => (
          <li key={`${q.query}-${q.timestamp}`}>
            <button type="button" onClick={() => onSelect(q.query)}>
              {q.query}
            </button>
            <span className="time-ago">{q.time_ago}</span>
          </li>
        ))}
      </ul>
    </aside>
  )
}
