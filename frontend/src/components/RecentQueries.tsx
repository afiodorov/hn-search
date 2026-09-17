import type { Me, RecentQuery } from '../types'

interface Props {
  queries: RecentQuery[]
  me: Me
  onSelect: (query: string) => void
  onDelete: (query: string) => void
  onSignOut: () => void
}

export function RecentQueries({
  queries,
  me,
  onSelect,
  onDelete,
  onSignOut,
}: Props) {
  if (queries.length === 0 && !me.login) return null

  return (
    <aside className="recent">
      <h2>Recent searches</h2>
      {/* The list is shared by every visitor, so deleting is for admins. A
          signed-in non-admin sees their name and no delete buttons. */}
      <div className="recent-auth">
        {me.login ? (
          <>
            <span className="recent-user" title={me.admin ? 'admin' : 'signed in'}>
              {me.login}
              {me.admin && <span className="recent-badge">admin</span>}
            </span>
            <button type="button" className="recent-link" onClick={onSignOut}>
              sign out
            </button>
          </>
        ) : me.configured ? (
          <a className="recent-link" href="/auth/login">
            Sign in with GitHub
          </a>
        ) : null}
      </div>
      <ul>
        {queries.map((q) => (
          // The row is the li: a delete button nested inside the query button
          // would be invalid HTML and unreachable by keyboard.
          <li key={`${q.query}-${q.timestamp}`}>
            <button type="button" onClick={() => onSelect(q.query)}>
              {q.query}
            </button>
            <span className="time-ago">{q.time_ago}</span>
            {me.admin && (
              <button
                type="button"
                className="recent-delete"
                onClick={() => onDelete(q.query)}
                aria-label={`Delete recent search: ${q.query}`}
                title="Delete — this cannot be undone"
              >
                ×
              </button>
            )}
          </li>
        ))}
      </ul>
    </aside>
  )
}
