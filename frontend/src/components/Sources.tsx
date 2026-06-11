import type { Source } from '../types'

interface Props {
  sources: Source[]
}

export function Sources({ sources }: Props) {
  if (sources.length === 0) return null

  return (
    <details className="sources">
      <summary>Sources ({sources.length})</summary>
      <div className="source-list">
        {sources.map((s, i) => (
          <article key={s.id} className="source card">
            <header>
              <span className="source-index">{i + 1}</span>
              <a href={s.url} target="_blank" rel="noopener noreferrer">
                {s.author}
              </a>
              <time>{s.timestamp.slice(0, 10)}</time>
            </header>
            <p>{s.text}</p>
          </article>
        ))}
      </div>
    </details>
  )
}
