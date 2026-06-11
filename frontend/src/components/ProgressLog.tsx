import type { ProgressEvent, SearchStatus } from '../types'

interface Props {
  steps: ProgressEvent[]
  status: SearchStatus
}

export function ProgressLog({ steps, status }: Props) {
  if (steps.length === 0) return null

  const totalMs = steps.reduce((sum, s) => sum + (s.ms ?? 0), 0)

  return (
    <div className="progress-log card">
      <ol>
        {steps.map((s, i) => (
          <li key={i} className={s.status}>
            <span className="step-indicator" aria-hidden="true">
              {s.status === 'done' ? '✓' : <span className="spinner" />}
            </span>
            <span className="step-label">{s.label}</span>
            <span className="step-timing">
              {s.status === 'done' ? (
                <>
                  {s.hit && <span className="badge">cached</span>}
                  {s.ms}ms
                </>
              ) : (
                '…'
              )}
            </span>
          </li>
        ))}
      </ol>
      {status === 'done' && (
        <div className="progress-total">
          Done in {(totalMs / 1000).toFixed(1)}s
        </div>
      )}
    </div>
  )
}
