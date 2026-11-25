import { useState } from 'react'
import type { SavedAnswer } from '../types'

interface Props {
  shareId: string | null
  /** The saved answer being shown, when this is not a live run. */
  saved: SavedAnswer | null
  onAskAgain: (query: string) => void
}

function savedOn(ts: number): string {
  return new Date(ts * 1000).toLocaleDateString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  })
}

/** Under a finished answer: its shareable link, and for a saved answer, when
 *  it was made and a way to ask the question afresh. */
export function ShareBar({ shareId, saved, onAskAgain }: Props) {
  // Keyed by shareId in App, so "copied" never carries over to a new answer.
  const [copied, setCopied] = useState(false)

  if (!shareId) return null
  const url = `${location.origin}/a/${shareId}`

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(url)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      // No clipboard (insecure context, denied): select the link instead.
      window.prompt('Copy this link', url)
    }
  }

  return (
    <div className="share-bar">
      {saved && (
        <span className="share-note">
          Saved answer from {savedOn(saved.created_at)}
          {' · '}
          <button
            type="button"
            className="share-link"
            onClick={() => onAskAgain(saved.query)}
          >
            ask again
          </button>
        </span>
      )}
      <button type="button" className="share-copy" onClick={copy}>
        {copied ? '✓ Link copied' : 'Copy link'}
      </button>
    </div>
  )
}
