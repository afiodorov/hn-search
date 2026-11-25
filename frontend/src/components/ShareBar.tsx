import { useState } from 'react'
import { toHnComment } from '../hnComment'
import type { SavedAnswer } from '../types'

interface Props {
  shareId: string | null
  /** The answer's Markdown, for "Copy as HN comment". */
  answer: string
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
export function ShareBar({ shareId, answer, saved, onAskAgain }: Props) {
  // Which button last copied. Keyed by shareId in App, so it never carries
  // over to a new answer.
  const [copied, setCopied] = useState<'link' | 'comment' | null>(null)

  if (!shareId) return null
  const url = `${location.origin}/a/${shareId}`

  const copy = async (what: 'link' | 'comment') => {
    const text = what === 'link' ? url : toHnComment(answer, url)
    try {
      await navigator.clipboard.writeText(text)
      setCopied(what)
      setTimeout(() => setCopied(null), 2000)
    } catch {
      // No clipboard (insecure context, denied): show it to copy by hand.
      window.prompt('Copy this', text)
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
      <button
        type="button"
        className="share-copy"
        onClick={() => copy('comment')}
        title="The answer as plain text for a Hacker News comment: citations as numbered links at the end"
      >
        {copied === 'comment' ? '✓ Comment copied' : 'Copy as HN comment'}
      </button>
      <button type="button" className="share-copy" onClick={() => copy('link')}>
        {copied === 'link' ? '✓ Link copied' : 'Copy link'}
      </button>
    </div>
  )
}
