import { useCallback, useEffect, useState } from 'react'
import { fetchMe, logout } from './api'
import { Answer } from './components/Answer'
import { Freshness } from './components/Freshness'
import { ProgressLog } from './components/ProgressLog'
import { RecentQueries } from './components/RecentQueries'
import { SearchBar } from './components/SearchBar'
import { ShareBar } from './components/ShareBar'
import { Sources } from './components/Sources'
import { useRecentQueries } from './hooks/useRecentQueries'
import { useSearch } from './hooks/useSearch'
import { useTheme } from './hooks/useTheme'
import type { Me } from './types'

const ANONYMOUS: Me = { login: null, admin: false, configured: false }

function queryFromUrl(): string {
  return new URLSearchParams(location.search).get('q') ?? ''
}

/** The id in a /a/<id> permalink, or null on any other page. */
function savedIdFromUrl(): string | null {
  return location.pathname.match(/^\/a\/([A-Za-z0-9_-]{8})\/?$/)?.[1] ?? null
}

export default function App() {
  const { status, steps, answer, sources, error, shareId, saved, run, load } =
    useSearch()
  const { queries, refresh, remove } = useRecentQueries(25)
  const { theme, toggle } = useTheme()
  const [input, setInput] = useState(queryFromUrl)

  // Who is looking. Only admins get delete buttons; everyone else sees the
  // same list read-only, and a sign-in link when the deployment offers one.
  const [me, setMe] = useState<Me>(ANONYMOUS)
  useEffect(() => {
    fetchMe().then(setMe, () => setMe(ANONYMOUS))
  }, [])
  const signOut = () =>
    logout().then(() => setMe({ ...me, login: null, admin: false }))

  const runSearch = useCallback(
    (q: string, push = true) => {
      const query = q.trim()
      if (!query) return
      setInput(query)
      if (push) {
        history.pushState({}, '', `/?q=${encodeURIComponent(query)}`)
      }
      run(query)
      window.scrollTo({ top: 0 })
    },
    [run],
  )

  // What the address bar asks for: a saved answer, a question to run, or
  // nothing.
  const followUrl = useCallback(() => {
    const id = savedIdFromUrl()
    if (id) {
      load(id).then((snap) => snap && setInput(snap.query))
      return
    }
    const q = queryFromUrl()
    if (q) runSearch(q, false)
  }, [load, runSearch])

  useEffect(() => {
    followUrl()
    const onPop = () => followUrl()
    addEventListener('popstate', onPop)
    return () => removeEventListener('popstate', onPop)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (status === 'done') refresh()
  }, [status, refresh])

  // Once a live answer is saved, the address bar becomes its permalink, so
  // what people copy from it is the answer they saw rather than a question
  // that reruns. replaceState: Back still leaves the page, as it would have.
  useEffect(() => {
    if (
      status === 'done' &&
      shareId &&
      !saved &&
      savedIdFromUrl() !== shareId
    ) {
      history.replaceState({}, '', `/a/${shareId}`)
    }
  }, [status, shareId, saved])

  return (
    <div className="layout">
      <header className="header">
        <a className="brand" href="/">
          <span className="logo">Y</span>
          <h1>HN Search</h1>
        </a>
        <button
          type="button"
          className="theme-toggle"
          onClick={toggle}
          aria-label="Toggle dark mode"
          title="Toggle dark mode"
        >
          {theme === 'dark' ? '☀' : '☾'}
        </button>
      </header>

      <main>
        <p className="tagline">
          Ask questions about Hacker News discussions, get answers with
          sources.
        </p>
        <Freshness />

        <SearchBar
          value={input}
          running={status === 'running'}
          onChange={setInput}
          onSearch={runSearch}
        />

        <ProgressLog steps={steps} status={status} />

        {error && <div className="error card">⚠ {error}</div>}

        <Answer answer={answer} status={status} />

        {status === 'done' && (
          <ShareBar
            key={shareId}
            shareId={shareId}
            saved={saved}
            onAskAgain={runSearch}
          />
        )}

        <Sources sources={sources} />
      </main>

      <RecentQueries
        queries={queries}
        me={me}
        onSelect={runSearch}
        onDelete={remove}
        onSignOut={signOut}
      />
    </div>
  )
}
