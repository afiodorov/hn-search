import { useCallback, useEffect, useState } from 'react'
import { fetchMe, logout } from './api'
import { Answer } from './components/Answer'
import { Freshness } from './components/Freshness'
import { ProgressLog } from './components/ProgressLog'
import { RecentQueries } from './components/RecentQueries'
import { SearchBar } from './components/SearchBar'
import { Sources } from './components/Sources'
import { useRecentQueries } from './hooks/useRecentQueries'
import { useSearch } from './hooks/useSearch'
import { useTheme } from './hooks/useTheme'
import type { Me } from './types'

const ANONYMOUS: Me = { login: null, admin: false, configured: false }

function queryFromUrl(): string {
  return new URLSearchParams(location.search).get('q') ?? ''
}

export default function App() {
  const { status, steps, answer, sources, error, run } = useSearch()
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
        history.pushState({}, '', `?q=${encodeURIComponent(query)}`)
      }
      run(query)
      window.scrollTo({ top: 0 })
    },
    [run],
  )

  useEffect(() => {
    const q = queryFromUrl()
    if (q) runSearch(q, false)

    const onPop = () => {
      const q = queryFromUrl()
      if (q) runSearch(q, false)
    }
    addEventListener('popstate', onPop)
    return () => removeEventListener('popstate', onPop)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (status === 'done') refresh()
  }, [status, refresh])

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
