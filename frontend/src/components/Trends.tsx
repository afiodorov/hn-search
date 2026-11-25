import { useCallback, useEffect, useMemo, useState } from 'react'
import type { FormEvent } from 'react'
import { fetchTrends } from '../api'
import type { TrendsData } from '../types'

const PER = 10_000
const PRESETS = [
  'rust, go, zig',
  'chatgpt, claude, gemini, deepseek',
  'kubernetes, docker, nomad',
  'remote work, return to office',
  'postgres, mysql, sqlite',
]

function termsFromUrl(): string {
  return new URLSearchParams(location.search).get('terms') ?? ''
}

function rate(count: number, rows: number): number {
  return rows ? (count * PER) / rows : 0
}

function fmtRate(v: number): string {
  return v >= 10 ? v.toFixed(0) : v.toFixed(1)
}

/** A round gridline step (1, 2 or 5 × 10^n), about four lines to `top`. */
function niceStep(top: number): number {
  if (top <= 0) return 1
  const raw = top / 4
  const mag = 10 ** Math.floor(Math.log10(raw))
  return [1, 2, 5, 10].find((m) => raw <= m * mag)! * mag
}

export function Trends() {
  const [input, setInput] = useState(termsFromUrl)
  const [data, setData] = useState<TrendsData | null>(null)
  const [error, setError] = useState<string | null>(null)
  // A link with terms starts out loading; the first fetch is the effect's.
  const [loading, setLoading] = useState(() => termsFromUrl().trim() !== '')
  const [copied, setCopied] = useState(false)

  // Only ever sets state once the fetch settles, so it is safe to start from
  // an effect; `load` is the event-handler version that resets first.
  const fetchInto = useCallback((raw: string) => {
    fetchTrends(raw).then(
      (d) => {
        setData(d)
        setLoading(false)
      },
      (e: Error) => {
        setError(e.message)
        setLoading(false)
      },
    )
  }, [])

  const load = useCallback(
    (raw: string) => {
      setError(null)
      if (!raw.trim()) {
        setData(null)
        setLoading(false)
        return
      }
      setLoading(true)
      fetchInto(raw)
    },
    [fetchInto],
  )

  useEffect(() => {
    const raw = termsFromUrl()
    if (raw.trim()) fetchInto(raw)
    const onPop = () => {
      setInput(termsFromUrl())
      load(termsFromUrl())
    }
    addEventListener('popstate', onPop)
    return () => removeEventListener('popstate', onPop)
  }, [fetchInto, load])

  const go = (raw: string) => {
    const terms = raw
      .split(',')
      .map((t) => t.trim())
      .filter(Boolean)
      .join(', ')
    if (!terms) return
    setInput(terms)
    history.pushState({}, '', `/trends?terms=${encodeURIComponent(terms)}`)
    load(terms)
  }

  const submit = (e: FormEvent) => {
    e.preventDefault()
    go(input)
  }

  const copy = () =>
    navigator.clipboard.writeText(location.href).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })

  return (
    <>
      <p className="tagline">
        How often Hacker News comments mention anything, month by month — exact
        counts from a full-text index of every comment since 2023.
      </p>

      <form className="search-bar" onSubmit={submit}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="rust, go, zig — up to five, comma-separated"
          aria-label="Terms to compare, comma-separated"
          autoFocus
        />
        <button type="submit" disabled={loading || !input.trim()}>
          {loading ? 'Counting…' : 'Compare'}
        </button>
      </form>

      <div className="presets">
        {PRESETS.map((p) => (
          <button
            key={p}
            type="button"
            className="preset"
            onClick={() => go(p)}
          >
            {p}
          </button>
        ))}
      </div>

      {error && <div className="error card">⚠ {error}</div>}

      {data && (
        <>
          <TrendChart data={data} />
          <div className="share-bar">
            <span className="share-note">
              Whole words, case-insensitive: “go” also counts the verb. The last
              month is partial; rates stay comparable.
            </span>
            <button type="button" className="share-copy" onClick={copy}>
              {copied ? '✓ Link copied' : 'Copy link'}
            </button>
          </div>
          <TrendTable data={data} />
        </>
      )}
    </>
  )
}

const H = 340
const M = { top: 16, right: 92, bottom: 32, left: 44 }

/** The element's rendered width, so the SVG is drawn at 1:1 and its text
 *  stays legible on a phone instead of being scaled down with the plot. */
function useWidth(fallback: number) {
  const [el, setEl] = useState<HTMLDivElement | null>(null)
  const [width, setWidth] = useState(fallback)
  useEffect(() => {
    if (!el) return
    const ro = new ResizeObserver(([entry]) =>
      setWidth(Math.max(300, Math.round(entry.contentRect.width))),
    )
    ro.observe(el)
    return () => ro.disconnect()
  }, [el])
  return [setEl, width] as const
}

function TrendChart({ data }: { data: TrendsData }) {
  const [hover, setHover] = useState<number | null>(null)
  const [plotRef, W] = useWidth(720)
  const { months, terms } = data
  const n = months.length

  const rates = useMemo(
    () => terms.map((_, i) => months.map((m) => rate(m.counts[i], m.rows))),
    [terms, months],
  )
  const max = Math.max(0, ...rates.flat())
  const step = niceStep(max)
  const ticks = Math.max(1, Math.ceil(max / step))
  const top = step * ticks

  const x = (j: number) =>
    M.left + (W - M.left - M.right) * (n > 1 ? j / (n - 1) : 0.5)
  const y = (v: number) => H - M.bottom - ((H - M.top - M.bottom) * v) / top

  // End labels, nudged apart so close lines don't print over each other.
  const ends = rates
    .map((r, i) => ({ i, y: y(r[n - 1] ?? 0) }))
    .sort((a, b) => a.y - b.y)
  for (let k = 1; k < ends.length; k++) {
    ends[k].y = Math.max(ends[k].y, ends[k - 1].y + 15)
  }

  const onMove = (e: React.PointerEvent<SVGRectElement>) => {
    const box = e.currentTarget.getBoundingClientRect()
    const px = ((e.clientX - box.left) / box.width) * W
    const j = Math.round(((px - M.left) / (W - M.left - M.right)) * (n - 1))
    setHover(Math.min(n - 1, Math.max(0, j)))
  }

  const hm = hover !== null ? months[hover] : null

  return (
    <figure className="trend card">
      <figcaption className="trend-legend">
        {terms.map((t, i) => (
          <span key={t.term} className="trend-key">
            <span className={`trend-swatch s${i + 1}`} />
            <strong>{t.term}</strong>
            <span className="trend-total">
              {t.count.toLocaleString()} comments
              {/* A first match in the archive's first month only means
                  the term predates the archive. */}
              {t.first &&
                t.first.timestamp.slice(0, 7) !== months[0]?.month && (
                  <>
                    {' · first '}
                    <a href={t.first.url} target="_blank" rel="noreferrer">
                      {t.first.timestamp.slice(0, 10)}
                    </a>
                  </>
                )}
            </span>
          </span>
        ))}
      </figcaption>

      <div className="trend-plot" ref={plotRef}>
        <svg
          viewBox={`0 0 ${W} ${H}`}
          role="img"
          aria-label={`Mentions per ${PER.toLocaleString()} comments by month: ${terms.map((t) => t.term).join(', ')}`}
        >
          {Array.from({ length: ticks + 1 }, (_, k) => (
            <g key={k} className="trend-grid">
              <line
                x1={M.left}
                x2={W - M.right}
                y1={y(step * k)}
                y2={y(step * k)}
              />
              <text x={M.left - 8} y={y(step * k)} dy="0.32em" textAnchor="end">
                {+(step * k).toFixed(2)}
              </text>
            </g>
          ))}
          {months.map((m, j) =>
            m.month.endsWith('-01') ? (
              <text
                key={m.month}
                className="trend-axis"
                x={x(j)}
                y={H - M.bottom + 20}
                textAnchor="middle"
              >
                {m.month.slice(0, 4)}
              </text>
            ) : null,
          )}

          {rates.map((r, i) => (
            <polyline
              key={terms[i].term}
              className={`trend-line s${i + 1}`}
              points={r.map((v, j) => `${x(j)},${y(v)}`).join(' ')}
            />
          ))}

          {ends.map(({ i, y: ly }) => (
            <text
              key={terms[i].term}
              className="trend-end"
              x={W - M.right + 8}
              y={ly}
              dy="0.32em"
            >
              {terms[i].term}
            </text>
          ))}

          {hover !== null && (
            <g>
              <line
                className="trend-crosshair"
                x1={x(hover)}
                x2={x(hover)}
                y1={M.top}
                y2={H - M.bottom}
              />
              {rates.map((r, i) => (
                <circle
                  key={terms[i].term}
                  className={`trend-dot s${i + 1}`}
                  cx={x(hover)}
                  cy={y(r[hover])}
                  r={4}
                />
              ))}
            </g>
          )}

          <rect
            x={M.left}
            y={M.top}
            width={W - M.left - M.right}
            height={H - M.top - M.bottom}
            fill="transparent"
            onPointerMove={onMove}
            onPointerLeave={() => setHover(null)}
          />
        </svg>

        {hm && hover !== null && (
          <div
            className="trend-tip card"
            style={{
              left: `${(x(hover) / W) * 100}%`,
              transform:
                hover > n / 2
                  ? 'translateX(calc(-100% - 12px))'
                  : 'translateX(12px)',
            }}
          >
            <div className="trend-tip-month">{hm.month}</div>
            {terms.map((t, i) => (
              <div key={t.term} className="trend-tip-row">
                <span className={`trend-swatch s${i + 1}`} />
                <span>{t.term}</span>
                <span className="trend-tip-val">
                  {hm.counts[i].toLocaleString()} ·{' '}
                  {fmtRate(rate(hm.counts[i], hm.rows))}/10k
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
      <div className="trend-unit">
        mentions per {PER.toLocaleString()} comments
      </div>
    </figure>
  )
}

function TrendTable({ data }: { data: TrendsData }) {
  return (
    <details className="sources">
      <summary>Table</summary>
      <div className="trend-table card">
        <table>
          <thead>
            <tr>
              <th>Month</th>
              {data.terms.map((t) => (
                <th key={t.term}>{t.term}</th>
              ))}
              <th>All comments</th>
            </tr>
          </thead>
          <tbody>
            {[...data.months].reverse().map((m) => (
              <tr key={m.month}>
                <td>{m.month}</td>
                {m.counts.map((c, i) => (
                  <td key={i}>
                    {c.toLocaleString()}{' '}
                    <span className="muted">({fmtRate(rate(c, m.rows))})</span>
                  </td>
                ))}
                <td>{m.rows.toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  )
}
