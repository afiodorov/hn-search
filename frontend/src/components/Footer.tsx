import { useState } from 'react'

const MCP_ADD = `claude mcp add --transport http hn-search ${location.origin}/mcp`

/** The machine-facing surface (`hn_search/api/agents.py`), where people who
 *  would use it can find it: an MCP server, a JSON API, and llms.txt. */
export function Footer() {
  const [copied, setCopied] = useState(false)
  const copy = () =>
    navigator.clipboard.writeText(MCP_ADD).then(
      () => {
        setCopied(true)
        setTimeout(() => setCopied(false), 2000)
      },
      () => {},
    )

  return (
    <footer className="footer">
      <p>
        <strong>Use it from your agent.</strong> An MCP server at{' '}
        <code>/mcp</code> (search, similar comments, exact keyword counts,
        cited answers); in Claude Code:
      </p>
      <button
        type="button"
        className="footer-cmd"
        onClick={copy}
        title="Copy to clipboard"
      >
        <code>{MCP_ADD}</code>
        <span className="footer-copy">{copied ? '✓ copied' : 'copy'}</span>
      </button>
      <p className="footer-links">
        <a href="/llms.txt">llms.txt</a>
        {' · '}
        <a href="/docs">JSON API</a>
        {' · '}
        <a
          href="https://github.com/afiodorov/hn-search"
          target="_blank"
          rel="noopener noreferrer"
        >
          Source on GitHub
        </a>
      </p>
    </footer>
  )
}
