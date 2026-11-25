/**
 * An answer's Markdown as a Hacker News comment.
 *
 * HN comments are nearly plain text (news.ycombinator.com/formatdoc): blank
 * lines separate paragraphs and a single newline does not break a line;
 * `*text*` is italics; lines indented two spaces are code, in monospace;
 * URLs are linked automatically and there is no link syntax. So:
 *
 * - citations `[[3]](url)` become `[3]`, listed with their URLs at the end,
 *   numbered as in the answer;
 * - any other link `[text](url)` keeps its text, with the URL after it;
 * - **bold** becomes *italics*, the only emphasis there is;
 * - headings, list items and quote lines each become their own paragraph,
 *   since a single newline would run them together;
 * - tables become an indented (monospace) block with aligned columns;
 * - inline code loses its backticks.
 */
export function toHnComment(markdown: string, permalink?: string): string {
  const notes = new Map<string, string>()

  const inline = (text: string): string =>
    text
      .replace(/\[\[(\d+)\]\]\(([^)\s]+)\)/g, (_, n: string, url: string) => {
        if (!notes.has(n)) notes.set(n, url)
        return `[${n}]`
      })
      .replace(
        /!?\[([^\]]+)\]\(([^)\s]+)\)/g,
        (_, text: string, url: string) =>
          text === url ? url : `${text} (${url})`,
      )
      .replace(/\*\*(\S(?:.*?\S)?)\*\*/g, '*$1*')
      .replace(/__(\S(?:.*?\S)?)__/g, '*$1*')
      .replace(/`([^`]+)`/g, '$1')

  const paragraphs: string[] = []
  let para: string[] = []
  const flush = () => {
    if (para.length) paragraphs.push(para.join(' '))
    para = []
  }

  const lines = markdown.replace(/\r\n?/g, '\n').split('\n')
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    const trimmed = line.trim()

    if (!trimmed) {
      flush()
      continue
    }

    // A fenced code block: kept verbatim, indented two spaces.
    if (trimmed.startsWith('```')) {
      flush()
      const code: string[] = []
      while (++i < lines.length && !lines[i].trim().startsWith('```')) {
        code.push('  ' + lines[i])
      }
      paragraphs.push(code.join('\n'))
      continue
    }

    // A table: consecutive |-rows, separator row dropped, columns padded.
    if (trimmed.startsWith('|')) {
      flush()
      const rows: string[][] = []
      for (; i < lines.length && lines[i].trim().startsWith('|'); i++) {
        const cells = lines[i]
          .trim()
          .replace(/^\||\|$/g, '')
          .split('|')
          .map((c) => inline(c.trim()).replace(/\*/g, ''))
        if (!cells.every((c) => /^:?-+:?$/.test(c))) rows.push(cells)
      }
      i--
      const widths = rows[0].map((_, c) =>
        Math.max(...rows.map((r) => (r[c] ?? '').length)),
      )
      paragraphs.push(
        rows
          .map(
            (r) =>
              '  ' +
              r
                .map((c, k) => c.padEnd(widths[k]))
                .join('  ')
                .trimEnd(),
          )
          .join('\n'),
      )
      continue
    }

    const heading = trimmed.match(/^#{1,6}\s+(.*)$/)
    if (heading) {
      flush()
      paragraphs.push(inline(heading[1]).replace(/^\*(.*)\*$/, '$1'))
      continue
    }

    const item = trimmed.match(/^([-*+]|\d+[.)])\s+(.*)$/)
    if (item) {
      flush()
      const bullet = /\d/.test(item[1]) ? item[1] : '-'
      para = [`${bullet} ${inline(item[2])}`]
      continue
    }

    if (trimmed.startsWith('>')) {
      flush()
      para = ['> ' + inline(trimmed.replace(/^>\s?/, ''))]
      continue
    }

    para.push(inline(trimmed))
  }
  flush()

  if (notes.size) {
    paragraphs.push(
      [...notes]
        .sort(([a], [b]) => Number(a) - Number(b))
        .map(([n, url]) => `[${n}] ${url}`)
        .join('\n\n'),
    )
  }
  if (permalink) paragraphs.push(`(via ${permalink})`)
  return paragraphs.join('\n\n') + '\n'
}
