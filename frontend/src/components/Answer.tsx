import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { SearchStatus } from '../types'

interface Props {
  answer: string
  status: SearchStatus
}

export function Answer({ answer, status }: Props) {
  if (!answer) return null

  return (
    <div className={`answer card${status === 'running' ? ' streaming' : ''}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: (props) => (
            <a {...props} target="_blank" rel="noopener noreferrer" />
          ),
        }}
      >
        {answer}
      </ReactMarkdown>
    </div>
  )
}
