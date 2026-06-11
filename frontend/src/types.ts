export interface ProgressEvent {
  type: 'progress'
  step: string
  label: string
  status: 'start' | 'done'
  ms: number | null
  hit: boolean | null
}

export interface Source {
  id: string
  author: string
  timestamp: string
  type: string
  text: string
  url: string
  distance: number
}

export interface RecentQuery {
  query: string
  timestamp: string
  time_ago: string
}

export type SearchStatus = 'idle' | 'running' | 'done' | 'error'
