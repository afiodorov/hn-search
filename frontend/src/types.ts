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

/** GET /auth/me. `login` is null when not signed in; `configured` is false when
 *  the deployment has no GitHub OAuth app, in which case there is no sign-in
 *  to offer and nobody can delete. */
export interface Me {
  login: string | null
  admin: boolean
  configured: boolean
}

export interface Stats {
  count: number
  max_id: number
  latest_timestamp: string
}
