import type { FormEvent } from 'react'

interface Props {
  value: string
  running: boolean
  onChange: (value: string) => void
  onSearch: (query: string) => void
}

export function SearchBar({ value, running, onChange, onSearch }: Props) {
  const submit = (e: FormEvent) => {
    e.preventDefault()
    onSearch(value)
  }

  return (
    <form className="search-bar" onSubmit={submit}>
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="What do people think about Rust vs Go?"
        aria-label="Your question"
        autoFocus
      />
      <button type="submit" disabled={running || !value.trim()}>
        {running ? 'Searching…' : 'Search'}
      </button>
    </form>
  )
}
