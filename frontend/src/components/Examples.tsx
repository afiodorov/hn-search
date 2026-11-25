interface Props {
  onSelect: (query: string) => void
}

// One per thing the planner does that a search box doesn't suggest on its
// own: exact counts over every comment, first mentions, opinion over time,
// and plain "what does HN think".
const EXAMPLES = [
  'How many comments mention ChatGPT each month?',
  'When did HN first mention DeepSeek?',
  "How has HN's view of remote work changed since 2023?",
  'What do people regret about moving to microservices?',
  'Best self-hosted alternatives to Google Photos?',
  'What does HN think about SQLite in production?',
]

/** Shown on the empty page, so a first visit has something to click. */
export function Examples({ onSelect }: Props) {
  return (
    <section className="examples" aria-label="Example questions">
      <h2>Try asking</h2>
      <div className="example-list">
        {EXAMPLES.map((q) => (
          <button
            key={q}
            type="button"
            className="example card"
            onClick={() => onSelect(q)}
          >
            {q}
          </button>
        ))}
      </div>
    </section>
  )
}
