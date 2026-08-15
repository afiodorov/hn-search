import argparse
import sys

from .agent import AgentState, create_agent_workflow


def main():
    parser = argparse.ArgumentParser(
        description="Ask questions about Hacker News discussions using RAG"
    )
    parser.add_argument("query", type=str, help="Your question about HN discussions")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("🔎 Hacker News RAG Search")
    print("=" * 70 + "\n")

    app = create_agent_workflow()

    initial_state = AgentState(
        messages=[],
        query=args.query,
        tool_calls=[],
        time_after=None,
        time_before=None,
        sources=[],
        parent_texts={},
        answer="",
    )

    try:
        final_state = app.invoke(initial_state)

        print("\n" + "-" * 70)
        print("💬 Answer:")
        print("-" * 70 + "\n")
        print(final_state["answer"])

        print("\n" + "-" * 70)
        print(f"📚 Based on {len(final_state['sources'])} HN comments/articles")
        print("-" * 70 + "\n")

    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
