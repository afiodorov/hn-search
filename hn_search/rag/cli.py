import argparse
import sys

from .graph import create_rag_workflow


def main():
    parser = argparse.ArgumentParser(
        description="Ask questions about Hacker News discussions using RAG"
    )
    parser.add_argument("query", type=str, help="Your question about HN discussions")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("🔎 Hacker News RAG Search")
    print("=" * 70 + "\n")

    app = create_rag_workflow()

    initial_state = {"query": args.query}

    try:
        final_state = app.invoke(initial_state)

        if final_state.get("error_message"):
            print(f"\n❌ Error: {final_state['error_message']}\n")
            sys.exit(1)

        print("\n" + "-" * 70)
        print("💬 Answer:")
        print("-" * 70 + "\n")
        print(final_state["answer"])

        print("\n" + "-" * 70)
        print(f"📚 Based on {len(final_state['search_results'])} HN comments/articles")
        print("-" * 70 + "\n")

    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
