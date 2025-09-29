import os

import gradio as gr

from .graph import create_rag_workflow


def hn_search_rag(query: str):
    if not query.strip():
        yield "Please enter a question.", "", ""
        return

    app = create_rag_workflow()
    initial_state = {"query": query}

    progress_log = []
    current_sources = ""
    current_answer = ""

    try:
        progress_log.append(f"🔍 Searching for: {query}")
        yield "\n".join(progress_log), "", ""

        for event in app.stream(initial_state):
            node_name = list(event.keys())[0]
            result_state = event[node_name]

            if result_state is None:
                continue

            node_messages = {
                "retrieve": "📚 Retrieving relevant HN comments...",
                "answer": "🤖 Generating answer with DeepSeek...",
            }

            if node_name in node_messages:
                progress_log.append(node_messages[node_name])
                yield "\n".join(progress_log), current_answer, current_sources

            if "search_results" in result_state and result_state["search_results"]:
                sources = []
                for i, r in enumerate(result_state["search_results"], 1):
                    hn_link = f"https://news.ycombinator.com/item?id={r['id']}"
                    sources.append(
                        f"**[{i}]** [{r['author']}]({hn_link}) ({r['timestamp']})\n\n{r['text'][:200]}..."
                    )
                current_sources = "\n\n---\n\n".join(sources)
                yield "\n".join(progress_log), current_answer, current_sources

            if "answer" in result_state:
                current_answer = result_state["answer"]
                yield "\n".join(progress_log), current_answer, current_sources

            if "error_message" in result_state:
                error_msg = f"❌ Error: {result_state['error_message']}"
                progress_log.append(error_msg)
                yield "\n".join(progress_log), "", ""
                return

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        progress_log.append(error_msg)
        yield "\n".join(progress_log), current_answer, current_sources
        return

    progress_log.append("✅ Complete!")
    yield "\n".join(progress_log), current_answer, current_sources


def create_interface():
    with gr.Blocks(title="🔎 Hacker News RAG Search") as demo:
        gr.Markdown(
            """
            # 🔎 Hacker News RAG Search

            Ask questions about Hacker News discussions and get AI-powered answers!
            """
        )

        query_input = gr.Textbox(
            label="Your Question",
            placeholder="What do people think about Rust vs Go?",
            lines=2,
        )

        search_button = gr.Button("🔍 Search", variant="primary")

        progress_output = gr.Textbox(
            label="Progress", lines=5, interactive=False, value="Ready to search..."
        )

        answer_output = gr.Markdown(label="💬 Answer", value="")

        with gr.Accordion("📚 Source Comments", open=False):
            sources_output = gr.Markdown(value="")

        search_button.click(
            fn=hn_search_rag,
            inputs=[query_input],
            outputs=[progress_output, answer_output, sources_output],
            show_progress="full",
        )

    return demo


demo = create_interface()

if __name__ == "__main__":
    print("🔎 Starting HN RAG Search Web Interface...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
    )
