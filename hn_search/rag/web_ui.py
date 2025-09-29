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
    with gr.Blocks(
        title="🔎 Hacker News RAG Search",
        head="""
        <script>
        window.updateUrlWithSearch = function(query) {
            query = query || '';
            console.log('🔗 Updating URL with:', query);

            const params = new URLSearchParams();
            if (query && query.trim()) {
                params.set('q', query.trim());
            }

            const newUrl = params.toString() ?
                window.location.pathname + '?' + params.toString() :
                window.location.pathname;

            window.history.pushState({}, '', newUrl);
            console.log('🔗 Updated URL to:', newUrl);
        };
        </script>
        """,
    ) as demo:
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
            elem_id="query_input",
        )

        search_button = gr.Button("🔍 Search", variant="primary")

        progress_output = gr.Textbox(
            label="Progress", lines=5, interactive=False, value="Ready to search..."
        )

        answer_output = gr.Markdown(label="💬 Answer", value="")

        with gr.Accordion("📚 Source Comments", open=False):
            sources_output = gr.Markdown(value="")

        # Hidden HTML component for JavaScript execution
        html_output = gr.HTML(visible=False)

        def search_and_update_url(query: str):
            """Search and update URL in browser."""
            query = query or ""
            for result in hn_search_rag(query):
                yield result + ("",)  # Add empty string for HTML output

        # Set up search action
        search_button.click(
            fn=search_and_update_url,
            inputs=[query_input],
            outputs=[progress_output, answer_output, sources_output, html_output],
            show_progress="full",
        )

        # Add JavaScript click handler to update URL
        search_button.click(
            fn=None,
            inputs=[query_input],
            outputs=[],
            js="(query) => { console.log('Updating URL:', query); window.updateUrlWithSearch(query); }",
        )

        # Handle URL parameters and auto-search on load
        def load_and_search_from_url(request: gr.Request):
            """Load query parameters from URL and auto-search if present."""
            if request:
                query = request.query_params.get("q", "")
                print(f"📎 Loading from URL: q='{query}'")

                if query:
                    print(f"🔍 Auto-searching for: {query}")
                    # Start the search immediately and return results
                    results = list(hn_search_rag(query))
                    if results:
                        # Get the final result
                        progress, answer, sources = results[-1]
                        return query, progress, answer, sources, ""
                    else:
                        return query, "Search completed", "", "", ""
                else:
                    return "", "Ready to search...", "", "", ""
            return "", "Ready to search...", "", "", ""

        # Set up load handler to populate fields and auto-search from URL
        demo.load(
            fn=load_and_search_from_url,
            inputs=[],
            outputs=[
                query_input,
                progress_output,
                answer_output,
                sources_output,
                html_output,
            ],
        )

    return demo


demo = create_interface()

if __name__ == "__main__":
    print("🔎 Starting HN RAG Search Web Interface...")
    print("✨ Features:")
    print("  • URL parameter support: ?q=query")
    print("  • Auto-search from URL parameters")
    print()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
    )
