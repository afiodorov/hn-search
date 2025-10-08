import os
import time

import gradio as gr

from hn_search.cache_config import redis_client
from hn_search.job_manager import JobManager
from hn_search.logging_config import get_logger

from .graph import create_rag_workflow

# Initialize job manager
job_manager = JobManager(redis_client)

logger = get_logger(__name__)


def hn_search_rag(query: str):
    if not query.strip():
        yield "Please enter a question.", "", "", ""
        return

    # Try to claim this job
    claimed, job_id = job_manager.try_claim_job(query)

    # Track query immediately so it shows up in recent queries
    job_manager.track_recent_query(query)

    progress_log = []
    current_sources = ""
    current_answer = ""
    current_recent = _format_recent_queries()
    logger.debug(f"Recent queries at start: {current_recent[:100]}...")

    if not claimed:
        # Another request is processing this query - poll for progress
        progress_log.append(f"🔍 Searching for: {query}")
        yield "\n".join(progress_log), "", "", current_recent

        # Poll for progress updates from the processing job
        timeout = job_manager.max_poll_time
        start_time = time.time()
        last_progress = ""

        while time.time() - start_time < timeout:
            # Check progress
            current_progress = job_manager.get_progress(job_id)
            if current_progress and current_progress != last_progress:
                # Mirror the progress from the processing job
                last_progress = current_progress
                yield current_progress, "", "", current_recent

            # Check if complete
            result = job_manager.get_result(job_id)
            if result:
                # Job completed - show final state
                final_progress = f"🔍 Searching for: {query}\n📚 Retrieving relevant HN comments...\n🤖 Generating answer with DeepSeek...\n✅ Complete!"
                # Refresh recent queries after completion
                current_recent = _format_recent_queries()
                yield (
                    final_progress,
                    result.get("answer", ""),
                    result.get("sources", ""),
                    current_recent,
                )
                return

            time.sleep(0.5)

        # Timeout - try to claim and process ourselves
        progress_log = [f"🔍 Searching for: {query}"]
        progress_log.append(
            "⚠️ Timeout waiting for other request, processing query now..."
        )
        yield "\n".join(progress_log), "", "", current_recent
        claimed, job_id = job_manager.try_claim_job(query)
        if not claimed:
            yield (
                "\n".join(progress_log) + "\n❌ Unable to process query",
                "",
                "",
                current_recent,
            )
            return

    # We claimed the job - process it
    app = create_rag_workflow()
    initial_state = {"query": query}

    try:
        progress_log.append(f"🔍 Searching for: {query}")
        job_manager.update_progress(job_id, "\n".join(progress_log))
        yield "\n".join(progress_log), "", "", current_recent

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
                job_manager.update_progress(job_id, "\n".join(progress_log))
                yield (
                    "\n".join(progress_log),
                    current_answer,
                    current_sources,
                    current_recent,
                )

            if "search_results" in result_state and result_state["search_results"]:
                sources = []
                for i, r in enumerate(result_state["search_results"], 1):
                    hn_link = f"https://news.ycombinator.com/item?id={r['id']}"
                    sources.append(
                        f"**[{i}]** [{r['author']}]({hn_link}) ({r['timestamp']})\n\n{r['text'][:200]}..."
                    )
                current_sources = "\n\n---\n\n".join(sources)
                yield (
                    "\n".join(progress_log),
                    current_answer,
                    current_sources,
                    current_recent,
                )

            if "answer" in result_state:
                current_answer = result_state["answer"]
                yield (
                    "\n".join(progress_log),
                    current_answer,
                    current_sources,
                    current_recent,
                )

            if "error_message" in result_state:
                error_msg = f"❌ Error: {result_state['error_message']}"
                progress_log.append(error_msg)
                job_manager.store_error(job_id, result_state["error_message"])
                yield "\n".join(progress_log), "", "", current_recent
                return

        # Store successful result for other waiting requests
        job_manager.store_result(
            job_id, {"answer": current_answer, "sources": current_sources}
        )

        # Refresh recent queries after storing result
        current_recent = _format_recent_queries()

        progress_log.append("✅ Complete!")
        yield "\n".join(progress_log), current_answer, current_sources, current_recent

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        progress_log.append(error_msg)
        job_manager.store_error(job_id, str(e))
        yield "\n".join(progress_log), current_answer, current_sources, current_recent
        return


def _format_recent_queries() -> str:
    """Format recent queries as markdown."""
    recent = job_manager.get_recent_queries(limit=10)

    if not recent:
        return "*No recent queries yet*"

    lines = []
    for item in recent:
        query = item["query"]
        time_ago = item["time_ago"]
        # Make queries clickable (URL-encoded)
        import urllib.parse

        encoded_query = urllib.parse.quote(query)
        lines.append(f"- **{time_ago}**: [{query}](?q={encoded_query})")

    return "\n".join(lines)


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

        with gr.Accordion("🕒 Recent Queries", open=False):
            recent_output = gr.Markdown(value="*Loading recent queries...*")

        # Auto-refresh recent queries every 3 seconds
        timer = gr.Timer(value=3, active=True)
        timer.tick(fn=_format_recent_queries, inputs=[], outputs=[recent_output])

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
            outputs=[
                progress_output,
                answer_output,
                sources_output,
                recent_output,
                html_output,
            ],
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
            recent_queries = _format_recent_queries()

            if request:
                query = request.query_params.get("q", "")
                logger.info(f"📎 Loading from URL: q='{query}'")

                if query:
                    logger.info(f"🔍 Auto-searching for: {query}")
                    # Yield initial state with query and recent queries immediately
                    yield query, "Ready to search...", "", "", recent_queries, ""

                    # Then stream search results
                    for progress, answer, sources, recent in hn_search_rag(query):
                        yield query, progress, answer, sources, recent, ""
                    return

            # No query - yield once to update recent queries immediately without loading state
            yield "", "Ready to search...", "", "", recent_queries, ""

        # Set up load handler to populate fields and auto-search from URL
        demo.load(
            fn=load_and_search_from_url,
            inputs=[],
            outputs=[
                query_input,
                progress_output,
                answer_output,
                sources_output,
                recent_output,
                html_output,
            ],
        )

    return demo


demo = create_interface()

if __name__ == "__main__":
    logger.info("🔎 Starting HN RAG Search Web Interface...")
    logger.info("✨ Features:")
    logger.info("  • URL parameter support: ?q=query")
    logger.info("  • Auto-search from URL parameters")
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
    )
