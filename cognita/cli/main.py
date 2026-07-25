"""Command-line interface entry point for Cognita RAG.

This module exposes a Click-based CLI that uses Rich for pretty terminal
output. Each command wraps its asynchronous work in a single ``asyncio.run``
call (via :func:`run_async`) so that event-loop-bound resources such as the
Qdrant and OpenAI async clients remain valid for the lifetime of the command.
"""
from __future__ import annotations

import asyncio
import os

import click
import uvicorn
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from cognita.config import Settings, get_settings
from cognita.core.llm import get_llm
from cognita.core.models import Citation, DocumentStatus, SearchResult
from cognita.core.vectorstore import get_vectorstore
from cognita.generation.generator import RAGGenerator
from cognita.generation.memory import ConversationMemory
from cognita.ingestion.pipeline import IngestionPipeline
from cognita.observability.logging import get_logger, setup_logging
from cognita.retrieval.hybrid import HybridRetriever
from cognita.retrieval.reranker import CrossEncoderReranker

# A single Rich Console instance drives all CLI output.
console: Console = Console()
logger = get_logger(__name__)


def run_async(coro):
    """Run an awaitable coroutine synchronously using ``asyncio.run``."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_banner(console: Console) -> None:
    """Print a styled banner when the CLI starts."""
    title = Text("Cognita RAG", style="bold cyan", justify="center")
    subtitle = Text("Production-grade RAG Knowledge Agent", style="dim", justify="center")
    content = Text.assemble(title, "\n", subtitle)
    console.print(Panel(content, border_style="cyan", padding=(1, 4)))


def _get_settings_safe() -> Settings | None:
    """Return cached settings, or ``None`` if they cannot be loaded."""
    try:
        return get_settings()
    except Exception as exc:
        logger.debug("Failed to load settings", error=str(exc))
        return None


def _resolve_top_k(top_k: int | None) -> int:
    """Resolve the effective ``top_k`` from the option or settings."""
    if top_k is not None:
        return top_k
    settings = _get_settings_safe()
    if settings is not None:
        for attr in ("retrieval_top_k", "top_k", "default_top_k"):
            value = getattr(settings, attr, None)
            if value is not None:
                return int(value)
    return 5


def _resolve_host(host: str | None) -> str:
    """Resolve the API host from the option or settings."""
    if host:
        return host
    settings = _get_settings_safe()
    if settings is not None:
        for attr in ("api_host", "host", "server_host"):
            value = getattr(settings, attr, None)
            if value:
                return str(value)
    return "0.0.0.0"


def _resolve_port(port: int | None) -> int:
    """Resolve the API port from the option or settings."""
    if port is not None:
        return port
    settings = _get_settings_safe()
    if settings is not None:
        for attr in ("api_port", "port", "server_port"):
            value = getattr(settings, attr, None)
            if value is not None:
                return int(value)
    return 8000


def _format_score(score: float | None) -> str:
    """Format a similarity score for display."""
    if score is None:
        return "-"
    try:
        return f"{float(score):.3f}"
    except (TypeError, ValueError):
        return str(score)


def _snippet(text: str | None, length: int = 80) -> str:
    """Collapse whitespace and truncate a snippet for table display."""
    if not text:
        return ""
    collapsed = " ".join(text.split())
    if len(collapsed) <= length:
        return collapsed
    return collapsed[:length].rstrip() + "..."


def _display_citations(citations: list[Citation]) -> None:
    """Render a Rich table of citations."""
    if not citations:
        console.print("[dim]No citations available.[/dim]")
        return
    table = Table(title="Sources", show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=3)
    table.add_column("Document", style="cyan", overflow="fold")
    table.add_column("Source", overflow="fold")
    table.add_column("Chunk", justify="right", width=5)
    table.add_column("Score", justify="right", width=6)
    table.add_column("Snippet", overflow="fold")
    for idx, citation in enumerate(citations, 1):
        table.add_row(
            str(idx),
            citation.document_title or "-",
            citation.source or "-",
            str(citation.chunk_index) if citation.chunk_index is not None else "-",
            _format_score(citation.score),
            _snippet(citation.content_snippet),
        )
    console.print(table)


def _display_search_results(results: list[SearchResult]) -> None:
    """Render a Rich table of retrieved search results as sources."""
    if not results:
        console.print("[dim]No sources retrieved.[/dim]")
        return
    table = Table(title="Sources", show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=3)
    table.add_column("Document", style="cyan", overflow="fold")
    table.add_column("Source", overflow="fold")
    table.add_column("Score", justify="right", width=6)
    table.add_column("Snippet", overflow="fold")
    for idx, result in enumerate(results, 1):
        content = getattr(result.chunk, "content", "") or ""
        table.add_row(
            str(idx),
            result.source_title or "-",
            result.source_path or "-",
            _format_score(result.score),
            _snippet(content),
        )
    console.print(table)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
def cli() -> None:
    """Cognita RAG - Production-grade RAG Knowledge Agent CLI.

    Ingest documents, query the knowledge base, and chat with a
    retrieval-augmented generation agent.
    """
    # Configure structured logging before any operations run. A failure here
    # (e.g. missing settings) must not prevent the CLI from producing a
    # user-friendly error message.
    try:
        setup_logging()
    except Exception as exc:
        logger.warning("Logging setup failed; continuing with defaults", error=str(exc))
    _print_banner(console)


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------

@cli.command()
def init() -> None:
    """Initialize the system (create vectorstore collection, load embedding model)."""
    console.print(Panel("[cyan]Initializing Cognita RAG system...[/cyan]", border_style="cyan"))
    try:

        async def _run_init() -> None:
            pipeline = IngestionPipeline()
            await pipeline.initialize()
            # Verify the vectorstore is reachable after collection creation.
            vectorstore = get_vectorstore()
            await vectorstore.health_check()

        with console.status("[cyan]Creating vectorstore collection and loading models...[/cyan]"):
            run_async(_run_init())

        console.print("[bold green]\u2713 System initialized successfully.[/bold green]")
        console.print("  \u2022 Vectorstore collection ready")
        console.print("  \u2022 Embedding model loaded")
        logger.info("System initialization completed")
    except Exception as exc:
        console.print(f"[bold red]\u2717 Initialization failed:[/bold red] {exc}")
        logger.error("Initialization failed", error=str(exc), exc_info=True)
        raise click.exceptions.Exit(1) from exc


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("path", type=click.Path(exists=True))
def ingest(path: str) -> None:
    """Ingest a file or directory into the knowledge base."""
    is_dir = os.path.isdir(path)
    label = "directory" if is_dir else "file"
    console.print(Panel(f"[cyan]Ingesting {label}:[/cyan] {path}", border_style="cyan"))

    try:

        async def _run_ingest():
            pipeline = IngestionPipeline()
            if is_dir:
                return await pipeline.ingest_directory(path)
            return [await pipeline.ingest_file(path)]

        with console.status(f"[cyan]Ingesting {label}...[/cyan]"):
            results = run_async(_run_ingest())
    except Exception as exc:
        console.print(f"[bold red]\u2717 Ingestion failed:[/bold red] {exc}")
        logger.error("Ingestion failed", path=path, error=str(exc), exc_info=True)
        raise click.exceptions.Exit(1) from exc

    if not results:
        console.print("[yellow]\u26a0 No files were ingested.[/yellow]")
        return

    table = Table(title="Ingestion Results", show_header=True, header_style="bold cyan")
    table.add_column("File", style="cyan", overflow="fold")
    table.add_column("Chunks", justify="right", width=7)
    table.add_column("Status", width=12)
    table.add_column("Latency (ms)", justify="right", width=12)

    total_chunks = 0
    succeeded = 0
    for res in results:
        if res.status == DocumentStatus.INDEXED:
            status_cell = "[green]\u2713 indexed[/green]"
            succeeded += 1
        elif res.status == DocumentStatus.FAILED:
            status_cell = "[red]\u2717 failed[/red]"
        else:
            status_cell = f"[yellow]{res.status.value}[/yellow]"
        chunks = res.chunks_indexed if res.chunks_indexed is not None else 0
        total_chunks += chunks or 0
        table.add_row(
            res.title or os.path.basename(path),
            str(chunks),
            status_cell,
            str(res.latency_ms) if res.latency_ms is not None else "-",
        )

    console.print(table)
    console.print(
        f"[bold green]\u2713 Done.[/bold green] "
        f"{succeeded}/{len(results)} file(s) ingested, {total_chunks} chunk(s) indexed."
    )
    # Surface any per-file errors beneath the summary.
    for res in results:
        if res.error:
            console.print(f"[red]  \u2022 {res.title or 'unknown'}: {res.error}[/red]")


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("question")
@click.option("--top-k", "top_k", type=int, default=None, help="Number of chunks to retrieve.")
@click.option("--thinking", is_flag=True, help="Enable thinking mode for the generator.")
@click.option(
    "--show-sources", "show_sources", is_flag=True, help="Show retrieved source documents."
)
def query(question: str, top_k: int | None, thinking: bool, show_sources: bool) -> None:
    """Ask a question and get a RAG-grounded answer."""
    resolved_top_k = _resolve_top_k(top_k)

    try:
        retriever = HybridRetriever()
        reranker = CrossEncoderReranker()
        generator = RAGGenerator()

        async def _run_query():
            with console.status("[cyan]Processing your question...[/cyan]") as status:
                status.update("[cyan]Retrieving relevant documents...[/cyan]")
                results = await retriever.retrieve(question, top_k=resolved_top_k)
                if not results:
                    return None, []
                status.update("[cyan]Reranking results...[/cyan]")
                results = await reranker.rerank(question, results, top_k=resolved_top_k)
                status.update("[cyan]Generating answer...[/cyan]")
                response = await generator.generate(
                    question, results, thinking=thinking
                )
            return response, results

        response, results = run_async(_run_query())
    except Exception as exc:
        console.print(f"[bold red]\u2717 Query failed:[/bold red] {exc}")
        logger.error("Query failed", error=str(exc), exc_info=True)
        raise click.exceptions.Exit(1) from exc

    if response is None:
        console.print("[yellow]\u26a0 No relevant documents found for your question.[/yellow]")
        return

    # Render the answer as Markdown.
    console.print(Panel(Markdown(response.answer or ""), title="Answer", border_style="green"))

    # Optional thinking trace.
    if thinking and getattr(response, "thinking", None):
        console.print(Panel(Markdown(response.thinking), title="Thinking", border_style="yellow"))

    # Optionally show source documents.
    if show_sources:
        citations = getattr(response, "citations", None) or []
        if citations:
            _display_citations(citations)
        else:
            _display_search_results(results)

    # Usage / latency footer.
    usage = getattr(response, "usage", None)
    model = getattr(response, "model", "") or "-"
    latency = getattr(response, "latency_ms", 0)
    if usage:
        console.print(
            f"[dim]Model: {model} | Latency: {latency}ms | Usage: {usage}[/dim]"
        )
    else:
        console.print(f"[dim]Model: {model} | Latency: {latency}ms[/dim]")


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------

@cli.command()
def chat() -> None:
    """Start an interactive chat session with the RAG agent."""
    settings = _get_settings_safe()
    max_turns = getattr(settings, "conversation_memory_turns", 10) if settings else 10
    memory = ConversationMemory(max_turns=max_turns)

    try:
        retriever = HybridRetriever()
        reranker = CrossEncoderReranker()
        generator = RAGGenerator()
    except Exception as exc:
        console.print(f"[bold red]\u2717 Failed to initialize chat components:[/bold red] {exc}")
        logger.error("Chat initialization failed", error=str(exc), exc_info=True)
        raise click.exceptions.Exit(1) from exc

    console.print(
        Panel(
            "[cyan]Interactive chat mode.[/cyan]\n"
            "Type [bold]exit[/bold] or [bold]quit[/bold] to leave the session.",
            title="Cognita Chat",
            border_style="cyan",
        )
    )

    async def _chat_session() -> None:
        while True:
            # Blocking prompt is fine here: no concurrent tasks run while we wait.
            try:
                user_input = click.prompt("You", prompt_suffix=" \u276f ")
            except (click.exceptions.Abort, EOFError, KeyboardInterrupt):
                console.print()
                break

            user_input = (user_input or "").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                console.print("[dim]Goodbye![/dim]")
                break

            # Prior turns only; the current question is passed as `query`.
            history = memory.get_messages()

            try:
                with console.status("[cyan]Retrieving relevant documents...[/cyan]") as status:
                    results = await retriever.retrieve(user_input)
                    if results:
                        status.update("[cyan]Reranking results...[/cyan]")
                        results = await reranker.rerank(user_input, results)

                if not results:
                    console.print("[yellow]\u26a0 No relevant documents found.[/yellow]")

                # Stream the response token by token.
                console.print("[bold cyan]Assistant[/bold cyan] \u276f ", end="")
                collected: list[str] = []
                async for token in generator.generate_stream(
                    user_input, results, conversation_history=history
                ):
                    console.print(token, end="")
                    collected.append(token)
                full_response = "".join(collected)
                console.print()  # newline after streamed tokens

                memory.add_user_message(user_input)
                memory.add_assistant_message(full_response)

                # Show sources after each response.
                if results:
                    console.print()
                    _display_search_results(results)
            except Exception as exc:
                console.print()
                console.print(f"[bold red]\u2717 Error during chat:[/bold red] {exc}")
                logger.error("Chat error", error=str(exc), exc_info=True)
                continue

    try:
        run_async(_chat_session())
    except KeyboardInterrupt:
        console.print()
    finally:
        memory.clear()


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

@cli.command(name="list")
def list_docs() -> None:
    """List document statistics (total chunks in the vectorstore)."""
    try:

        async def _run_list() -> int:
            vectorstore = get_vectorstore()
            return await vectorstore.count()

        with console.status("[cyan]Counting chunks in vectorstore...[/cyan]"):
            count = run_async(_run_list())
    except Exception as exc:
        console.print(f"[bold red]\u2717 Failed to retrieve statistics:[/bold red] {exc}")
        logger.error("List failed", error=str(exc), exc_info=True)
        raise click.exceptions.Exit(1) from exc

    table = Table(title="Document Statistics", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Total chunks", str(count))
    console.print(table)


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("document_id")
def delete(document_id: str) -> None:
    """Delete a document and all its chunks from the vectorstore."""
    # Confirm before deletion.
    if not click.confirm(f"Delete document '{document_id}' and all its chunks?", default=False):
        console.print("[yellow]Cancelled.[/yellow]")
        return

    try:

        async def _run_delete() -> int:
            vectorstore = get_vectorstore()
            return await vectorstore.delete_by_document(document_id)

        with console.status(f"[cyan]Deleting document {document_id}...[/cyan]"):
            deleted = run_async(_run_delete())
    except Exception as exc:
        console.print(f"[bold red]\u2717 Deletion failed:[/bold red] {exc}")
        logger.error("Delete failed", document_id=document_id, error=str(exc), exc_info=True)
        raise click.exceptions.Exit(1) from exc

    if deleted and deleted > 0:
        console.print(
            f"[bold green]\u2713 Deleted {deleted} chunk(s) for document "
            f"[cyan]{document_id}[/cyan].[/bold green]"
        )
    else:
        console.print(
            f"[yellow]\u26a0 No chunks found for document [cyan]{document_id}[/cyan].[/yellow]"
        )


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------

@cli.command()
def health() -> None:
    """Check the health of system components (LLM, vectorstore)."""
    async def _run_health() -> dict[str, bool]:
        statuses: dict[str, bool] = {}
        try:
            llm = get_llm()
            statuses["LLM"] = await llm.health_check()
        except Exception as exc:
            logger.debug("LLM health check raised", error=str(exc))
            statuses["LLM"] = False
        try:
            vectorstore = get_vectorstore()
            statuses["Vectorstore"] = await vectorstore.health_check()
        except Exception as exc:
            logger.debug("Vectorstore health check raised", error=str(exc))
            statuses["Vectorstore"] = False
        return statuses

    try:
        with console.status("[cyan]Checking system health...[/cyan]"):
            statuses = run_async(_run_health())
    except Exception as exc:
        console.print(f"[bold red]\u2717 Health check failed:[/bold red] {exc}")
        logger.error("Health check failed", error=str(exc), exc_info=True)
        raise click.exceptions.Exit(1) from exc

    table = Table(title="System Health", show_header=True, header_style="bold cyan")
    table.add_column("Component", style="cyan")
    table.add_column("Status")

    def _status_cell(ok: bool) -> str:
        if ok:
            return "[bold green]\u25cf healthy[/bold green]"
        return "[bold red]\u25cf unhealthy[/bold red]"

    table.add_row("LLM", _status_cell(statuses.get("LLM", False)))
    table.add_row("Vectorstore", _status_cell(statuses.get("Vectorstore", False)))
    console.print(table)

    if all(statuses.values()):
        console.print("[bold green]\u2713 All systems operational.[/bold green]")
    else:
        console.print("[bold red]\u2717 One or more components are unhealthy.[/bold red]")


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--host", default=None, help="Host to bind the API server to.")
@click.option("--port", type=int, default=None, help="Port to bind the API server to.")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development.")
def serve(host: str | None, port: int | None, reload: bool) -> None:
    """Start the API server."""
    resolved_host = _resolve_host(host)
    resolved_port = _resolve_port(port)

    console.print(
        Panel(
            f"Starting API server at [cyan]http://{resolved_host}:{resolved_port}[/cyan]"
            + (" [yellow](reload enabled)[/yellow]" if reload else ""),
            title="Cognita API",
            border_style="cyan",
        )
    )

    try:
        from cognita.api.app import create_app
    except Exception as exc:
        console.print(f"[bold red]\u2717 Failed to import API app:[/bold red] {exc}")
        logger.error("API app import failed", error=str(exc), exc_info=True)
        raise click.exceptions.Exit(1) from exc

    if reload:
        # With reload enabled, uvicorn requires an import string. Use the
        # factory form so the app is re-created on each reload.
        uvicorn.run(
            "cognita.api.app:create_app",
            host=resolved_host,
            port=resolved_port,
            reload=True,
            factory=True,
        )
    else:
        app = create_app()
        uvicorn.run(app, host=resolved_host, port=resolved_port)


if __name__ == "__main__":
    cli()
