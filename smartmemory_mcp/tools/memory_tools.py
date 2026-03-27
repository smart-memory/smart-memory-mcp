"""Memory CRUD, search, list, and stats MCP tools."""

import logging
from typing import Any, Dict, List, Optional

from .common import get_backend, graceful

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers — all dict-safe (backends return dicts, not objects)
# ---------------------------------------------------------------------------


def _item_field(item, field: str, default=""):
    """Get a field from a dict or object."""
    if isinstance(item, dict):
        return item.get(field, default)
    return getattr(item, field, default)


def _relative_age(dt) -> str:
    """Convert a UTC datetime to a human-readable relative age string."""
    if dt is None:
        return ""
    try:
        from datetime import UTC, datetime

        now = datetime.now(UTC)
        if isinstance(dt, str):
            dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        aware_dt = dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt
        days = (now - aware_dt).days
        if days <= 0:
            return "today"
        if days == 1:
            return "1d ago"
        if days < 30:
            return f"{days}d ago"
        if days < 365:
            return f"{days // 30}mo ago"
        return f"{days // 365}y ago"
    except Exception:
        return ""


def _format_catalog(query: str, results: list) -> str:
    """Format search results as a compact ranked catalog."""
    if not results:
        return f"No results found for query: {query}"

    lines = [f"Memory catalog for '{query}' — {len(results)} results", "─" * 43]
    for i, item in enumerate(results, 1):
        meta = _item_field(item, "metadata", {}) or {}
        if isinstance(meta, str):
            meta = {}

        content = str(_item_field(item, "content", "")).replace("\n", " ")
        title = str(meta.get("title", "")).replace("\n", " ") if isinstance(meta, dict) else ""
        snippet_src = title if title else content
        snippet = (snippet_src[:100] + "...") if len(snippet_src) > 100 else snippet_src

        item_id = _item_field(item, "item_id", _item_field(item, "id", "?"))
        mtype = _item_field(item, "memory_type", "?")
        score = _item_field(item, "score", None)
        score_str = f"  {score:.3f}" if score is not None else ""
        age_str = _relative_age(_item_field(item, "transaction_time", _item_field(item, "created_at", None)))
        stale = meta.get("stale", False) if isinstance(meta, dict) else False
        stale_str = "  ⚠ stale" if stale else ""

        lines.append(f"{i}. {item_id}  {mtype}{score_str}  {age_str}{stale_str}")
        lines.append(f"   {snippet}")
        lines.append("")

    lines.append("─" * 43)
    lines.append("Use memory_get(item_id) to read any item in full.")
    return "\n".join(lines)


def _format_turn_content(user_turn: str, assistant_turn: str) -> str:
    """Format a user/assistant turn pair for storage."""
    return f"USER: {user_turn.strip()}\nASSISTANT: {assistant_turn.strip()}"


def _format_recall(query: str, results: list, session_id: str = None, drift_warnings: list = None) -> str:
    """Format recalled turns as a prompt-ready context string."""
    if not results:
        if session_id:
            return "No prior turns found for this session."
        return f"No prior turns found for: {query}"

    lines = [f"Relevant prior context for '{query}':", ""]
    for i, item in enumerate(results, 1):
        age_str = _relative_age(_item_field(item, "transaction_time", _item_field(item, "created_at", None)))
        age_part = f"{age_str} — " if age_str else ""
        content = str(_item_field(item, "content", "")).replace("\n", " ↵ ")
        snippet = (content[:300] + "…") if len(content) > 300 else content
        lines.append(f"[{i}] {age_part}{snippet}")
        lines.append("")

    item_ids = ", ".join(
        str(_item_field(item, "item_id", _item_field(item, "id", "")))
        for item in results
        if _item_field(item, "item_id", _item_field(item, "id", ""))
    )
    lines.append(
        f"(Retrieved {len(results)} turns."
        + (f" Use memory_get(item_id) for full content. IDs: {item_ids}" if item_ids else "")
        + ")"
    )

    if drift_warnings:
        lines.append("")
        lines.append("⚠ Anchor drift warnings:")
        for dw in drift_warnings:
            severity = _item_field(dw, "severity", "unknown")
            acontent = str(_item_field(dw, "anchor_content", ""))[:100]
            dscore = _item_field(dw, "drift_score", 0)
            missing = _item_field(dw, "missing_keywords", [])[:5]
            missing_str = ", ".join(missing) if missing else "none"
            lines.append(f"  [{severity}] {acontent} (drift={dscore:.2f}, missing: {missing_str})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# FREE tier tools (4)
# ---------------------------------------------------------------------------


def register_free(mcp):
    """Register FREE-tier memory tools (ingest, search, recall, get)."""

    @mcp.tool()
    @graceful
    def memory_ingest(content: str, memory_type: str = "semantic") -> str:
        """Ingest content through the full NLP pipeline with entity extraction."""
        backend = get_backend()
        result = backend.ingest(content, memory_type=memory_type)
        if isinstance(result, dict):
            item_id = result.get("item_id", "unknown")
        else:
            item_id = str(result)
        return f"Memory ingested (pipeline). Item ID: {item_id}"

    @mcp.tool()
    @graceful
    def memory_search(
        query: str,
        top_k: int = 5,
        memory_type: Optional[str] = None,
        enable_hybrid: bool = True,
        catalog_mode: bool = True,
        decompose: bool = False,
    ) -> str:
        """Search memories using semantic similarity with optional hybrid mode."""
        backend = get_backend()
        results = backend.search(
            query,
            top_k=top_k * 3,
            memory_type=memory_type,
            enable_hybrid=enable_hybrid,
            decompose_query=decompose,
        )

        # CORE-ORIGIN-1: apply search tier policy
        try:
            from smartmemory.origin_policy import filter_by_tiers, get_default_tiers
            results = filter_by_tiers(results, get_default_tiers("search"))
        except Exception:
            pass

        results = results[:top_k]

        if not results:
            return f"No results found for query: {query}"

        if catalog_mode:
            return _format_catalog(query, results)

        output = [f"Found {len(results)} results for '{query}':\n"]
        for i, item in enumerate(results, 1):
            content = str(_item_field(item, "content", ""))
            preview = content[:200] + "..." if len(content) > 200 else content
            item_id = _item_field(item, "item_id", _item_field(item, "id", "?"))
            mtype = _item_field(item, "memory_type", "?")
            score = _item_field(item, "score", None)
            score_str = f" (score: {score:.3f})" if score is not None else ""

            # CORE-PROPS-1: confidence display
            confidence = _item_field(item, "confidence", None)
            conf_str = f" [conf: {confidence:.2f}]" if confidence is not None else ""

            # CORE-PROPS-1: stale marker
            stale = _item_field(item, "stale", False)
            stale_prefix = "⚠" if stale else ""
            stale_suffix = " [STALE]" if stale else ""

            # CORE-PROPS-1: lineage (derived_from)
            derived = _item_field(item, "derived_from", "")
            lineage_str = f" [→{str(derived)[:8]}]" if derived else ""

            # CORE-PROPS-1: low-confidence tilde marker
            id_display = f"~[{item_id}]" if confidence is not None and confidence < 0.5 else f"[{item_id}]"

            output.append(f"{i}. {stale_prefix}{id_display} ({mtype}){score_str}{conf_str}{lineage_str}{stale_suffix}")
            output.append(f"   {preview}\n")

        return "\n".join(output)

    @mcp.tool()
    @graceful
    def memory_recall(query: str, session_id: Optional[str] = None, top_k: int = 5) -> str:
        """Retrieve relevant prior conversation turns for prompt injection."""
        if top_k < 1:
            return "top_k must be at least 1."

        backend = get_backend()
        # Try native recall first (LocalBackend has it)
        if hasattr(backend, "recall") and not session_id:
            result = backend.recall(cwd=query, top_k=top_k)
            if isinstance(result, str):
                return result

        # Fallback: search working memory
        fetch_k = top_k * 5 if session_id else top_k
        results = backend.search(query, top_k=fetch_k, memory_type="working")
        return _format_recall(query, results[:top_k], session_id=session_id)

    @mcp.tool()
    @graceful
    def memory_get(item_id: str) -> str:
        """Retrieve a memory item by ID with full content and metadata."""
        backend = get_backend()
        item = backend.get(item_id)

        if item is None:
            return f"Memory item not found: {item_id}"

        content = _item_field(item, "content", "")
        mtype = _item_field(item, "memory_type", "unknown")
        meta = _item_field(item, "metadata", {})

        parts = [
            f"Memory Item: {item_id}",
            f"Type: {mtype}",
            f"Content: {content}",
        ]
        if meta:
            parts.append(f"Metadata: {meta}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# PRO tier tools (9)
# ---------------------------------------------------------------------------


def register_pro(mcp):
    """Register PRO-tier memory tools."""

    @mcp.tool()
    @graceful
    def memory_add(
        content: str,
        memory_type: str = "semantic",
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """Add a memory item directly without the NLP pipeline."""
        backend = get_backend()
        item_metadata = metadata or {}
        if tags:
            item_metadata["tags"] = tags
        item_id = backend.add(content, memory_type=memory_type, metadata=item_metadata)
        return f"Memory added (direct). Item ID: {item_id}"

    @mcp.tool()
    @graceful
    def memory_update(item_id: str, content: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Update a memory item's content or metadata."""
        backend = get_backend()
        result = backend.update(item_id, content=content, metadata=metadata)
        return str(result) if result else f"Updated: {item_id}"

    @mcp.tool()
    @graceful
    def memory_delete(item_id: str) -> str:
        """Delete a memory item by ID."""
        backend = get_backend()
        success = backend.delete(item_id)
        return f"Deleted: {item_id}" if success else f"Failed to delete: {item_id}"

    @mcp.tool()
    @graceful
    def memory_clear(confirm: bool = False) -> str:
        """Clear all memories permanently (requires confirm=True)."""
        if not confirm:
            return "Safety check: Set confirm=True to actually delete all memories."
        backend = get_backend()
        result = backend.clear_user_memories(confirm=True)
        return str(result)

    @mcp.tool()
    @graceful
    def memory_list(limit: int = 20, offset: int = 0) -> str:
        """List memories with pagination."""
        backend = get_backend()
        result = backend.list_memories(limit=limit, offset=offset)

        # Handle both list and dict responses
        if isinstance(result, dict):
            items = result.get("items", [])
            total = result.get("total", len(items))
        elif isinstance(result, list):
            items = result
            total = len(items)
        else:
            return "Unexpected response format."

        if not items:
            return "No memories found."

        output = [f"Showing {len(items)} of {total} memories:\n"]
        for item in items:
            item_id = _item_field(item, "item_id", _item_field(item, "id", "?"))
            content = str(_item_field(item, "content", ""))
            mtype = _item_field(item, "memory_type", "?")
            preview = content[:100] + "..." if len(content) > 100 else content
            output.append(f"- [{item_id}] ({mtype}): {preview}")

        return "\n".join(output)

    @mcp.tool()
    @graceful
    def memory_stats() -> str:
        """Get memory count statistics grouped by type."""
        backend = get_backend()
        result = backend.get_all_items_debug()

        if isinstance(result, dict):
            total = result.get("total_items", 0)
            by_type = result.get("items_by_type", {})
        else:
            return f"Stats: {result}"

        output = ["Memory Statistics:\n", f"Total memories: {total}", "\nBy type:"]
        for mtype, count in sorted(by_type.items()):
            output.append(f"  - {mtype}: {count}")

        return "\n".join(output)

    @mcp.tool()
    @graceful
    def memory_distill(user_turn: str, assistant_turn: str, session_id: Optional[str] = None) -> str:
        """Ingest a conversation turn pair into working memory for later recall."""
        backend = get_backend()
        content = _format_turn_content(user_turn, assistant_turn)
        meta: Dict[str, Any] = {"role": "turn", "distill_version": "1"}
        if session_id:
            meta["conversation_id"] = session_id
        item_id = backend.add(content, memory_type="working", metadata=meta)
        return f"Turn stored: {item_id}"

    @mcp.tool()
    @graceful
    def memory_search_advanced(query: str, algorithm: str = "query_traversal", max_results: int = 15) -> str:
        """Advanced search using Similarity Graph Traversal (SSG) algorithms."""
        try:
            from smartmemory.retrieval.ssg_traversal import SimilarityGraphTraversal
        except ImportError:
            return "Advanced search requires local backend with smartmemory installed."

        backend = get_backend()
        mem = getattr(backend, "_mem", None)
        if mem is None:
            return "Advanced search requires local backend."

        ssg = SimilarityGraphTraversal(mem)
        if algorithm == "query_traversal":
            results = ssg.query_traversal(query, max_results=max_results)
        elif algorithm == "triangulation_fulldim":
            results = ssg.triangulation_fulldim(query, max_results=max_results)
        else:
            return f"Unknown algorithm: {algorithm}. Use 'query_traversal' or 'triangulation_fulldim'"

        if not results:
            return f"No results found for query: {query}"

        output = [f"SSG ({algorithm}) found {len(results)} results for '{query}':\n"]
        for i, item in enumerate(results, 1):
            content = str(_item_field(item, "content", ""))
            preview = content[:200] + "..." if len(content) > 200 else content
            item_id = _item_field(item, "item_id", _item_field(item, "id", "?"))
            mtype = _item_field(item, "memory_type", "?")
            output.append(f"{i}. [{item_id}] ({mtype})")
            output.append(f"   {preview}\n")

        return "\n".join(output)

    @mcp.tool()
    @graceful
    def memory_search_by_metadata(metadata_key: str, metadata_value: str, top_k: int = 10) -> str:
        """Search memories by exact metadata key-value match."""
        backend = get_backend()
        results = backend.search_by_metadata(metadata_key, metadata_value, top_k=top_k)

        if not results:
            return f"No memories found with {metadata_key}={metadata_value}"

        output = [f"Found {len(results)} memories with {metadata_key}={metadata_value}:\n"]
        for item in results:
            content = str(_item_field(item, "content", ""))
            preview = content[:150] + "..." if len(content) > 150 else content
            item_id = _item_field(item, "item_id", _item_field(item, "id", "?"))
            mtype = _item_field(item, "memory_type", "?")
            output.append(f"- [{item_id}] ({mtype}): {preview}")

        return "\n".join(output)
