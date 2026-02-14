"""
SmartMemory MCP Server — thin HTTP client to remote SmartMemory API.

No local infrastructure required. All operations go through the REST API.

Env vars:
    SMARTMEMORY_API_URL      — API base URL (default: https://api.smartmemory.ai)
    SMARTMEMORY_API_KEY      — API key (sk_...) for authentication
    SMARTMEMORY_WORKSPACE_ID — Default workspace (default: "default")
"""

import os
import json
from typing import Optional

import httpx
from fastmcp import FastMCP

API_URL = os.environ.get("SMARTMEMORY_API_URL", "https://api.smartmemory.ai")
API_KEY = os.environ.get("SMARTMEMORY_API_KEY", "")
DEFAULT_WORKSPACE = os.environ.get("SMARTMEMORY_WORKSPACE_ID", "default")

mcp = FastMCP("smartmemory-memory")


def _headers(workspace_id: Optional[str] = None) -> dict:
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "X-Workspace-Id": workspace_id or DEFAULT_WORKSPACE,
    }


def _request(method: str, path: str, workspace_id: Optional[str] = None, **kwargs):
    try:
        r = httpx.request(method, f"{API_URL}{path}", headers=_headers(workspace_id), timeout=30, **kwargs)
        r.raise_for_status()
        return r.json() if r.status_code != 204 else None
    except httpx.ConnectError:
        return {"error": f"SmartMemory API unreachable at {API_URL}. Check SMARTMEMORY_API_URL."}
    except httpx.HTTPStatusError as e:
        return {"error": f"API error {e.response.status_code}: {e.response.text}"}
    except Exception as e:
        return {"error": f"Request failed: {e}"}


def _fmt_error(result) -> Optional[str]:
    """If result is an error dict, return the message. Otherwise None."""
    if isinstance(result, dict) and "error" in result:
        return result["error"]
    return None


# ---------------------------------------------------------------------------
# Memory tools
# ---------------------------------------------------------------------------


@mcp.tool()
def memory_add(
    content: str,
    memory_type: str = "semantic",
    metadata: Optional[str] = None,
    use_pipeline: bool = False,
    workspace_id: Optional[str] = None,
) -> str:
    """Store a memory in SmartMemory.

    Args:
        content: The text content to store
        memory_type: Memory type — semantic, episodic, procedural, working, decision, opinion, observation
        metadata: Optional JSON string of metadata (e.g. '{"source": "claude-code"}')
        use_pipeline: Run full extraction pipeline (entities, relations, enrichment). Default: False
        workspace_id: Workspace to store in (default: from SMARTMEMORY_WORKSPACE_ID env var)
    """
    body = {
        "content": content,
        "memory_type": memory_type,
        "metadata": json.loads(metadata) if metadata else {},
        "use_pipeline": use_pipeline,
    }
    result = _request("POST", "/memory/add", workspace_id=workspace_id, json=body)
    err = _fmt_error(result)
    if err:
        return err
    item_id = result.get("id", "unknown")
    return f"Memory stored. ID: {item_id}"


@mcp.tool()
def memory_search(
    query: str,
    top_k: int = 5,
    memory_type: Optional[str] = None,
    workspace_id: Optional[str] = None,
) -> str:
    """Search memories by semantic similarity.

    Args:
        query: Search query (matched by meaning, not just keywords)
        top_k: Number of results to return (default: 5)
        memory_type: Filter by type — semantic, episodic, procedural, etc. (optional)
        workspace_id: Workspace to search in (default: from SMARTMEMORY_WORKSPACE_ID env var)
    """
    body = {"query": query, "top_k": top_k, "enable_hybrid": True}
    if memory_type:
        body["memory_type"] = memory_type
    result = _request("POST", "/memory/search", workspace_id=workspace_id, json=body)
    err = _fmt_error(result)
    if err:
        return err
    if not result:
        return f"No results found for: {query}"
    items = result if isinstance(result, list) else [result]
    lines = [f"Found {len(items)} results for '{query}':\n"]
    for i, item in enumerate(items, 1):
        content = item.get("content", "")
        preview = content[:200] + "..." if len(content) > 200 else content
        score = item.get("score")
        score_str = f" (score: {score:.3f})" if isinstance(score, (int, float)) else ""
        lines.append(f"{i}. [{item.get('item_id', '?')}] ({item.get('memory_type', '?')}){score_str}")
        lines.append(f"   {preview}\n")
    return "\n".join(lines)


@mcp.tool()
def memory_get(
    item_id: str,
    workspace_id: Optional[str] = None,
) -> str:
    """Get a specific memory by ID.

    Args:
        item_id: The memory item ID
        workspace_id: Workspace (default: from SMARTMEMORY_WORKSPACE_ID env var)
    """
    result = _request("GET", f"/memory/{item_id}", workspace_id=workspace_id)
    err = _fmt_error(result)
    if err:
        return err
    if not result:
        return f"Memory not found: {item_id}"
    parts = [
        f"ID: {result.get('item_id', item_id)}",
        f"Type: {result.get('memory_type', '?')}",
        f"Content: {result.get('content', '')}",
    ]
    if result.get("metadata"):
        parts.append(f"Metadata: {json.dumps(result['metadata'], indent=2)}")
    return "\n".join(parts)


@mcp.tool()
def memory_delete(
    item_id: str,
    workspace_id: Optional[str] = None,
) -> str:
    """Delete a memory by ID.

    Args:
        item_id: The memory item ID to delete
        workspace_id: Workspace (default: from SMARTMEMORY_WORKSPACE_ID env var)
    """
    result = _request("DELETE", f"/memory/{item_id}", workspace_id=workspace_id)
    err = _fmt_error(result)
    if err:
        return err
    return f"Memory deleted: {item_id}"


@mcp.tool()
def memory_ingest(
    content: str,
    memory_type: str = "semantic",
    workspace_id: Optional[str] = None,
) -> str:
    """Ingest content through the full SmartMemory pipeline.

    Runs entity extraction, relation detection, enrichment, linking, and grounding.
    Use this for important content that should be deeply indexed.

    Args:
        content: Text to ingest
        memory_type: Memory type (default: semantic)
        workspace_id: Workspace (default: from SMARTMEMORY_WORKSPACE_ID env var)
    """
    body = {"content": content, "context": {"memory_type": memory_type}}
    result = _request("POST", "/memory/ingest", workspace_id=workspace_id, json=body)
    err = _fmt_error(result)
    if err:
        return err
    item_id = result.get("item_id", "unknown")
    entities = result.get("entities_extracted", 0)
    relations = result.get("relations_extracted", 0)
    return f"Ingested. ID: {item_id}, entities: {entities}, relations: {relations}"


@mcp.tool()
def memory_stats(
    workspace_id: Optional[str] = None,
) -> str:
    """Get memory statistics for the workspace.

    Args:
        workspace_id: Workspace (default: from SMARTMEMORY_WORKSPACE_ID env var)
    """
    result = _request("GET", "/memory/health", workspace_id=workspace_id)
    err = _fmt_error(result)
    if err:
        return err
    total = result.get("total_items", 0)
    by_type = result.get("items_by_type", {})
    health = result.get("health_score", 0)
    lines = [f"Memory Statistics:\n", f"Total: {total}", f"Health: {health:.0%}", "\nBy type:"]
    for mtype, count in sorted(by_type.items()):
        if count > 0:
            lines.append(f"  {mtype}: {count}")
    if not any(count > 0 for count in by_type.values()):
        lines.append("  (empty workspace)")
    return "\n".join(lines)


@mcp.tool()
def memory_health() -> str:
    """Check if the SmartMemory API is reachable and healthy."""
    try:
        r = httpx.get(f"{API_URL}/health", timeout=10)
        r.raise_for_status()
        return f"SmartMemory API healthy. URL: {API_URL}, workspace: {DEFAULT_WORKSPACE}"
    except Exception as e:
        return f"SmartMemory API unreachable at {API_URL}: {e}"


# ---------------------------------------------------------------------------
# Code indexing tools (parser runs locally, results POST to API)
# ---------------------------------------------------------------------------

import code_tools  # noqa: F401 — registers code_index, code_search, code_dependencies on mcp


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    import sys
    if "--http" in sys.argv:
        port = 8011
        for i, arg in enumerate(sys.argv):
            if arg == "--port" and i + 1 < len(sys.argv):
                port = int(sys.argv[i + 1])
        mcp.run(transport="http", host="0.0.0.0", port=port)
    else:
        mcp.run()


if __name__ == "__main__":
    main()
