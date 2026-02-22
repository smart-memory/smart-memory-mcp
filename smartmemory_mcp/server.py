"""
SmartMemory MCP Server — thin HTTP client to remote SmartMemory API.

No local infrastructure required. All operations go through the REST API.

Auth: Call the `login` tool with email/password to authenticate, or fall back
to env vars for API-key auth. Tokens are held in-process — no reload needed
to switch users or teams.

Env vars (fallback only):
    SMARTMEMORY_API_URL   — API base URL (default: https://api.smartmemory.ai)
    SMARTMEMORY_API_KEY   — API key (sk_...) for authentication
    SMARTMEMORY_TEAM_ID   — Default team (default: "default")
"""

import os
import json
from typing import Optional

import httpx
from fastmcp import FastMCP

API_URL = os.environ.get("SMARTMEMORY_API_URL", "https://api.smartmemory.ai")

# Mutable session state — updated by login/refresh tools at runtime.
_session: dict = {
    "access_token": os.environ.get("SMARTMEMORY_API_KEY", ""),
    "refresh_token": "",
    # SMARTMEMORY_TEAM_ID is an explicit override; otherwise auto-discovered from /auth/me.
    "team_id": os.environ.get("SMARTMEMORY_TEAM_ID", os.environ.get("SMARTMEMORY_WORKSPACE_ID", "")),
    "user_email": "",
    "_bootstrapped": False,
}

mcp = FastMCP("smartmemory-memory")


def _bootstrap_from_api_key() -> None:
    """On first use, call /auth/me with the API key to discover user identity and default_team_id.

    This ensures the session team_id always matches the authenticated user's actual
    default team — not a potentially-stale SMARTMEMORY_TEAM_ID env var.
    """
    if _session["_bootstrapped"] or not _session["access_token"]:
        return
    _session["_bootstrapped"] = True
    try:
        r = httpx.get(
            f"{API_URL}/auth/me",
            headers={
                "Authorization": f"Bearer {_session['access_token']}",
                "Content-Type": "application/json",
            },
            timeout=10,
        )
        if r.status_code == 200:
            user = r.json()
            _session["user_email"] = user.get("email", "")
            # Only override team_id if not explicitly set via env var.
            if not os.environ.get("SMARTMEMORY_TEAM_ID") and not os.environ.get("SMARTMEMORY_WORKSPACE_ID"):
                discovered = user.get("default_team_id") or ""
                if discovered:
                    _session["team_id"] = discovered
    except Exception:
        pass  # Silently skip — bootstrap is best-effort; tool calls will surface real errors.


def _headers(workspace_id: Optional[str] = None) -> dict:
    _bootstrap_from_api_key()
    return {
        "Authorization": f"Bearer {_session['access_token']}",
        "Content-Type": "application/json",
        "X-Team-Id": workspace_id or _session["team_id"],
    }


def _request(method: str, path: str, workspace_id: Optional[str] = None, timeout: int = 30, **kwargs):
    try:
        r = httpx.request(method, f"{API_URL}{path}", headers=_headers(workspace_id), timeout=timeout, **kwargs)
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
# Auth tools
# ---------------------------------------------------------------------------


@mcp.tool()
def login(
    api_key: Optional[str] = None,
    team_id: Optional[str] = None,
) -> str:
    """Set API key credentials for this session and discover user identity from /auth/me.

    SmartMemory uses Clerk-based SSO — email/password login is not available via MCP.
    Use an API key (sk_...) from your SmartMemory account settings instead.

    After calling login, all other tools use the new credentials automatically.
    No MCP reload needed — just call login again to switch API keys.

    Args:
        api_key: SmartMemory API key (sk_...) — defaults to SMARTMEMORY_API_KEY env var
        team_id: Team to use — if omitted, auto-discovered from the API key's user profile
    """
    key = api_key or os.environ.get("SMARTMEMORY_API_KEY", "")
    if not key:
        return "No API key. Pass api_key or set SMARTMEMORY_API_KEY env var."

    _session["access_token"] = key
    _session["refresh_token"] = ""
    _session["user_email"] = ""
    _session["_bootstrapped"] = False  # Force re-bootstrap with new key

    # Discover user identity from /auth/me
    try:
        r = httpx.get(
            f"{API_URL}/auth/me",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            timeout=15,
        )
        r.raise_for_status()
        user = r.json()
        _session["user_email"] = user.get("email", "")
        _session["team_id"] = team_id or user.get("default_team_id") or _session["team_id"]
        _session["_bootstrapped"] = True
    except httpx.HTTPStatusError as e:
        return f"API key validation failed ({e.response.status_code}): {e.response.text}"
    except Exception as e:
        return f"Login failed: {e}"

    return f"Logged in as {_session['user_email']}, team: {_session['team_id']}"


@mcp.tool()
def refresh_token() -> str:
    """Refresh the access token using the stored refresh token.

    Call this if you get 401 errors — tokens expire after 1 hour.
    """
    if not _session["refresh_token"]:
        return "No refresh token stored. Call login first."

    try:
        r = httpx.post(
            f"{API_URL}/auth/refresh",
            json={"refresh_token": _session["refresh_token"]},
            headers={"Content-Type": "application/json"},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
    except httpx.HTTPStatusError as e:
        return f"Refresh failed ({e.response.status_code}): {e.response.text}. Call login again."
    except Exception as e:
        return f"Refresh failed: {e}"

    _session["access_token"] = data.get("access_token", _session["access_token"])
    _session["refresh_token"] = data.get("refresh_token", _session["refresh_token"])

    return f"Token refreshed. User: {_session['user_email']}, team: {_session['team_id']}"


@mcp.tool()
def switch_team(team_id: str) -> str:
    """Switch to a different team/workspace without re-authenticating.

    Args:
        team_id: The team ID to switch to
    """
    _session["team_id"] = team_id
    return f"Switched to team: {team_id}. User: {_session['user_email']}"


@mcp.tool()
def whoami() -> str:
    """Show current session info — logged-in user, team, and API URL."""
    if not _session["access_token"]:
        return f"Not authenticated. API: {API_URL}. Call login to authenticate."
    return (
        f"User: {_session['user_email'] or '(API key auth)'}\n"
        f"Team: {_session['team_id']}\n"
        f"API: {API_URL}\n"
        f"Has refresh token: {'yes' if _session['refresh_token'] else 'no'}"
    )


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
        workspace_id: Workspace to store in (default: current session team)
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
        workspace_id: Workspace to search in (default: current session team)
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
        workspace_id: Workspace (default: current session team)
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
        workspace_id: Workspace (default: current session team)
    """
    result = _request("DELETE", f"/memory/{item_id}", workspace_id=workspace_id)
    err = _fmt_error(result)
    if err:
        return err
    return f"Memory deleted: {item_id}"


@mcp.tool()
def memory_clear_all(
    nuclear: bool = False,
    workspace_id: Optional[str] = None,
) -> str:
    """Delete all memories for the current user in the workspace.

    WARNING: Cannot be undone.

    Args:
        nuclear: If true, deletes EVERYTHING (graph + cache + vectors). Use for dev/cleanup only.
        workspace_id: Workspace (default: current session team)
    """
    params = {}
    if nuclear:
        params["nuclear"] = "true"
    result = _request("DELETE", "/memory/clear-all", workspace_id=workspace_id, params=params)
    err = _fmt_error(result)
    if err:
        return err
    deleted = result.get("deleted_count", "?")
    msg = f"Cleared all memories. Deleted: {deleted}"
    if result.get("cache_cleared"):
        msg += ", cache cleared"
    if result.get("vectors_cleared"):
        msg += ", vectors cleared"
    return msg


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
        workspace_id: Workspace (default: current session team)
    """
    body = {"content": content, "context": {"memory_type": memory_type}}
    # Full pipeline (LLM extraction + grounding) can take 60-120s — use a long timeout.
    result = _request("POST", "/memory/ingest", workspace_id=workspace_id, timeout=120, json=body)
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
        workspace_id: Workspace (default: current session team)
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
        return f"SmartMemory API healthy. URL: {API_URL}, team: {_session['team_id']}"
    except Exception as e:
        return f"SmartMemory API unreachable at {API_URL}: {e}"


# ---------------------------------------------------------------------------
# Code indexing tools (parser runs locally, results POST to API)
# ---------------------------------------------------------------------------

from smartmemory_mcp import code_tools  # noqa: F401 — registers code_index, code_search, code_dependencies on mcp


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
        mcp.run(transport="http", host="0.0.0.0", port=port, show_banner=False)
    else:
        mcp.run(show_banner=False)


if __name__ == "__main__":
    main()
