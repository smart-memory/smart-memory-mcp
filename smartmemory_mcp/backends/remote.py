"""RemoteBackend — httpx client to the SmartMemory hosted API.

Extracted from server.py. Session state is instance-level (not module globals).
_request() never raises — returns error dicts on all failure modes.
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

import httpx

from .models import MemoryResult, normalize_item, normalize_items


class RemoteBackend:
    """HTTP client implementing MemoryBackend protocol for the hosted SmartMemory API."""

    def __init__(
        self,
        api_url: str | None = None,
        api_key: str | None = None,
        team_id: str | None = None,
    ) -> None:
        self._api_url = (api_url or os.environ.get("SMARTMEMORY_API_URL", "https://api.smartmemory.ai")).rstrip("/")
        self._session: dict[str, str | bool] = {
            "access_token": api_key or os.environ.get("SMARTMEMORY_API_KEY", ""),
            "refresh_token": "",
            "team_id": team_id or os.environ.get("SMARTMEMORY_TEAM_ID", os.environ.get("SMARTMEMORY_WORKSPACE_ID", "")),
            "user_email": "",
            "_bootstrapped": False,
        }

    # --- Session bootstrap -------------------------------------------------------

    def _bootstrap_from_api_key(self) -> None:
        """On first use, call /auth/me to discover user identity and default team."""
        if self._session["_bootstrapped"] or not self._session["access_token"]:
            return
        self._session["_bootstrapped"] = True
        try:
            r = httpx.get(
                f"{self._api_url}/auth/me",
                headers={
                    "Authorization": f"Bearer {self._session['access_token']}",
                    "Content-Type": "application/json",
                },
                timeout=10,
            )
            if r.status_code == 200:
                user = r.json()
                self._session["user_email"] = user.get("email", "")
                if not self._session["team_id"]:
                    discovered = user.get("default_team_id") or ""
                    if discovered:
                        self._session["team_id"] = discovered
        except Exception:
            pass  # bootstrap is best-effort

    def _headers(self, workspace_id: str | None = None) -> dict[str, str]:
        """Build request headers with auth and workspace context."""
        self._bootstrap_from_api_key()
        return {
            "Authorization": f"Bearer {self._session['access_token']}",
            "Content-Type": "application/json",
            "X-Workspace-Id": workspace_id or str(self._session.get("team_id", "")),
        }

    def _request(
        self,
        method: str,
        path: str,
        workspace_id: str | None = None,
        timeout: int = 30,
        **kwargs: Any,
    ) -> Any:
        """Execute an API request. Never raises — returns error dict on any failure."""
        try:
            r = httpx.request(
                method,
                f"{self._api_url}{path}",
                headers=self._headers(workspace_id),
                timeout=timeout,
                **kwargs,
            )
            r.raise_for_status()
            return r.json() if r.status_code != 204 else None
        except httpx.ConnectError:
            return {"error": f"SmartMemory API unreachable at {self._api_url}. Check SMARTMEMORY_API_URL."}
        except httpx.HTTPStatusError as e:
            return {"error": f"API error {e.response.status_code}: {e.response.text}"}
        except Exception as e:
            return {"error": f"Request failed: {e}"}

    def request(self, method: str, path: str, **kwargs: Any) -> Any:
        """Public request method for tools that need REST calls not in the protocol."""
        return self._request(method, path, **kwargs)

    @staticmethod
    def _fmt_error(result: Any) -> str | None:
        """If result is an error dict, return the message. Otherwise None."""
        if isinstance(result, dict) and "error" in result:
            return result["error"]
        return None

    # --- Auth --------------------------------------------------------------------

    def login(self, api_key: str | None = None, team_id: str | None = None) -> str:
        """Set API key and discover user identity from /auth/me."""
        key = api_key or os.environ.get("SMARTMEMORY_API_KEY", "")
        if not key:
            return "No API key. Pass api_key or set SMARTMEMORY_API_KEY env var."
        self._session["access_token"] = key
        self._session["refresh_token"] = ""
        self._session["user_email"] = ""
        self._session["_bootstrapped"] = False
        try:
            r = httpx.get(
                f"{self._api_url}/auth/me",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                timeout=15,
            )
            r.raise_for_status()
            user = r.json()
            self._session["user_email"] = user.get("email", "")
            self._session["team_id"] = team_id or user.get("default_team_id") or str(self._session["team_id"])
            self._session["_bootstrapped"] = True
        except httpx.HTTPStatusError as e:
            return f"API key validation failed ({e.response.status_code}): {e.response.text}"
        except Exception as e:
            return f"Login failed: {e}"
        return f"Logged in as {self._session['user_email']}, team: {self._session['team_id']}"

    def whoami(self) -> str:
        """Return current session info."""
        if not self._session["access_token"]:
            return f"Not authenticated. API: {self._api_url}. Call login to authenticate."
        return (
            f"User: {self._session['user_email'] or '(API key auth)'}\n"
            f"Team: {self._session['team_id']}\n"
            f"API: {self._api_url}\n"
            f"Backend: remote"
        )

    def switch_team(self, team_id: str) -> str:
        """Switch to a different team without re-authenticating."""
        self._session["team_id"] = team_id
        return f"Switched to team: {team_id}. User: {self._session['user_email']}"

    # --- MemoryBackend protocol: implemented (have REST routes) ------------------

    def add(self, content: str, memory_type: str = "semantic", **kwargs: Any) -> str:
        """POST /memory/add. Returns item_id string."""
        body: dict[str, Any] = {"content": content, "memory_type": memory_type}
        if metadata := kwargs.get("metadata"):
            body["metadata"] = metadata if isinstance(metadata, dict) else json.loads(metadata)
        if kwargs.get("use_pipeline"):
            body["use_pipeline"] = True
        result = self._request("POST", "/memory/add", json=body) or {}
        if isinstance(result, dict):
            return result.get("item_id", result.get("id", str(result)))
        return str(result)

    def get(self, item_id: str, **kwargs: Any) -> MemoryResult | None:
        """GET /memory/{item_id}."""
        result = self._request("GET", f"/memory/{item_id}")
        if result is None or self._fmt_error(result):
            return None
        return normalize_item(result)

    def update(self, item_id: str, **kwargs: Any) -> dict[str, Any]:
        """PUT /memory/{item_id}."""
        body: dict[str, Any] = {}
        for key in ("content", "memory_type", "metadata"):
            if key in kwargs:
                body[key] = kwargs[key]
        return self._request("PUT", f"/memory/{item_id}", json=body) or {}

    def delete(self, item_id: str, **kwargs: Any) -> bool:
        """DELETE /memory/{item_id}. Returns True on success."""
        result = self._request("DELETE", f"/memory/{item_id}")
        if result is None:
            return True  # 204 No Content = success
        if isinstance(result, dict) and self._fmt_error(result):
            return False
        return True

    def search(self, query: str, top_k: int = 5, **kwargs: Any) -> list[MemoryResult]:
        """POST /memory/search."""
        body: dict[str, Any] = {"query": query, "top_k": top_k}
        if kwargs.get("enable_hybrid", True):
            body["enable_hybrid"] = True
        # Accept both decompose and decompose_query
        decompose = kwargs.get("decompose") or kwargs.get("decompose_query")
        if decompose:
            body["decompose"] = True
        for key in ("memory_type", "include_reference"):
            if key in kwargs and kwargs[key]:
                body[key] = kwargs[key]
        result = self._request("POST", "/memory/search", json=body)
        if isinstance(result, dict) and self._fmt_error(result):
            return [result]
        raw = result if isinstance(result, list) else []
        return normalize_items(raw)

    def search_by_metadata(self, metadata_key: str, metadata_value: str, top_k: int = 10, **kwargs: Any) -> list[MemoryResult]:
        """GET /memory/by-metadata — exact metadata match."""
        params = {"metadata_key": metadata_key, "metadata_value": metadata_value}
        result = self._request("GET", "/memory/by-metadata", params=params)
        if isinstance(result, dict):
            if self._fmt_error(result):
                return [result]
            return normalize_items([result])  # Single item returned by service
        raw = result if isinstance(result, list) else []
        return normalize_items(raw)

    def recall(self, cwd: str | None = None, top_k: int = 10, **kwargs: Any) -> str:
        """Client-side recall — no /memory/recall endpoint in the hosted API."""
        requested = max(1, top_k)
        recent_k = max(1, (requested + 1) // 2)
        semantic_k = max(0, requested - recent_k)
        recent = self.search("", top_k=recent_k)
        semantic = self.search(cwd or "", top_k=semantic_k) if cwd and semantic_k else []
        seen: set[str] = set()
        items: list[MemoryResult] = []
        for r in recent + semantic:
            iid = r["item_id"]
            if iid and iid not in seen:
                seen.add(iid)
                items.append(r)
        recall_floor = float(os.environ.get("SMARTMEMORY_RECALL_FLOOR", "0.3"))
        items = [r for r in items if (r.get("confidence") if r.get("confidence") is not None else 1.0) >= recall_floor]
        items = [r for r in items if not r.get("reference")]
        if not items:
            return ""
        lines = ["## SmartMemory Context"]
        for item in items[:top_k]:
            conf = item.get("confidence", 1.0)
            conf_marker = "~" if isinstance(conf, (int, float)) and conf < 0.5 else ""
            stale_marker = "" if not item.get("stale") else "!"
            lines.append(f"- {stale_marker}{conf_marker}[{item['memory_type']}] {item['content'][:200]}")
        return "\n".join(lines)

    def ingest(self, content: str, memory_type: str = "semantic", **kwargs: Any) -> dict[str, Any] | str:
        """POST /memory/ingest (full pipeline)."""
        context: dict[str, Any] = {"memory_type": memory_type}
        # Merge metadata as top-level context keys (service merges context into pipeline state)
        metadata = kwargs.get("metadata")
        if metadata and isinstance(metadata, dict):
            context.update(metadata)
        body: dict[str, Any] = {"content": content, "context": context}
        result = self._request("POST", "/memory/ingest", timeout=120, json=body)
        if err := self._fmt_error(result):
            return {"error": err}
        return result or {}

    def clear_user_memories(self, **kwargs: Any) -> dict[str, Any]:
        """DELETE /memory/clear-all."""
        params: dict[str, str] = {}
        if kwargs.get("nuclear"):
            params["nuclear"] = "true"
        return self._request("DELETE", "/memory/clear-all", params=params) or {}

    def stats(self, **kwargs: Any) -> dict[str, Any]:
        """GET /memory/health."""
        return self._request("GET", "/memory/health") or {}

    def health(self) -> dict[str, Any]:
        """GET /health — API-level health check."""
        try:
            r = httpx.get(f"{self._api_url}/health", timeout=10)
            r.raise_for_status()
            return {"healthy": True, "api_url": self._api_url}
        except Exception as e:
            return {"healthy": False, "error": str(e), "api_url": self._api_url}

    def list_memories(self, **kwargs: Any) -> list[MemoryResult]:
        """GET /memory/list — list all memories."""
        params: dict[str, str] = {}
        if "limit" in kwargs:
            params["limit"] = str(kwargs["limit"])
        if "offset" in kwargs:
            params["offset"] = str(kwargs["offset"])
        result = self._request("GET", "/memory/list", params=params or None)
        if isinstance(result, dict):
            if self._fmt_error(result):
                return []
            # Service returns paginated dict with "items" and "total"
            raw = result.get("items", [])
        else:
            raw = result if isinstance(result, list) else []
        return normalize_items(raw)

    # --- MemoryBackend protocol: NOT available in remote mode --------------------

    def ingest_structured(self, items: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        """Not available in remote mode."""
        raise NotImplementedError("Not available in remote mode. Use local backend.")

    def get_all_items_debug(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Not available in remote mode."""
        raise NotImplementedError("Not available in remote mode. Use local backend.")

    def run_evolution_cycle(self, **kwargs: Any) -> dict[str, Any]:
        """Not available in remote mode."""
        raise NotImplementedError("Not available in remote mode. Use local backend.")

    def commit_working_to_episodic(self, **kwargs: Any) -> dict[str, Any]:
        """Not available in remote mode."""
        raise NotImplementedError("Not available in remote mode. Use local backend.")

    def commit_working_to_procedural(self, **kwargs: Any) -> dict[str, Any]:
        """Not available in remote mode."""
        raise NotImplementedError("Not available in remote mode. Use local backend.")

    def run_evolver(self, evolver_name: str, **kwargs: Any) -> dict[str, Any]:
        """Not available in remote mode."""
        raise NotImplementedError("Not available in remote mode. Use local backend.")

    def run_clustering(self, **kwargs: Any) -> dict[str, Any]:
        """Not available in remote mode."""
        raise NotImplementedError("Not available in remote mode. Use local backend.")

    def reflect(self, **kwargs: Any) -> dict[str, Any]:
        """Not available in remote mode."""
        raise NotImplementedError("Not available in remote mode. Use local backend.")

    def summary(self, **kwargs: Any) -> dict[str, Any]:
        """Not available in remote mode."""
        raise NotImplementedError("Not available in remote mode. Use local backend.")

    def orphaned_notes(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Not available in remote mode."""
        raise NotImplementedError("Not available in remote mode. Use local backend.")

    def find_old_notes(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Not available in remote mode."""
        raise NotImplementedError("Not available in remote mode. Use local backend.")

    def personalize(self, query: str, **kwargs: Any) -> dict[str, Any]:
        """Not available in remote mode."""
        raise NotImplementedError("Not available in remote mode. Use local backend.")

    def update_from_feedback(self, item_id: str, feedback: str, **kwargs: Any) -> dict[str, Any]:
        """Not available in remote mode."""
        raise NotImplementedError("Not available in remote mode. Use local backend.")

    def ground(self, item_id: str, **kwargs: Any) -> dict[str, Any]:
        """Not available in remote mode."""
        raise NotImplementedError("Not available in remote mode. Use local backend.")

    def link(self, source_id: str, target_id: str, **kwargs: Any) -> dict[str, Any]:
        """Not available in remote mode."""
        raise NotImplementedError("Not available in remote mode. Use local backend.")

    def add_edge(self, source_id: str, target_id: str, relation: str, **kwargs: Any) -> dict[str, Any]:
        """Not available in remote mode."""
        raise NotImplementedError("Not available in remote mode. Use local backend.")

    def get_links(self, item_id: str, **kwargs: Any) -> list[dict[str, Any]]:
        """Not available in remote mode."""
        raise NotImplementedError("Not available in remote mode. Use local backend.")

    def get_neighbors(self, item_id: str, **kwargs: Any) -> dict[str, Any]:
        """Not available in remote mode."""
        raise NotImplementedError("Not available in remote mode. Use local backend.")

    def find_shortest_path(self, source_id: str, target_id: str, **kwargs: Any) -> dict[str, Any]:
        """Not available in remote mode."""
        raise NotImplementedError("Not available in remote mode. Use local backend.")
