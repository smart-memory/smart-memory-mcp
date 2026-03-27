"""Agent recall profile MCP tools (rewritten for unified MCP -- no MongoDB)."""

import json
import logging
from typing import Dict, Optional

from .common import get_backend, graceful

logger = logging.getLogger(__name__)


def _item_field(item, field, default=""):
    if isinstance(item, dict):
        return item.get(field, default)
    return getattr(item, field, default)


def register(mcp):
    """Register agent recall profile tools with the MCP server."""

    @mcp.tool()
    @graceful
    def agent_set_recall_profile(
        agent_id: str,
        memory_type_weights: Optional[Dict[str, float]] = None,
    ) -> str:
        """Set an agent's recall profile for personality-aware search re-ranking."""
        backend = get_backend()

        if memory_type_weights:
            for k, v in memory_type_weights.items():
                if not isinstance(v, (int, float)) or v < 0:
                    return f"Error: weight for '{k}' must be non-negative number, got {v}"

        profile = {"memory_type_weights": memory_type_weights or {}}
        content = json.dumps(profile)

        # Search for existing profile to update
        existing = backend.search(f"recall profile {agent_id}", top_k=10, memory_type="procedural")
        for item in (existing or []):
            meta = _item_field(item, "metadata", {}) or {}
            if meta.get("recall_profile") and meta.get("agent_id") == agent_id:
                item_id = _item_field(item, "item_id", _item_field(item, "id", ""))
                if item_id:
                    try:
                        backend.update(item_id, content=content)
                        if memory_type_weights:
                            weight_str = ", ".join(f"{k}: {v}x" for k, v in memory_type_weights.items())
                            return f"Recall profile updated for {agent_id}: {weight_str}"
                        return f"Recall profile cleared for {agent_id}."
                    except Exception:
                        logger.warning("Failed to update existing profile for %s, creating new", agent_id)

        # No existing profile found -- create new
        backend.add(
            content=content,
            memory_type="procedural",
            metadata={
                "recall_profile": True,
                "agent_id": agent_id,
                "entity_type": "recall_profile",
                "tags": ["recall-profile", f"agent-{agent_id}"],
            },
        )

        if memory_type_weights:
            weight_str = ", ".join(f"{k}: {v}x" for k, v in memory_type_weights.items())
            return f"Recall profile set for {agent_id}: {weight_str}"
        return f"Recall profile cleared for {agent_id}."

    @mcp.tool()
    @graceful
    def agent_get_recall_profile(agent_id: str) -> str:
        """Get an agent's recall profile."""
        backend = get_backend()

        results = backend.search(f"recall profile {agent_id}", top_k=10, memory_type="procedural")

        for item in (results or []):
            meta = _item_field(item, "metadata", {}) or {}
            if meta.get("recall_profile") and meta.get("agent_id") == agent_id:
                raw_content = _item_field(item, "content", "")
                try:
                    profile = json.loads(raw_content)
                    return f"Recall profile for {agent_id}: {json.dumps(profile, indent=2)}"
                except (json.JSONDecodeError, TypeError):
                    return f"Recall profile for {agent_id}: {raw_content}"

        return f"Agent {agent_id} has no recall profile (default behavior)."
