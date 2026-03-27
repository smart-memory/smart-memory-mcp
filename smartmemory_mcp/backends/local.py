"""Local backend — delegates to smartmemory_app.storage (optional dependency)."""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


class LocalBackend:
    """Wraps smartmemory package for local-mode operations."""

    def __init__(self) -> None:
        try:
            from smartmemory_app.storage import get_memory
            self._get_memory = get_memory
            self._mem = get_memory()
        except ImportError:
            raise RuntimeError(
                "Local backend requires the smartmemory package.\n"
                "Install with: pip install smartmemory"
            )

    # -- Core CRUD --

    def add(self, content: str, memory_type: str = "semantic", metadata: dict | None = None, **kwargs: Any) -> str:
        """Store a memory item."""
        from smartmemory.models.memory_item import MemoryItem
        item = MemoryItem(content=content, memory_type=memory_type, metadata=metadata or {})
        return self._mem.add(item)

    def get(self, item_id: str, **kwargs: Any) -> dict | None:
        """Retrieve a memory by ID."""
        result = self._mem.get(item_id)
        if result is None:
            return None
        return result.to_dict() if hasattr(result, "to_dict") else result

    def update(self, item_id: str, content: str | None = None, metadata: dict | None = None, **kwargs: Any) -> str:
        """Update an existing memory."""
        item = self._mem.get(item_id)
        if item is None:
            return f"Not found: {item_id}"
        if content is not None:
            item.content = content
        if metadata is not None:
            item.metadata.update(metadata)
        self._mem.update(item)
        return f"Updated: {item_id}"

    def delete(self, item_id: str, **kwargs: Any) -> bool:
        """Delete a memory by ID."""
        return self._mem.delete(item_id)

    # -- Search --

    def search(self, query: str, top_k: int = 5, **kwargs: Any) -> list[dict]:
        """Semantic search."""
        from smartmemory_app.storage import search
        return search(query, top_k, **kwargs)

    def search_by_metadata(self, metadata_key: str, metadata_value: str, top_k: int = 10, **kwargs: Any) -> list[dict]:
        """Search by metadata field."""
        return self._mem.search_by_metadata(metadata_key, metadata_value, top_k=top_k)

    # -- Ingest & Recall --

    def ingest(self, content: str, memory_type: str = "episodic", **kwargs: Any) -> str:
        """Full pipeline ingestion."""
        from smartmemory_app.storage import ingest
        # storage.ingest() uses 'properties' not 'metadata'
        if "metadata" in kwargs:
            kwargs["properties"] = kwargs.pop("metadata")
        return ingest(content, memory_type, **kwargs)

    def recall(self, cwd: str | None = None, top_k: int = 10, **kwargs: Any) -> str:
        """Context-aware recall."""
        from smartmemory_app.storage import recall
        return recall(cwd, top_k)

    def ingest_structured(self, data: dict, schema: str | None = None, schema_name: str | None = None, **kwargs: Any) -> str:
        """Structured data ingestion."""
        name = schema or schema_name
        return self._mem.ingest_structured(data, schema=name)

    # -- Listing & Stats --

    def list_memories(self, limit: int = 100, offset: int = 0, **kwargs: Any) -> list[dict]:
        """List memories with pagination."""
        return self._mem.list_memories(limit=limit, offset=offset)

    def clear_user_memories(self, confirm: bool = False, **kwargs: Any) -> str:
        """Clear all user memories."""
        if not confirm:
            return "Pass confirm=True to clear all memories."
        self._mem.clear_user_memories()
        return "All memories cleared."

    def get_all_items_debug(self, **kwargs: Any) -> dict:
        """Get debug stats."""
        return self._mem.get_all_items_debug()

    def stats(self, **kwargs: Any) -> dict:
        """Memory statistics."""
        return self.get_all_items_debug(**kwargs)

    # -- Evolution --

    def run_evolution_cycle(self, **kwargs: Any) -> dict:
        """Trigger evolution cycle."""
        return self._mem.run_evolution_cycle(**kwargs)

    def commit_working_to_episodic(self, **kwargs: Any) -> dict:
        """Commit working memory to episodic."""
        return self._mem.commit_working_to_episodic(**kwargs)

    def commit_working_to_procedural(self, **kwargs: Any) -> dict:
        """Commit working memory to procedural."""
        return self._mem.commit_working_to_procedural(**kwargs)

    def run_evolver(self, evolver_class: Any, **kwargs: Any) -> dict:
        """Run a specific evolver."""
        return self._mem.run_evolver(evolver_class, **kwargs)

    def run_clustering(self, **kwargs: Any) -> dict:
        """Run clustering."""
        return self._mem.run_clustering(**kwargs)

    # -- Insight --

    def reflect(self, **kwargs: Any) -> str:
        """Reflective analysis."""
        return self._mem.reflect(**kwargs)

    def summary(self, **kwargs: Any) -> dict:
        """Memory summary."""
        return self._mem.summary(**kwargs)

    def orphaned_notes(self, **kwargs: Any) -> list:
        """Find orphaned notes."""
        return self._mem.orphaned_notes(**kwargs)

    def find_old_notes(self, days: int = 90, **kwargs: Any) -> list:
        """Find notes older than the given number of days."""
        return self._mem.find_old_notes(days, **kwargs)

    def personalize(self, user_id: str = "mcp-user", traits: dict | None = None, preferences: dict | None = None, **kwargs: Any) -> str:
        """Personalize memory system."""
        return self._mem.personalize(user_id=user_id, traits=traits or {}, preferences=preferences or {}, **kwargs)

    def update_from_feedback(self, feedback: dict | None = None, memory_type: str = "semantic", **kwargs: Any) -> str:
        """Update from user feedback."""
        return self._mem.update_from_feedback(feedback=feedback or {}, memory_type=memory_type, **kwargs)

    def ground(self, item_id: str, **kwargs: Any) -> dict:
        """Ground a memory item."""
        return self._mem.ground(item_id=item_id, **kwargs)

    # -- Graph --

    def link(self, source_id: str, target_id: str, link_type: str = "RELATES_TO", **kwargs: Any) -> str:
        """Link two memories."""
        return self._mem.link(source_id, target_id, link_type=link_type)

    def add_edge(self, source_id: str, target_id: str, relation_type: str, **kwargs: Any) -> str:
        """Add a graph edge."""
        return self._mem.add_edge(source_id, target_id, relation_type=relation_type, **kwargs)

    def get_links(self, item_id: str, **kwargs: Any) -> list:
        """Get links for an item."""
        return self._mem.get_links(item_id)

    def get_neighbors(self, item_id: str, **kwargs: Any) -> dict:
        """Get graph neighbors."""
        return self._mem.get_neighbors(item_id)

    def find_shortest_path(self, start_id: str, end_id: str, **kwargs: Any) -> list:
        """Find shortest path between two items."""
        return self._mem.find_shortest_path(start_id, end_id, **kwargs)

    # -- Auth (no-ops for local) --

    def login(self, api_key: str, **kwargs: Any) -> str:
        """No-op in local mode."""
        return "Local mode — no authentication required."

    def whoami(self, **kwargs: Any) -> str:
        """Local mode session info."""
        return "Local mode — single user."

    def switch_team(self, team_id: str, **kwargs: Any) -> str:
        """No-op in local mode."""
        return "Local mode — teams not applicable."
