"""Portability tools — export, import, and migrate memories between backends."""

import json
import logging
import tempfile
from pathlib import Path

from .common import get_backend, graceful

logger = logging.getLogger(__name__)


def _get_items(backend) -> list[dict]:
    """Get all memory items from backend, handling list or dict response."""
    result = backend.list_memories(limit=100_000, offset=0)
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        return result.get("items", [])
    return []


def _item_to_record(item: dict) -> dict:
    """Convert a memory item (dict) to an export record."""
    meta = item["metadata"]
    return {
        "content": item["content"],
        "memory_type": item["memory_type"],
        "metadata": meta,
        "created_at": item.get("created_at", ""),
        "tags": meta.get("tags", []) if isinstance(meta, dict) else [],
        "origin": item.get("origin", ""),
    }


def _import_record(backend, record: dict) -> None:
    """Import a single record into backend, preserving metadata, tags, and origin."""
    content = record.get("content", "")
    if not content:
        raise ValueError("Empty content")
    memory_type = record.get("memory_type", "semantic")
    metadata = record.get("metadata", {}) or {}
    tags = record.get("tags", [])
    origin = record.get("origin", "")
    if tags:
        metadata["tags"] = tags
    if origin:
        metadata["origin"] = origin
    backend.ingest(content, memory_type=memory_type, metadata=metadata)


def register(mcp):
    """Register portability tools with the MCP server (3 tools)."""

    @mcp.tool()
    @graceful
    def memory_export(path: str) -> str:
        """Export all memories to a JSONL file for backup or migration."""
        backend = get_backend()
        items = _get_items(backend)

        if not items:
            return "No memories to export."

        export_path = Path(path).expanduser().resolve()
        export_path.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with open(export_path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(_item_to_record(item), default=str) + "\n")
                count += 1

        return f"Exported {count} memories to {export_path}"

    @mcp.tool()
    @graceful
    def memory_import(path: str) -> str:
        """Import memories from a JSONL file, re-ingesting each through the pipeline."""
        backend = get_backend()
        import_path = Path(path).expanduser().resolve()

        if not import_path.exists():
            return f"File not found: {import_path}"

        success_count = 0
        fail_count = 0

        with open(import_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    _import_record(backend, record)
                    success_count += 1
                except Exception as e:
                    logger.warning("Failed to import line %d: %s", line_num, e)
                    fail_count += 1

        return f"Import complete: {success_count} succeeded, {fail_count} failed from {import_path}"

    @mcp.tool()
    @graceful
    def memory_migrate(target: str) -> str:
        """Migrate all memories to a different backend (local or remote)."""
        if target not in ("local", "remote"):
            return f"Invalid target: {target}. Must be 'local' or 'remote'."

        from smartmemory_mcp.backends.dispatch import reset_backend
        from smartmemory_mcp.tools.common import reset_backend as reset_common_backend

        # Step 1: Export from current backend
        backend = get_backend()
        items = _get_items(backend)

        if not items:
            return "No memories to migrate."

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", prefix="smartmemory_migrate_",
            delete=False, encoding="utf-8",
        )
        tmp_path = Path(tmp.name)

        for item in items:
            tmp.write(json.dumps(_item_to_record(item), default=str) + "\n")
        tmp.close()

        # Step 2: Switch config to target
        try:
            from smartmemory_app.config import load_config, save_config
            cfg = load_config()
            original_mode = cfg.mode
        except ImportError:
            tmp_path.unlink(missing_ok=True)
            return "Config management requires the smartmemory package."

        cfg.mode = target
        save_config(cfg)
        reset_backend()
        reset_common_backend()

        # Step 3: Import into new backend
        try:
            target_backend = get_backend()
            success_count = 0
            fail_count = 0

            with open(tmp_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        _import_record(target_backend, record)
                        success_count += 1
                    except Exception as e:
                        logger.warning("Failed to import line %d: %s", line_num, e)
                        fail_count += 1

            if fail_count > 0 and success_count == 0:
                cfg.mode = original_mode
                save_config(cfg)
                reset_backend()
                reset_common_backend()
                return (
                    f"Migration failed: all {fail_count} items failed. "
                    f"Config reverted to '{original_mode}'. Temp file: {tmp_path}"
                )

            tmp_path.unlink(missing_ok=True)
            return (
                f"Migration to '{target}' complete: {success_count} succeeded, "
                f"{fail_count} failed out of {len(items)} total."
            )

        except Exception as e:
            cfg.mode = original_mode
            save_config(cfg)
            reset_backend()
            reset_common_backend()
            return (
                f"Migration failed: {e}\n"
                f"Config reverted to '{original_mode}'. Temp file: {tmp_path}"
            )
