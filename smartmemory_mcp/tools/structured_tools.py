"""Structured ingestion MCP tools."""

import logging
from typing import Any, Dict

from .common import get_backend, graceful

logger = logging.getLogger(__name__)


def register(mcp):
    """Register structured ingestion tools with the MCP server (1 tool)."""

    @mcp.tool()
    @graceful
    def memory_ingest_structured(data: Dict[str, Any], schema_name: str) -> str:
        """Ingest structured data via a registered handler, bypassing NLP pipeline."""
        backend = get_backend()
        item_id = backend.ingest_structured(data, schema=schema_name)
        return f"Structured item ingested. Schema: {schema_name}, Item ID: {item_id}"
