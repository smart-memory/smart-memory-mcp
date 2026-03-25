"""Contract tests for CORE-PROPS-1 Phase 2: stale markers in MCP search results.

Verifies that memory_search output includes ⚠ prefix and [STALE] suffix for stale items.
"""

from unittest.mock import patch


class TestStaleDisplay:
    def _call_search(self, mock_items, **kwargs):
        """Call memory_search with mocked _request returning mock_items."""
        import smartmemory_mcp.server as srv

        fn = None
        for tool in srv.mcp._tool_manager._tools.values():
            if tool.name == "memory_search":
                fn = tool.fn
                break

        assert fn is not None, "memory_search tool not found"

        def mock_request(method, path, workspace_id=None, json=None, **kw):
            return mock_items

        with patch.object(srv, "_request", side_effect=mock_request):
            result = fn(query="test", **kwargs)

        return result

    def test_stale_marker_in_output(self):
        """Stale items get ⚠ prefix and [STALE] suffix."""
        items = [
            {
                "item_id": "stale123",
                "memory_type": "semantic",
                "content": "Stale information",
                "confidence": 0.8,
                "stale": True,
            }
        ]
        result = self._call_search(items)
        assert "⚠" in result
        assert "[STALE]" in result
        assert "Stale information" in result

    def test_no_stale_marker_when_false(self):
        """Non-stale items have no ⚠ or [STALE]."""
        items = [
            {
                "item_id": "fresh123",
                "memory_type": "semantic",
                "content": "Fresh information",
                "confidence": 0.9,
                "stale": False,
            }
        ]
        result = self._call_search(items)
        assert "⚠" not in result
        assert "[STALE]" not in result
        assert "Fresh information" in result

    def test_no_stale_field_no_marker(self):
        """Items without stale field have no marker."""
        items = [
            {
                "item_id": "nofld123",
                "memory_type": "working",
                "content": "No stale field",
            }
        ]
        result = self._call_search(items)
        assert "⚠" not in result
        assert "[STALE]" not in result
