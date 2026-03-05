"""Unit tests for standalone MCP memory_search decompose param."""

import importlib
from unittest.mock import patch, MagicMock
from fastmcp import FastMCP


class TestServerDecompose:
    def _call_search(self, **kwargs):
        """Call memory_search and capture the REST body sent to _request."""
        # Re-import to get fresh module; extract the original function from the tool
        import smartmemory_mcp.server as srv

        captured_body = {}

        def mock_request(method, path, workspace_id=None, json=None, **kw):
            captured_body.update(json or {})
            return []

        # Find the memory_search tool and call its underlying function
        fn = None
        for tool in srv.mcp._tool_manager._tools.values():
            if tool.name == "memory_search":
                fn = tool.fn
                break

        assert fn is not None, "memory_search tool not found"

        with patch.object(srv, "_request", side_effect=mock_request):
            result = fn(**kwargs)

        return result, captured_body

    def test_decompose_false_not_in_body(self):
        _, body = self._call_search(query="auth")
        assert "decompose" not in body

    def test_decompose_true_in_body(self):
        _, body = self._call_search(query="auth", decompose=True)
        assert body["decompose"] is True

    def test_enable_hybrid_default_true(self):
        """TECHDEBT-SEARCH-1: standalone MCP sends enable_hybrid=True by default."""
        _, body = self._call_search(query="auth")
        assert body["enable_hybrid"] is True
