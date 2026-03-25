"""Contract tests for CORE-PROPS-1 Phase 4: derived_from in MCP search results."""

from unittest.mock import patch


class TestLineageDisplay:
    def _call_search(self, mock_items):
        import smartmemory_mcp.server as srv
        fn = None
        for tool in srv.mcp._tool_manager._tools.values():
            if tool.name == "memory_search":
                fn = tool.fn
                break
        assert fn is not None

        def mock_request(method, path, workspace_id=None, json=None, **kw):
            return mock_items

        with patch.object(srv, "_request", side_effect=mock_request):
            return fn(query="test")

    def test_derived_from_shown(self):
        items = [{"item_id": "derived-1", "memory_type": "semantic", "content": "Derived item",
                  "confidence": 0.8, "derived_from": "source-abc12345"}]
        result = self._call_search(items)
        assert "[→source-a]" in result

    def test_no_derived_from_no_marker(self):
        items = [{"item_id": "root-1", "memory_type": "episodic", "content": "Root item",
                  "confidence": 0.9}]
        result = self._call_search(items)
        assert "[→" not in result
