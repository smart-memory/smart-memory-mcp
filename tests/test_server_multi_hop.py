"""Unit tests for RLM-1c multi-hop params in MCP memory_search tool."""

from unittest.mock import patch


class MockBackend:
    def __init__(self, items=None):
        self._items = items or []
        self.last_search_kwargs = {}

    def search(self, query, top_k=5, **kwargs):
        self.last_search_kwargs = {"query": query, "top_k": top_k, **kwargs}
        return self._items


class TestServerMultiHop:
    def _call_search(self, **kwargs):
        """Call memory_search and capture kwargs sent to backend.search()."""
        import smartmemory_mcp.server as srv

        mock_backend = MockBackend()

        fn = None
        for tool in srv.mcp._tool_manager._tools.values():
            if tool.name == "memory_search":
                fn = tool.fn
                break

        assert fn is not None, "memory_search tool not found"

        with patch("smartmemory_mcp.tools.common._backend", mock_backend):
            result = fn(**kwargs)

        return result, mock_backend.last_search_kwargs

    def test_multi_hop_false_by_default(self):
        _, search_kwargs = self._call_search(query="auth")
        assert search_kwargs.get("multi_hop") is False

    def test_multi_hop_true_forwarded(self):
        _, search_kwargs = self._call_search(
            query="auth", multi_hop=True, max_hops=2, budget_ms=500
        )
        assert search_kwargs["multi_hop"] is True
        assert search_kwargs["max_hops"] == 2
        assert search_kwargs["budget_ms"] == 500

    def test_multi_hop_defaults_forwarded(self):
        _, search_kwargs = self._call_search(query="auth", multi_hop=True)
        assert search_kwargs["multi_hop"] is True
        assert search_kwargs["max_hops"] == 3
        assert search_kwargs["budget_ms"] == 1500


class TestRemoteBackendMultiHop:
    """Test that RemoteBackend.search() serializes multi-hop params."""

    def test_remote_search_forwards_multi_hop(self):
        from unittest.mock import MagicMock
        from smartmemory_mcp.backends.remote import RemoteBackend

        backend = RemoteBackend.__new__(RemoteBackend)
        backend._base_url = "http://test:9001"
        backend._headers = {}

        captured_body = {}

        def mock_request(method, path, **kwargs):
            captured_body.update(kwargs.get("json", {}))
            return []

        backend._request = mock_request

        backend.search("auth", top_k=5, multi_hop=True, max_hops=2, budget_ms=800)

        assert captured_body.get("multi_hop") is True
        assert captured_body.get("max_hops") == 2
        assert captured_body.get("budget_ms") == 800

    def test_remote_search_omits_multi_hop_when_false(self):
        from unittest.mock import MagicMock
        from smartmemory_mcp.backends.remote import RemoteBackend

        backend = RemoteBackend.__new__(RemoteBackend)
        backend._base_url = "http://test:9001"
        backend._headers = {}

        captured_body = {}

        def mock_request(method, path, **kwargs):
            captured_body.update(kwargs.get("json", {}))
            return []

        backend._request = mock_request

        backend.search("auth", top_k=5)

        assert "multi_hop" not in captured_body
