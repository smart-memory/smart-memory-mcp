"""Evolution, clustering, and synthesis MCP tools."""

import logging

from .common import get_backend, graceful

logger = logging.getLogger(__name__)


def register(mcp):
    """Register evolution tools with the MCP server."""

    @mcp.tool()
    @graceful
    def evolution_trigger() -> str:
        """Trigger a memory evolution cycle — runs clustering, deduplication, and consolidation."""
        backend = get_backend()
        backend.run_evolution_cycle()
        return "Memory evolution cycle completed successfully."

    @mcp.tool()
    @graceful
    def evolution_dream() -> str:
        """Run dream phase — promote working memory to episodic and procedural."""
        backend = get_backend()

        episodic_ids = backend.commit_working_to_episodic()
        procedural_ids = backend.commit_working_to_procedural()

        ep_count = len(episodic_ids or [])
        proc_count = len(procedural_ids or [])
        return (
            f"Dream phase completed.\n"
            f"Episodic memories created: {ep_count}\n"
            f"Procedural memories created: {proc_count}"
        )

    @mcp.tool()
    @graceful
    def evolution_status() -> str:
        """Get status of memory evolution processes."""
        backend = get_backend()

        try:
            working_items = backend.search("*", memory_type="working", top_k=100) or []
            working_count = len(working_items)
        except Exception:
            working_count = 0

        status = "ready" if working_count >= 1 else "idle"
        return (
            f"Evolution status: {status}\n"
            f"Working memory items pending: {working_count}"
        )

    @mcp.tool()
    @graceful
    def evolution_synthesize_opinions() -> str:
        """Run opinion synthesis — detect patterns in episodic memories and form opinions."""
        from smartmemory.plugins.evolvers.opinion_synthesis import OpinionSynthesisEvolver

        backend = get_backend()
        evolver = OpinionSynthesisEvolver()
        backend.run_evolver(evolver, log=logger)
        return "Opinion synthesis completed."

    @mcp.tool()
    @graceful
    def evolution_synthesize_observations() -> str:
        """Run observation synthesis — create entity summaries from scattered facts."""
        from smartmemory.plugins.evolvers.observation_synthesis import ObservationSynthesisEvolver

        backend = get_backend()
        evolver = ObservationSynthesisEvolver()
        backend.run_evolver(evolver, log=logger)
        return "Observation synthesis completed."

    @mcp.tool()
    @graceful
    def evolution_reinforce_opinions() -> str:
        """Run opinion reinforcement — update confidence scores based on new evidence."""
        from smartmemory.plugins.evolvers.opinion_reinforcement import OpinionReinforcementEvolver

        backend = get_backend()
        evolver = OpinionReinforcementEvolver()
        backend.run_evolver(evolver, log=logger)
        return "Opinion reinforcement completed."

    @mcp.tool()
    @graceful
    def clustering_run(distance_threshold: float = 0.1) -> str:
        """Run entity clustering/deduplication — merge duplicate entities by vector similarity."""
        backend = get_backend()
        result = backend.run_clustering()

        merged = result.get("merged_count", 0)
        clusters = result.get("clusters_found", 0)
        total = result.get("total_entities", 0)
        return (
            f"Clustering completed.\n"
            f"Total entities: {total}\n"
            f"Clusters found: {clusters}\n"
            f"Entities merged: {merged}"
        )
