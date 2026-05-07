"""Decision lifecycle MCP tools — create, query, supersede, retract, reinforce, contradict."""

import logging
from typing import List, Optional

from .common import get_backend, graceful

logger = logging.getLogger(__name__)


def register(mcp):
    """Register decision tools with the MCP server (10 tools)."""

    @mcp.tool()
    @graceful
    def decision_create(
        content: str,
        decision_type: str = "inference",
        confidence: float = 0.8,
        evidence_ids: Optional[List[str]] = None,
        domain: Optional[str] = None,
        tags: Optional[List[str]] = None,
        source_trace_id: Optional[str] = None,
        rejected_alternatives: Optional[List[str]] = None,
        rationale: Optional[str] = None,
        constraints: Optional[List[str]] = None,
    ) -> str:
        """Create a new decision with provenance tracking.

        CORE-EXPERTISE-1 Phase 1: rejected_alternatives, rationale, constraints
        capture the "why" structure that makes decisions usable expertise.
        """
        try:
            from smartmemory.decisions.manager import DecisionManager

            backend = get_backend()
            manager = DecisionManager(backend)
            decision = manager.create(
                content=content,
                decision_type=decision_type,
                confidence=confidence,
                source_trace_id=source_trace_id,
                evidence_ids=evidence_ids or [],
                domain=domain,
                tags=tags or [],
                rejected_alternatives=rejected_alternatives or [],
                rationale=rationale,
                constraints=constraints or [],
            )

            lines = [
                f"Decision created: {decision.decision_id}",
                f"Type: {decision.decision_type}",
                f"Confidence: {decision.confidence}",
            ]
            if decision.rejected_alternatives:
                lines.append(f"Rejected alternatives: {decision.rejected_alternatives}")
            if decision.rationale:
                lines.append(f"Rationale: {decision.rationale}")
            if decision.constraints:
                lines.append(f"Constraints: {decision.constraints}")
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"Failed to create decision: {e}", exc_info=True)
            raise

    @mcp.tool()
    @graceful
    def decision_get(decision_id: str) -> str:
        """Retrieve a decision by ID."""
        try:
            from smartmemory.decisions.manager import DecisionManager

            backend = get_backend()
            manager = DecisionManager(backend)
            decision = manager.get_decision(decision_id)

            if not decision:
                return f"Decision not found: {decision_id}"

            return str(decision.to_dict())
        except Exception as e:
            logger.error(f"Failed to get decision: {e}", exc_info=True)
            raise

    @mcp.tool()
    @graceful
    def decision_list(
        domain: Optional[str] = None,
        decision_type: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 50,
    ) -> str:
        """List active decisions with optional filtering."""
        try:
            from smartmemory.decisions.queries import DecisionQueries

            backend = get_backend()
            queries = DecisionQueries(backend)
            decisions = queries.get_active_decisions(
                domain=domain,
                decision_type=decision_type,
                min_confidence=min_confidence,
                limit=limit,
            )

            if not decisions:
                return "No active decisions found."

            output = [f"Found {len(decisions)} active decisions:\n"]
            for d in decisions:
                output.append(
                    f"- [{d.decision_id}] ({d.decision_type}, conf={d.confidence:.2f}): "
                    f"{d.content[:100]}"
                )
            return "\n".join(output)
        except Exception as e:
            logger.error(f"Failed to list decisions: {e}", exc_info=True)
            raise

    @mcp.tool()
    @graceful
    def decision_search(topic: str, limit: int = 20) -> str:
        """Search for active decisions related to a topic."""
        try:
            from smartmemory.decisions.queries import DecisionQueries

            backend = get_backend()
            queries = DecisionQueries(backend)
            decisions = queries.get_decisions_about(topic=topic, limit=limit)

            if not decisions:
                return f"No decisions found about: {topic}"

            output = [f"Found {len(decisions)} decisions about '{topic}':\n"]
            for d in decisions:
                output.append(
                    f"- [{d.decision_id}] ({d.decision_type}, conf={d.confidence:.2f}): "
                    f"{d.content[:100]}"
                )
            return "\n".join(output)
        except Exception as e:
            logger.error(f"Failed to search decisions: {e}", exc_info=True)
            raise

    @mcp.tool()
    @graceful
    def decision_supersede(
        decision_id: str,
        new_content: str,
        reason: str,
        new_decision_type: str = "inference",
        new_confidence: float = 0.8,
    ) -> str:
        """Replace a decision with a new one, marking the old as superseded."""
        try:
            from smartmemory.decisions.manager import DecisionManager
            from smartmemory.models.decision import Decision

            backend = get_backend()
            manager = DecisionManager(backend)
            new_decision = Decision(
                content=new_content,
                decision_type=new_decision_type,
                confidence=new_confidence,
            )
            result = manager.supersede(decision_id, new_decision, reason=reason)

            return f"Decision {decision_id} superseded.\nNew decision: {result.decision_id}"
        except ValueError as e:
            return f"Decision not found: {e}"
        except Exception as e:
            logger.error(f"Failed to supersede decision: {e}", exc_info=True)
            raise

    @mcp.tool()
    @graceful
    def decision_retract(decision_id: str, reason: str) -> str:
        """Retract a decision, marking it as no longer valid."""
        try:
            from smartmemory.decisions.manager import DecisionManager

            backend = get_backend()
            manager = DecisionManager(backend)
            manager.retract(decision_id, reason=reason)
            return f"Decision retracted: {decision_id}"
        except ValueError as e:
            return f"Decision not found: {e}"
        except Exception as e:
            logger.error(f"Failed to retract decision: {e}", exc_info=True)
            raise

    @mcp.tool()
    @graceful
    def decision_reinforce(decision_id: str, evidence_id: str) -> str:
        """Record supporting evidence for a decision."""
        try:
            from smartmemory.decisions.manager import DecisionManager

            backend = get_backend()
            manager = DecisionManager(backend)
            decision = manager.reinforce(decision_id, evidence_id)

            return (
                f"Decision reinforced: {decision_id}\n"
                f"New confidence: {decision.confidence:.2f}\n"
                f"Reinforcement count: {decision.reinforcement_count}"
            )
        except ValueError as e:
            return f"Decision not found: {e}"
        except Exception as e:
            logger.error(f"Failed to reinforce decision: {e}", exc_info=True)
            raise

    @mcp.tool()
    @graceful
    def decision_contradict(decision_id: str, evidence_id: str) -> str:
        """Record contradicting evidence against a decision."""
        try:
            from smartmemory.decisions.manager import DecisionManager

            backend = get_backend()
            manager = DecisionManager(backend)
            decision = manager.contradict(decision_id, evidence_id)

            return (
                f"Decision contradicted: {decision_id}\n"
                f"New confidence: {decision.confidence:.2f}\n"
                f"Contradiction count: {decision.contradiction_count}"
            )
        except ValueError as e:
            return f"Decision not found: {e}"
        except Exception as e:
            logger.error(f"Failed to contradict decision: {e}", exc_info=True)
            raise

    @mcp.tool()
    @graceful
    def decision_provenance(decision_id: str) -> str:
        """Get full provenance chain for a decision."""
        try:
            from smartmemory.decisions.queries import DecisionQueries

            backend = get_backend()
            queries = DecisionQueries(backend)
            provenance = queries.get_decision_provenance(decision_id)

            if provenance["decision"] is None:
                return f"Decision not found: {decision_id}"

            parts = [f"Provenance for {decision_id}:"]
            parts.append(f"Decision: {provenance['decision'].content}")
            if provenance.get("reasoning_trace"):
                parts.append(f"Reasoning trace: {provenance['reasoning_trace']}")
            if provenance.get("evidence"):
                parts.append(f"Evidence: {len(provenance['evidence'])} items")
            if provenance.get("superseded"):
                parts.append(f"Superseded: {len(provenance['superseded'])} decisions")
            return "\n".join(parts)
        except Exception as e:
            logger.error(f"Failed to get provenance: {e}", exc_info=True)
            raise

    @mcp.tool()
    @graceful
    def decision_find_conflicts(decision_id: str) -> str:
        """Find existing decisions that may conflict with this one."""
        try:
            from smartmemory.decisions.manager import DecisionManager

            backend = get_backend()
            manager = DecisionManager(backend)
            decision = manager.get_decision(decision_id)

            if not decision:
                return f"Decision not found: {decision_id}"

            conflicts = manager.find_conflicts(decision)

            if not conflicts:
                return f"No conflicts found for decision {decision_id}"

            output = [f"Found {len(conflicts)} conflicts for {decision_id}:\n"]
            for c in conflicts:
                output.append(f"- [{c.decision_id}] ({c.decision_type}): {c.content[:100]}")
            return "\n".join(output)
        except Exception as e:
            logger.error(f"Failed to find conflicts: {e}", exc_info=True)
            raise
