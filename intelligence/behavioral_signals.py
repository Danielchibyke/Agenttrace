import logging
from typing import Optional
from tracer.node import Node, NodeType

logger = logging.getLogger(__name__)


class BehavioralSignalEngine:
    """
    Computes universal behavioral signals that detect
    failure patterns regardless of task type.
    Works immediately with zero historical data.
    Generalizes across all agent frameworks and tasks.
    """

    # thresholds
    REPETITION_THRESHOLD = 0.7
    MAX_REASONING_HOPS = 8
    MAX_CONSECUTIVE_ERRORS = 2
    MIN_PROGRESS_DELTA = -0.1
    LATENCY_SPIKE_MULTIPLIER = 3.0
    MIN_TOOL_DIVERSITY = 0.2

    def __init__(self):
        self._response_history: list[str] = []
        self._tool_history: list[str] = []
        self._error_history: list[bool] = []
        self._latency_history: list[float] = []
        self._progress_history: list[float] = []
        self._reasoning_hop_count: int = 0
        self._alerts: list[dict] = []

    def process_node(self, node: Node) -> list[dict]:
        """
        Process a new node and return any alerts fired.
        Called after every node is captured.
        """
        alerts = []

        if node.node_type == NodeType.REASONING:
            self._reasoning_hop_count += 1
            if node.response_text:
                self._response_history.append(
                    node.response_text
                )
            if node.latency_ms:
                self._latency_history.append(node.latency_ms)

            # check reasoning hop explosion
            alert = self._check_reasoning_hops()
            if alert:
                alerts.append(alert)

            # check repetition
            alert = self._check_repetition()
            if alert:
                alerts.append(alert)

            # check latency spike
            alert = self._check_latency_spike(node.latency_ms)
            if alert:
                alerts.append(alert)

        elif node.node_type == NodeType.TOOL_CALL:
            if node.tool_name:
                self._tool_history.append(node.tool_name)

            # check tool diversity
            alert = self._check_tool_diversity()
            if alert:
                alerts.append(alert)

        # track errors
        is_error = node.status == "error"
        self._error_history.append(is_error)
        if is_error:
            alert = self._check_consecutive_errors()
            if alert:
                alerts.append(alert)

        self._alerts.extend(alerts)
        return alerts

    def update_progress(
        self, progress_score: float
    ) -> Optional[dict]:
        """
        Called when a new progress score is computed.
        Returns alert if progress is declining.
        """
        self._progress_history.append(progress_score)
        return self._check_progress_decline()

    def _check_reasoning_hops(self) -> Optional[dict]:
        if self._reasoning_hop_count > self.MAX_REASONING_HOPS:
            return {
                "type": "reasoning_explosion",
                "severity": "high",
                "message": (
                    f"Agent has made "
                    f"{self._reasoning_hop_count} reasoning "
                    f"hops without completing task. "
                    f"Possible decision paralysis."
                ),
                "value": self._reasoning_hop_count,
                "threshold": self.MAX_REASONING_HOPS,
            }
        return None

    def _check_repetition(self) -> Optional[dict]:
        if len(self._response_history) < 3:
            return None

        last_three = self._response_history[-3:]
        score = self._compute_repetition_score(last_three)

        if score > self.REPETITION_THRESHOLD:
            return {
                "type": "repetition_detected",
                "severity": "high",
                "message": (
                    "Agent is repeating itself. "
                    "Possible reasoning loop detected."
                ),
                "value": round(score, 3),
                "threshold": self.REPETITION_THRESHOLD,
            }
        return None

    def _check_consecutive_errors(self) -> Optional[dict]:
        if len(self._error_history) < self.MAX_CONSECUTIVE_ERRORS:
            return None

        recent = self._error_history[
            -self.MAX_CONSECUTIVE_ERRORS:
        ]
        if all(recent):
            return {
                "type": "consecutive_errors",
                "severity": "critical",
                "message": (
                    f"{self.MAX_CONSECUTIVE_ERRORS} consecutive "
                    f"errors detected. Agent is failing repeatedly."
                ),
                "value": self.MAX_CONSECUTIVE_ERRORS,
                "threshold": self.MAX_CONSECUTIVE_ERRORS,
            }
        return None

    def _check_tool_diversity(self) -> Optional[dict]:
        if len(self._tool_history) < 5:
            return None

        recent = self._tool_history[-5:]
        unique = len(set(recent))
        diversity = unique / len(recent)

        if diversity < self.MIN_TOOL_DIVERSITY:
            return {
                "type": "low_tool_diversity",
                "severity": "medium",
                "message": (
                    "Agent is repeatedly using the same tool. "
                    "Possible stuck execution pattern."
                ),
                "value": round(diversity, 3),
                "threshold": self.MIN_TOOL_DIVERSITY,
            }
        return None

    def _check_latency_spike(
        self, current_latency: Optional[float]
    ) -> Optional[dict]:
        if not current_latency:
            return None
        if len(self._latency_history) < 3:
            return None

        avg = sum(self._latency_history[:-1]) / (
            len(self._latency_history) - 1
        )
        if avg == 0:
            return None

        spike = current_latency / avg
        if spike > self.LATENCY_SPIKE_MULTIPLIER:
            return {
                "type": "latency_spike",
                "severity": "medium",
                "message": (
                    f"Latency spike detected — "
                    f"{round(spike, 1)}x average. "
                    f"Agent may be struggling."
                ),
                "value": round(spike, 1),
                "threshold": self.LATENCY_SPIKE_MULTIPLIER,
            }
        return None

    def _check_progress_decline(self) -> Optional[dict]:
        if len(self._progress_history) < 3:
            return None

        recent = self._progress_history[-3:]
        delta = recent[-1] - recent[0]

        if delta < self.MIN_PROGRESS_DELTA:
            return {
                "type": "progress_declining",
                "severity": "high",
                "message": (
                    "Agent progress is declining. "
                    "Moving away from task goal."
                ),
                "value": round(delta, 3),
                "threshold": self.MIN_PROGRESS_DELTA,
            }
        return None

    def _compute_repetition_score(
        self, texts: list[str]
    ) -> float:
        """
        Simple repetition score based on word overlap.
        Returns 0.0 to 1.0 — higher means more repetitive.
        """
        if len(texts) < 2:
            return 0.0

        sets = [set(t.lower().split()) for t in texts]
        overlaps = []

        for i in range(len(sets) - 1):
            if not sets[i] or not sets[i + 1]:
                continue
            intersection = sets[i] & sets[i + 1]
            union = sets[i] | sets[i + 1]
            if union:
                overlaps.append(len(intersection) / len(union))

        return sum(overlaps) / len(overlaps) if overlaps else 0.0

    def get_health_status(self) -> dict:
        """
        Current health summary of the execution.
        """
        critical = [
            a for a in self._alerts
            if a["severity"] == "critical"
        ]
        high = [
            a for a in self._alerts
            if a["severity"] == "high"
        ]
        medium = [
            a for a in self._alerts
            if a["severity"] == "medium"
        ]

        if critical:
            status = "critical"
        elif high:
            status = "warning"
        elif medium:
            status = "caution"
        else:
            status = "healthy"

        return {
            "status": status,
            "total_alerts": len(self._alerts),
            "critical": len(critical),
            "high": len(high),
            "medium": len(medium),
            "reasoning_hops": self._reasoning_hop_count,
            "recent_alerts": self._alerts[-5:],
        }