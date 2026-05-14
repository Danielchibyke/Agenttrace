import logging
from typing import Optional
from intelligence.behavioral_signals import BehavioralSignalEngine
from intelligence.pattern_library import PatternLibrary
from intelligence.progress_scorer import ProgressScorer
from tracer.node import Node

logger = logging.getLogger(__name__)

DRIFT_ALERT_THRESHOLD = 0.02
CRITICAL_ALERT_THRESHOLD = 0.08


class DriftDetector:
    """
    Combines behavioral signals and geometric pattern
    matching into a single drift detection system.

    Layer 1 — behavioral signals
    Works immediately. No historical data needed.
    Generalizes across all task types.

    Layer 2 — geometric pattern matching
    Works when pattern library has enough history.
    Task-specific. High precision.
    """

    def __init__(
        self,
        pattern_library: PatternLibrary,
        task_type: str,
        goal_embedding: list[float] = None,
    ):
        self.pattern_library = pattern_library
        self.task_type = task_type
        self.behavioral_engine = BehavioralSignalEngine()
        self.progress_scorer = (
            ProgressScorer(goal_embedding)
            if goal_embedding else None
        )
        self._drift_history: list[float] = []
        self._alerts: list[dict] = []

    async def process_node(
        self, node: Node
    ) -> dict:
        """
        Process a new node through both detection layers.
        Returns current health assessment.
        Called after every node is captured.
        """
        result = {
            "node_id": node.node_id,
            "node_type": node.node_type.value,
            "step_number": node.step_number,
            "alerts": [],
            "drift_score": 0.0,
            "classification": "healthy",
            "progress_score": None,
            "progress_trend": None,
            "action_required": False,
        }

        # Layer 1 — behavioral signals
        behavioral_alerts = (
            self.behavioral_engine.process_node(node)
        )
        result["alerts"].extend(behavioral_alerts)

        # progress scoring
        if (self.progress_scorer and
                node.embedding_vector):
            progress = self.progress_scorer.score(
                node.embedding_vector
            )
            result["progress_score"] = progress
            result["progress_trend"] = (
                self.progress_scorer.get_trend()
            )

            progress_alert = (
                self.progress_scorer.get_velocity() < -0.1
            )
            if progress_alert:
                result["alerts"].append({
                    "type": "progress_declining",
                    "severity": "high",
                    "message": "Agent moving away from goal.",
                    "value": progress,
                })

        # Layer 2 — geometric pattern matching
        if node.embedding_vector:
            drift = await (
                self.pattern_library.compute_drift_score(
                    node.embedding_vector,
                    self.task_type,
                )
            )
            result["drift_score"] = drift["drift_score"]
            result["classification"] = (
                drift["classification"]
            )
            result["patterns_available"] = (
                drift["patterns_available"]
            )
            self._drift_history.append(
                drift["drift_score"]
            )

            # geometric drift alert
            if drift["drift_score"] > DRIFT_ALERT_THRESHOLD:
                severity = (
                    "critical"
                    if drift["drift_score"] >
                    CRITICAL_ALERT_THRESHOLD
                    else "high"
                )
                result["alerts"].append({
                    "type": "geometric_drift",
                    "severity": severity,
                    "message": (
                        f"Trajectory heading toward "
                        f"failure region. "
                        f"Drift score: "
                        f"{drift['drift_score']:.2f}"
                    ),
                    "value": drift["drift_score"],
                })

        # determine if action required
        critical_alerts = [
            a for a in result["alerts"]
            if a["severity"] == "critical"
        ]
        high_alerts = [
            a for a in result["alerts"]
            if a["severity"] == "high"
        ]
        result["action_required"] = (
            len(critical_alerts) > 0 or
            len(high_alerts) >= 2
        )

        if result["alerts"]:
            self._alerts.extend(result["alerts"])
            logger.warning(
                f"Drift alerts at step "
                f"{node.step_number}: "
                f"{[a['type'] for a in result['alerts']]}"
            )

        return result

    def get_trajectory_summary(self) -> dict:
        behavioral = (
            self.behavioral_engine.get_health_status()
        )
        progress = (
            self.progress_scorer.summary()
            if self.progress_scorer else None
        )

        avg_drift = (
            sum(self._drift_history) /
            len(self._drift_history)
            if self._drift_history else 0.0
        )

        return {
            "behavioral_health": behavioral,
            "progress": progress,
            "average_drift_score": round(avg_drift, 4),
            "peak_drift_score": (
                max(self._drift_history)
                if self._drift_history else 0.0
            ),
            "total_alerts": len(self._alerts),
            "action_required": any(
                a["severity"] in ["critical", "high"]
                for a in self._alerts[-5:]
            ),
        }