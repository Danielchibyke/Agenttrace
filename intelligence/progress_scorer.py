import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


class ProgressScorer:
    """
    Measures how much closer the agent is getting
    to its goal at each step.
    Goal-relative — works for any task type.
    """

    def __init__(self, goal_embedding: list[float]):
        self.goal_embedding = np.array(
            goal_embedding, dtype=np.float32
        )
        self._scores: list[float] = []
        self._initial_distance: Optional[float] = None

    def score(
        self, current_embedding: list[float]
    ) -> float:
        """
        Compute progress score for current node.
        Returns value between 0.0 and 1.0.
        1.0 = at goal. 0.0 = as far as initial state.
        Negative = moving away from goal.
        """
        current = np.array(
            current_embedding, dtype=np.float32
        )

        distance = self._cosine_distance(
            current, self.goal_embedding
        )

        if self._initial_distance is None:
            self._initial_distance = distance
            score = 0.5
        elif self._initial_distance == 0:
            score = 1.0
        else:
            score = 1.0 - (
                distance / self._initial_distance
            )

        self._scores.append(score)
        return round(float(score), 4)

    def get_trend(self) -> str:
        """
        Returns the trend of recent progress scores.
        improving, declining, stalled, or insufficient_data
        """
        if len(self._scores) < 3:
            return "insufficient_data"

        recent = self._scores[-3:]
        delta = recent[-1] - recent[0]

        if delta > 0.05:
            return "improving"
        elif delta < -0.05:
            return "declining"
        else:
            return "stalled"

    def get_velocity(self) -> float:
        """
        Rate of progress change.
        Positive = accelerating toward goal.
        Negative = moving away.
        """
        if len(self._scores) < 2:
            return 0.0
        return round(
            self._scores[-1] - self._scores[-2], 4
        )

    def _cosine_distance(
        self,
        a: np.ndarray,
        b: np.ndarray
    ) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 1.0
        similarity = np.dot(a, b) / (norm_a * norm_b)
        return float(1.0 - similarity)

    def summary(self) -> dict:
        return {
            "current_score": (
                self._scores[-1] if self._scores else None
            ),
            "trend": self.get_trend(),
            "velocity": self.get_velocity(),
            "total_measurements": len(self._scores),
        }