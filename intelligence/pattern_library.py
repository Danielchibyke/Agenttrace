import json
import logging
import numpy as np
import asyncio
from typing import Optional
from storage.database import DatabaseWriter

logger = logging.getLogger(__name__)

CREATE_PATTERNS_TABLE = """
CREATE TABLE IF NOT EXISTS execution_patterns (
    pattern_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    task_type TEXT NOT NULL,
    outcome TEXT NOT NULL,
    trajectory_vector vector(896),
    start_vector vector(896),
    end_vector vector(896),
    progress_scores JSONB,
    repetition_score FLOAT,
    error_count INTEGER,
    total_steps INTEGER,
    reasoning_hops INTEGER,
    behavioral_signals JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_patterns_task_type
    ON execution_patterns(task_type);
CREATE INDEX IF NOT EXISTS idx_patterns_outcome
    ON execution_patterns(outcome);
"""


class PatternLibrary:
    """
    Stores and queries execution patterns.
    The ground truth for drift detection and prediction.
    Grows smarter with every run.
    """

    def __init__(self, db_writer: DatabaseWriter):
        self.db = db_writer
        self._success_centroids: dict[
            str, list[np.ndarray]
        ] = {}
        self._failure_centroids: dict[
            str, list[np.ndarray]
        ] = {}
        self._loaded = False

    async def setup(self):
        async with self.db.pool.acquire() as conn:
            await conn.execute(CREATE_PATTERNS_TABLE)
        await self._load_into_memory()
        logger.info("PatternLibrary initialized.")

    async def store_pattern(
        self,
        session_id: str,
        task_type: str,
        outcome: str,
        node_embeddings: list[list[float]],
        progress_scores: list[float] = None,
        repetition_score: float = None,
        error_count: int = 0,
        total_steps: int = 0,
        reasoning_hops: int = 0,
        behavioral_signals: list[dict] = None,
    ) -> str:
        """
        Store a completed execution as a pattern.
        Called after every session completes.
        """
        import uuid
        pattern_id = str(uuid.uuid4())

        if not node_embeddings:
            logger.warning(
                f"No embeddings for session {session_id}"
            )
            return None

        matrix = np.array(
            node_embeddings, dtype=np.float32
        )
        trajectory_vector = matrix.mean(axis=0).tolist()
        start_vector = matrix[0].tolist()
        end_vector = matrix[-1].tolist()

        def vec_str(v):
            return "[" + ",".join(str(x) for x in v) + "]"

        async with self.db.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO execution_patterns (
                    pattern_id, session_id, task_type,
                    outcome, trajectory_vector,
                    start_vector, end_vector,
                    progress_scores, repetition_score,
                    error_count, total_steps,
                    reasoning_hops, behavioral_signals
                ) VALUES (
                    $1,$2,$3,$4,
                    $5::vector,$6::vector,$7::vector,
                    $8,$9,$10,$11,$12,$13
                )
                """,
                pattern_id, session_id, task_type,
                outcome,
                vec_str(trajectory_vector),
                vec_str(start_vector),
                vec_str(end_vector),
                json.dumps(progress_scores or []),
                repetition_score,
                error_count,
                total_steps,
                reasoning_hops,
                json.dumps(behavioral_signals or []),
            )

        # update in-memory centroids
        centroid = np.array(
            trajectory_vector, dtype=np.float32
        )
        if outcome == "success":
            if task_type not in self._success_centroids:
                self._success_centroids[task_type] = []
            self._success_centroids[task_type].append(
                centroid
            )
        else:
            if task_type not in self._failure_centroids:
                self._failure_centroids[task_type] = []
            self._failure_centroids[task_type].append(
                centroid
            )

        logger.info(
            f"Pattern stored: {outcome} — "
            f"task_type={task_type}"
        )
        return pattern_id

    async def compute_drift_score(
        self,
        current_embedding: list[float],
        task_type: str,
    ) -> dict:
        """
        Compare current position against known patterns.
        Returns drift score and classification.
        Works immediately from behavioral signals.
        Gets more precise as pattern library grows.
        """
        current = np.array(
            current_embedding, dtype=np.float32
        )

        success_centroids = self._success_centroids.get(
            task_type, []
        )
        failure_centroids = self._failure_centroids.get(
            task_type, []
        )

        success_sim = self._max_similarity(
            current, success_centroids
        )
        failure_sim = self._max_similarity(
            current, failure_centroids
        )

        # drift score — 0 = safe, 1 = certain failure
        if success_sim == 0 and failure_sim == 0:
            drift_score = 0.0
            classification = "unknown"
            confidence = "low"
        elif failure_sim == 0:
            drift_score = 0.0
            classification = "healthy"
            confidence = "medium"
        elif success_sim == 0:
            drift_score = 1.0
            classification = "failing"
            confidence = "medium"
        else:
            drift_score = failure_sim / (
                failure_sim + success_sim
            )
            classification = (
                "failing" if drift_score > 0.6
                else "drifting" if drift_score > 0.4
                else "healthy"
            )
            confidence = "high"

        return {
            "drift_score": round(float(drift_score), 4),
            "classification": classification,
            "confidence": confidence,
            "success_similarity": round(
                float(success_sim), 4
            ),
            "failure_similarity": round(
                float(failure_sim), 4
            ),
            "patterns_available": {
                "success": len(success_centroids),
                "failure": len(failure_centroids),
            }
        }

    def _max_similarity(
        self,
        vector: np.ndarray,
        centroids: list[np.ndarray]
    ) -> float:
        if not centroids:
            return 0.0

        similarities = []
        for centroid in centroids:
            norm_v = np.linalg.norm(vector)
            norm_c = np.linalg.norm(centroid)
            if norm_v == 0 or norm_c == 0:
                continue
            sim = float(
                np.dot(vector, centroid) /
                (norm_v * norm_c)
            )
            similarities.append(sim)

        return max(similarities) if similarities else 0.0

    async def _load_into_memory(self):
        """Load all patterns into RAM on startup."""
        async with self.db.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT task_type, outcome,
                trajectory_vector::text
                FROM execution_patterns
                """
            )

        for row in rows:
            vec_str = row["trajectory_vector"]
            if not vec_str:
                continue
            vec = np.array(
                [
                    float(x)
                    for x in vec_str.strip("[]").split(",")
                ],
                dtype=np.float32
            )
            task_type = row["task_type"]
            outcome = row["outcome"]

            if outcome == "success":
                if task_type not in self._success_centroids:
                    self._success_centroids[task_type] = []
                self._success_centroids[task_type].append(vec)
            else:
                if task_type not in self._failure_centroids:
                    self._failure_centroids[task_type] = []
                self._failure_centroids[task_type].append(vec)

        total = sum(
            len(v) for v in self._success_centroids.values()
        ) + sum(
            len(v) for v in self._failure_centroids.values()
        )
        logger.info(
            f"Loaded {total} patterns into memory."
        )
        self._loaded = True

    async def get_stats(self) -> dict:
        async with self.db.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT task_type, outcome, COUNT(*) as count
                FROM execution_patterns
                GROUP BY task_type, outcome
                ORDER BY task_type, outcome
                """
            )
        return [dict(row) for row in rows]