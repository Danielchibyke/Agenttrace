import sys
import os
sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))

import asyncio
import json
import logging
import numpy as np
from storage.database import DatabaseWriter

logger = logging.getLogger(__name__)


class TrajectoryCollector:
    """
    Collects and prepares trajectory data from
    the database for mathematical analysis.
    """

    def __init__(self, db_writer: DatabaseWriter):
        self.db = db_writer

    async def get_all_trajectories(self) -> list[dict]:
        """
        Fetch all sessions with their full trajectory
        vectors and outcomes from pattern library.
        """
        async with self.db.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT
                    ep.pattern_id,
                    ep.session_id,
                    ep.outcome,
                    ep.task_type,
                    ep.total_steps,
                    ep.reasoning_hops,
                    ep.error_count,
                    ep.trajectory_vector::text as trajectory_vector,
                    ep.start_vector::text as start_vector,
                    ep.end_vector::text as end_vector
                FROM execution_patterns ep
                WHERE ep.trajectory_vector IS NOT NULL
                ORDER BY ep.created_at
            """)

        trajectories = []
        for row in rows:
            traj_vec = self._parse_vector(
                row["trajectory_vector"]
            )
            start_vec = self._parse_vector(
                row["start_vector"]
            )
            end_vec = self._parse_vector(
                row["end_vector"]
            )

            if traj_vec is None:
                continue

            trajectories.append({
                "pattern_id": row["pattern_id"],
                "session_id": row["session_id"],
                "outcome": row["outcome"],
                "task_type": row["task_type"],
                "total_steps": row["total_steps"],
                "reasoning_hops": row["reasoning_hops"],
                "error_count": row["error_count"],
                "trajectory_vector": np.array(
                    traj_vec, dtype=np.float32
                ),
                "start_vector": np.array(
                    start_vec, dtype=np.float32
                ) if start_vec else None,
                "end_vector": np.array(
                    end_vec, dtype=np.float32
                ) if end_vec else None,
            })

        return trajectories

    async def get_step_vectors(
        self, session_id: str
    ) -> list[dict]:
        """
        Fetch individual node vectors for a session
        in step order. Used for temporal analysis.
        """
        async with self.db.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT
                    node_id,
                    node_type,
                    step_number,
                    tool_name,
                    status,
                    latency_ms,
                    embedding_vector::text as embedding_vector
                FROM nodes
                WHERE session_id = $1
                AND embedding_vector IS NOT NULL
                ORDER BY step_number
            """, session_id)

        steps = []
        for row in rows:
            vec = self._parse_vector(
                row["embedding_vector"]
            )
            if vec is None:
                continue
            steps.append({
                "node_id": row["node_id"],
                "node_type": row["node_type"],
                "step_number": row["step_number"],
                "tool_name": row.get("tool_name"),
                "status": row.get("status"),
                "latency_ms": row.get("latency_ms"),
                "vector": np.array(vec, dtype=np.float32),
            })

        return steps

    async def get_sessions_by_outcome(self) -> dict:
        """
        Get all session IDs grouped by outcome.
        """
        async with self.db.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT session_id, outcome
                FROM execution_patterns
                WHERE trajectory_vector IS NOT NULL
            """)

        result = {"success": [], "failure": [], "partial": []}
        for row in rows:
            outcome = row["outcome"]
            if outcome in result:
                result[outcome].append(row["session_id"])
        return result

    def _parse_vector(self, vec_str) -> list[float]:
        try:
            if not vec_str:
                return None
            if isinstance(vec_str, str):
                return [
                    float(x)
                    for x in vec_str.strip("[]").split(",")
                ]
            return None
        except Exception:
            return None