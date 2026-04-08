import json
import uuid
import logging
import asyncio
import numpy as np
from typing import Optional
from storage.database import DatabaseWriter

logger = logging.getLogger(__name__)

CREATE_PATTERNS_TABLE = """
CREATE TABLE IF NOT EXISTS execution_patterns (
    pattern_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    task_type TEXT NOT NULL,
    task_cluster_id TEXT,
    outcome TEXT NOT NULL,
    trajectory_vector vector(896),
    start_vector vector(896),
    end_vector vector(896),
    progress_scores JSONB,
    repetition_score FLOAT,
    error_count INTEGER DEFAULT 0,
    total_steps INTEGER DEFAULT 0,
    reasoning_hops INTEGER DEFAULT 0,
    behavioral_signals JSONB,
    is_representative BOOLEAN DEFAULT FALSE,
    similarity_to_centroid FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_patterns_task_type
    ON execution_patterns(task_type);
CREATE INDEX IF NOT EXISTS idx_patterns_outcome
    ON execution_patterns(outcome);
CREATE INDEX IF NOT EXISTS idx_patterns_cluster
    ON execution_patterns(task_cluster_id);
CREATE INDEX IF NOT EXISTS idx_patterns_representative
    ON execution_patterns(is_representative);
"""

CREATE_TASK_CLUSTERS_TABLE = """
CREATE TABLE IF NOT EXISTS task_clusters (
    cluster_id TEXT PRIMARY KEY,
    task_type TEXT NOT NULL,
    centroid_vector vector(896),
    pattern_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    last_updated TIMESTAMPTZ DEFAULT NOW()
);
"""

# similarity threshold for deduplication
DEDUP_THRESHOLD = 0.97

# minimum distance to form a new cluster
NEW_CLUSTER_THRESHOLD = 0.3


class PatternLibrary:
    """
    Automatic pattern classification and storage.
    No manual labeling required.
    Self-organizing — sorts patterns automatically.
    Self-pruning — removes redundant patterns.
    Fast — all queries run from RAM.
    """

    def __init__(self, db_writer: DatabaseWriter):
        self.db = db_writer

        # in-memory structures for fast querying
        self._success_vectors: dict[
            str, list[np.ndarray]
        ] = {}
        self._failure_vectors: dict[
            str, list[np.ndarray]
        ] = {}
        self._task_clusters: dict[
            str, dict
        ] = {}
        self._loaded = False
        self._lock = asyncio.Lock()

    async def setup(self):
        async with self.db.pool.acquire() as conn:
            await conn.execute(CREATE_PATTERNS_TABLE)
            await conn.execute(CREATE_TASK_CLUSTERS_TABLE)
        await self._load_into_memory()
        logger.info("PatternLibrary ready.")

    # -------------------------------------------------
    # AUTOMATIC INTAKE — called after every session
    # -------------------------------------------------

    async def ingest(
        self,
        session_id: str,
        goal_text: str,
        goal_embedding: list[float],
        node_embeddings: list[list[float]],
        behavioral_health: dict,
        progress_scores: list[float] = None,
        error_count: int = 0,
        total_steps: int = 0,
        reasoning_hops: int = 0,
    ) -> dict:
        """
        Automatic intake pipeline.
        Classifies task type, labels outcome,
        deduplicates, and stores — all automatically.
        Runs in background without blocking agent.
        """
        if not node_embeddings:
            return {"status": "skipped", "reason": "no embeddings"}

        matrix = np.array(node_embeddings, dtype=np.float32)
        trajectory_vector = matrix.mean(axis=0)
        start_vector = matrix[0]
        end_vector = matrix[-1]

        # step 1 — auto classify task type
        task_type, cluster_id = await self._classify_task(
            goal_embedding, goal_text
        )

        # step 2 — auto label outcome
        outcome = self._label_outcome(
            behavioral_health, error_count, progress_scores
        )

        # step 3 — deduplication check
        is_duplicate = await self._is_duplicate(
            trajectory_vector, task_type, outcome
        )
        if is_duplicate:
            logger.info(
                f"Pattern deduplicated — "
                f"too similar to existing {outcome} pattern"
            )
            return {
                "status": "deduplicated",
                "task_type": task_type,
                "outcome": outcome,
            }

        # step 4 — compute representativeness
        similarity = self._similarity_to_centroid(
            trajectory_vector, task_type, outcome
        )
        is_representative = similarity > 0.8

        # step 5 — store pattern
        pattern_id = await self._store(
            session_id=session_id,
            task_type=task_type,
            cluster_id=cluster_id,
            outcome=outcome,
            trajectory_vector=trajectory_vector,
            start_vector=start_vector,
            end_vector=end_vector,
            progress_scores=progress_scores or [],
            error_count=error_count,
            total_steps=total_steps,
            reasoning_hops=reasoning_hops,
            behavioral_signals=behavioral_health.get(
                "recent_alerts", []
            ),
            is_representative=is_representative,
            similarity=similarity,
        )

        # step 6 — update in-memory structures
        async with self._lock:
            self._update_memory(
                task_type, outcome, trajectory_vector
            )

        # step 7 — update cluster stats
        await self._update_cluster(
            cluster_id, task_type, outcome
        )

        logger.info(
            f"Pattern ingested — "
            f"task={task_type} "
            f"outcome={outcome} "
            f"representative={is_representative}"
        )

        return {
            "status": "stored",
            "pattern_id": pattern_id,
            "task_type": task_type,
            "outcome": outcome,
            "is_representative": is_representative,
            "cluster_id": cluster_id,
        }

    # -------------------------------------------------
    # AUTO TASK CLASSIFICATION
    # -------------------------------------------------

    async def _classify_task(
        self,
        goal_embedding: list[float],
        goal_text: str,
    ) -> tuple[str, str]:
        """
        Automatically classify task type by comparing
        goal embedding against existing clusters.
        Creates new cluster if no match found.
        """
        if not goal_embedding:
            return "unknown", "unknown"

        goal_vec = np.array(
            goal_embedding, dtype=np.float32
        )

        best_cluster = None
        best_similarity = 0.0

        for cluster_id, cluster in (
            self._task_clusters.items()
        ):
            centroid = cluster["centroid"]
            sim = self._cosine_similarity(
                goal_vec, centroid
            )
            if sim > best_similarity:
                best_similarity = sim
                best_cluster = cluster_id

        # close enough to existing cluster
        if (best_cluster and
                best_similarity > (1 - NEW_CLUSTER_THRESHOLD)):
            task_type = self._task_clusters[
                best_cluster
            ]["task_type"]
            return task_type, best_cluster

        # create new cluster
        cluster_id = str(uuid.uuid4())
        task_type = self._infer_task_type(goal_text)

        self._task_clusters[cluster_id] = {
            "task_type": task_type,
            "centroid": goal_vec,
            "count": 1,
        }

        vec_str = (
            "[" + ",".join(str(x) for x in goal_vec) + "]"
        )

        async with self.db.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO task_clusters (
                    cluster_id, task_type, centroid_vector
                ) VALUES ($1, $2, $3::vector)
                ON CONFLICT (cluster_id) DO NOTHING
                """,
                cluster_id, task_type, vec_str
            )

        logger.info(
            f"New task cluster created: "
            f"{task_type} ({cluster_id[:8]})"
        )
        return task_type, cluster_id

    def _infer_task_type(self, goal_text: str) -> str:
        """
        Infer task type from goal text keywords.
        Falls back to 'general' if no match.
        """
        goal_lower = goal_text.lower()

        type_keywords = {
            "code": [
                "code", "write", "function", "debug",
                "implement", "program", "script"
            ],
            "search": [
                "search", "find", "lookup", "research",
                "information", "query"
            ],
            "analysis": [
                "analyze", "analyse", "evaluate",
                "assess", "review", "compare"
            ],
            "generation": [
                "generate", "create", "produce",
                "draft", "write", "compose"
            ],
            "calculation": [
                "calculate", "compute", "math",
                "solve", "equation", "number"
            ],
            "retrieval": [
                "retrieve", "fetch", "get", "load",
                "read", "extract"
            ],
        }

        for task_type, keywords in type_keywords.items():
            if any(k in goal_lower for k in keywords):
                return task_type

        return "general"

    # -------------------------------------------------
    # AUTO OUTCOME LABELING
    # -------------------------------------------------

    def _label_outcome(
        self,
        behavioral_health: dict,
        error_count: int,
        progress_scores: list[float] = None,
    ) -> str:
        """
        Automatically label outcome from behavioral signals.
        No manual labeling required.
        """
        status = behavioral_health.get("status", "healthy")

        if status == "critical":
            return "failure"

        if error_count >= 3:
            return "failure"

        if progress_scores and len(progress_scores) >= 3:
            final_progress = progress_scores[-1]
            if final_progress < 0.2:
                return "failure"

        if status == "warning":
            return "partial"

        return "success"

    # -------------------------------------------------
    # DEDUPLICATION
    # -------------------------------------------------

    async def _is_duplicate(
        self,
        trajectory_vector: np.ndarray,
        task_type: str,
        outcome: str,
    ) -> bool:
        """
        Check if pattern is too similar to existing ones.
        Keeps library lean and representative.
        """
        if outcome == "success":
            existing = self._success_vectors.get(
                task_type, []
            )
        else:
            existing = self._failure_vectors.get(
                task_type, []
            )

        for vec in existing:
            sim = self._cosine_similarity(
                trajectory_vector, vec
            )
            if sim > DEDUP_THRESHOLD:
                return True

        return False

    # -------------------------------------------------
    # DRIFT SCORING — fast in-memory query
    # -------------------------------------------------

    async def compute_drift_score(
        self,
        current_embedding: list[float],
        task_type: str,
    ) -> dict:
        """
        Fast in-memory drift computation.
        No database query — pure RAM operation.
        """
        current = np.array(
            current_embedding, dtype=np.float32
        )

        success_vecs = self._success_vectors.get(
            task_type, []
        ) + self._success_vectors.get("general", [])

        failure_vecs = self._failure_vectors.get(
            task_type, []
        ) + self._failure_vectors.get("general", [])

        success_sim = self._max_similarity(
            current, success_vecs
        )
        failure_sim = self._max_similarity(
            current, failure_vecs
        )

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
                "success": len(success_vecs),
                "failure": len(failure_vecs),
            }
        }

    # -------------------------------------------------
    # INTERNAL HELPERS
    # -------------------------------------------------

    def _update_memory(
        self,
        task_type: str,
        outcome: str,
        vector: np.ndarray,
    ):
        if outcome == "success":
            if task_type not in self._success_vectors:
                self._success_vectors[task_type] = []
            self._success_vectors[task_type].append(vector)
        else:
            if task_type not in self._failure_vectors:
                self._failure_vectors[task_type] = []
            self._failure_vectors[task_type].append(vector)

    def _similarity_to_centroid(
        self,
        vector: np.ndarray,
        task_type: str,
        outcome: str,
    ) -> float:
        if outcome == "success":
            existing = self._success_vectors.get(
                task_type, []
            )
        else:
            existing = self._failure_vectors.get(
                task_type, []
            )

        if not existing:
            return 0.5

        centroid = np.mean(
            np.array(existing, dtype=np.float32),
            axis=0
        )
        return float(
            self._cosine_similarity(vector, centroid)
        )

    async def _update_cluster(
        self,
        cluster_id: str,
        task_type: str,
        outcome: str,
    ):
        success_increment = (
            1 if outcome == "success" else 0
        )
        failure_increment = (
            1 if outcome != "success" else 0
        )

        async with self.db.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE task_clusters SET
                    pattern_count = pattern_count + 1,
                    success_count = success_count + $1,
                    failure_count = failure_count + $2,
                    last_updated = NOW()
                WHERE cluster_id = $3
                """,
                success_increment,
                failure_increment,
                cluster_id
            )

    async def _store(self, **kwargs) -> str:
        pattern_id = str(uuid.uuid4())

        def vec_str(v):
            if isinstance(v, np.ndarray):
                return (
                    "[" + ",".join(str(x) for x in v) + "]"
                )
            return (
                "[" + ",".join(str(x) for x in v) + "]"
            )

        async with self.db.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO execution_patterns (
                    pattern_id, session_id, task_type,
                    task_cluster_id, outcome,
                    trajectory_vector, start_vector,
                    end_vector, progress_scores,
                    error_count, total_steps,
                    reasoning_hops, behavioral_signals,
                    is_representative,
                    similarity_to_centroid
                ) VALUES (
                    $1,$2,$3,$4,$5,
                    $6::vector,$7::vector,$8::vector,
                    $9,$10,$11,$12,$13,$14,$15
                )
                """,
                pattern_id,
                kwargs["session_id"],
                kwargs["task_type"],
                kwargs["cluster_id"],
                kwargs["outcome"],
                vec_str(kwargs["trajectory_vector"]),
                vec_str(kwargs["start_vector"]),
                vec_str(kwargs["end_vector"]),
                json.dumps(kwargs["progress_scores"]),
                kwargs["error_count"],
                kwargs["total_steps"],
                kwargs["reasoning_hops"],
                json.dumps(kwargs["behavioral_signals"]),
                kwargs["is_representative"],
                kwargs["similarity"],
            )

        return pattern_id

    async def _load_into_memory(self):
        """Load all patterns into RAM on startup."""
        try:
            async with self.db.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT task_type, outcome,
                    trajectory_vector::text
                    FROM execution_patterns
                    """
                )
                clusters = await conn.fetch(
                    """
                    SELECT cluster_id, task_type,
                    centroid_vector::text
                    FROM task_clusters
                    """
                )

            for row in rows:
                vec_str = row["trajectory_vector"]
                if not vec_str:
                    continue
                vec = np.array(
                    [
                        float(x)
                        for x in
                        vec_str.strip("[]").split(",")
                    ],
                    dtype=np.float32
                )
                self._update_memory(
                    row["task_type"],
                    row["outcome"],
                    vec
                )

            for row in clusters:
                vec_str = row["centroid_vector"]
                if not vec_str:
                    continue
                vec = np.array(
                    [
                        float(x)
                        for x in
                        vec_str.strip("[]").split(",")
                    ],
                    dtype=np.float32
                )
                self._task_clusters[
                    row["cluster_id"]
                ] = {
                    "task_type": row["task_type"],
                    "centroid": vec,
                    "count": 1,
                }

            success_total = sum(
                len(v)
                for v in self._success_vectors.values()
            )
            failure_total = sum(
                len(v)
                for v in self._failure_vectors.values()
            )
            logger.info(
                f"Loaded {success_total} success "
                f"and {failure_total} failure patterns. "
                f"{len(self._task_clusters)} clusters."
            )
            self._loaded = True

        except Exception as e:
            logger.error(
                f"Pattern library load failed: {e}"
            )

    def _cosine_similarity(
        self,
        a: np.ndarray,
        b: np.ndarray,
    ) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(
            np.dot(a, b) / (norm_a * norm_b)
        )

    def _max_similarity(
        self,
        vector: np.ndarray,
        vectors: list[np.ndarray],
    ) -> float:
        if not vectors:
            return 0.0
        sims = [
            self._cosine_similarity(vector, v)
            for v in vectors
        ]
        return max(sims)

    async def get_stats(self) -> dict:
        success_total = sum(
            len(v)
            for v in self._success_vectors.values()
        )
        failure_total = sum(
            len(v)
            for v in self._failure_vectors.values()
        )
        return {
            "total_patterns": (
                success_total + failure_total
            ),
            "success_patterns": success_total,
            "failure_patterns": failure_total,
            "task_clusters": len(self._task_clusters),
            "cluster_details": {
                k: v["task_type"]
                for k, v in self._task_clusters.items()
            }
        }