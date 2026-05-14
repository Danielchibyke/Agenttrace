import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class DriftAnalyzer:
    """
    Mathematical proof engine for trajectory drift theory.

    Proves three claims:
    1. Geometric separability of success vs failure
    2. Temporal detectability before failure
    3. Predictive validity of trajectory projection
    """

    def __init__(self):
        self.success_centroid: Optional[np.ndarray] = None
        self.failure_centroid: Optional[np.ndarray] = None
        self.success_vectors: list[np.ndarray] = []
        self.failure_vectors: list[np.ndarray] = []

    # ─────────────────────────────────────────────
    # CLAIM 1 — GEOMETRIC SEPARABILITY
    # ─────────────────────────────────────────────

    def prove_separability(
        self, trajectories: list[dict]
    ) -> dict:
        """
        Prove that success and failure trajectories
        occupy distinct regions in HD space.

        Returns silhouette score and cluster statistics.
        """
        success = [
            t for t in trajectories
            if t["outcome"] == "success"
        ]
        failure = [
            t for t in trajectories
            if t["outcome"] in ["failure", "error"]
        ]

        if len(success) < 2 or len(failure) < 2:
            return {
                "proven": False,
                "reason": (
                    f"Insufficient data. "
                    f"Need 2+ success and 2+ failure. "
                    f"Have {len(success)} success "
                    f"and {len(failure)} failure."
                ),
                "success_count": len(success),
                "failure_count": len(failure),
            }

        success_vecs = np.array([
            t["trajectory_vector"] for t in success
        ])
        failure_vecs = np.array([
            t["trajectory_vector"] for t in failure
        ])

        self.success_vectors = list(success_vecs)
        self.failure_vectors = list(failure_vecs)

        # compute centroids
        self.success_centroid = success_vecs.mean(axis=0)
        self.failure_centroid = failure_vecs.mean(axis=0)

        # inter-cluster distance
        inter_distance = self._cosine_distance(
            self.success_centroid,
            self.failure_centroid
        )

        # intra-cluster distances
        success_intra = self._mean_intra_distance(
            success_vecs
        )
        failure_intra = self._mean_intra_distance(
            failure_vecs
        )

        # silhouette score
        silhouette = self._compute_silhouette(
            success_vecs, failure_vecs
        )

        # davies-bouldin index (lower = better separation)
        davies_bouldin = self._davies_bouldin(
            success_vecs, failure_vecs
        )

        # variance explained by cluster membership
        all_vecs = np.vstack([success_vecs, failure_vecs])
        global_centroid = all_vecs.mean(axis=0)
        total_variance = np.var(all_vecs, axis=0).mean()

        between_variance = (
            len(success) * self._cosine_distance(
                self.success_centroid, global_centroid
            ) ** 2 +
            len(failure) * self._cosine_distance(
                self.failure_centroid, global_centroid
            ) ** 2
        ) / len(trajectories)

        proven = silhouette > 0.1 or inter_distance > 0.05

        return {
            "proven": proven,
            "silhouette_score": round(float(silhouette), 4),
            "inter_cluster_distance": round(
                float(inter_distance), 4
            ),
            "success_intra_distance": round(
                float(success_intra), 4
            ),
            "failure_intra_distance": round(
                float(failure_intra), 4
            ),
            "davies_bouldin_index": round(
                float(davies_bouldin), 4
            ),
            "between_cluster_variance": round(
                float(between_variance), 4
            ),
            "success_count": len(success),
            "failure_count": len(failure),
            "interpretation": self._interpret_separability(
                silhouette, inter_distance
            ),
        }

    # ─────────────────────────────────────────────
    # CLAIM 2 — TEMPORAL DETECTABILITY
    # ─────────────────────────────────────────────

    def prove_temporal_detection(
        self,
        step_vectors: list[dict],
        outcome: str,
        threshold: float = 0.55,
    ) -> dict:
        """
        Prove that geometric drift is detectable
        before the final failure output.

        step_vectors: ordered list of node vectors
        outcome: actual outcome of this session
        threshold: drift score that triggers alert
        """
        if not self.success_centroid is not None:
            return {
                "proven": False,
                "reason": "No centroids computed. Run prove_separability first."
            }

        if len(step_vectors) < 3:
            return {
                "proven": False,
                "reason": f"Too few steps: {len(step_vectors)}"
            }

        drift_timeline = []
        detection_step = None

        for i, step in enumerate(step_vectors):
            vec = step["vector"]

            sim_success = self._cosine_similarity(
                vec, self.success_centroid
            )
            sim_failure = self._cosine_similarity(
                vec, self.failure_centroid
            )

            total = sim_success + sim_failure
            drift_score = (
                sim_failure / total if total > 0 else 0.0
            )

            drift_timeline.append({
                "step": i,
                "node_type": step.get("node_type"),
                "step_number": step.get("step_number"),
                "drift_score": round(float(drift_score), 4),
                "sim_success": round(float(sim_success), 4),
                "sim_failure": round(float(sim_failure), 4),
            })

            if (
                detection_step is None and
                drift_score > threshold and
                i < len(step_vectors) - 1
            ):
                detection_step = i

        final_drift = drift_timeline[-1]["drift_score"]
        max_drift = max(d["drift_score"] for d in drift_timeline)
        avg_drift = sum(
            d["drift_score"] for d in drift_timeline
        ) / len(drift_timeline)

        is_failure = outcome in ["failure", "error"]
        total_steps = len(step_vectors)

        if detection_step is not None:
            lead_time = total_steps - 1 - detection_step
            detection_pct = (
                detection_step / total_steps * 100
            )
        else:
            lead_time = 0
            detection_pct = 100.0

        proven = (
            is_failure and
            detection_step is not None and
            lead_time > 0
        )

        return {
            "proven": proven,
            "outcome": outcome,
            "total_steps": total_steps,
            "detection_step": detection_step,
            "lead_time_steps": lead_time,
            "detection_at_pct": round(detection_pct, 1),
            "final_drift_score": round(final_drift, 4),
            "max_drift_score": round(max_drift, 4),
            "avg_drift_score": round(avg_drift, 4),
            "drift_timeline": drift_timeline,
            "interpretation": (
                f"Drift detected at step {detection_step} "
                f"({detection_pct:.0f}% through execution). "
                f"{lead_time} steps before end."
                if detection_step is not None
                else "No drift threshold crossed."
            ),
        }

    # ─────────────────────────────────────────────
    # CLAIM 3 — PREDICTIVE VALIDITY
    # ─────────────────────────────────────────────

    def prove_prediction(
        self,
        step_vectors: list[dict],
        outcome: str,
        lookahead: int = 3,
    ) -> dict:
        """
        Prove that trajectory velocity vector points
        toward failure cluster before failure occurs.

        Computes velocity at each step and projects
        forward to see if projection lands near
        failure centroid.
        """
        if self.success_centroid is None:
            return {
                "proven": False,
                "reason": "No centroids computed."
            }

        if len(step_vectors) < 3:
            return {
                "proven": False,
                "reason": "Too few steps for velocity."
            }

        prediction_timeline = []
        correct_predictions = 0
        total_predictions = 0

        for i in range(1, len(step_vectors) - 1):
            curr = step_vectors[i]["vector"]
            prev = step_vectors[i-1]["vector"]

            # velocity vector
            velocity = curr - prev

            # acceleration if we have enough history
            if i >= 2:
                prev_prev = step_vectors[i-2]["vector"]
                prev_velocity = prev - prev_prev
                acceleration = velocity - prev_velocity
            else:
                acceleration = np.zeros_like(velocity)

            # project forward
            projected = (
                curr +
                lookahead * velocity +
                0.5 * lookahead**2 * acceleration
            )

            # normalize projection
            norm = np.linalg.norm(projected)
            if norm > 0:
                projected = projected / norm

            # check where projection lands
            sim_success = self._cosine_similarity(
                projected, self.success_centroid
            )
            sim_failure = self._cosine_similarity(
                projected, self.failure_centroid
            )

            predicts_failure = sim_failure > sim_success
            actual_failure = outcome in ["failure", "error"]

            velocity_magnitude = float(
                np.linalg.norm(velocity)
            )

            prediction_timeline.append({
                "step": i,
                "velocity_magnitude": round(
                    velocity_magnitude, 4
                ),
                "projected_sim_success": round(
                    float(sim_success), 4
                ),
                "projected_sim_failure": round(
                    float(sim_failure), 4
                ),
                "predicts_failure": predicts_failure,
                "correct": predicts_failure == actual_failure,
            })

            total_predictions += 1
            if predicts_failure == actual_failure:
                correct_predictions += 1

        accuracy = (
            correct_predictions / total_predictions * 100
            if total_predictions > 0 else 0
        )

        # find earliest correct prediction of failure
        early_prediction = None
        if outcome in ["failure", "error"]:
            for p in prediction_timeline:
                if p["predicts_failure"]:
                    early_prediction = p["step"]
                    break

        proven = accuracy > 60 and total_predictions > 0

        return {
            "proven": proven,
            "outcome": outcome,
            "prediction_accuracy": round(accuracy, 1),
            "correct_predictions": correct_predictions,
            "total_predictions": total_predictions,
            "earliest_failure_prediction": early_prediction,
            "lookahead_steps": lookahead,
            "prediction_timeline": prediction_timeline,
            "interpretation": (
                f"Trajectory prediction {accuracy:.0f}% accurate "
                f"({correct_predictions}/{total_predictions}). "
                f"Earliest failure predicted at step "
                f"{early_prediction}."
                if early_prediction
                else f"Prediction accuracy: {accuracy:.0f}%."
            ),
        }

    # ─────────────────────────────────────────────
    # MATH HELPERS
    # ─────────────────────────────────────────────

    def _cosine_similarity(
        self, a: np.ndarray, b: np.ndarray
    ) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _cosine_distance(
        self, a: np.ndarray, b: np.ndarray
    ) -> float:
        return 1.0 - self._cosine_similarity(a, b)

    def _mean_intra_distance(
        self, vectors: np.ndarray
    ) -> float:
        if len(vectors) < 2:
            return 0.0
        total = 0.0
        count = 0
        for i in range(len(vectors)):
            for j in range(i+1, len(vectors)):
                total += self._cosine_distance(
                    vectors[i], vectors[j]
                )
                count += 1
        return total / count if count > 0 else 0.0

    def _compute_silhouette(
        self,
        success_vecs: np.ndarray,
        failure_vecs: np.ndarray,
    ) -> float:
        """
        Simplified silhouette score for two clusters.
        """
        scores = []

        for vec in success_vecs:
            a = np.mean([
                self._cosine_distance(vec, other)
                for other in success_vecs
                if not np.array_equal(vec, other)
            ]) if len(success_vecs) > 1 else 0.0

            b = np.mean([
                self._cosine_distance(vec, other)
                for other in failure_vecs
            ]) if len(failure_vecs) > 0 else 0.0

            if max(a, b) > 0:
                scores.append((b - a) / max(a, b))

        for vec in failure_vecs:
            a = np.mean([
                self._cosine_distance(vec, other)
                for other in failure_vecs
                if not np.array_equal(vec, other)
            ]) if len(failure_vecs) > 1 else 0.0

            b = np.mean([
                self._cosine_distance(vec, other)
                for other in success_vecs
            ]) if len(success_vecs) > 0 else 0.0

            if max(a, b) > 0:
                scores.append((b - a) / max(a, b))

        return float(np.mean(scores)) if scores else 0.0

    def _davies_bouldin(
        self,
        success_vecs: np.ndarray,
        failure_vecs: np.ndarray,
    ) -> float:
        """
        Davies-Bouldin index for two clusters.
        Lower is better — indicates better separation.
        """
        s_centroid = success_vecs.mean(axis=0)
        f_centroid = failure_vecs.mean(axis=0)

        s_scatter = np.mean([
            self._cosine_distance(v, s_centroid)
            for v in success_vecs
        ])
        f_scatter = np.mean([
            self._cosine_distance(v, f_centroid)
            for v in failure_vecs
        ])

        centroid_dist = self._cosine_distance(
            s_centroid, f_centroid
        )

        if centroid_dist == 0:
            return float('inf')

        return float(
            (s_scatter + f_scatter) / centroid_dist
        )

    def _interpret_separability(
        self, silhouette: float, inter_distance: float
    ) -> str:
        if silhouette > 0.5:
            return (
                "STRONG separation — success and failure "
                "occupy clearly distinct HD regions"
            )
        elif silhouette > 0.25:
            return (
                "MODERATE separation — meaningful geometric "
                "distinction between outcomes"
            )
        elif silhouette > 0.1 or inter_distance > 0.1:
            return (
                "WEAK separation — some geometric signal "
                "present but needs more data"
            )
        else:
            return (
                "NO CLEAR separation — need more patterns "
                "or better failure task design"
            )