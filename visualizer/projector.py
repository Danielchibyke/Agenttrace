# import numpy as np
# import logging
# from typing import Optional

# logger = logging.getLogger(__name__)

# class HDProjector:
#     """
#     Reduces high dimensional vectors to 3D coordinates
#     for visualization using UMAP.

#     For small batches (under 10 nodes) falls back to
#     PCA which works without minimum sample requirements.
#     """

#     def __init__(self, n_components: int = 3):
#         self.n_components = n_components
#         self._cache: dict = {}
#         self._fitted_matrix = None
#         self._fitted_coords = None

#     def _cache_key(self, vectors: list) -> str:
#         """Generate cache key from vectors."""
#         data = str([v[:5] for v in vectors[:3]])
#         return hashlib.md5(
#             data.encode()
#         ).hexdigest()[:16]

#     def project(
#         self,
#         vectors: list[list[float]],
#         labels: list[str] = None
#     ) -> list[dict]:
#         if not vectors:
#             return []

#         matrix = np.array(vectors, dtype=np.float32)
#         n_samples = len(matrix)

#         if n_samples == 1:
#             return [{
#                 "x": 0.0, "y": 0.0, "z": 0.0,
#                 "index": 0,
#                 "label": labels[0] if labels else "node_0"
#             }]

#         coords = self._reduce(matrix, n_samples)

#         results = []
#         for i, coord in enumerate(coords):
#             results.append({
#                 "x": float(coord[0]),
#                 "y": float(coord[1]),
#                 "z": float(coord[2]) if len(coord) > 2 else 0.0,
#                 "index": i,
#                 "label": labels[i] if labels else f"node_{i}"
#             })

#         return results

#     def _reduce(
#         self,
#         matrix: np.ndarray,
#         n_samples: int
#     ) -> np.ndarray:
#         if n_samples >= 10:
#             try:
#                 import umap
#                 reducer = umap.UMAP(
#                     n_components=self.n_components,
#                     n_neighbors=min(5, n_samples - 1),
#                     min_dist=0.1,
#                     metric="cosine",
#                     random_state=42,
#                     transform_seed=42,
#                 )
#                 return reducer.fit_transform(matrix)
#             except Exception as e:
#                 logger.warning(
#                     f"UMAP failed, falling back to PCA: {e}"
#                 )

#         return self._pca_reduce(matrix)

#     def _pca_reduce(
#         self, matrix: np.ndarray
#     ) -> np.ndarray:
#         from sklearn.decomposition import PCA
#         n_components = min(
#             self.n_components,
#             matrix.shape[0],
#             matrix.shape[1]
#         )
#         pca = PCA(
#             n_components=n_components,
#             random_state=42
#         )
#         reduced = pca.fit_transform(matrix)

#         if reduced.shape[1] < 3:
#             padding = np.zeros((
#                 reduced.shape[0],
#                 3 - reduced.shape[1]
#             ))
#             reduced = np.hstack([reduced, padding])

#         return reduced

#     def project_incremental(
#         self,
#         existing_coords: list[dict],
#         new_vector: list[float],
#         all_vectors: list[list[float]]
#     ) -> list[dict]:
#         """
#         Re-project all vectors including a new one.
#         Called each time a new node arrives in real time.
#         """
#         return self.project(
#             all_vectors,
#             labels=[c["label"] for c in existing_coords] + ["new"]
#         )


import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class HDProjector:
    """
    HD to 3D projector.
    Uses PCA for speed and determinism.
    Falls back to UMAP for quality when dataset is large.
    Caches results to avoid recomputation.
    """

    def __init__(self, n_components: int = 3):
        self.n_components = n_components
        self._cache: dict = {}

    def project(
        self,
        vectors: list[list[float]],
        labels: list[str] = None,
    ) -> list[dict]:
        if not vectors:
            return []

        matrix = np.array(vectors, dtype=np.float32)
        n_samples = len(matrix)

        if n_samples == 1:
            return [{
                "x": 0.0, "y": 0.0, "z": 0.0,
                "index": 0,
                "label": labels[0] if labels else "node_0",
            }]

        coords = self._reduce(matrix, n_samples)

        return [
            {
                "x": float(coord[0]),
                "y": float(coord[1]),
                "z": float(coord[2]) if len(coord) > 2 else 0.0,
                "index": i,
                "label": labels[i] if labels else f"node_{i}",
            }
            for i, coord in enumerate(coords)
        ]

    def _reduce(
        self,
        matrix: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        """
        PCA for speed — always deterministic, always parallel.
        UMAP only for large datasets where quality matters.
        """
        # always use PCA first — fast and deterministic
        pca_coords = self._pca_reduce(matrix)

        # only attempt UMAP for larger datasets
        # and only when we have enough samples
        if n_samples >= 15:
            try:
                import umap
                # use random_state=None for parallelism
                # sacrifice reproducibility for speed
                reducer = umap.UMAP(
                    n_components=self.n_components,
                    n_neighbors=min(10, n_samples - 1),
                    min_dist=0.1,
                    metric="cosine",
                    random_state=None,
                    n_jobs=-1,
                    low_memory=True,
                )
                return reducer.fit_transform(matrix)
            except Exception as e:
                logger.debug(
                    f"UMAP failed, using PCA: {e}"
                )
                return pca_coords

        return pca_coords

    def _pca_reduce(
        self, matrix: np.ndarray
    ) -> np.ndarray:
        from sklearn.decomposition import PCA

        n_components = min(
            self.n_components,
            matrix.shape[0],
            matrix.shape[1],
        )

        # center the data
        matrix_centered = matrix - matrix.mean(axis=0)

        pca = PCA(
            n_components=n_components,
            random_state=42,
        )
        reduced = pca.fit_transform(matrix_centered)

        if reduced.shape[1] < 3:
            padding = np.zeros((
                reduced.shape[0],
                3 - reduced.shape[1],
            ))
            reduced = np.hstack([reduced, padding])

        return reduced