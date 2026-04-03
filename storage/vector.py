import asyncpg
import os
import json
from dotenv import load_dotenv

load_dotenv()

class VectorStorage:
    """
    Stores and queries HD vectors in pgvector.
    Works alongside DatabaseWriter which handles
    the raw node data.
    """

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def store_embedding(
        self,
        node_id: str,
        embedding: list[float]
    ):
        """Store embedding vector for a node."""
        vector_str = "[" + ",".join(str(x) for x in embedding) + "]"
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE nodes
                SET embedding_vector = $1::vector
                WHERE node_id = $2
                """,
                vector_str,
                node_id
            )

    async def store_embeddings_batch(
        self,
        embeddings: dict[str, list[float]]
    ):
        """Store multiple embeddings at once."""
        async with self.pool.acquire() as conn:
            for node_id, embedding in embeddings.items():
                if embedding is None:
                    continue
                vector_str = (
                    "[" + ",".join(str(x) for x in embedding) + "]"
                )
                await conn.execute(
                    """
                    UPDATE nodes
                    SET embedding_vector = $1::vector
                    WHERE node_id = $2
                    """,
                    vector_str,
                    node_id
                )

    async def find_similar_nodes(
        self,
        embedding: list[float],
        limit: int = 10,
        session_id: str = None
    ) -> list[dict]:
        """
        Find nodes most similar to a given vector.
        Core of the HD space pattern matching.
        Optional filter by session.
        """
        vector_str = "[" + ",".join(str(x) for x in embedding) + "]"

        query = """
            SELECT
                node_id,
                session_id,
                task_id,
                node_type,
                step_number,
                tool_name,
                status,
                1 - (embedding_vector <=> $1::vector) AS similarity
            FROM nodes
            WHERE embedding_vector IS NOT NULL
            {session_filter}
            ORDER BY embedding_vector <=> $1::vector
            LIMIT $2
        """

        if session_id:
            query = query.format(
                session_filter="AND session_id = $3"
            )
            params = [vector_str, limit, session_id]
        else:
            query = query.format(session_filter="")
            params = [vector_str, limit]

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [dict(row) for row in rows]

    async def find_similar_to_node(
        self,
        node_id: str,
        limit: int = 10
    ) -> list[dict]:
        """
        Find nodes similar to a specific node by its ID.
        Used for pattern matching during live execution.
        """
        async with self.pool.acquire() as conn:
            source = await conn.fetchrow(
                "SELECT embedding_vector FROM nodes WHERE node_id = $1",
                node_id
            )
            if not source or not source["embedding_vector"]:
                return []

            rows = await conn.fetch(
                """
                SELECT
                    node_id,
                    session_id,
                    task_id,
                    node_type,
                    step_number,
                    tool_name,
                    status,
                    1 - (embedding_vector <=> $1::vector) AS similarity
                FROM nodes
                WHERE embedding_vector IS NOT NULL
                AND node_id != $2
                ORDER BY embedding_vector <=> $1::vector
                LIMIT $3
                """,
                source["embedding_vector"],
                node_id,
                limit
            )
            return [dict(row) for row in rows]

    async def get_session_trajectory(
        self,
        session_id: str
    ) -> list[dict]:
        """
        Get all embedded nodes for a session in order.
        This is the raw trajectory through HD space.
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    node_id,
                    parent_id,
                    node_type,
                    step_number,
                    embedding_vector::text,
                    tool_name,
                    status,
                    latency_ms,
                    timestamp
                FROM nodes
                WHERE session_id = $1
                AND embedding_vector IS NOT NULL
                ORDER BY step_number
                """,
                session_id
            )
            return [dict(row) for row in rows]