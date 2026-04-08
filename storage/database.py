import json
import asyncpg
import os
from dotenv import load_dotenv
from tracer.node import Node

load_dotenv()

CREATE_NODES_TABLE = """
CREATE TABLE IF NOT EXISTS nodes (
    node_id TEXT PRIMARY KEY,
    parent_id TEXT,
    session_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    step_number INTEGER,
    node_type TEXT NOT NULL,
    timestamp TIMESTAMPTZ,
    latency_ms FLOAT,
    prompt_text TEXT,
    response_text TEXT,
    model_name TEXT,
    tokens_in INTEGER,
    tokens_out INTEGER,
    tool_name TEXT,
    input_params JSONB,
    raw_output TEXT,
    status TEXT,
    error_message TEXT,
    conversation_snapshot JSONB,
    system_prompt_snapshot TEXT,
    memory_snapshot JSONB,
    agent_state_snapshot JSONB,
    tool_config_snapshot JSONB,
    embedding_vector vector(896),
    progress_score FLOAT,
    error_flag BOOLEAN DEFAULT FALSE,
    repetition_score FLOAT,
    goal_embedding vector(896)
);
"""

CREATE_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_nodes_session 
    ON nodes(session_id);
CREATE INDEX IF NOT EXISTS idx_nodes_task 
    ON nodes(task_id);
CREATE INDEX IF NOT EXISTS idx_nodes_parent 
    ON nodes(parent_id);
CREATE INDEX IF NOT EXISTS idx_nodes_type 
    ON nodes(node_type);
"""
CREATE_MICRO_NODES_TABLE = """
CREATE TABLE IF NOT EXISTS micro_nodes (
    micro_id TEXT PRIMARY KEY,
    parent_node_id TEXT REFERENCES nodes(node_id),
    session_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    content TEXT,
    index INTEGER,
    token_index INTEGER,
    timestamp TIMESTAMPTZ,
    logprob FLOAT,
    tier INTEGER DEFAULT 1,
    embedding_vector vector(896)
);
CREATE INDEX IF NOT EXISTS idx_micro_session
    ON micro_nodes(session_id);
CREATE INDEX IF NOT EXISTS idx_micro_parent
    ON micro_nodes(parent_node_id);
CREATE INDEX IF NOT EXISTS idx_micro_tier
    ON micro_nodes(tier);
"""
class DatabaseWriter:
    def __init__(self):
        self.db_url = os.getenv("DATABASE_URL")
        self.pool = None

    async def connect(self):
        self.pool = await asyncpg.create_pool(
            self.db_url,
            min_size=2,
            max_size=10
        )
        await self._setup_schema()

    async def _setup_schema(self):
        async with self.pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            await conn.execute(CREATE_NODES_TABLE)
            await conn.execute(CREATE_MICRO_NODES_TABLE)
            await conn.execute(CREATE_INDEXES)

    async def write_nodes(self, nodes: list[Node]):
        async with self.pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO nodes (
                    node_id, parent_id, session_id, task_id,
                    step_number, node_type, timestamp, latency_ms,
                    prompt_text, response_text, model_name,
                    tokens_in, tokens_out, tool_name, input_params,
                    raw_output, status, error_message,
                    conversation_snapshot, system_prompt_snapshot,
                    memory_snapshot, agent_state_snapshot,
                    tool_config_snapshot
                ) VALUES (
                    $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,
                    $12,$13,$14,$15,$16,$17,$18,$19,$20,
                    $21,$22,$23
                )
               ON CONFLICT (node_id) DO UPDATE SET
                    response_text = EXCLUDED.response_text,
                    latency_ms = EXCLUDED.latency_ms,
                    tokens_in = EXCLUDED.tokens_in,
                    tokens_out = EXCLUDED.tokens_out,
                    raw_output = EXCLUDED.raw_output,
                    status = EXCLUDED.status,
                    error_message = EXCLUDED.error_message,
                    prompt_text = EXCLUDED.prompt_text,
                    tool_name = EXCLUDED.tool_name,
                    input_params = EXCLUDED.input_params
                """,
                [
                    (
                        n.node_id, n.parent_id, n.session_id,
                        n.task_id, n.step_number, n.node_type.value,
                        n.timestamp, n.latency_ms, n.prompt_text,
                        n.response_text, n.model_name, n.tokens_in,
                        n.tokens_out, n.tool_name,
                        json.dumps(n.input_params) if n.input_params else None,
                        str(n.raw_output) if n.raw_output else None,
                        n.status, n.error_message,
                        json.dumps(n.conversation_snapshot) if n.conversation_snapshot else None,
                        n.system_prompt_snapshot,
                        json.dumps(n.memory_snapshot) if n.memory_snapshot else None,
                        json.dumps(n.agent_state_snapshot) if n.agent_state_snapshot else None,
                        json.dumps(n.tool_config_snapshot) if n.tool_config_snapshot else None,
                    )
                    for n in nodes
                ]
            )

    async def get_session_nodes(self, session_id: str) -> list[dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM nodes WHERE session_id = $1 ORDER BY step_number",
                session_id
            )
            return [dict(row) for row in rows]

    async def get_node(self, node_id: str) -> dict:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM nodes WHERE node_id = $1",
                node_id
            )
            return dict(row) if row else None

    async def close(self):
        if self.pool:
            await self.pool.close()
            
    async def write_micro_nodes(
            self,
            micro_nodes: list
        ) -> None:
            async with self.pool.acquire() as conn:
                await conn.executemany(
                    """
                    INSERT INTO micro_nodes (
                        micro_id, parent_node_id, session_id,
                        task_id, content, index, token_index,
                        timestamp, logprob, tier
                    ) VALUES (
                        $1,$2,$3,$4,$5,$6,$7,$8,$9,$10
                    )
                    ON CONFLICT (micro_id) DO NOTHING
                    """,
                    [
                        (
                            m.micro_id,
                            m.parent_node_id,
                            m.session_id,
                            m.task_id,
                            m.content,
                            m.index,
                            m.token_index,
                            m.timestamp,
                            m.logprob,
                            1,
                        )
                        for m in micro_nodes
                    ]
                )

    async def update_micro_node_embedding(
            self,
            micro_id: str,
            embedding: list[float]
        ) -> None:
            vector_str = (
                "[" + ",".join(str(x) for x in embedding) + "]"
            )
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE micro_nodes
                    SET embedding_vector = $1::vector
                    WHERE micro_id = $2
                    """,
                    vector_str,
                    micro_id
                )

    async def get_session_micro_trajectory(
            self,
            session_id: str,
            parent_node_id: str = None
        ) -> list[dict]:
            query = """
                SELECT
                    micro_id, parent_node_id, content,
                    index, token_index, timestamp, logprob,
                    embedding_vector::text as embedding_vector
                FROM micro_nodes
                WHERE session_id = $1
                AND embedding_vector IS NOT NULL
                {filter}
                ORDER BY token_index
            """
            if parent_node_id:
                query = query.format(
                    filter="AND parent_node_id = $2"
                )
                params = [session_id, parent_node_id]
            else:
                query = query.format(filter="")
                params = [session_id]

            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                return [dict(row) for row in rows]           