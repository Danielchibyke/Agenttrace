import asyncio
import json
import logging
import os
import numpy as np
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv
from storage.database import DatabaseWriter
from storage.vector import VectorStorage
from embeddings.encoder import EmbeddingEncoder
from visualizer.projector import HDProjector

load_dotenv()
logger = logging.getLogger(__name__)

app = FastAPI(title="AgentTrace HD Visualizer")

db_writer = DatabaseWriter()
projector = HDProjector(n_components=3)

# connected websocket clients
active_connections: list[WebSocket] = []

# session state for live projection
session_vectors: dict[str, list] = {}
session_nodes: dict[str, list] = {}


@app.on_event("startup")
async def startup():
    await db_writer.connect()
    logger.info("Visualizer server started.")


@app.on_event("shutdown")
async def shutdown():
    await db_writer.close()


@app.get("/")
async def root():
    with open("visualizer/static/index.html") as f:
        return HTMLResponse(f.read())


@app.get("/sessions")
async def get_sessions():
    """Get all available sessions."""
    async with db_writer.pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT
                session_id,
                task_id,
                COUNT(*) as node_count,
                MIN(timestamp) as started_at,
                MAX(timestamp) as last_updated
            FROM nodes
            GROUP BY session_id, task_id
            ORDER BY MIN(timestamp) DESC
        """)
        return [dict(row) for row in rows]


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get full projection for a completed session."""
    vector_storage = VectorStorage(db_writer.pool)
    trajectory = await vector_storage.get_session_trajectory(session_id)

    if not trajectory:
        return {"error": "Session not found or no embeddings yet"}

    vectors = []
    labels = []
    metadata = []

    for node in trajectory:
        vec_str = node.get("embedding_vector")
        if not vec_str:
            continue

        vec = _parse_vector(vec_str)
        if vec is None:
            continue

        vectors.append(vec)
        labels.append(f"{node['node_type']}_{node['step_number']}")
        metadata.append({
            "node_id": node["node_id"],
            "node_type": node["node_type"],
            "step_number": node["step_number"],
            "tool_name": node.get("tool_name"),
            "status": node.get("status"),
            "latency_ms": node.get("latency_ms"),
            "timestamp": str(node.get("timestamp")),
        })

    if not vectors:
        return {"error": "No embedded nodes found"}

    coords = projector.project(vectors, labels)

    for i, coord in enumerate(coords):
        if i < len(metadata):
            coord.update(metadata[i])

    return {
        "session_id": session_id,
        "node_count": len(coords),
        "nodes": coords
    }


@app.websocket("/ws/live/{session_id}")
async def live_session(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for live session visualization.
    Streams new nodes to the frontend as they arrive.
    """
    await websocket.accept()
    active_connections.append(websocket)
    logger.info(f"Client connected for session {session_id}")

    seen_node_ids = set()
    session_vectors[session_id] = []
    session_nodes[session_id] = []

    try:
        while True:
            nodes = await db_writer.get_session_nodes(session_id)
            new_nodes = [
                n for n in nodes
                if n["node_id"] not in seen_node_ids
                and n.get("embedding_vector") is not None
            ]

            if new_nodes:
                vector_storage = VectorStorage(db_writer.pool)
                trajectory = await vector_storage.get_session_trajectory(
                    session_id
                )

                vectors = []
                meta = []

                for node in trajectory:
                    vec_str = node.get("embedding_vector")
                    if not vec_str:
                        continue
                    vec = _parse_vector(vec_str)
                    if vec is None:
                        continue
                    vectors.append(vec)
                    meta.append({
                        "node_id": node["node_id"],
                        "node_type": node["node_type"],
                        "step_number": node["step_number"],
                        "tool_name": node.get("tool_name"),
                        "status": node.get("status"),
                        "latency_ms": node.get("latency_ms"),
                    })

                if vectors:
                    coords = projector.project(vectors, [
                        f"{m['node_type']}_{m['step_number']}"
                        for m in meta
                    ])

                    for i, coord in enumerate(coords):
                        if i < len(meta):
                            coord.update(meta[i])

                    for node in new_nodes:
                        seen_node_ids.add(node["node_id"])

                    await websocket.send_json({
                        "event": "update",
                        "session_id": session_id,
                        "nodes": coords,
                        "latest_step": max(
                            m["step_number"] for m in meta
                        )
                    })

            await asyncio.sleep(1)

    except WebSocketDisconnect:
        active_connections.remove(websocket)
        logger.info(f"Client disconnected from session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)


async def push_node_to_clients(session_id: str, node_data: dict):
    """
    Called by tracer when a new node is captured.
    Pushes update to all connected WebSocket clients.
    """
    for connection in active_connections:
        try:
            await connection.send_json({
                "event": "new_node",
                "session_id": session_id,
                "node": node_data
            })
        except Exception as e:
            logger.error(f"Failed to push to client: {e}")


def _parse_vector(vec_str: str) -> Optional[list[float]]:
    """Parse vector string from PostgreSQL into list of floats."""
    try:
        if isinstance(vec_str, str):
            clean = vec_str.strip("[]")
            return [float(x) for x in clean.split(",")]
        return None
    except Exception:
        return None


app.mount(
    "/static",
    StaticFiles(directory="visualizer/static"),
    name="static"
)