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
    
@app.get("/session/{session_id}/micro")
async def get_session_micro(
    session_id: str,
    parent_node_id: str = None
):
    """Get micro node trajectory for a session."""
    trajectory = await db_writer.get_session_micro_trajectory(
        session_id, parent_node_id
    )

    if not trajectory:
        return {"error": "No micro nodes found"}

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
            "micro_id": node["micro_id"],
            "parent_node_id": node["parent_node_id"],
            "content": node.get("content", ""),
            "token_index": node.get("token_index", 0),
            "timestamp": str(node.get("timestamp")),
        })

    if not vectors:
        return {"error": "No embedded micro nodes found"}

    coords = projector.project(
        vectors,
        [f"token_{m['token_index']}" for m in meta]
    )

    for i, coord in enumerate(coords):
        if i < len(meta):
            coord.update(meta[i])
            coord["node_type"] = "micro"

    return {
        "session_id": session_id,
        "micro_count": len(coords),
        "nodes": coords
    }
@app.get("/session/{session_id}/combined")
async def get_session_combined(session_id: str):
    """
    Returns both parent nodes and micro nodes together.
    Parent nodes are landmarks.
    Micro nodes are the detailed trajectory.
    """
    # get parent nodes
    parent_nodes = await db_writer.get_session_nodes(
        session_id
    )

    # get micro nodes
    micro_trajectory = (
        await db_writer.get_session_micro_trajectory(
            session_id
        )
    )

    if not parent_nodes and not micro_trajectory:
        return {"error": "Session not found"}

    all_vectors = []
    all_meta = []

    # add parent nodes
    for node in parent_nodes:
        vec_str = node.get("embedding_vector")
        if not vec_str:
            continue
        vec = _parse_vector(
            vec_str if isinstance(vec_str, str)
            else str(vec_str)
        )
        if vec is None:
            continue
        all_vectors.append(vec)
        all_meta.append({
            "node_id": node["node_id"],
            "node_type": node["node_type"],
            "step_number": node["step_number"],
            "tool_name": node.get("tool_name"),
            "status": node.get("status"),
            "latency_ms": node.get("latency_ms"),
            "is_parent": True,
            "size": "large",
        })

    # add micro nodes
    for node in micro_trajectory:
        vec_str = node.get("embedding_vector")
        if not vec_str:
            continue
        vec = _parse_vector(vec_str)
        if vec is None:
            continue
        all_vectors.append(vec)
        all_meta.append({
            "node_id": node["micro_id"],
            "node_type": "micro",
            "step_number": node.get("token_index", 0),
            "content": node.get("content", ""),
            "parent_node_id": node.get("parent_node_id"),
            "is_parent": False,
            "size": "small",
        })

    if not all_vectors:
        return {"error": "No embedded nodes found"}

    coords = projector.project(
        all_vectors,
        [
            f"{m['node_type']}_{m['step_number']}"
            for m in all_meta
        ]
    )

    for i, coord in enumerate(coords):
        if i < len(all_meta):
            coord.update(all_meta[i])

    return {
        "session_id": session_id,
        "total_count": len(coords),
        "parent_count": len(parent_nodes),
        "micro_count": len(micro_trajectory),
        "nodes": coords,
    }    


@app.websocket("/ws/live/{session_id}")
async def live_session(websocket: WebSocket, session_id: str):
    await websocket.accept()
    active_connections.append(websocket)
    logger.info(f"Client connected for session {session_id}")

    seen_node_ids = set()

    try:
        while True:
            # fetch ALL nodes for session
            # regardless of whether they have embeddings
            async with db_writer.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT
                        node_id, parent_id, node_type,
                        step_number, tool_name, status,
                        latency_ms, timestamp,
                        embedding_vector::text as embedding_vector
                    FROM nodes
                    WHERE session_id = $1
                    ORDER BY step_number
                """, session_id)

            all_nodes = [dict(r) for r in rows]

            if not all_nodes:
                await asyncio.sleep(0.5)
                continue

            new_nodes = [
                n for n in all_nodes
                if n["node_id"] not in seen_node_ids
            ]

            if new_nodes:
                for n in new_nodes:
                    seen_node_ids.add(n["node_id"])

                # separate embedded vs not yet embedded
                embedded = [
                    n for n in all_nodes
                    if n.get("embedding_vector") is not None
                ]
                pending = [
                    n for n in all_nodes
                    if n.get("embedding_vector") is None
                ]

                coords = []

                # project embedded nodes properly
                if embedded:
                    vectors = []
                    meta = []
                    for node in embedded:
                        vec = _parse_vector(node["embedding_vector"])
                        if vec is None:
                            continue
                        vectors.append(vec)
                        meta.append(node)

                    if vectors:
                        projected = projector.project(
                            vectors,
                            [f"{m['node_type']}_{m['step_number']}"
                             for m in meta]
                        )
                        for i, coord in enumerate(projected):
                            if i < len(meta):
                                coord.update({
                                    "node_id": meta[i]["node_id"],
                                    "node_type": meta[i]["node_type"],
                                    "step_number": meta[i]["step_number"],
                                    "tool_name": meta[i].get("tool_name"),
                                    "status": meta[i].get("status"),
                                    "latency_ms": meta[i].get("latency_ms"),
                                    "has_embedding": True,
                                })
                        coords.extend(projected)

                # add pending nodes at placeholder positions
                for node in pending:
                    coords.append({
                        "node_id": node["node_id"],
                        "node_type": node["node_type"],
                        "step_number": node["step_number"],
                        "tool_name": node.get("tool_name"),
                        "status": node.get("status"),
                        "latency_ms": node.get("latency_ms"),
                        "has_embedding": False,
                        # placeholder position near origin
                        # will update when embedding arrives
                        "x": 0.0,
                        "y": float(node["step_number"]) * 0.1,
                        "z": 0.0,
                    })

                await websocket.send_json({
                    "event": "update",
                    "session_id": session_id,
                    "nodes": coords,
                    "latest_step": max(
                        n["step_number"] for n in all_nodes
                    ),
                    "total_nodes": len(all_nodes),
                    "embedded_count": len(embedded),
                    "pending_count": len(pending),
                })

            await asyncio.sleep(0.3)

    except WebSocketDisconnect:
        if websocket in active_connections:
            active_connections.remove(websocket)
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