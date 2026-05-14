import asyncio
import json
import logging
import os
from typing import Optional
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from dotenv import load_dotenv
from storage.database import DatabaseWriter
from storage.vector import VectorStorage
from visualizer.projector import HDProjector

load_dotenv()
logger = logging.getLogger(__name__)

app = FastAPI(title="AgentTrace HD Visualizer")

db_writer = DatabaseWriter()
projector = HDProjector(n_components=3)

# projection cache — avoids rerunning UMAP on every poll
_proj_cache: dict[str, dict] = {}


def _cached_project(
    vectors: list,
    labels: list,
    cache_key: str,
) -> list:
    cached = _proj_cache.get(cache_key)
    if cached and cached["count"] == len(vectors):
        return cached["coords"]
    coords = projector.project(vectors, labels)
    _proj_cache[cache_key] = {
        "count": len(vectors),
        "coords": coords,
    }
    return coords


def _parse_vector(vec_str) -> Optional[list[float]]:
    try:
        if isinstance(vec_str, str):
            return [
                float(x)
                for x in vec_str.strip("[]").split(",")
            ]
        return None
    except Exception:
        return None


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
    vector_storage = VectorStorage(db_writer.pool)
    trajectory = await vector_storage.get_session_trajectory(
        session_id
    )
    if not trajectory:
        return {"error": "Session not found or no embeddings yet"}

    vectors, labels, metadata = [], [], []
    for node in trajectory:
        vec = _parse_vector(node.get("embedding_vector"))
        if not vec:
            continue
        vectors.append(vec)
        labels.append(
            f"{node['node_type']}_{node['step_number']}"
        )
        metadata.append({
            "node_id": node["node_id"],
            "node_type": node["node_type"],
            "step_number": node["step_number"],
            "tool_name": node.get("tool_name"),
            "status": node.get("status"),
            "latency_ms": node.get("latency_ms"),
            "is_parent": True,
            "has_embedding": True,
        })

    if not vectors:
        return {"error": "No embedded nodes found"}

    coords = _cached_project(
        vectors, labels, f"session_{session_id}"
    )
    for i, coord in enumerate(coords):
        if i < len(metadata):
            coord.update(metadata[i])

    return {
        "session_id": session_id,
        "node_count": len(coords),
        "nodes": coords,
    }


@app.get("/session/{session_id}/combined")
async def get_session_combined(session_id: str):
    parent_nodes = await db_writer.get_session_nodes(
        session_id
    )
    micro_trajectory = (
        await db_writer.get_session_micro_trajectory(
            session_id
        )
    )

    if not parent_nodes and not micro_trajectory:
        return {"error": "Session not found"}

    all_vectors, all_meta = [], []

    for node in parent_nodes:
        vec = _parse_vector(
            str(node.get("embedding_vector", ""))
        )
        if not vec:
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
            "has_embedding": True,
        })

    for node in micro_trajectory:
        vec = _parse_vector(node.get("embedding_vector"))
        if not vec:
            continue
        all_vectors.append(vec)
        all_meta.append({
            "node_id": node["micro_id"],
            "node_type": "micro",
            "step_number": node.get("token_index", 0),
            "content": node.get("content", ""),
            "parent_node_id": node.get("parent_node_id"),
            "is_parent": False,
            "has_embedding": True,
        })

    if not all_vectors:
        return {"error": "No embedded nodes found"}

    coords = _cached_project(
        all_vectors,
        [
            f"{m['node_type']}_{m['step_number']}"
            for m in all_meta
        ],
        f"combined_{session_id}",
    )
    for i, coord in enumerate(coords):
        if i < len(all_meta):
            coord.update(all_meta[i])

    return {
        "session_id": session_id,
        "total_count": len(coords),
        "nodes": coords,
    }


async def _build_payload(
    session_id: str,
    seen_node_ids: set,
    seen_micro_ids: set,
) -> Optional[list]:
    """
    Build SSE payload.
    Returns None if nothing new to send.
    Uses projection cache to avoid rerunning UMAP.
    """
    try:
        async with db_writer.pool.acquire() as conn:
            parent_rows = await conn.fetch("""
                SELECT node_id, node_type, step_number,
                tool_name, status, latency_ms,
                embedding_vector::text as embedding_vector
                FROM nodes
                WHERE session_id = $1
                ORDER BY step_number
            """, session_id)

            micro_rows = await conn.fetch("""
                SELECT micro_id, content, token_index,
                embedding_vector::text as embedding_vector
                FROM micro_nodes
                WHERE session_id = $1
                ORDER BY token_index
            """, session_id)

    except Exception as e:
        logger.error(f"DB fetch failed: {e}")
        return None

    parent_nodes = [dict(r) for r in parent_rows]
    micro_nodes = [dict(r) for r in micro_rows]

    # check for anything new or newly embedded
    new_parents = [
        n for n in parent_nodes
        if n["node_id"] not in seen_node_ids
    ]
    new_micros = [
        n for n in micro_nodes
        if n["micro_id"] not in seen_micro_ids
    ]
    newly_embedded_parents = [
        n for n in parent_nodes
        if n["node_id"] in seen_node_ids
        and n.get("embedding_vector")
    ]
    newly_embedded_micros = [
        n for n in micro_nodes
        if n["micro_id"] in seen_micro_ids
        and n.get("embedding_vector")
    ]

    has_updates = (
        new_parents or new_micros or
        newly_embedded_parents or newly_embedded_micros
    )
    if not has_updates:
        return None

    # mark new nodes as seen
    for n in new_parents:
        seen_node_ids.add(n["node_id"])
    for n in new_micros:
        seen_micro_ids.add(n["micro_id"])

    payload = []

    # --- parent nodes ---
    embedded_parents = [
        n for n in parent_nodes
        if n.get("embedding_vector")
    ]

    if embedded_parents:
        vectors, meta = [], []
        for n in embedded_parents:
            vec = _parse_vector(n["embedding_vector"])
            if vec:
                vectors.append(vec)
                meta.append(n)

        if vectors:
            try:
                coords = _cached_project(
                    vectors,
                    [
                        f"{m['node_type']}_{m['step_number']}"
                        for m in meta
                    ],
                    f"parents_{session_id}",
                )
                for i, coord in enumerate(coords):
                    if i < len(meta):
                        payload.append({
                            **coord,
                            "node_id": meta[i]["node_id"],
                            "node_type": meta[i]["node_type"],
                            "step_number": meta[i]["step_number"],
                            "tool_name": meta[i].get("tool_name"),
                            "status": meta[i].get("status"),
                            "latency_ms": meta[i].get("latency_ms"),
                            "is_parent": True,
                            "has_embedding": True,
                        })
            except Exception as e:
                logger.error(f"Parent projection error: {e}")

    # unembedded parents — placeholder
    for n in parent_nodes:
        if not n.get("embedding_vector"):
            payload.append({
                "node_id": n["node_id"],
                "node_type": n["node_type"],
                "step_number": n["step_number"],
                "tool_name": n.get("tool_name"),
                "status": n.get("status"),
                "is_parent": True,
                "has_embedding": False,
                "x": 0.0,
                "y": float(n.get("step_number") or 0) * 0.3,
                "z": 0.0,
            })

    # --- micro nodes ---
    embedded_micros = [
        n for n in micro_nodes
        if n.get("embedding_vector")
    ]

    micro_coords = {}
    if embedded_micros:
        vectors, meta = [], []
        for n in embedded_micros:
            vec = _parse_vector(n["embedding_vector"])
            if vec:
                vectors.append(vec)
                meta.append(n)

        if vectors:
            try:
                coords = _cached_project(
                    vectors,
                    [
                        f"token_{m['token_index']}"
                        for m in meta
                    ],
                    f"micros_{session_id}",
                )
                for i, coord in enumerate(coords):
                    if i < len(meta):
                        micro_coords[
                            meta[i]["micro_id"]
                        ] = coord
            except Exception as e:
                logger.error(f"Micro projection error: {e}")

    for n in micro_nodes:
        micro_id = n["micro_id"]
        if micro_id in micro_coords:
            coord = micro_coords[micro_id]
            payload.append({
                **coord,
                "node_id": micro_id,
                "node_type": "micro",
                "step_number": n.get("token_index", 0),
                "content": n.get("content", ""),
                "is_parent": False,
                "has_embedding": True,
            })
        else:
            idx = n.get("token_index", 0)
            radius = 0.5 + (idx * 0.02)
            payload.append({
                "node_id": micro_id,
                "node_type": "micro",
                "step_number": idx,
                "content": n.get("content", ""),
                "is_parent": False,
                "has_embedding": False,
                "x": radius * 0.5,
                "y": float(idx) * 0.05,
                "z": radius * 0.3,
            })

    return payload if payload else None


@app.get("/stream/{session_id}")
async def stream_session(session_id: str):
    """
    SSE endpoint for live streaming.
    Stable — never drops connection on error.
    Projection cached — no repeated UMAP computation.
    """
    async def event_generator():
        seen_node_ids: set = set()
        seen_micro_ids: set = set()

        # send connected event immediately
        yield "data: {\"event\": \"connected\"}\n\n"

        poll_count = 0
        while True:
            try:
                payload = await _build_payload(
                    session_id,
                    seen_node_ids,
                    seen_micro_ids,
                )
                if payload:
                    yield (
                        f"data: "
                        f"{json.dumps({'event': 'update', 'nodes': payload})}"
                        f"\n\n"
                    )

                poll_count += 1
                # heartbeat every 10 seconds
                if poll_count % 20 == 0:
                    yield (
                        "data: {\"event\": \"heartbeat\"}\n\n"
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"SSE poll error: {e}")
                # never break — keep connection alive

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
    )


app.mount(
    "/static",
    StaticFiles(directory="visualizer/static"),
    name="static"
)