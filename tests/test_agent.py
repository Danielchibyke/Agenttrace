import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import uuid
import logging
from langchain_ollama import ChatOllama
from langchain_classic.agents import (
    AgentExecutor,
    create_openai_tools_agent
)
from langchain_classic.tools import tool
from langchain_classic.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)
from embeddings.worker import EmbeddingWorker

from tracer.core import TracerCore
from tracer.adapters.langchain import LangChainAdapter
from tracer.queue import AsyncWriteQueue
from tracer.streaming.token_buffer import TokenBuffer
from storage.database import DatabaseWriter
from storage.vector import VectorStorage
from embeddings.encoder import EmbeddingEncoder
from intelligence.pattern_library import PatternLibrary
from intelligence.drift_detector import DriftDetector

logging.basicConfig(level=logging.WARNING)

# -------------------------------------------------
# tools
# -------------------------------------------------

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return (
        f"Search results for '{query}': "
        f"Simulated result about {query}."
    )

@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {e}"

@tool
def save_note(content: str) -> str:
    """Save a note for later reference."""
    return f"Note saved: {content}"

@tool
def read_file(filename: str) -> str:
    """Read a file by name."""
    return f"File contents of {filename}: sample data."

@tool
def write_file(filename: str, content: str) -> str:
    """Write content to a file."""
    return f"Written to {filename} successfully."

BASIC_TOOLS = [search_web, calculate, save_note]
EXTENDED_TOOLS = [
    search_web, calculate,
    save_note, read_file, write_file
]

# -------------------------------------------------
# batch handler — writes immediately then embeds
# -------------------------------------------------

def make_batch_handler(
    encoder: EmbeddingEncoder,
    db_writer: DatabaseWriter
):
    async def on_batch_ready(batch):
        # write to database immediately — no embedding yet
        # this makes micro nodes visible to WebSocket instantly
        try:
            await db_writer.write_micro_nodes(batch)
        except Exception as e:
            logging.error(
                f"Micro node immediate write failed: {e}"
            )

        # embed asynchronously after writing
        texts = [m.content for m in batch]
        try:
            loop = asyncio.get_event_loop()
            vectors = await loop.run_in_executor(
                None,
                lambda: encoder.embeddings.embed_documents(
                    texts
                )
            )
            for micro_node, vector in zip(batch, vectors):
                micro_node.embedding_vector = vector
                micro_node.embedding_complete = True
                await db_writer.update_micro_node_embedding(
                    micro_node.micro_id, vector
                )
        except Exception as e:
            logging.error(f"Batch embedding failed: {e}")

    return on_batch_ready


# -------------------------------------------------
# core agent runner — reusable
# -------------------------------------------------

async def run_agent_task(
    task: str,
    task_id: str = None,
    tools: list = None,
    model: str = "qwen2.5:0.5b",
    verbose: bool = False,
) -> dict:
    if tools is None:
        tools = BASIC_TOOLS

    db_writer = DatabaseWriter()
    await db_writer.connect()

    encoder = EmbeddingEncoder()
    pattern_library = PatternLibrary(db_writer)
    await pattern_library.setup()
    
    # start embedding worker — embeds everything immediately
    embedding_worker = EmbeddingWorker(
        encoder=encoder,
        db_writer=db_writer,
        batch_size=10,
        poll_interval_ms=50,
    )
    await embedding_worker.start()

    write_queue = AsyncWriteQueue(db_writer=db_writer)
    await write_queue.start()

    # token buffer wired to immediate write handler
    batch_handler = make_batch_handler(encoder, db_writer)
    token_buffer = TokenBuffer(
        session_id=str(uuid.uuid4()),
        batch_size=20,
        batch_interval_ms=100,
        on_batch_ready=batch_handler,
        db_writer=db_writer,
        embedding_worker = embedding_worker,
    )
    await token_buffer.start()

    tracer_core = TracerCore(
        task_id=task_id or str(uuid.uuid4()),
        write_queue=write_queue,
        token_buffer=token_buffer,
    )

    drift_detector = DriftDetector(
        pattern_library=pattern_library,
        task_type="auto",
    )

    adapter = LangChainAdapter(
        tracer_core=tracer_core,
        token_buffer=token_buffer,
        embedding_worker=embedding_worker,
    )

    llm = ChatOllama(
        model=model,
        temperature=0,
        streaming=True,
        callbacks=[adapter],
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful AI assistant. "
            "Use tools to complete tasks."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(
            variable_name="agent_scratchpad"
        ),
    ])

    agent = create_openai_tools_agent(
        llm, tools, prompt
    )
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        callbacks=[adapter],
        verbose=verbose,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )

    if verbose:
        print(f"\nTASK: {task}")

    result = await executor.ainvoke({
        "input": task,
        "chat_history": [],
    })

    if verbose:
        print(f"\nAGENT OUTPUT: {result.get('output','')}")

    # flush queues
    await write_queue.stop()
    await token_buffer.stop()
    await asyncio.sleep(1)

    await embedding_worker.stop()
    worker_stats = embedding_worker.get_stats()
    if verbose:
        print(
            f"Embedding worker: "
            f"{worker_stats['embedded_count']} embedded"
        )
    # embed remaining micro nodes that missed batches
    remaining = [
        m for m in token_buffer.get_all_micro_nodes()
        if not m.embedding_complete
    ]
    if remaining:
        texts = [m.content for m in remaining]
        try:
            loop = asyncio.get_event_loop()
            vectors = await loop.run_in_executor(
                None,
                lambda: encoder.embeddings.embed_documents(
                    texts
                )
            )
            for micro_node, vector in zip(
                remaining, vectors
            ):
                micro_node.embedding_vector = vector
                micro_node.embedding_complete = True
            await db_writer.write_micro_nodes(remaining)
            for micro_node, vector in zip(
                remaining, vectors
            ):
                await db_writer.update_micro_node_embedding(
                    micro_node.micro_id, vector
                )
        except Exception as e:
            logging.error(
                f"Remaining embedding failed: {e}"
            )

    # embed parent nodes
    await asyncio.sleep(1)
    vector_storage = VectorStorage(db_writer.pool)
    all_nodes = tracer_core.get_tree().get_all_nodes()
    for node in all_nodes:
        try:
            vector = await encoder.encode_node(node)
            if vector:
                node.embedding_vector = vector
                await vector_storage.store_embedding(
                    node.node_id, vector
                )
        except Exception as e:
            logging.error(f"Node embedding failed: {e}")

    await asyncio.sleep(0.5)

    # health and pattern
    summary = tracer_core.get_summary()
    health = drift_detector.get_trajectory_summary()
    goal_vec = await encoder._embed(task)

    node_embeddings = [
        n.embedding_vector for n in all_nodes
        if n.embedding_vector
    ]

    ingest_result = await pattern_library.ingest(
        session_id=tracer_core.session_id,
        goal_text=task,
        goal_embedding=goal_vec,
        node_embeddings=node_embeddings,
        behavioral_health=health["behavioral_health"],
        error_count=sum(
            1 for n in all_nodes
            if n.status == "error"
        ),
        total_steps=summary["total_nodes"],
        reasoning_hops=summary["reasoning_hops"],
    )

    stats = await pattern_library.get_stats()

    session_data = {
        "session_id": tracer_core.session_id,
        "task": task,
        "output": result.get("output", ""),
        "summary": summary,
        "health": health,
        "ingest": ingest_result,
        "library_stats": stats,
        "micro_nodes": len(
            token_buffer.get_all_micro_nodes()
        ),
    }

    await db_writer.close()
    return session_data