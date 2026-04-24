import sys
import os
sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))

import asyncio
import uuid
import logging
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_classic.agents import (
    AgentExecutor,
    create_openai_tools_agent
)
from langchain_classic.tools import tool
from langchain_classic.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

from tracer.core import TracerCore
from tracer.adapters.langchain import LangChainAdapter
from tracer.queue import AsyncWriteQueue
from tracer.streaming.token_buffer import TokenBuffer
from storage.database import DatabaseWriter
from embeddings.encoder import EmbeddingEncoder
from embeddings.worker import EmbeddingWorker
from intelligence.pattern_library import PatternLibrary
from intelligence.drift_detector import DriftDetector

logging.basicConfig(level=logging.WARNING)

# -------------------------------------------------
# model constants
# -------------------------------------------------

OllamaModel = "qwen2.5:0.5b"
GeminiModel = "gemini-1.5-flash"
OpenAIModel = "gpt-3.5-turbo"
GroqModel = "llama-3.1-8b-instant"

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
# llm builder
# -------------------------------------------------

def build_llm(model: str, adapter):
    """Build LLM for given model with adapter callbacks."""
    if model == OllamaModel:
        return ChatOllama(
            model=model,
            temperature=0,
            streaming=True,
            callbacks=[adapter],
        )
    if model == GeminiModel:
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=0,
            streaming=True,
            callbacks=[adapter],
            convert_system_message_to_human=True,
        )
    if model == OpenAIModel:
        return ChatOpenAI(
            model=model,
            temperature=0,
            streaming=True,
            callbacks=[adapter],
        )
    if model == GroqModel:
        return ChatGroq(
            model=model,
            temperature=0,
            streaming=True,
            callbacks=[adapter],
            max_tokens=1024,
        )
    raise ValueError(
        f"Unknown model: {model}. "
        f"Available: {[OllamaModel, GeminiModel, OpenAIModel, GroqModel]}"
    )

# -------------------------------------------------
# core agent runner
# -------------------------------------------------

async def run_agent_task(
    task: str,
    task_id: str = None,
    tools: list = None,
    model: str = GroqModel,
    verbose: bool = False,
) -> dict:
    if tools is None:
        tools = BASIC_TOOLS

    # shared session id
    session_id = str(uuid.uuid4())
    task_id = task_id or str(uuid.uuid4())

    # setup database
    db_writer = DatabaseWriter()
    await db_writer.connect()

    encoder = EmbeddingEncoder()

    # setup pattern library
    pattern_library = PatternLibrary(db_writer)
    await pattern_library.setup()

    # setup Redis embedding worker
    embedding_worker = EmbeddingWorker()
    await embedding_worker.start()

    # setup write queue
    write_queue = AsyncWriteQueue(db_writer=db_writer)
    await write_queue.start()

    # setup token buffer — uses shared session_id
    token_buffer = TokenBuffer(
        session_id=session_id,
        batch_size=20,
        batch_interval_ms=100,
        db_writer=db_writer,
        embedding_worker=embedding_worker,
    )
    await token_buffer.start()

    # setup tracer core — uses same session_id
    tracer_core = TracerCore(
        task_id=task_id,
        write_queue=write_queue,
        token_buffer=token_buffer,
    )
    # override session_id to match token_buffer
    tracer_core.session_id = session_id
    tracer_core.tree.session_id = session_id

    # setup drift detector
    drift_detector = DriftDetector(
        pattern_library=pattern_library,
        task_type="auto",
    )

    # setup adapter
    adapter = LangChainAdapter(
        tracer_core=tracer_core,
        token_buffer=token_buffer,
        embedding_worker=embedding_worker,
    )

    # build llm
    llm = build_llm(model, adapter)

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

    agent = create_openai_tools_agent(llm, tools, prompt)
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

    # run agent
    result = await executor.ainvoke({
        "input": task,
        "chat_history": [],
    })

    if verbose:
        print(f"\nAGENT OUTPUT: {result.get('output', '')}")

    # flush queues
    await write_queue.stop()
    await token_buffer.stop()

    # wait for embedding queue to drain
    print("Waiting for embeddings...")
    max_wait = 30
    waited = 0
    while waited < max_wait:
        await asyncio.sleep(1)
        waited += 1
        sizes = embedding_worker.queue_client.get_queue_size()
        node_q = sizes.get("node_queue", 0)
        micro_q = sizes.get("micro_queue", 0)
        if node_q == 0 and micro_q == 0:
            print(f"Embeddings done in {waited}s.")
            break
        if waited % 5 == 0:
            print(
                f"Queue: {node_q} nodes "
                f"{micro_q} micros remaining..."
            )

    await asyncio.sleep(0.5)
    await embedding_worker.stop()

    # fetch summary and health
    summary = tracer_core.get_summary()
    health = drift_detector.get_trajectory_summary()

    # embed goal for pattern classification
    goal_vec = await encoder._embed(task)

    # fetch node embeddings from database
    all_nodes = tracer_core.get_tree().get_all_nodes()
    node_embeddings = []
    async with db_writer.pool.acquire() as conn:
        for node in all_nodes:
            row = await conn.fetchrow(
                """
                SELECT embedding_vector::text
                FROM nodes WHERE node_id = $1
                """,
                node.node_id
            )
            if row and row["embedding_vector"]:
                vec = [
                    float(x) for x in
                    row["embedding_vector"]
                    .strip("[]").split(",")
                ]
                node.embedding_vector = vec
                node_embeddings.append(vec)

    # ingest pattern
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