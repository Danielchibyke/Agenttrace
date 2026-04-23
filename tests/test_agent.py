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

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

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

OllamaModel = "qwen2.5:0.5b"
geminiModel = "gemini-1.5-flash-100b"  # or "gemini-1.5-pro-flash-100b" for Pro users
OpenAIModel = "gpt-3.5-turbo"
GroqModel = "llama-3.1-8b-instant"
llms = [OllamaModel, geminiModel, OpenAIModel, GroqModel]

# -------------------------------------------------
# core agent runner — reusable
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

    # setup database
    db_writer = DatabaseWriter()
    await db_writer.connect()

    encoder = EmbeddingEncoder()

    # setup pattern library
    pattern_library = PatternLibrary(db_writer)
    await pattern_library.setup()

    # setup Redis embedding worker
    # pushes to embedding service — never blocks agent
    embedding_worker = EmbeddingWorker()
    await embedding_worker.start()

    # setup write queue
    write_queue = AsyncWriteQueue(db_writer=db_writer)
    await write_queue.start()
    
    #  # CREATE PARENT NODE FIRST
    # parent_node_id = str(uuid.uuid4())
    # session_id = str(uuid.uuid4())
    
    # setup token buffer
    # writes micro skeletons immediately to database
    # pushes content to Redis for embedding
    token_buffer = TokenBuffer(
        session_id=str(uuid.uuid4()),
        # parent_node_id=parent_node_id,
        batch_size=20,
        batch_interval_ms=100,
        db_writer=db_writer,
        embedding_worker=embedding_worker,
    )
    await token_buffer.start()

    # setup tracer core
    tracer_core = TracerCore(
        task_id=task_id or str(uuid.uuid4()),
        write_queue=write_queue,
        token_buffer=token_buffer,
    )

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

    # setup llm
    
    
    if(model == OllamaModel):
    #ollam llm with streaming and callbacks to capture tokens and micro skeletons
    
        llm = ChatOllama(
            model=model,
            temperature=0,
            streaming=True,
            callbacks=[adapter],
        )
    else:
        if(model == geminiModel):
    # gemini llm with streaming and callbacks to capture tokens and micro skeletons
            llm = ChatGoogleGenerativeAI(
                model=model,
                temperature=0,
                streaming=True,
                callbacks=[adapter],
                convert_system_message_to_human=True,
            )
        else:
            if(model == OpenAIModel):
    
    # openai llm with streaming and callbacks to capture tokens and micro skeletons
                llm = ChatOpenAI(
                    model=model,
                    temperature=0,
                    streaming=True,
                    callbacks=[adapter],
                )
            else:
                if(model == GroqModel):
    # groq llm with streaming and callbacks to capture tokens and micro skeletons
                    llm = ChatGroq(
                        model=model,
                        temperature=0,
                        streaming=True,
                        callbacks=[adapter],
                        max_tokens=1024,
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

    # run agent
    result = await executor.ainvoke({
        "input": task,
        "chat_history": [],
    })

    if verbose:
        print(
            f"\nAGENT OUTPUT: "
            f"{result.get('output', '')}"
        )

    # flush queues
    await write_queue.stop()
    await token_buffer.stop()

    # give embedding service time to process queue
    # embeddings happen in separate process — just wait
    await asyncio.sleep(3)

    worker_stats = embedding_worker.get_stats()
    await embedding_worker.stop()

    if verbose:
        print(
            f"Queued for embedding: "
            f"{worker_stats['queued_count']}"
        )
        print(
            f"Queue remaining: "
            f"{worker_stats['queue_sizes']}"
        )

    # fetch final state from database
    summary = tracer_core.get_summary()
    health = drift_detector.get_trajectory_summary()
    goal_vec = await encoder._embed(task)

    all_nodes = tracer_core.get_tree().get_all_nodes()

    # fetch embeddings from db
    # embedding service wrote them asynchronously
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