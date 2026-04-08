import asyncio
import uuid
import os
import logging
from dotenv import load_dotenv
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

from tracer.core import TracerCore
from tracer.adapters.langchain import LangChainAdapter
from tracer.queue import AsyncWriteQueue
from tracer.streaming.token_buffer import TokenBuffer
from storage.database import DatabaseWriter
from storage.vector import VectorStorage
from embeddings.encoder import EmbeddingEncoder
from replay.replay import ReplayEngine
from intelligence.pattern_library import PatternLibrary
from intelligence.drift_detector import DriftDetector
from intelligence.behavioral_signals import BehavioralSignalEngine

load_dotenv()
logging.basicConfig(level=logging.WARNING)

# -------------------------------------------------
# test tools
# -------------------------------------------------

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Search results for '{query}': Simulated result."

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

TOOLS = [search_web, calculate, save_note]

# -------------------------------------------------
# micro node handler — Track A visualizer dispatch
# -------------------------------------------------

async def on_micro_node_visualizer(micro_node):
    """
    Called immediately for every token captured.
    Sends raw token to visualizer without waiting
    for embedding.
    """
    pass  # wired to WebSocket in server.py


# -------------------------------------------------
# batch handler — Track B embedding pipeline
# -------------------------------------------------

async def make_batch_handler(
    encoder: EmbeddingEncoder,
    db_writer: DatabaseWriter
):
    async def on_batch_ready(batch):
        """
        Called every 100ms with a batch of micro nodes.
        Embeds them and stores to database.
        """
        texts = [m.content for m in batch]
        try:
            loop = asyncio.get_event_loop()
            vectors = await loop.run_in_executor(
                None,
                lambda: encoder.embeddings.embed_documents(texts)
            )
            for micro_node, vector in zip(batch, vectors):
                micro_node.embedding_vector = vector
                micro_node.embedding_complete = True

            await db_writer.write_micro_nodes(batch)

            for micro_node, vector in zip(batch, vectors):
                await db_writer.update_micro_node_embedding(
                    micro_node.micro_id, vector
                )

        except Exception as e:
            logging.error(f"Batch embedding failed: {e}")

    return on_batch_ready


# -------------------------------------------------
# main pipeline
# -------------------------------------------------

async def run_agent(task: str, task_id: str = None):
    print(f"\n{'='*50}")
    print(f"TASK: {task}")
    print(f"{'='*50}\n")

    # setup database
    db_writer = DatabaseWriter()
    await db_writer.connect()
    print("Database connected.")
    
    # setup pattern library
    pattern_library = PatternLibrary(db_writer)
    await pattern_library.setup()

    # setup drift detector
    drift_detector = DriftDetector(
        pattern_library=pattern_library,
        task_type="general",
    )

    # setup encoder
    encoder = EmbeddingEncoder()

    # setup write queue
    write_queue = AsyncWriteQueue(db_writer=db_writer)
    await write_queue.start()

    # setup token buffer with both track handlers
    batch_handler = await make_batch_handler(
        encoder, db_writer
    )
    token_buffer = TokenBuffer(
        session_id=str(uuid.uuid4()),
        batch_size=20,
        batch_interval_ms=100,
        on_visualizer_dispatch=on_micro_node_visualizer,
        on_batch_ready=batch_handler,
    )
    await token_buffer.start()
    print("Token buffer started.")

    # setup tracer core
    tracer_core = TracerCore(
        task_id=task_id or str(uuid.uuid4()),
        write_queue=write_queue,
        token_buffer=token_buffer,
    )

    # setup langchain adapter with token buffer
    adapter = LangChainAdapter(
        tracer_core=tracer_core,
        token_buffer=token_buffer,
        on_micro_node=on_micro_node_visualizer,
    )

    # setup llm with streaming enabled
    llm = ChatOllama(
        model="qwen2.5:0.5b",
        temperature=0,
        streaming=True,
        callbacks=[adapter],
    )
    # setup prompt
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful AI assistant. "
            "Use tools to complete tasks thoroughly."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(
            variable_name="agent_scratchpad"
        ),
    ])

    # build agent
    agent = create_openai_tools_agent(llm, TOOLS, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=TOOLS,
        callbacks=[adapter],
        verbose=False,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )

    # run agent
    print("Agent running...\n")
    result = await executor.ainvoke({
        "input": task,
        "chat_history": [],
    })

    print(f"\nAGENT OUTPUT:\n{result['output']}\n")

    # stop write queue first
    await write_queue.stop()

    # stop token buffer — flushes remaining tokens
    await token_buffer.stop()

    # wait for all batch embeddings to complete
    await asyncio.sleep(2)

    # embed any remaining micro nodes that missed batches
    print("Embedding remaining micro nodes...")
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
                lambda: encoder.embeddings.embed_documents(texts)
            )
            for micro_node, vector in zip(remaining, vectors):
                micro_node.embedding_vector = vector
                micro_node.embedding_complete = True
            await db_writer.write_micro_nodes(remaining)
            for micro_node, vector in zip(remaining, vectors):
                await db_writer.update_micro_node_embedding(
                    micro_node.micro_id, vector
                )
            print(f"Embedded {len(remaining)} remaining micro nodes.")
        except Exception as e:
            print(f"Remaining embedding failed: {e}")
    # embed parent nodes
    print("Embedding parent nodes...")
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
            print(f"Failed to embed node: {e}")

    await asyncio.sleep(0.5)

    # print summary
    summary = tracer_core.get_summary()
    micro_nodes = token_buffer.get_all_micro_nodes()
    embedded_micro = sum(
        1 for m in micro_nodes if m.embedding_complete
    )

    print(f"EXECUTION SUMMARY:")
    print(f"  Session ID:       {tracer_core.session_id}")
    print(f"  Parent nodes:     {summary['total_nodes']}")
    print(f"  Reasoning hops:   {summary['reasoning_hops']}")
    print(f"  Micro nodes:      {len(micro_nodes)}")
    print(f"  Embedded micro:   {embedded_micro}")

    await db_writer.close()
    return tracer_core.session_id

# store pattern from this session
# auto ingest pattern — runs in background
    goal_text = (
        "Search for information about AI agents, "
        "calculate 15 * 24, then save a note "
        "summarizing what you found."
    )

    # embed the goal for task classification
    goal_vec = await encoder._embed(goal_text)

    node_embeddings = [
        n.embedding_vector
        for n in tracer_core.get_tree().get_all_nodes()
        if n.embedding_vector
    ]

    health = drift_detector.get_trajectory_summary()

    ingest_result = await pattern_library.ingest(
        session_id=tracer_core.session_id,
        goal_text=goal_text,
        goal_embedding=goal_vec,
        node_embeddings=node_embeddings,
        behavioral_health=health["behavioral_health"],
        error_count=sum(
            1 for n in
            tracer_core.get_tree().get_all_nodes()
            if n.status == "error"
        ),
        total_steps=summary["total_nodes"],
        reasoning_hops=summary["reasoning_hops"],
    )

    stats = await pattern_library.get_stats()

    print(f"\nHEALTH SUMMARY:")
    print(
        f"  Status:        "
        f"{health['behavioral_health']['status']}"
    )
    print(
        f"  Total alerts:  "
        f"{health['total_alerts']}"
    )
    print(
        f"  Pattern:       "
        f"{ingest_result['status']} — "
        f"{ingest_result.get('outcome', 'n/a')}"
    )
    print(
        f"  Task type:     "
        f"{ingest_result.get('task_type', 'n/a')}"
    )
    print(f"\nLIBRARY STATS:")
    print(
        f"  Total patterns:  "
        f"{stats['total_patterns']}"
    )
    print(
        f"  Success:         "
        f"{stats['success_patterns']}"
    )
    print(
        f"  Failure:         "
        f"{stats['failure_patterns']}"
    )
    print(
        f"  Task clusters:   "
        f"{stats['task_clusters']}"
    )


# -------------------------------------------------
# observe mode
# -------------------------------------------------

async def demo_observe(session_id: str):
    print(f"\n{'='*50}")
    print(f"OBSERVE MODE — session {session_id}")
    print(f"{'='*50}\n")

    db_writer = DatabaseWriter()
    await db_writer.connect()

    replay = ReplayEngine(db_writer)
    observation = await replay.observe(session_id)

    if "error" in observation:
        print(f"Observe error: {observation['error']}")
        await db_writer.close()
        return

    summary = observation.get("summary", {})
    print(f"Total steps:     {summary.get('total_steps', 0)}")
    print(f"Reasoning steps: {summary.get('reasoning_steps', 0)}")
    print(f"Tool calls:      {summary.get('tool_calls', 0)}")
    print(f"Errors:          {summary.get('errors', 0)}")

    await db_writer.close()


# -------------------------------------------------
# entry point
# -------------------------------------------------

if __name__ == "__main__":
    async def main():
        session_id = await run_agent(
            task=(
                "Search for information about AI agents, "
                "calculate 15 * 24, then save a note "
                "summarizing what you found."
            ),
            task_id="test_task_001"
        )
        await demo_observe(session_id)

    asyncio.run(main())