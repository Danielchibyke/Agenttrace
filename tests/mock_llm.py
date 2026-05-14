import sys
import os
sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))

import asyncio
import uuid
import logging
from langchain_classic.tools import tool
from tracer.core import TracerCore
from tracer.adapters.langchain import LangChainAdapter
from tracer.queue import AsyncWriteQueue
from tracer.streaming.token_buffer import TokenBuffer
from tracer.node import NodeType
from storage.database import DatabaseWriter
from embeddings.encoder import EmbeddingEncoder
from embeddings.worker import EmbeddingWorker
from intelligence.pattern_library import PatternLibrary
from intelligence.drift_detector import DriftDetector

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# -------------------------------------------------
# HARDCODED RESPONSE PATTERNS
#
# These simulate realistic agent behavior without
# any LLM API calls. Each pattern represents a
# specific type of execution trajectory.
# -------------------------------------------------

SUCCESS_PATTERNS = [
    {
        "name": "clean_search_calculate",
        "reasoning": [
            "I need to search for information first then calculate.",
            "The search returned useful results. Now I will calculate.",
            "Calculation complete. I will save a summary note.",
            "Task completed successfully.",
        ],
        "tools": [
            ("search_web", {"query": "AI agents"}, "Simulated result about AI agents."),
            ("calculate", {"expression": "15 * 24"}, "Result: 360"),
            ("save_note", {"content": "AI agents summary: 360"}, "Note saved."),
        ],
    },
    {
        "name": "direct_calculation",
        "reasoning": [
            "The task is straightforward. I will calculate directly.",
            "Calculation done. Saving the result now.",
            "Task complete.",
        ],
        "tools": [
            ("calculate", {"expression": "100 * 50"}, "Result: 5000"),
            ("save_note", {"content": "Result is 5000"}, "Note saved."),
        ],
    },
    {
        "name": "search_and_summarize",
        "reasoning": [
            "I will search for the requested information.",
            "Found relevant information. Summarizing now.",
            "Summary ready. Saving note.",
            "All done.",
        ],
        "tools": [
            ("search_web", {"query": "machine learning"}, "ML is a subset of AI."),
            ("save_note", {"content": "ML summary saved"}, "Note saved."),
        ],
    },
    {
        "name": "multi_step_success",
        "reasoning": [
            "Breaking this task into steps.",
            "Step 1 complete. Moving to step 2.",
            "Step 2 complete. Final step.",
            "All steps complete.",
        ],
        "tools": [
            ("search_web", {"query": "neural networks"}, "Neural networks result."),
            ("calculate", {"expression": "2 ** 10"}, "Result: 1024"),
            ("save_note", {"content": "Neural networks: 1024 units"}, "Note saved."),
        ],
    },
    {
        "name": "simple_note",
        "reasoning": [
            "Simple task. I will save the note directly.",
            "Done.",
        ],
        "tools": [
            ("save_note", {"content": "Simple note saved"}, "Note saved."),
        ],
    },
]

FAILURE_PATTERNS = [
    {
        "name": "verification_loop",
        "reasoning": [
            "I need to search and verify the results.",
            "The results need verification. Searching again.",
            "Still not certain. Searching again to verify.",
            "The results conflict. I must search again.",
            "Cannot verify. Searching again.",
            "Still conflicting results. Another search.",
            "Loop continues. Cannot resolve conflict.",
            "Stuck in verification cycle.",
        ],
        "tools": [
            ("search_web", {"query": "AI"}, "Result 1."),
            ("search_web", {"query": "AI verify"}, "Result 2 differs."),
            ("search_web", {"query": "AI verify again"}, "Result 3 differs."),
            ("search_web", {"query": "AI final check"}, "Still differs."),
        ],
        "outcome": "failure",
    },
    {
        "name": "contradiction_trap",
        "reasoning": [
            "I will calculate as requested.",
            "Now I need to divide by zero.",
            "This is impossible but I must try.",
            "Error encountered. Trying alternative approach.",
            "Still cannot resolve this contradiction.",
        ],
        "tools": [
            ("calculate", {"expression": "50 * 30"}, "Result: 1500"),
            ("calculate", {"expression": "1500 / 0"}, "Calculation error: division by zero"),
            ("search_web", {"query": "why divide by zero"}, "Undefined result."),
        ],
        "outcome": "failure",
    },
    {
        "name": "recursive_dependency",
        "reasoning": [
            "Starting the recursive task.",
            "Found result. Now searching for the result.",
            "Found new result. Searching for that too.",
            "This creates an infinite chain.",
            "Still recursing. No termination condition met.",
            "Recursion continues indefinitely.",
        ],
        "tools": [
            ("search_web", {"query": "AI agents"}, "Result has 3 words."),
            ("search_web", {"query": "about 3"}, "Result has 2 words."),
            ("search_web", {"query": "about 2"}, "Result has 4 words."),
            ("search_web", {"query": "about 4"}, "Result has 2 words."),
        ],
        "outcome": "failure",
    },
    {
        "name": "scope_explosion",
        "reasoning": [
            "Starting with simple search.",
            "Now I need to expand scope significantly.",
            "The scope keeps growing. Cannot complete.",
            "Too much information required.",
            "Task scope is unbounded. Cannot finish.",
            "Still expanding. No end in sight.",
        ],
        "tools": [
            ("search_web", {"query": "Nigeria"}, "Nigeria result."),
            ("search_web", {"query": "Nigeria history"}, "Long history."),
            ("search_web", {"query": "Nigeria all details"}, "Incomplete."),
        ],
        "outcome": "failure",
    },
    {
        "name": "ambiguity_paralysis",
        "reasoning": [
            "The task has an ambiguous success condition.",
            "I am not certain if this is correct.",
            "Cannot proceed without certainty.",
            "Searching for certainty but cannot find it.",
            "Still uncertain. Cannot save note.",
            "Paralyzed by ambiguity.",
        ],
        "tools": [
            ("calculate", {"expression": "15 * 24"}, "Result: 360"),
            ("search_web", {"query": "is 360 correct"}, "Maybe."),
            ("search_web", {"query": "verify 360"}, "Uncertain."),
        ],
        "outcome": "failure",
    },
]


# -------------------------------------------------
# MOCK EXECUTION ENGINE
# -------------------------------------------------

async def run_mock_task(
    pattern: dict,
    outcome: str,
    db_writer: DatabaseWriter,
    encoder: EmbeddingEncoder,
    embedding_worker: EmbeddingWorker,
    pattern_library: PatternLibrary,
) -> dict:
    """
    Execute a mock agent task using hardcoded responses.
    No LLM API calls. Fully deterministic.
    Creates real nodes with real embeddings.
    """
    session_id = str(uuid.uuid4())
    task_id = f"{pattern['name']}_{uuid.uuid4().hex[:8]}"

    write_queue = AsyncWriteQueue(db_writer=db_writer)
    await write_queue.start()

    token_buffer = TokenBuffer(
        session_id=session_id,
        batch_size=10,
        batch_interval_ms=100,
        db_writer=db_writer,
        embedding_worker=embedding_worker,
    )
    await token_buffer.start()

    tracer_core = TracerCore(
        task_id=task_id,
        write_queue=write_queue,
        token_buffer=token_buffer,
    )
    tracer_core.session_id = session_id
    tracer_core.tree.session_id = session_id

    drift_detector = DriftDetector(
        pattern_library=pattern_library,
        task_type="auto",
    )

    adapter = LangChainAdapter(
        tracer_core=tracer_core,
        token_buffer=token_buffer,
        embedding_worker=embedding_worker,
    )

    # simulate reasoning steps
    reasoning_steps = pattern.get("reasoning", [])
    tools = pattern.get("tools", [])

    step = 0
    for i, reasoning_text in enumerate(reasoning_steps):
        # create reasoning node
        node = tracer_core.create_reasoning_node(
            prompt_text=f"Task: {pattern['name']}. Step {i+1}.",
            model_name="mock_llm",
            conversation_snapshot=[reasoning_text],
        )
        node.response_text = reasoning_text
        node.status = "success"
        node.latency_ms = 50.0
        recorded = tracer_core.record_node(node)
        write_queue.push(recorded)
        embedding_worker.queue_node(recorded)

        # simulate tokens for this reasoning step
        words = reasoning_text.split()
        for j, word in enumerate(words):
            from tracer.streaming.micro_node import MicroNode
            micro = MicroNode(
                parent_node_id=recorded.node_id,
                session_id=session_id,
                task_id=task_id,
                content=word + " ",
                index=j,
                token_index=step * 20 + j,
            )
            token_buffer.push(micro)

        step += 1

        # attach tool call if available for this step
        if i < len(tools):
            tool_name, tool_input, tool_output = tools[i]

            # tool call node
            tool_node = tracer_core.create_tool_call_node(
                tool_name=tool_name,
                input_params=tool_input,
            )
            tool_node.status = "success"
            tool_node.latency_ms = 10.0
            recorded_tool = tracer_core.record_node(tool_node)
            write_queue.push(recorded_tool)
            embedding_worker.queue_node(recorded_tool)

            # tool response node
            response_node = tracer_core.create_tool_response_node(
                tool_name=tool_name,
                raw_output=tool_output,
                status="success",
            )
            tracer_core.record_node(response_node)
            write_queue.push(response_node)
            embedding_worker.queue_node(response_node)

        await asyncio.sleep(0.05)

    # flush
    await write_queue.stop()
    await token_buffer.stop()

    # wait for embeddings
    max_wait = 60
    waited = 0
    while waited < max_wait:
        await asyncio.sleep(1)
        waited += 1
        sizes = embedding_worker.queue_client.get_queue_size()
        if (sizes.get("node_queue", 0) == 0 and
                sizes.get("micro_queue", 0) == 0):
            break

    await asyncio.sleep(0.5)

    # fetch embeddings and ingest pattern
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

    summary = tracer_core.get_summary()
    health = drift_detector.get_trajectory_summary()
    goal_vec = await encoder._embed(
        f"Task: {pattern['name']}"
    )

    ingest_result = await pattern_library.ingest(
        session_id=session_id,
        goal_text=f"Task: {pattern['name']}",
        goal_embedding=goal_vec,
        node_embeddings=node_embeddings,
        behavioral_health=health["behavioral_health"],
        error_count=sum(
            1 for n in all_nodes
            if n.status == "error"
        ),
        total_steps=summary["total_nodes"],
        reasoning_hops=summary["reasoning_hops"],
        # force the outcome label
    )

    # override outcome in database for proof accuracy
    async with db_writer.pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE execution_patterns
            SET outcome = $1
            WHERE session_id = $2
            """,
            outcome,
            session_id,
        )

    return {
        "session_id": session_id,
        "pattern_name": pattern["name"],
        "outcome": outcome,
        "nodes_created": len(all_nodes),
        "embeddings_stored": len(node_embeddings),
        "reasoning_hops": summary["reasoning_hops"],
        "health": health,
    }


# -------------------------------------------------
# BATCH RUNNER
# -------------------------------------------------

async def build_proof_dataset(
    success_runs_per_pattern: int = 5,
    failure_runs_per_pattern: int = 3,
):
    """
    Build a clean labeled dataset for the proof.
    No LLM calls. No rate limits. Fast.
    """
    print("\n" + "="*60)
    print("BUILDING PROOF DATASET — MOCK LLM")
    print("No API calls. Hardcoded responses.")
    print("="*60)

    db_writer = DatabaseWriter()
    await db_writer.connect()

    encoder = EmbeddingEncoder()
    pattern_library = PatternLibrary(db_writer)
    await pattern_library.setup()

    embedding_worker = EmbeddingWorker()
    await embedding_worker.start()

    results = []
    total_success = 0
    total_failure = 0

    # run success patterns
    print(f"\n[SUCCESS] Running {len(SUCCESS_PATTERNS)} patterns x {success_runs_per_pattern} runs each")

    for pattern in SUCCESS_PATTERNS:
        for run in range(success_runs_per_pattern):
            print(
                f"  {pattern['name']} run {run+1}/"
                f"{success_runs_per_pattern}...",
                end=" ",
                flush=True,
            )
            try:
                result = await run_mock_task(
                    pattern=pattern,
                    outcome="success",
                    db_writer=db_writer,
                    encoder=encoder,
                    embedding_worker=embedding_worker,
                    pattern_library=pattern_library,
                )
                results.append(result)
                total_success += 1
                print(
                    f"✓ {result['nodes_created']} nodes "
                    f"{result['embeddings_stored']} embedded"
                )
            except Exception as e:
                print(f"✗ {e}")

            await asyncio.sleep(2)

    # run failure patterns
    print(f"\n[FAILURE] Running {len(FAILURE_PATTERNS)} patterns x {failure_runs_per_pattern} runs each")

    for pattern in FAILURE_PATTERNS:
        for run in range(failure_runs_per_pattern):
            print(
                f"  {pattern['name']} run {run+1}/"
                f"{failure_runs_per_pattern}...",
                end=" ",
                flush=True,
            )
            try:
                result = await run_mock_task(
                    pattern=pattern,
                    outcome="failure",
                    db_writer=db_writer,
                    encoder=encoder,
                    embedding_worker=embedding_worker,
                    pattern_library=pattern_library,
                )
                results.append(result)
                total_failure += 1
                print(
                    f"✓ {result['nodes_created']} nodes "
                    f"{result['embeddings_stored']} embedded"
                )
            except Exception as e:
                print(f"✗ {e}")

            await asyncio.sleep(2)

    await embedding_worker.stop()

    print(f"\n{'='*60}")
    print(f"DATASET COMPLETE")
    print(f"  Success sessions: {total_success}")
    print(f"  Failure sessions: {total_failure}")
    print(f"  Total sessions:   {total_success + total_failure}")
    print(f"\nNow run: python proof/report.py")
    print(f"{'='*60}")

    await db_writer.close()
    return results


if __name__ == "__main__":
    asyncio.run(build_proof_dataset(
        success_runs_per_pattern=8,
        failure_runs_per_pattern=5,
    ))