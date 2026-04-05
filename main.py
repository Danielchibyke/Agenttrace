import asyncio
import uuid
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
from langchain_classic.tools import tool
from langchain_classic.prompts import ChatPromptTemplate, MessagesPlaceholder

from tracer.core import TracerCore
from tracer.adapters.langchain import LangChainAdapter
from tracer.queue import AsyncWriteQueue
from storage.database import DatabaseWriter
from storage.vector import VectorStorage
from embeddings.encoder import EmbeddingEncoder
from replay.replay import ReplayEngine

load_dotenv()

# -------------------------------------------------
# define some simple test tools for the agent
# -------------------------------------------------

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Search results for '{query}': This is a simulated result."

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

    # setup write queue
    write_queue = AsyncWriteQueue(db_writer=db_writer)
    await write_queue.start()
    print("Write queue started.")

    # setup tracer core
    tracer_core = TracerCore(
        task_id=task_id or str(uuid.uuid4()),
        write_queue=write_queue
    )

    # setup langchain adapter
    adapter = LangChainAdapter(tracer_core=tracer_core)

    # setup llm
    llm = ChatOllama(
    model="qwen2.5:0.5b",
    temperature=0,
)
    # setup prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. "
                   "Use tools to complete tasks thoroughly."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # build agent
    agent = create_openai_tools_agent(llm, TOOLS, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=TOOLS,
        callbacks=[adapter],
        verbose=True,
        return_intermediate_steps=True
    )

    # run agent
    result = await executor.ainvoke(
        {"input": task, "chat_history": []},
        config={"callbacks": [adapter]} 
    )

    print(f"\nAGENT OUTPUT: {result['output']}")

    # stop queue and flush remaining nodes
    await write_queue.stop()

    # wait briefly to ensure all nodes are flushed
    await asyncio.sleep(1)
   # compute embeddings concurrently as agent runs
    # embed nodes one by one as soon as they're in the db
    print("\nComputing embeddings...")
    encoder = EmbeddingEncoder()
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
                print(f"  Embedded: {node.node_type.value} "
                      f"step {node.step_number}")
        except Exception as e:
            print(f"  Failed to embed node {node.node_id}: {e}")

    print(f"Done. {len(all_nodes)} nodes embedded.")

    # final flush — ensures last node is visible
    await asyncio.sleep(0.5)

    summary = tracer_core.get_summary()
    print(f"\nEXECUTION SUMMARY:")
    print(f"  Session ID:      {tracer_core.session_id}")
    print(f"  Total nodes:     {summary['total_nodes']}")
    print(f"  Reasoning hops:  {summary['reasoning_hops']}")

    await db_writer.close()
    return tracer_core.session_id


async def demo_observe(session_id: str):
    print(f"\n{'='*50}")
    print(f"OBSERVE MODE — session {session_id}")
    print(f"{'='*50}\n")

    db_writer = DatabaseWriter()
    await db_writer.connect()

    replay = ReplayEngine(db_writer)
    observation = await replay.observe(session_id)

    print(f"Total steps:     {observation['summary']['total_steps']}")
    print(f"Reasoning steps: {observation['summary']['reasoning_steps']}")
    print(f"Tool calls:      {observation['summary']['tool_calls']}")
    print(f"Errors:          {observation['summary']['errors']}")
    print(f"Last reasoning:  {observation['summary']['last_reasoning']}")

    await db_writer.close()
    return tracer_core.session_id

# -------------------------------------------------
# entry point
# -------------------------------------------------

if __name__ == "__main__":
    async def main():
        session_id = await run_agent(
            task="Search for information about AI agents, "
                 "calculate 15 * 24, then save a note "
                 "summarizing what you found.",
            task_id="test_task_001"
        )
        await demo_observe(session_id)

    asyncio.run(main())