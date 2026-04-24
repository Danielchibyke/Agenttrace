import asyncio
import logging
from tests.test_agent import run_agent_task

logging.basicConfig(level=logging.WARNING)

DEFAULT_TASK = (
    "Search for information about japan hall, "
    "calculate 15 * 24, then save a note "
    "summarizing what you found."
)


async def main():
    result = await run_agent_task(
        task=DEFAULT_TASK,
        task_id="main_task",
        verbose=True,
    )

    print(f"\n{'='*50}")
    print(f"SESSION:  {result['session_id']}")
    print(f"OUTPUT:   {result['output'][:100]}...")
    print(
        f"HEALTH:   "
        f"{result['health']['behavioral_health']['status']}"
    )
    print(
        f"ALERTS:   "
        f"{result['health']['total_alerts']}"
    )
    print(
        f"PATTERN:  "
        f"{result['ingest'].get('outcome', 'n/a')} — "
        f"{result['ingest'].get('task_type', 'n/a')}"
    )
    print(
        f"LIBRARY:  "
        f"{result['library_stats']['total_patterns']} "
        f"total patterns"
    )
    print(
        f"MICROS:   "
        f"{result['micro_nodes']} tokens captured"
    )


if __name__ == "__main__":
    asyncio.run(main())