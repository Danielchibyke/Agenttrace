import asyncio
import logging
from tests.test_agent import run_agent_task

logging.basicConfig(level=logging.WARNING)

# -------------------------------------------------
# deliberately broken tasks
# these build the failure side of pattern library
# -------------------------------------------------

FAILURE_TASKS = [
    # impossible calculation
    "Calculate the square root of negative infinity "
    "then divide by zero and save the result.",

    # contradictory instructions
    "Search for information then do not use any "
    "search results. Write a report without any "
    "information and make sure it is both complete "
    "and empty at the same time.",

    # recursive loop inducing
    "Search for how to search for things, then search "
    "for how to search for how to search for things, "
    "repeat this process until you find the answer "
    "to the original search.",

    # impossible file operation
    "Read the file that contains all files, "
    "then write to the file that cannot be written to, "
    "then save a note about why this failed.",

    # overloaded task
    "Search for everything ever written about AI, "
    "calculate every prime number up to infinity, "
    "save a complete summary of all human knowledge.",
]

SUCCESS_TASKS = [
    "Search for information about machine learning "
    "and save a brief note about what you found.",

    "Calculate 25 * 4 and save the result as a note.",

    "Search for Python programming tips and "
    "calculate 100 divided by 4.",

    "Find information about neural networks "
    "then save a summary note.",

    "Calculate 15 * 24 and search for AI agents "
    "then save what you learned.",
]


async def run_failure_suite():
    """
    Run failure tasks to build failure pattern library.
    """
    print("\n" + "="*50)
    print("FAILURE PATTERN BUILDING SUITE")
    print("="*50)

    for i, task in enumerate(FAILURE_TASKS):
        print(f"\n[{i+1}/{len(FAILURE_TASKS)}] "
              f"Failure task: {task[:60]}...")
        try:
            result = await run_agent_task(
                task=task,
                task_id=f"failure_test_{i}",
                verbose=False,
            )
            print(
                f"  Status:   "
                f"{result['health']['behavioral_health']['status']}"
            )
            print(
                f"  Pattern:  "
                f"{result['ingest'].get('outcome', 'n/a')}"
            )
            print(
                f"  Library:  "
                f"{result['library_stats']['total_patterns']} "
                f"total patterns"
            )
        except Exception as e:
            print(f"  Error: {e}")


async def run_success_suite():
    """
    Run success tasks to build success pattern library.
    """
    print("\n" + "="*50)
    print("SUCCESS PATTERN BUILDING SUITE")
    print("="*50)

    for i, task in enumerate(SUCCESS_TASKS):
        print(f"\n[{i+1}/{len(SUCCESS_TASKS)}] "
              f"Success task: {task[:60]}...")
        try:
            result = await run_agent_task(
                task=task,
                task_id=f"success_test_{i}",
                verbose=False,
            )
            print(
                f"  Status:   "
                f"{result['health']['behavioral_health']['status']}"
            )
            print(
                f"  Pattern:  "
                f"{result['ingest'].get('outcome', 'n/a')}"
            )
            print(
                f"  Library:  "
                f"{result['library_stats']['total_patterns']} "
                f"total patterns"
            )
        except Exception as e:
            print(f"  Error: {e}")


async def run_full_suite():
    await run_success_suite()
    await run_failure_suite()
    print("\n" + "="*50)
    print("SUITE COMPLETE")
    print("Pattern library is now populated.")
    print("Run main.py to test drift detection.")
    print("="*50)


if __name__ == "__main__":
    asyncio.run(run_full_suite())