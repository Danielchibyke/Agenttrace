import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asyncio
import logging
from storage.database import DatabaseWriter
from intelligence.pattern_library import PatternLibrary

logging.basicConfig(level=logging.WARNING)


async def analyze_library():
    """
    Inspect the current state of the pattern library.
    Shows what's been learned so far.
    """
    db_writer = DatabaseWriter()
    await db_writer.connect()

    pattern_library = PatternLibrary(db_writer)
    await pattern_library.setup()

    stats = await pattern_library.get_stats()

    print("\n" + "="*50)
    print("PATTERN LIBRARY ANALYSIS")
    print("="*50)
    print(
        f"Total patterns:   {stats['total_patterns']}"
    )
    print(
        f"Success patterns: {stats['success_patterns']}"
    )
    print(
        f"Failure patterns: {stats['failure_patterns']}"
    )
    print(
        f"Task clusters:    {stats['task_clusters']}"
    )

    if stats['task_clusters'] > 0:
        print("\nCluster breakdown:")
        for cluster_id, task_type in (
            stats['cluster_details'].items()
        ):
            print(
                f"  {cluster_id[:8]}... → {task_type}"
            )

    # raw database stats
    async with db_writer.pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT task_type, outcome, COUNT(*) as count
            FROM execution_patterns
            GROUP BY task_type, outcome
            ORDER BY task_type, outcome
            """
        )

    if rows:
        print("\nDetailed breakdown:")
        for row in rows:
            print(
                f"  {row['task_type']:15} "
                f"{row['outcome']:10} "
                f"{row['count']} patterns"
            )

    await db_writer.close()


if __name__ == "__main__":
    asyncio.run(analyze_library())