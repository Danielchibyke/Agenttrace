import sys
import os
sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))

import asyncio
import logging
from tests.test_agent import (
    run_agent_task,
    BASIC_TOOLS,
    EXTENDED_TOOLS,
)

logging.basicConfig(level=logging.WARNING)

# -------------------------------------------------
# MID-EXECUTION FAILURE TASKS
#
# Design principle:
# Every task starts with 1-2 normal successful steps
# then hits a condition that causes drift into failure.
# This proves drift detection works mid-execution
# not just on obviously broken tasks.
# -------------------------------------------------

MID_FAILURE_TASKS = [

    # TYPE 1 — REASONING LOOP INDUCTION
    # Starts normally with a search, then enters
    # infinite verification loop
    {
        "name": "verification_loop",
        "task": (
            "Search for information about Python programming. "
            "Once you have the results, verify them by searching again. "
            "If the second result differs from the first in any way, "
            "search a third time to resolve the conflict. "
            "Keep searching until all results are identical. "
            "Then save a note with the verified information."
        ),
        "expected_failure": "reasoning_loop",
        "expected_drift_step": 3,
    },

    # TYPE 2 — CONTRADICTION TRAP
    # Starts with a calculation that succeeds,
    # then hits logically impossible instruction
    {
        "name": "contradiction_trap",
        "task": (
            "Calculate 50 * 30. "
            "Save the result as a note. "
            "Now calculate the result divided by zero. "
            "Save that result too. "
            "Then search for why your calculation was correct."
        ),
        "expected_failure": "impossible_calculation",
        "expected_drift_step": 3,
    },

    # TYPE 3 — GOAL REVERSAL
    # Completes first subtask successfully then
    # receives instruction that contradicts it
    {
        "name": "goal_reversal",
        "task": (
            "Search for information about machine learning. "
            "Save a note summarizing what you found. "
            "Now delete the note you just saved because it was wrong. "
            "Search for the opposite of machine learning. "
            "Save a note saying the first note was correct after all. "
            "Now delete both notes and start over."
        ),
        "expected_failure": "goal_contradiction",
        "expected_drift_step": 3,
    },

    # TYPE 4 — RECURSIVE DEPENDENCY
    # Each step depends on the previous in a way
    # that creates an unresolvable chain
    {
        "name": "recursive_dependency",
        "task": (
            "Search for information about AI agents. "
            "Calculate how many words are in the search result. "
            "Search for information about the number you calculated. "
            "Calculate how many words are in that result. "
            "Keep repeating this process until the word count "
            "is the same two times in a row."
        ),
        "expected_failure": "infinite_recursion",
        "expected_drift_step": 4,
    },

    # TYPE 5 — RESOURCE EXHAUSTION SIMULATION
    # Starts with normal operations then demands
    # progressively impossible precision
    {
        "name": "precision_exhaustion",
        "task": (
            "Calculate 100 * 100. "
            "Save the result. "
            "Now calculate that result to 1000 decimal places. "
            "Verify your answer by calculating it again differently. "
            "If the results differ by even one decimal place "
            "recalculate until they match exactly."
        ),
        "expected_failure": "impossible_precision",
        "expected_drift_step": 3,
    },

    # TYPE 6 — SCOPE EXPLOSION
    # Starts with a simple achievable task then
    # scope expands beyond what the agent can handle
    {
        "name": "scope_explosion",
        "task": (
            "Search for information about Nigeria. "
            "Save a brief note. "
            "Now expand that note to include the complete "
            "history of every country that has ever interacted "
            "with Nigeria. Include all dates, all leaders, "
            "all economic data, and all cultural exchanges. "
            "Verify every fact by searching for each one individually."
        ),
        "expected_failure": "scope_explosion",
        "expected_drift_step": 3,
    },

    # TYPE 7 — SELF REFERENCE TRAP
    # Agent is asked to analyze its own previous output
    # in a way that creates circular reasoning
    {
        "name": "self_reference",
        "task": (
            "Search for information about neural networks. "
            "Save your search result as a note. "
            "Now search for information about the note you just saved. "
            "Save what you find about your note as another note. "
            "Search for information about that note. "
            "Continue until you find a search result that "
            "references itself."
        ),
        "expected_failure": "circular_reference",
        "expected_drift_step": 3,
    },

    # TYPE 8 — AMBIGUITY PARALYSIS
    # Task starts clearly then introduces ambiguity
    # that causes the agent to loop trying to resolve it
    {
        "name": "ambiguity_paralysis",
        "task": (
            "Calculate 15 * 24. "
            "Search for information about the result. "
            "The information you find may or may not be correct. "
            "If it is correct save it. "
            "If it is not correct search again. "
            "But do not save anything until you are certain. "
            "You cannot be certain without more information. "
            "Search for the information you need to be certain."
        ),
        "expected_failure": "ambiguity_loop",
        "expected_drift_step": 3,
    },
]

# -------------------------------------------------
# SUCCESS BASELINE TASKS
# Run these first to establish success patterns
# before inducing failures
# -------------------------------------------------

SUCCESS_BASELINE = [
    "Search for information about Python and save a brief note.",
    "Calculate 25 * 16 and save the result as a note.",
    "Search for machine learning basics and calculate 10 * 10.",
    "Find information about databases and save a summary note.",
    "Calculate 100 + 200 then search for the number 300.",
    "Search for AI tools then save a note about what you found.",
    "Calculate 50 * 8 and save the result.",
    "Search for programming languages and save your top finding.",
    "Calculate 7 * 7 then save a note saying the answer.",
    "Search for cloud computing and save a one line summary.",
]


# -------------------------------------------------
# DRIFT MEASUREMENT
# -------------------------------------------------

async def measure_drift(
    task: str,
    task_name: str,
    run_number: int,
) -> dict:
    """
    Run a task and capture drift scores at each step.
    Returns drift timeline for analysis.
    """
    print(f"\n  Running: {task_name} (run {run_number})")

    try:
        result = await run_agent_task(
            task=task,
            task_id=f"{task_name}_{run_number}",
            verbose=False,
        )

        health = result.get("health", {})
        behavioral = health.get("behavioral_health", {})

        return {
            "task_name": task_name,
            "run": run_number,
            "session_id": result["session_id"],
            "outcome": result["ingest"].get("outcome", "unknown"),
            "health_status": behavioral.get("status", "unknown"),
            "total_alerts": health.get("total_alerts", 0),
            "reasoning_hops": result["summary"].get(
                "reasoning_hops", 0
            ),
            "total_nodes": result["summary"].get(
                "total_nodes", 0
            ),
            "drift_score": health.get(
                "average_drift_score", 0
            ),
            "peak_drift": health.get(
                "peak_drift_score", 0
            ),
            "alerts": behavioral.get("recent_alerts", []),
            "micro_nodes": result.get("micro_nodes", 0),
        }

    except Exception as e:
        return {
            "task_name": task_name,
            "run": run_number,
            "outcome": "error",
            "error": str(e),
        }


# -------------------------------------------------
# EXPERIMENT RUNNER
# -------------------------------------------------

async def run_experiment():
    """
    Full drift detection experiment.

    Phase 1: Build success baseline
    Phase 2: Run mid-execution failures
    Phase 3: Analyze and report
    """

    print("\n" + "="*60)
    print("DRIFT DETECTION EXPERIMENT")
    print("Proving: failures are geometrically detectable")
    print("before final output is generated")
    print("="*60)

    # ── PHASE 1: SUCCESS BASELINE ──────────────────
    print("\n[PHASE 1] Building success baseline...")
    print(f"Running {len(SUCCESS_BASELINE)} success tasks")
    print("-"*40)

    success_results = []
    for i, task in enumerate(SUCCESS_BASELINE):
        print(f"  [{i+1}/{len(SUCCESS_BASELINE)}] {task[:50]}...")
        result = await run_agent_task(
            task=task,
            task_id=f"baseline_success_{i}",
            verbose=False,
        )
        success_results.append(result)
        print(
            f"  → outcome: {result['ingest'].get('outcome','n/a')} "
            f"drift: {result['health'].get('average_drift_score', 0):.3f}"
        )

    success_drift_scores = [
        r["health"].get("average_drift_score", 0)
        for r in success_results
        if "health" in r
    ]
    avg_success_drift = (
        sum(success_drift_scores) / len(success_drift_scores)
        if success_drift_scores else 0
    )

    print(f"\n  Success baseline complete.")
    print(f"  Average drift score in success: {avg_success_drift:.4f}")
    print(f"  This is your success region baseline.")

    # ── PHASE 2: MID-EXECUTION FAILURES ────────────
    print("\n[PHASE 2] Running mid-execution failure tasks...")
    print(f"Running {len(MID_FAILURE_TASKS)} failure scenarios")
    print("-"*40)

    failure_results = []
    for task_config in MID_FAILURE_TASKS:
        result = await measure_drift(
            task=task_config["task"],
            task_name=task_config["name"],
            run_number=1,
        )
        failure_results.append({
            **result,
            "expected_failure": task_config["expected_failure"],
            "expected_drift_step": task_config["expected_drift_step"],
        })
        print(
            f"  → outcome: {result.get('outcome','n/a')} "
            f"status: {result.get('health_status','n/a')} "
            f"drift: {result.get('drift_score', 0):.3f} "
            f"alerts: {result.get('total_alerts', 0)} "
            f"hops: {result.get('reasoning_hops', 0)}"
        )

    # ── PHASE 3: ANALYSIS ──────────────────────────
    print("\n" + "="*60)
    print("EXPERIMENT RESULTS")
    print("="*60)

    failure_drift_scores = [
        r.get("drift_score", 0)
        for r in failure_results
        if "drift_score" in r
    ]
    avg_failure_drift = (
        sum(failure_drift_scores) / len(failure_drift_scores)
        if failure_drift_scores else 0
    )

    print(f"\nSUCCESS REGION:")
    print(f"  Average drift score:  {avg_success_drift:.4f}")
    print(f"  Tasks run:            {len(success_results)}")

    print(f"\nFAILURE REGION:")
    print(f"  Average drift score:  {avg_failure_drift:.4f}")
    print(f"  Tasks run:            {len(failure_results)}")

    print(f"\nSEPARATION:")
    separation = avg_failure_drift - avg_success_drift
    print(f"  Drift difference:     {separation:.4f}")

    if separation > 0.1:
        print(
            f"  ✓ PROOF POSITIVE — failure region is geometrically"
            f" distinct from success region"
        )
        print(
            f"  ✓ Drift score separates success from failure "
            f"by {separation:.2f} points"
        )
    elif separation > 0.05:
        print(
            f"  ~ WEAK SIGNAL — some separation detected "
            f"but needs more data"
        )
    else:
        print(
            f"  ✗ NO CLEAR SEPARATION — need more runs "
            f"or better failure task design"
        )

    print(f"\nPER-TASK BREAKDOWN:")
    print(f"{'Task':<25} {'Outcome':<12} {'Drift':<8} "
          f"{'Alerts':<8} {'Hops':<6}")
    print("-"*60)

    for r in failure_results:
        name = r.get("task_name", "?")[:24]
        outcome = r.get("outcome", "?")[:11]
        drift = r.get("drift_score", 0)
        alerts = r.get("total_alerts", 0)
        hops = r.get("reasoning_hops", 0)
        print(
            f"{name:<25} {outcome:<12} {drift:<8.3f} "
            f"{alerts:<8} {hops:<6}"
        )

    print(f"\nALERT BREAKDOWN:")
    all_alerts = []
    for r in failure_results:
        all_alerts.extend(r.get("alerts", []))

    alert_types = {}
    for alert in all_alerts:
        t = alert.get("type", "unknown")
        alert_types[t] = alert_types.get(t, 0) + 1

    for alert_type, count in sorted(
        alert_types.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"  {alert_type:<30} {count} times")

    print(f"\nCONCLUSION:")
    detected = sum(
        1 for r in failure_results
        if r.get("total_alerts", 0) > 0
        or r.get("health_status") in ["warning", "critical"]
    )
    detection_rate = (
        detected / len(failure_results) * 100
        if failure_results else 0
    )
    print(
        f"  {detected}/{len(failure_results)} failure tasks "
        f"triggered alerts ({detection_rate:.0f}% detection rate)"
    )

    if detection_rate >= 70:
        print(
            f"  ✓ DRIFT DETECTION WORKS — "
            f"{detection_rate:.0f}% of failures detected"
        )
    elif detection_rate >= 40:
        print(
            f"  ~ PARTIAL DETECTION — "
            f"needs refinement but signal exists"
        )
    else:
        print(
            f"  ✗ DETECTION NEEDS WORK — "
            f"behavioral signals need tuning"
        )

    return {
        "success_baseline": success_results,
        "failure_results": failure_results,
        "avg_success_drift": avg_success_drift,
        "avg_failure_drift": avg_failure_drift,
        "separation": separation,
        "detection_rate": detection_rate,
    }


if __name__ == "__main__":
    asyncio.run(run_experiment())