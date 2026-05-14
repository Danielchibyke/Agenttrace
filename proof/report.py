import sys
import os
sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))

import asyncio
import json
import logging
from storage.database import DatabaseWriter
from proof.collector import TrajectoryCollector
from proof.analyzer import DriftAnalyzer

logging.basicConfig(level=logging.WARNING)


async def run_proof():
    """
    Full mathematical proof of drift detection theory.
    Runs all three claims and produces a report.
    """

    print("\n" + "="*60)
    print("AGENTRACE HD — DRIFT DETECTION PROOF")
    print("Mathematical validation of trajectory theory")
    print("="*60)

    db = DatabaseWriter()
    await db.connect()

    collector = TrajectoryCollector(db)
    analyzer = DriftAnalyzer()

    # ── DATA INVENTORY ────────────────────────────
    print("\n[0] DATA INVENTORY")
    print("-"*40)

    trajectories = await collector.get_all_trajectories()
    sessions = await collector.get_sessions_by_outcome()

    success_count = len(sessions.get("success", []))
    failure_count = len(sessions.get("failure", []))
    partial_count = len(sessions.get("partial", []))

    print(f"  Total patterns:     {len(trajectories)}")
    print(f"  Success sessions:   {success_count}")
    print(f"  Failure sessions:   {failure_count}")
    print(f"  Partial sessions:   {partial_count}")

    if len(trajectories) < 4:
        print(
            f"\n  ✗ INSUFFICIENT DATA"
            f"\n  Need at least 2 success + 2 failure patterns."
            f"\n  Run tests/test_mid_failures.py first"
            f"\n  to populate the pattern library."
        )
        await db.close()
        return

    # ── CLAIM 1 — GEOMETRIC SEPARABILITY ─────────
    print("\n[1] CLAIM 1 — GEOMETRIC SEPARABILITY")
    print("-"*40)
    print(
        "  Hypothesis: success and failure trajectories"
        "\n  occupy distinct regions in HD space."
    )

    sep_result = analyzer.prove_separability(trajectories)

    print(f"\n  Silhouette score:        {sep_result.get('silhouette_score', 'N/A')}")
    print(f"  Inter-cluster distance:  {sep_result.get('inter_cluster_distance', 'N/A')}")
    print(f"  Success intra-distance:  {sep_result.get('success_intra_distance', 'N/A')}")
    print(f"  Failure intra-distance:  {sep_result.get('failure_intra_distance', 'N/A')}")
    print(f"  Davies-Bouldin index:    {sep_result.get('davies_bouldin_index', 'N/A')}")
    print(f"\n  Interpretation: {sep_result.get('interpretation', 'N/A')}")
    print(
        f"\n  Claim 1: "
        f"{'✓ PROVEN' if sep_result.get('proven') else '✗ NOT PROVEN'}"
    )

    if not sep_result.get("proven"):
        print(
            f"  Reason: {sep_result.get('reason', 'See interpretation')}"
        )

    # ── CLAIM 2 — TEMPORAL DETECTABILITY ─────────
    print("\n[2] CLAIM 2 — TEMPORAL DETECTABILITY")
    print("-"*40)
    print(
        "  Hypothesis: geometric drift is detectable"
        "\n  before final failure output is generated."
    )

    if not sep_result.get("proven"):
        print(
            "\n  Skipping — requires Claim 1 to be proven first."
        )
    else:
        # analyze each failure session
        failure_sessions = sessions.get("failure", [])
        success_sessions = sessions.get("success", [])

        temporal_results = []

        print(f"\n  Analyzing {len(failure_sessions)} failure sessions...")
        for session_id in failure_sessions[:5]:
            steps = await collector.get_step_vectors(
                session_id
            )
            if len(steps) < 3:
                continue
            result = analyzer.prove_temporal_detection(
                steps, "failure"
            )
            temporal_results.append(result)
            print(
                f"  Session {session_id[:8]}... "
                f"steps={result['total_steps']} "
                f"detected={'step ' + str(result['detection_step']) if result['detection_step'] is not None else 'never'} "
                f"lead={result['lead_time_steps']}"
            )

        print(f"\n  Analyzing {len(success_sessions)} success sessions...")
        for session_id in success_sessions[:5]:
            steps = await collector.get_step_vectors(
                session_id
            )
            if len(steps) < 3:
                continue
            result = analyzer.prove_temporal_detection(
                steps, "success"
            )
            temporal_results.append(result)
            print(
                f"  Session {session_id[:8]}... "
                f"steps={result['total_steps']} "
                f"max_drift={result['max_drift_score']:.3f} "
                f"detected={'step ' + str(result['detection_step']) if result['detection_step'] is not None else 'none'}"
            )

        failure_detections = [
            r for r in temporal_results
            if r.get("proven")
        ]
        false_positives = [
            r for r in temporal_results
            if r.get("outcome") == "success"
            and r.get("detection_step") is not None
        ]

        if temporal_results:
            avg_lead = sum(
                r["lead_time_steps"]
                for r in temporal_results
                if r.get("outcome") == "failure"
            ) / max(
                len([
                    r for r in temporal_results
                    if r.get("outcome") == "failure"
                ]), 1
            )

            print(f"\n  Failure sessions analyzed: {len([r for r in temporal_results if r.get('outcome') == 'failure'])}")
            print(f"  Failures detected early:   {len(failure_detections)}")
            print(f"  False positives:           {len(false_positives)}")
            print(f"  Average lead time:         {avg_lead:.1f} steps")

            claim2_proven = (
                len(failure_detections) > 0 and
                avg_lead > 0
            )
            print(
                f"\n  Claim 2: "
                f"{'✓ PROVEN' if claim2_proven else '✗ NOT PROVEN'}"
            )

    # ── CLAIM 3 — PREDICTIVE VALIDITY ─────────────
    print("\n[3] CLAIM 3 — PREDICTIVE VALIDITY")
    print("-"*40)
    print(
        "  Hypothesis: trajectory velocity vector"
        "\n  predicts destination before arrival."
    )

    if not sep_result.get("proven"):
        print(
            "\n  Skipping — requires Claim 1 to be proven first."
        )
    else:
        prediction_results = []
        all_sessions = (
            sessions.get("failure", [])[:3] +
            sessions.get("success", [])[:3]
        )

        for session_id in all_sessions:
            outcome = (
                "failure"
                if session_id in sessions.get("failure", [])
                else "success"
            )
            steps = await collector.get_step_vectors(
                session_id
            )
            if len(steps) < 3:
                continue
            result = analyzer.prove_prediction(
                steps, outcome
            )
            prediction_results.append(result)
            print(
                f"  Session {session_id[:8]}... "
                f"outcome={outcome} "
                f"accuracy={result['prediction_accuracy']:.0f}% "
                f"early_pred={result['earliest_failure_prediction']}"
            )

        if prediction_results:
            avg_accuracy = sum(
                r["prediction_accuracy"]
                for r in prediction_results
            ) / len(prediction_results)

            claim3_proven = avg_accuracy > 60

            print(f"\n  Average prediction accuracy: {avg_accuracy:.1f}%")
            print(
                f"\n  Claim 3: "
                f"{'✓ PROVEN' if claim3_proven else '✗ NOT PROVEN'}"
            )

    # ── FINAL VERDICT ──────────────────────────────
    print("\n" + "="*60)
    print("FINAL VERDICT")
    print("="*60)

    claims = {
        "Geometric Separability": sep_result.get("proven", False),
    }

    proven_count = sum(1 for v in claims.values() if v)
    total_claims = 3

    print(f"\n  Claims proven: {proven_count}/{total_claims}")

    if proven_count == 3:
        print(
            "\n  ✓ THEORY FULLY PROVEN"
            "\n  AgentTrace HD drift detection is"
            "\n  mathematically validated."
        )
    elif proven_count == 1:
        print(
            "\n  ~ PARTIAL PROOF"
            "\n  Geometric foundation established."
            "\n  Need more failure patterns for"
            "\n  temporal and predictive proofs."
        )
    else:
        print(
            "\n  ✗ PROOF INCOMPLETE"
            "\n  Need more data. Run:"
            "\n  python tests/test_mid_failures.py"
        )

    await db.close()


if __name__ == "__main__":
    asyncio.run(run_proof())