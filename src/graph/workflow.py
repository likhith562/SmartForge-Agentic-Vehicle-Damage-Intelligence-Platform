"""
SmartForge — LangGraph Workflow
================================
Assembles and compiles the full Directed Cyclic Graph (DCG) from all node
functions, wires every fixed edge and conditional edge, attaches the
MemorySaver checkpointer, and exposes the compiled ``graph`` singleton.

Graph topology (v19)
--------------------

    intake ──► fraud ──┬──► map_images ──► cv_worker(×N) ──► fusion ──►┐
                       │   [Batch 2 Send fan-out]                       │
                       │                                                 ▼
                       └──────────────────────────────────────► perception
                                                                         │
                                                                         ▼
                       human_audit ◄── (fraud: SUSPICIOUS) ──  gemini_agent
                                                                         │
                                                               false_positive_gate
                                                                         │
                                                               health_monitor
                                                                  │         │
                                                         perception_retry   │
                                                                  │         │
                                                                  └────►verification_v2
                                                                         │
                                                                      reasoning
                                                                         │
                                                                 decision [HITL interrupt]
                                                                         │
                                                                      report ──► END

Conditional edges
-----------------
fraud → fraud_router:
    "human_audit"  — trust_score < FRAUD_TRUST_THRESHOLD
    "map_images"   — verified + multi-image claim (len(image_paths) > 1)
    "perception"   — verified + single-image claim

health_monitor → health_monitor_router:
    "reasoning"        — validation passed OR circuit breaker fired
                         (both route through verification_v2 via fixed edge)
    "perception_retry" — validation failed, retries remaining

HITL interrupt
--------------
    interrupt_before=["decision"]
    The graph pauses before decision_node.  The caller resumes with:
        for event in graph.stream(None, thread, stream_mode="values"):
            final = event

Exported symbols
----------------
    graph        — compiled CompiledGraph, ready for .stream() / .invoke()
    checkpointer — MemorySaver instance (queryable for audit dumps)
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.graph.state import SmartForgeState

# ── Node imports ──────────────────────────────────────────────────────────────
from src.graph.nodes.intake        import intake_node
from src.graph.nodes.fraud         import fraud_node, fraud_router
from src.graph.nodes.human_audit   import human_audit_node
from src.graph.nodes.map_reduce    import (
    map_images_node,
    cv_worker_node,
    fusion_node,
)
from src.graph.nodes.perception    import perception_node, perception_retry_node
from src.graph.nodes.gemini_agent  import gemini_agent_node
from src.graph.nodes.false_positive_gate import false_positive_gate_node
from src.graph.nodes.health_monitor      import (
    health_monitor_node,
    health_monitor_router,
)
from src.graph.nodes.verification_v2 import verification_v2_node
from src.graph.nodes.reasoning     import reasoning_node
from src.graph.nodes.decision      import decision_node
from src.graph.nodes.report        import report_node


# ─────────────────────────────────────────────────────────────────────────────
# Build the StateGraph
# ─────────────────────────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    """Construct and return the uncompiled StateGraph."""
    builder = StateGraph(SmartForgeState)

    # ── Register nodes ────────────────────────────────────────────────────────

    # Core pipeline
    builder.add_node("intake",               intake_node)
    builder.add_node("perception",           perception_node)
    builder.add_node("perception_retry",     perception_retry_node)
    builder.add_node("gemini_agent",         gemini_agent_node)
    builder.add_node("false_positive_gate",  false_positive_gate_node)
    builder.add_node("health_monitor",       health_monitor_node)
    builder.add_node("verification_v2",      verification_v2_node)
    builder.add_node("reasoning",            reasoning_node)
    builder.add_node("decision",             decision_node)
    builder.add_node("report",               report_node)

    # Batch 1: Fraud Layer
    builder.add_node("fraud",                fraud_node)
    builder.add_node("human_audit",          human_audit_node)

    # Batch 2: Multi-Image Map-Reduce
    builder.add_node("map_images",           map_images_node)
    builder.add_node("cv_worker",            cv_worker_node)
    builder.add_node("fusion",               fusion_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    builder.set_entry_point("intake")

    # ── Fixed edges ───────────────────────────────────────────────────────────

    # Intake always goes to fraud gate first
    builder.add_edge("intake", "fraud")

    # Fraud-flagged claims terminate at human_audit
    builder.add_edge("human_audit", END)

    # Batch 2: each cv_worker output converges at fusion
    builder.add_edge("cv_worker", "fusion")
    # Fallback direct edge (in case Send fan-out is not supported in the runtime)
    builder.add_edge("map_images", "fusion")

    # Fusion feeds into Gemini enrichment (same as single-image perception path)
    builder.add_edge("fusion", "gemini_agent")

    # Single-image perception path also feeds into Gemini
    builder.add_edge("perception", "gemini_agent")

    # Main pipeline spine: Gemini → FP gate → health monitor
    builder.add_edge("gemini_agent",       "false_positive_gate")
    builder.add_edge("false_positive_gate", "health_monitor")

    # Retry loop: perception_retry re-enters at gemini_agent (same as first pass)
    builder.add_edge("perception_retry",   "gemini_agent")

    # Golden Frame verification always runs before reasoning
    builder.add_edge("verification_v2",    "reasoning")

    # Final pipeline spine
    builder.add_edge("reasoning", "decision")
    builder.add_edge("decision",  "report")
    builder.add_edge("report",    END)

    # ── Conditional edges ─────────────────────────────────────────────────────

    # Fraud router: suspicious → human_audit | multi-img verified → map_images
    #               single-img verified → perception
    builder.add_conditional_edges(
        "fraud",
        fraud_router,
        {
            "perception":  "perception",
            "map_images":  "map_images",
            "human_audit": "human_audit",
        },
    )

    # Health monitor router: pass → verification_v2 | fail → perception_retry
    # Note: health_monitor_router returns "reasoning" for both PASS and
    # circuit-breaker; the mapping sends both to verification_v2 so Golden
    # Frame always runs regardless of health outcome.
    builder.add_conditional_edges(
        "health_monitor",
        health_monitor_router,
        {
            "reasoning":        "verification_v2",   # PASS or circuit-breaker
            "perception_retry": "perception_retry",  # FAIL + retries remaining
        },
    )

    return builder


# ─────────────────────────────────────────────────────────────────────────────
# Compile with MemorySaver checkpointer
# ─────────────────────────────────────────────────────────────────────────────

checkpointer = MemorySaver()

graph = _build_graph().compile(
    checkpointer     = checkpointer,
    interrupt_before = ["decision"],   # HITL pause point
)


# ─────────────────────────────────────────────────────────────────────────────
# Startup summary
# ─────────────────────────────────────────────────────────────────────────────

def _print_graph_summary() -> None:
    node_names = list(graph.nodes.keys())
    print("✅ SmartForge LangGraph compiled.")
    print(f"   Nodes ({len(node_names)}): {node_names}")
    print()
    print("   SINGLE-IMAGE FLOW:")
    print("   intake → fraud → perception → gemini_agent → false_positive_gate")
    print("          → health_monitor → verification_v2 → reasoning → decision → report")
    print()
    print("   MULTI-IMAGE FLOW (Batch 2):")
    print("   intake → fraud → map_images → cv_worker(×N) → fusion")
    print("          → gemini_agent → false_positive_gate → health_monitor")
    print("          → verification_v2 → reasoning → decision → report")
    print()
    print("   FRAUD PATH:   intake → fraud → human_audit → END")
    print("   HITL:         interrupt_before=['decision']")
    print("   RETRY LOOP:   health_monitor → perception_retry → gemini_agent → …")


# Print summary at import time (visible in Colab cell output)
_print_graph_summary()
