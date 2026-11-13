"""Assemble the LangGraph StateGraph with a SQLite checkpointer.

The checkpointer is what makes `interrupt()` in human_review work: the state is
saved, the process can even exit, and a later `Command(resume=...)` with the
same thread_id picks the run back up.
"""
from __future__ import annotations

import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from .config import CFG, path
from .nodes import (
    TriageState,
    assess_risk,
    classify,
    draft,
    extract,
    human_review,
    out_of_scope,
    retrieve,
    route_after_verify,
    route_scope,
    verify_citations,
)


def build_graph(checkpointer=None):
    g = StateGraph(TriageState)

    g.add_node("classify", classify)
    g.add_node("extract", extract)
    g.add_node("assess_risk", assess_risk)
    g.add_node("retrieve", retrieve)
    g.add_node("draft", draft)
    g.add_node("verify_citations", verify_citations)
    g.add_node("out_of_scope", out_of_scope)
    g.add_node("human_review", human_review)

    g.add_edge(START, "classify")
    g.add_edge("classify", "extract")
    g.add_edge("extract", "assess_risk")
    g.add_conditional_edges("assess_risk", route_scope, {"retrieve": "retrieve", "out_of_scope": "out_of_scope"})
    g.add_edge("retrieve", "draft")
    g.add_edge("draft", "verify_citations")
    g.add_conditional_edges("verify_citations", route_after_verify, {"draft": "draft", "human_review": "human_review"})
    g.add_edge("out_of_scope", "human_review")
    g.add_edge("human_review", END)

    return g.compile(checkpointer=checkpointer)


def sqlite_checkpointer() -> SqliteSaver:
    conn = sqlite3.connect(str(path("checkpoint_db")), check_same_thread=False)
    return SqliteSaver(conn)


def build_app():
    """Compiled graph with persistence, ready for invoke/stream."""
    return build_graph(sqlite_checkpointer())


if __name__ == "__main__":
    # python -m triage.graph  -> prints the graph as Mermaid (paste into https://mermaid.live).
    print(build_graph().get_graph().draw_mermaid())
