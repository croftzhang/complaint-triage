#!/usr/bin/env python
"""Step 8 - Run the LangGraph triage pipeline end to end, with a human-in-the-loop pause.

  python scripts/08_demo.py                       # first in-scope complaint from the test set
  python scripts/08_demo.py --index 7             # a specific test row
  python scripts/08_demo.py --text "Someone withdrew $800 from my checking account ..."
  python scripts/08_demo.py --auto-approve        # non-interactive (CI / scripted runs)
  python scripts/08_demo.py --thread my-run-1     # reuse a thread id to inspect checkpoints
  python scripts/08_demo.py --product-group "Checking or savings account"   # skip the classifier

Each node's update is streamed as it happens; at human_review the graph stops
(interrupt) and waits for your decision, then resumes from the SQLite checkpoint.
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from langgraph.types import Command

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from triage.config import CFG, path  # noqa: E402
from triage.data_io import load_split  # noqa: E402
from triage.graph import build_app  # noqa: E402
from triage.taxonomy import REG_E_SCOPE  # noqa: E402


def pick_complaint(args) -> dict:
    if args.text:
        return {"complaint_id": "adhoc", "narrative": args.text, "submitted_via": "Web", "tags": args.tags}
    test = load_split(path("processed_dir"), "test")
    if args.index is None:
        candidates = test.index[test["product_group"].isin(REG_E_SCOPE)]
        args.index = int(candidates[0]) if len(candidates) else 0
    row = test.iloc[args.index]
    return {
        "complaint_id": str(row.complaint_id), "narrative": str(row.narrative),
        "submitted_via": str(row.submitted_via), "tags": None if row.tags is None or str(row.tags) == "nan" else str(row.tags),
        "_true_product_group": str(row.product_group),
    }


def show(update: dict) -> None:
    for node, out in update.items():
        if node == "__interrupt__":
            continue
        print(f"\n--- {node} ---")
        for k, v in out.items():
            if k == "trace":
                continue
            s = json.dumps(v, ensure_ascii=False, default=str)
            print(f"  {k}: {s[:600]}{'...' if len(s) > 600 else ''}")


def ask_reviewer(payload: dict, auto: bool) -> dict:
    print("\n================ HUMAN REVIEW ================")
    print(f"product: {payload['product_group']} | escalate: {payload['escalate']} {payload['risk_reasons']}")
    print(f"confidence: {payload['confidence']} | cited: {payload['cited_sections']} | issues: {payload['citation_issues']}")
    print(f"\nDRAFT:\n{payload['draft']}\n\nNOTES: {payload['notes_for_reviewer']}")
    print("==============================================")
    if auto:
        print("(auto-approve)")
        return {"action": "approve"}
    while True:
        choice = input("approve / edit / reject ? ").strip().lower()
        if choice in ("approve", "a"):
            return {"action": "approve"}
        if choice in ("reject", "r"):
            return {"action": "reject"}
        if choice in ("edit", "e"):
            return {"action": "edit", "text": input("edited reply: ").strip()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int)
    ap.add_argument("--text")
    ap.add_argument("--tags", default=None, help="e.g. 'Older American'")
    ap.add_argument("--product-group", default=None, help="skip the classifier and use this product group")
    ap.add_argument("--auto-approve", action="store_true")
    ap.add_argument("--thread", default=None)
    args = ap.parse_args()

    state = pick_complaint(args)
    true_group = state.pop("_true_product_group", None)
    if args.product_group:
        state["product_group_override"] = args.product_group
    print(f"complaint {state['complaint_id']}" + (f" (true group: {true_group})" if true_group else ""))
    print(state["narrative"][:800], "\n")

    app = build_app()
    config = {"configurable": {"thread_id": args.thread or f"demo-{uuid.uuid4().hex[:8]}"}}
    print(f"thread_id = {config['configurable']['thread_id']}")

    interrupt_payload = None
    for update in app.stream(state, config, stream_mode="updates"):
        show(update)
        if "__interrupt__" in update:
            interrupt_payload = update["__interrupt__"][0].value

    if interrupt_payload is None:
        print("graph finished without human review?")
        return

    decision = ask_reviewer(interrupt_payload, args.auto_approve)
    for update in app.stream(Command(resume=decision), config, stream_mode="updates"):
        show(update)

    final = app.get_state(config).values
    print("\n================ RESULT ================")
    print(f"decision: {final.get('review_decision')}  | relief_prob: {final.get('relief_prob')}")
    print(f"final response:\n{final.get('final_response') or '(rejected - nothing sent)'}")
    print("\ntrace:")
    for line in final.get("trace", []):
        print("  ", line)


if __name__ == "__main__":
    main()
