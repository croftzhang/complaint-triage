"""LangGraph state and node functions for complaint triage.

Flow (see graph.py):
  classify -> extract -> assess_risk -> [in Reg E scope?]
      yes: retrieve -> draft -> verify_citations -> (retry draft) -> human_review
      no : out_of_scope -> human_review
"""
from __future__ import annotations

import operator
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt

from .config import CFG
from .features import FeatureSpec
from .llm import get_llm
from .models import classifier, relief
from .prompts import DRAFT_SYSTEM, EXTRACTION_SYSTEM, OUT_OF_SCOPE_TEMPLATE, draft_user_message
from .reg_e import N_RE, sections_in_text
from .schemas import DraftResponse, Extraction
from .taxonomy import REG_E_SCOPE

# How LangChain asks Claude for structured output. "json_schema" uses Anthropic's
# native structured outputs and is the method recommended for Claude 5.x models;
# "function_calling" (forced tool use) is the legacy fallback.
STRUCTURED_METHOD = CFG["llm"].get("structured_output_method", "json_schema")


def normalise_sections(values) -> list[str]:
    """['§ 1005.6', '1005.11'] -> ['1005.11', '1005.6'] (section numbers only, no § sign)."""
    return sorted({m for v in (values or []) for m in N_RE.findall(str(v))})


class TriageState(TypedDict, total=False):
    # input
    complaint_id: str
    narrative: str
    submitted_via: str
    tags: str | None
    product_group_override: str | None   # set when the product is already known (skips the model)
    # classify
    product_group: str
    product_confidence: float
    reg_e_in_scope: bool
    # extract
    extraction: dict[str, Any]
    # assess_risk
    relief_prob: float | None
    escalate: bool
    risk_reasons: list[str]
    # retrieve
    retrieved: list[dict[str, Any]]
    # draft / verify
    draft: str
    cited_sections: list[str]
    draft_confidence: str
    notes_for_reviewer: str
    citation_issues: list[str]
    draft_attempts: int
    # human_review
    review_decision: str
    final_response: str
    # append-only log of what each node did
    trace: Annotated[list[str], operator.add]


# ---------------------------------------------------------------- nodes

def classify(state: TriageState) -> dict:
    if state.get("product_group_override"):
        label, score, how = state["product_group_override"], 1.0, "override"
    else:
        label, score = classifier().predict(state["narrative"])
        how = "model"
    in_scope = label in REG_E_SCOPE
    return {
        "product_group": label,
        "product_confidence": score,
        "reg_e_in_scope": in_scope,
        "trace": [f"classify ({how}): {label} (p={score:.2f}) -> Reg E scope={in_scope}"],
    }


def extract(state: TriageState) -> dict:
    llm = get_llm("fast").with_structured_output(Extraction, method=STRUCTURED_METHOD)
    result: Extraction = llm.invoke([SystemMessage(content=EXTRACTION_SYSTEM), HumanMessage(content=state["narrative"])])
    return {"extraction": result.model_dump(), "trace": [f"extract: {result.transaction_type}, amount={result.amount_usd}"]}


def assess_risk(state: TriageState) -> dict:
    risk = CFG["risk"]
    reasons: list[str] = []
    tags = state.get("tags") or ""
    for t in risk["vulnerable_tags"]:
        if t in tags:
            reasons.append(f"vulnerable consumer tag: {t}")
    amount = state["extraction"].get("amount_usd")
    if amount is not None and amount >= risk["amount_threshold"]:
        reasons.append(f"amount ${amount:,.0f} >= ${risk['amount_threshold']:,}")

    prob = None
    spec_path = Path(CFG["paths"]["processed_dir"]) / "feature_spec.json"
    if relief().backend != "disabled" and spec_path.exists():
        from .retrieval import embed_query

        spec = FeatureSpec.load(spec_path)
        feats = spec.transform_one(
            embed_query(state["narrative"]), state["product_group"], state.get("submitted_via") or "Web", tags, state["narrative"]
        )
        prob = relief().predict(feats)
        if prob is not None and prob >= CFG["relief"]["threshold"]:
            reasons.append(f"TF relief model p={prob:.2f} >= {CFG['relief']['threshold']}")

    return {
        "relief_prob": prob,
        "escalate": bool(reasons),
        "risk_reasons": reasons,
        "trace": [f"assess_risk: escalate={bool(reasons)} reasons={reasons} relief_prob={prob}"],
    }


def retrieve(state: TriageState) -> dict:
    from .retrieval import retrieve as _retrieve

    ex = state["extraction"]
    query = f"{ex.get('transaction_type', '')}: {ex.get('summary', '')} {state['narrative'][:300]}"
    docs = _retrieve(query, CFG["retrieval"]["k"])
    retrieved = [{"section": d.metadata["section"], "heading": d.metadata["heading"], "text": d.page_content} for d in docs]
    return {"retrieved": retrieved, "trace": [f"retrieve: sections={[r['section'] for r in retrieved]}"]}


def draft(state: TriageState) -> dict:
    attempts = state.get("draft_attempts", 0) + 1
    feedback = None
    if state.get("citation_issues"):
        feedback = "; ".join(state["citation_issues"])
    llm = get_llm("strong").with_structured_output(DraftResponse, method=STRUCTURED_METHOD)
    result: DraftResponse = llm.invoke([
        SystemMessage(content=DRAFT_SYSTEM),
        HumanMessage(content=draft_user_message(state["narrative"], state["extraction"], state["retrieved"], feedback)),
    ])
    cited = normalise_sections(result.cited_sections)  # the model sometimes writes "§ 1005.6"
    return {
        "draft": result.response_text,
        "cited_sections": cited,
        "draft_confidence": result.confidence,
        "notes_for_reviewer": result.notes_for_reviewer,
        "draft_attempts": attempts,
        "trace": [f"draft (attempt {attempts}): cites={cited} confidence={result.confidence}"],
    }


def verify_citations(state: TriageState) -> dict:
    """Every '§ 1005.x' in the draft must come from a retrieved chunk."""
    allowed = {r["section"] for r in state["retrieved"]}
    used = sections_in_text(state["draft"]) | set(normalise_sections(state.get("cited_sections")))
    issues = [f"cited § {s} is not in the retrieved excerpts" for s in sorted(used - allowed)]
    if not used and state.get("draft_confidence") != "low":
        issues.append("no Regulation E section cited although confidence is not 'low'")
    return {"citation_issues": issues, "trace": [f"verify_citations: {'ok' if not issues else issues}"]}


def out_of_scope(state: TriageState) -> dict:
    text = OUT_OF_SCOPE_TEMPLATE.format(product_group=state["product_group"], complaint_id=state.get("complaint_id", "n/a"))
    return {
        "retrieved": [], "draft": text, "cited_sections": [], "draft_confidence": "high",
        "notes_for_reviewer": "Outside Regulation E scope; routed by product group without RAG.",
        "citation_issues": [], "trace": ["out_of_scope: template reply, no RAG"],
    }


def human_review(state: TriageState) -> dict:
    """Pause the graph. The caller resumes with Command(resume={"action": ..., "text": ...})."""
    decision = interrupt({
        "complaint_id": state.get("complaint_id"),
        "product_group": state["product_group"],
        "escalate": state["escalate"],
        "risk_reasons": state["risk_reasons"],
        "draft": state["draft"],
        "cited_sections": state.get("cited_sections", []),
        "confidence": state.get("draft_confidence"),
        "notes_for_reviewer": state.get("notes_for_reviewer"),
        "citation_issues": state.get("citation_issues", []),
        "instructions": "resume with {'action': 'approve'} | {'action': 'edit', 'text': '...'} | {'action': 'reject'}",
    })
    action = decision.get("action", "reject")
    if action == "approve":
        final = state["draft"]
    elif action == "edit":
        final = decision.get("text", state["draft"])
    else:
        final = ""
    return {"review_decision": action, "final_response": final, "trace": [f"human_review: {action}"]}


# ---------------------------------------------------------------- routing

def route_scope(state: TriageState) -> str:
    return "retrieve" if state["reg_e_in_scope"] else "out_of_scope"


def route_after_verify(state: TriageState) -> str:
    if state["citation_issues"] and state["draft_attempts"] < CFG["graph"]["max_draft_attempts"]:
        return "draft"
    return "human_review"
