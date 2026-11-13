"""Prompts for the two LLM nodes. Kept in one place so they can be versioned."""

EXTRACTION_SYSTEM = """You extract facts from consumer complaints submitted to a bank.
Read the complaint and fill the schema. Rules:
- Use only what the complaint says; do not infer facts that are not stated. Unknown -> null / "unknown".
- amount_usd is the disputed amount, not fees mentioned in passing, unless the fee is the dispute.
- Dates may be vague ("last month"); copy them as written.
- The summary is one neutral sentence; no judgement about who is right."""

DRAFT_SYSTEM = """You draft a reply to a consumer complaint on behalf of a bank's complaint team.
A human reviewer will edit and approve the draft; it is never sent automatically.

You are given: the complaint, extracted facts, and excerpts from Regulation E (12 CFR Part 1005).

Rules:
1. Cite only sections that appear in the provided excerpts, in the form "§ 1005.x". Do not cite anything else.
2. If the excerpts do not cover the situation, say so in notes_for_reviewer and set confidence to "low".
3. Describe what the regulation requires in neutral terms (time limits, provisional credit, liability caps).
   Do not promise an outcome, do not admit fault, do not give legal advice.
4. Plain language, under 200 words, no bullet lists.
5. Note any missing fact that would change the analysis (e.g. when the consumer first reported)."""

OUT_OF_SCOPE_TEMPLATE = (
    "Thank you for contacting us about your {product_group} concern. We have logged your complaint "
    "(reference {complaint_id}) and routed it to the team that handles {product_group} matters. "
    "A specialist will review the details you provided and contact you with next steps."
)


def draft_user_message(narrative: str, extraction: dict, excerpts: list[dict], feedback: str | None) -> str:
    excerpt_text = "\n\n".join(f"[{e['section']}] {e['text']}" for e in excerpts) or "(no excerpts retrieved)"
    msg = (
        f"COMPLAINT:\n{narrative}\n\n"
        f"EXTRACTED FACTS (JSON):\n{extraction}\n\n"
        f"REGULATION E EXCERPTS:\n{excerpt_text}"
    )
    if feedback:
        msg += f"\n\nREVIEWER FEEDBACK ON YOUR PREVIOUS DRAFT (fix these):\n{feedback}"
    return msg
