"""Pydantic schemas for LLM structured outputs (LangChain with_structured_output)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Extraction(BaseModel):
    """Facts pulled from the complaint narrative. Unknown values stay null."""

    summary: str = Field(description="One neutral sentence summarising what the consumer says happened.")
    transaction_type: Literal[
        "unauthorized_transaction",
        "failed_or_delayed_transfer",
        "fee_dispute",
        "stop_payment_or_preauthorized",
        "account_access_or_closure",
        "other",
    ] = Field(description="Best-fitting category of the core problem.")
    amount_usd: float | None = Field(default=None, description="Disputed dollar amount, if stated.")
    date_noticed: str | None = Field(default=None, description="When the consumer noticed the problem, as written (may be vague).")
    date_reported: str | None = Field(default=None, description="When the consumer reported it to the company, as written.")
    reported_within_60_days: Literal["yes", "no", "unknown"] = Field(
        default="unknown", description="Whether the consumer reported within 60 days of the statement, if determinable."
    )
    requested_remedy: str = Field(description="What the consumer is asking for (refund, investigation, fee reversal, ...).")


class DraftResponse(BaseModel):
    """A draft reply for a human reviewer; never sent automatically."""

    response_text: str = Field(description="Draft reply to the consumer, under 200 words, plain language.")
    cited_sections: list[str] = Field(default_factory=list, description="Regulation E sections cited, e.g. ['1005.6', '1005.11'].")
    confidence: Literal["high", "medium", "low"] = Field(description="How well the retrieved regulation text supports the draft.")
    notes_for_reviewer: str = Field(description="Open questions, missing facts, or caveats for the human reviewer.")
