"""Claude via LangChain, plus a fake LLM for running the pipeline without an API key.

    export ANTHROPIC_API_KEY=...        # real models from config.yaml
    export TRIAGE_FAKE_LLM=1            # deterministic fake structured outputs (smoke test)
"""
from __future__ import annotations

import os
import re
from typing import Any

from dotenv import load_dotenv

from .config import CFG

load_dotenv()


class _FakeStructured:
    """Mimics `ChatAnthropic(...).with_structured_output(schema)` for offline runs."""

    def __init__(self, schema):
        self.schema = schema

    def invoke(self, messages: list[Any]) -> Any:
        text = " ".join(getattr(m, "content", str(m)) for m in messages)
        name = self.schema.__name__
        if name == "Extraction":
            m = re.search(r"\$([\d,]+(?:\.\d+)?)", text)
            amount = float(m.group(1).replace(",", "")) if m else None
            ttype = "unauthorized_transaction" if "unauthorized" in text.lower() or "did not" in text.lower() else "other"
            return self.schema(
                summary="(fake) consumer reports a problem with an electronic transaction.",
                transaction_type=ttype, amount_usd=amount, reported_within_60_days="unknown",
                requested_remedy="(fake) refund",
            )
        if name == "DraftResponse":
            sections = sorted(set(re.findall(r"\[(1005\.\d+)\]", text)))[:2]
            cites = " and ".join(f"§ {s}" for s in sections) or "the applicable provisions"
            return self.schema(
                response_text=f"(fake draft) Thank you for your complaint. Under {cites} we will review the transaction and respond within the required time limits.",
                cited_sections=sections, confidence="medium" if sections else "low",
                notes_for_reviewer="Fake LLM output - set ANTHROPIC_API_KEY and unset TRIAGE_FAKE_LLM for real drafts.",
            )
        raise ValueError(f"fake LLM has no template for {name}")


class _FakeLLM:
    def with_structured_output(self, schema, **kwargs):  # kwargs (e.g. method=) are ignored offline
        return _FakeStructured(schema)


def get_llm(tier: str = "fast"):
    """tier: 'fast' (extraction) or 'strong' (drafting)."""
    if os.environ.get("TRIAGE_FAKE_LLM") == "1":
        return _FakeLLM()
    from langchain_anthropic import ChatAnthropic

    kwargs = {"model": CFG["llm"][tier], "max_tokens": CFG["llm"]["max_tokens"]}
    # Claude 5.x models reject the `temperature` parameter ("Extra inputs are not
    # permitted"). Only send it when config.yaml sets a non-null value, e.g. for
    # older models that still accept it.
    temperature = CFG["llm"].get("temperature")
    if temperature is not None:
        kwargs["temperature"] = temperature
    return ChatAnthropic(**kwargs)
