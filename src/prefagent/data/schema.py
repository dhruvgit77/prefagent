"""Record types shared by every stage. Validating rows with pydantic means a malformed
record fails loudly at the stage that produced it, not three stages later in training."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

Subset = Literal["helpful-base", "harmless-base", "gsm8k"]
Split = Literal["train", "dev", "eval", "label_agreement", "gsm8k_label"]


class PromptRecord(BaseModel):
    """A seed prompt. For HH-RLHF prompts, the human-labelled pair is kept alongside
    so the `human` condition and the label-agreement metric use the same prompts."""

    id: str                                   # stable, e.g. "hh-helpful-base-train-01234"
    prompt: str
    subset: Subset
    split: Split
    prompt_tokens: int                        # length in the policy tokenizer
    human_chosen: Optional[str] = None        # HH-RLHF only
    human_rejected: Optional[str] = None      # HH-RLHF only
    gold_answer: Optional[str] = None         # GSM8K only: the final numeric answer


class PreferencePair(BaseModel):
    """One DPO training example: TRL's DPOTrainer reads prompt / chosen / rejected."""

    prompt_id: str
    prompt: str
    chosen: str
    rejected: str
    condition: str                            # experiments.yaml condition name
    # Fraction of final judge verdicts agreeing with this ordering (1.0 = unanimous).
    # Drives disagreement-aware DPO; None for human / heuristic labels.
    agreement: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    meta: dict = Field(default_factory=dict)  # scores, rationales, permutations, call ids

    @model_validator(mode="after")
    def _distinct(self) -> "PreferencePair":
        # Identical chosen/rejected gives a DPO loss gradient of exactly zero — a wasted
        # row that also distorts reward-accuracy metrics.
        if self.chosen.strip() == self.rejected.strip():
            raise ValueError(f"{self.prompt_id}: chosen and rejected are identical")
        return self
