"""Load a policy (base or base + LoRA adapter), generate answers, score log-probs.

Runs on CUDA, Apple MPS or CPU. Generation is greedy so differences between models come
from their weights, not sampling luck.
"""

from __future__ import annotations

import logging
from pathlib import Path

from prefagent.train.dpo import device_kind

log = logging.getLogger(__name__)


def load_policy(model_name: str, adapter: Path | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = device_kind()
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"                      # required for batched generation
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter))
    model.to(dev).eval()
    return model, tok


def chat_text(tok, prompt: str) -> str:
    return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True)


def generate(model, tok, prompts: list[str], max_new_tokens: int, batch_size: int = 8,
             repetition_penalty: float = 1.0) -> list[str]:
    import torch

    outs = []
    for i in range(0, len(prompts), batch_size):
        batch = [chat_text(tok, p) for p in prompts[i:i + batch_size]]
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 repetition_penalty=repetition_penalty,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        new = gen[:, enc["input_ids"].shape[1]:]   # strip the prompt: answers only
        outs.extend(t.strip() for t in tok.batch_decode(new, skip_special_tokens=True))
        log.info("generated %d/%d", min(i + batch_size, len(prompts)), len(prompts))
    return outs


def response_logprob(model, tok, prompt: str, response: str) -> float:
    """Sum of log-probs of the response tokens given the chat-formatted prompt."""
    import torch

    prefix = chat_text(tok, prompt)
    full = prefix + response + tok.eos_token
    p_ids = tok(prefix, add_special_tokens=False, return_tensors="pt")["input_ids"]
    f_ids = tok(full, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
    n_prompt = p_ids.shape[1]
    with torch.no_grad():
        logits = model(f_ids).logits[0, :-1].float()
    targets = f_ids[0, 1:]
    logps = torch.log_softmax(logits, dim=-1).gather(1, targets[:, None]).squeeze(1)
    return float(logps[n_prompt - 1:].sum())


def implicit_reward_accuracy(model, tok, pairs: list[dict]) -> dict:
    """DPO's implicit reward on unseen human-labelled pairs:
        margin = [log π(c) − log π_ref(c)] − [log π(r) − log π_ref(r)]
    The reference is the same model with the adapter switched off. accuracy = share of
    pairs where the trained policy moved toward the HUMAN-preferred answer — a judge-free
    measure of alignment with human preferences."""
    margins = []
    for p in pairs:
        lp = {k: response_logprob(model, tok, p["prompt"], p[k])
              for k in ("human_chosen", "human_rejected")}
        with model.disable_adapter():
            ref = {k: response_logprob(model, tok, p["prompt"], p[k])
                   for k in ("human_chosen", "human_rejected")}
        margins.append((lp["human_chosen"] - ref["human_chosen"])
                       - (lp["human_rejected"] - ref["human_rejected"]))
    return {"accuracy": sum(m > 0 for m in margins) / len(margins),
            "mean_margin": sum(margins) / len(margins), "per_pair": margins}
