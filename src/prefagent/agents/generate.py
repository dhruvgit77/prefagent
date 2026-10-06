"""Candidate generation — the DIVERSITY factor of the 2×2.

  single_persona  one system prompt, sampled n times  (diversity from sampling only)
  multi_persona   n different personas, once each     (diversity from personas)
  human_pair      the two HH-RLHF responses, no API call (for label-accuracy checks)

Both API sources use the same model, temperature and candidate count, so persona vs
sampling is the only difference between them.
"""

from __future__ import annotations

from prefagent.llm.client import ChatRequest, LLMClient

SOURCES = ("single_persona", "multi_persona", "human_pair")


def generate(client: LLMClient | None, cfg: dict, rec: dict, source: str) -> dict:
    if source == "human_pair":
        if rec.get("human_chosen") is None:
            raise ValueError(f"{rec['id']} has no human pair")
        # Order is irrelevant: judges see a shuffled order and we record which is which.
        texts = [rec["human_chosen"], rec["human_rejected"]]
        return {"prompt_id": rec["id"], "source": source,
                "candidates": [{"text": t, "persona": None, "sample_idx": 0,
                                "finish_reason": "human"} for t in texts],
                "dropped": 0}

    g = cfg["generation"]
    n = g["n_candidates"]
    if source == "single_persona":
        plan = [(g["single_persona"][0], k) for k in range(n)]
    elif source == "multi_persona":
        plan = [(persona, 0) for persona in g["multi_persona"]]
    else:
        raise ValueError(f"unknown source {source!r}")

    suffix = f" {g['length_instruction']}" if g.get("length_instruction") else ""
    reqs = [ChatRequest(model=g["model"], temperature=g["temperature"], top_p=g["top_p"],
                        max_tokens=g["max_tokens"], sample_idx=k,
                        messages=[{"role": "system", "content": persona + suffix},
                                  {"role": "user", "content": rec["prompt"]}],
                        tag=f"gen:{source}:{rec['id']}")
            for persona, k in plan]
    candidates, dropped = [], 0
    for (persona, k), req in zip(plan, reqs):
        resp = client.chat(req)
        text = resp.text.strip()
        # Truncated or empty answers are dropped, not kept: a cut-off "chosen" response
        # would teach the policy to stop mid-sentence. The drop count is reported.
        if not text or resp.finish_reason == "length":
            dropped += 1
            continue
        candidates.append({"text": text, "persona": g["multi_persona"].index(persona)
                           if source == "multi_persona" else 0,
                           "sample_idx": k, "finish_reason": resp.finish_reason})
    return {"prompt_id": rec["id"], "source": source, "candidates": candidates,
            "dropped": dropped}
