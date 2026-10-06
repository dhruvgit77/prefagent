"""Pairwise win-rate with an independent jury.

For each prompt, every juror compares model X's answer with model Y's answer TWICE, with
the A/B positions swapped. A juror's verdict counts only if both orders agree; otherwise
it is a tie (this cancels position bias instead of averaging it). Jurors are then combined
by strict majority; no majority → tie. Outcome per prompt: 1 = X wins, 0.5 = tie, 0 = Y wins.
"""

from __future__ import annotations

import json
import logging

from prefagent.llm.client import ChatRequest, LLMClient
from prefagent.llm.parsing import extract_json

log = logging.getLogger(__name__)


def pairwise_system(instructions: str) -> str:
    return (
        "You are an impartial expert judge comparing two AI assistant responses to the "
        f"same user request.\n{instructions}\n"
        "Decide which response is better overall, or 'tie' if they are equally good.\n"
        'Respond with ONLY a JSON object: {"winner": "A" | "B" | "tie", '
        '"rationale": "under 50 words"}'
    )


def pairwise_user(prompt: str, a: str, b: str) -> str:
    return f"## User request\n{prompt}\n\n## Response A\n{a}\n\n## Response B\n{b}"


def juror_verdict(client: LLMClient, juror: str, system: str, prompt: str, x: str, y: str,
                  temperature: float, retries: int = 3) -> dict:
    """One juror, both orders. Returns {'winner': 'x'|'y'|'tie', 'orders': [...]}."""
    orders = []
    for order, (a, b) in (("xy", (x, y)), ("yx", (y, x))):
        pick = None
        for attempt in range(1 + retries):
            resp = client.chat(ChatRequest(
                model=juror, temperature=temperature, max_tokens=300, json_mode=True,
                attempt=attempt, tag=f"winrate:{juror}:{order}",
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": pairwise_user(prompt, a, b)}]))
            obj = extract_json(resp.text)
            w = str(obj.get("winner", "")).strip().lower() if obj else ""
            if w in ("a", "b", "tie"):
                pick = w
                break
        if pick is None:
            pick = "tie"   # unparseable after retries: counts as no preference
            log.warning("%s: unparseable verdict, counted as tie", juror)
        orders.append({"order": order, "pick": pick})
    to_model = lambda o: ("tie" if o["pick"] == "tie" else
                          ("x" if (o["pick"] == "a") == (o["order"] == "xy") else "y"))
    picks = [to_model(o) for o in orders]
    return {"winner": picks[0] if picks[0] == picks[1] else "tie", "orders": orders}


def compare(client: LLMClient, cfg: dict, prompt: str, x: str, y: str) -> dict:
    wr = cfg["win_rate"]
    system = pairwise_system(cfg["judging"]["rubric"]["instructions"])
    verdicts = {j: juror_verdict(client, j, system, prompt, x, y, wr["temperature"])
                for j in wr["jury"]}
    n = len(verdicts)
    x_votes = sum(v["winner"] == "x" for v in verdicts.values())
    y_votes = sum(v["winner"] == "y" for v in verdicts.values())
    outcome = 1.0 if x_votes > n / 2 else 0.0 if y_votes > n / 2 else 0.5
    return {"outcome": outcome, "jurors": verdicts}


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)
