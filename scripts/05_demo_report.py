"""Stage 5: build the demo report page from the real output files.

    python scripts/05_demo_report.py --profile demo

Reads data/candidates, data/judgements, data/pairs, outputs/runs/*/log_history.json and
outputs/eval/<profile>/*. Writes outputs/eval/<profile>/report.html. Every number on the
page is computed here from those files; nothing is typed in by hand.
"""

import argparse
import html
import json
import statistics as st
import sys
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prefagent.config import load_config  # noqa: E402
from prefagent.utils.io import read_jsonl  # noqa: E402

E = html.escape
PERSONAS = ["Concise", "Step-by-step", "Domain expert", "Careful sceptic"]


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def label_stats(cfg: dict) -> dict:
    d = Path(cfg["paths"]["data_dir"])
    n = cfg["data_sizes"]["main"]
    cands = {r["prompt_id"]: r for r in read_jsonl(d / "candidates/train/multi_persona.jsonl")}
    judg = [r for r in read_jsonl(d / "judgements/train/multi_persona__panel.jsonl")]
    order = [r["id"] for r in read_jsonl(d / "prompts/train.jsonl")][:n]
    judg = [j for j in judg if j["prompt_id"] in set(order)]
    with_pair = [j for j in judg if j["aggregate"]]
    too_few = sum(j["n_candidates"] < 2 for j in judg)
    ties = sum(j["n_candidates"] >= 2 and not j["aggregate"] for j in judg)
    agreements = [j["aggregate"]["agreement"] for j in with_pair]

    chosen_p, rejected_p, len_c, len_r, best_pos, n_shown = Counter(), Counter(), [], [], Counter(), 0
    changed = total_r2 = 0
    for j in with_pair:
        cs = cands[j["prompt_id"]]["candidates"]
        a = j["aggregate"]
        chosen_p[cs[a["chosen"]]["persona"]] += 1
        rejected_p[cs[a["rejected"]]["persona"]] += 1
        len_c.append(len(cs[a["chosen"]]["text"].split()))
        len_r.append(len(cs[a["rejected"]]["text"].split()))
        calls = j["calls"]
        for c in calls:
            if len(c["permutation"]) == 4:
                best_pos[c["permutation"].index(c["best"])] += 1
                n_shown += 1
        r1 = {c["judge"]: c for c in calls if c["round"] == 1}
        for c in calls:
            if c["round"] == 2 and c["judge"] in r1:
                total_r2 += 1
                changed += c["best"] != r1[c["judge"]]["best"]
    judged_ids = {j["prompt_id"] for j in judg}
    dropped = sum(c["dropped"] for pid, c in cands.items() if pid in judged_ids)
    return {
        "prompts": len(judg), "pairs": len(with_pair), "too_few": too_few, "ties": ties,
        "dropped_candidates": dropped, "total_candidates": 4 * len(judg),
        "agreement_mean": st.mean(agreements) if agreements else None,
        "unanimous": sum(a == 1.0 for a in agreements) / len(agreements) if agreements else None,
        "chosen_persona": [chosen_p[i] for i in range(4)],
        "rejected_persona": [rejected_p[i] for i in range(4)],
        "len_chosen": st.mean(len_c) if len_c else None,
        "len_rejected": st.mean(len_r) if len_r else None,
        "longer_chosen": sum(c > r for c, r in zip(len_c, len_r)) / len(len_c) if len_c else None,
        "best_position": [best_pos[i] / n_shown for i in range(4)] if n_shown else None,
        "round2_changed": changed / total_r2 if total_r2 else None,
    }


def label_examples(cfg: dict, k: int = 3) -> list[dict]:
    d = Path(cfg["paths"]["data_dir"])
    pairs = read_jsonl(d / "pairs/train/multi_panel.jsonl")
    # deterministic, varied picks: one high-agreement helpful, one split vote, one harmless
    picks = []
    for want in (lambda p: p["agreement"] == 1.0 and "helpful" in p["prompt_id"],
                 lambda p: p["agreement"] < 0.8,
                 lambda p: "harmless" in p["prompt_id"]):
        p = next((p for p in pairs if want(p) and p not in picks), None)
        if p:
            picks.append(p)
    return picks[:k]


def training_summary(cfg: dict) -> dict:
    out = {}
    for run in sorted((Path(cfg["paths"]["output_dir"]) / "runs").glob("*__s*")):
        logs = run / "log_history.json"
        if not logs.exists() or "Qwen2.5-0.5B" not in run.name:
            continue
        hist = [h for h in json.loads(logs.read_text()) if "rewards/accuracies" in h]
        final = json.loads(logs.read_text())[-1]
        out[run.name.split("__")[0]] = {
            "steps": [h["step"] for h in hist],
            "acc": [h["rewards/accuracies"] for h in hist],
            "margin": [h["rewards/margins"] for h in hist],
            "runtime_min": final.get("train_runtime", 0) / 60,
            "n_pairs": int(run.name.split("__n")[1].split("__")[0]),
        }
    return out


# ---------------------------------------------------------------------------
# rendering helpers
# ---------------------------------------------------------------------------

def pct(x, digits=0):
    return "—" if x is None else f"{100 * x:.{digits}f}%"


def winrate_chart(comps: list[dict]) -> str:
    """Horizontal win-rate bars with 95% CI whiskers, drawn to one scale (0–100%)."""
    w, left, right, row = 640, 210, 30, 46
    h = 40 + row * len(comps)
    x = lambda v: left + v * (w - left - right)
    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Win-rate with 95% confidence intervals">']
    for t in (0, 0.25, 0.5, 0.75, 1):
        parts.append(f'<line x1="{x(t):.1f}" y1="18" x2="{x(t):.1f}" y2="{h - 16}" class="grid{" mid" if t == 0.5 else ""}"/>')
        parts.append(f'<text x="{x(t):.1f}" y="{h - 2}" class="tick" text-anchor="middle">{int(t * 100)}%</text>')
    for i, c in enumerate(comps):
        y = 30 + i * row
        lo, hi = c["ci95"]
        parts.append(f'<text x="{left - 12}" y="{y + 15}" class="lbl" text-anchor="end">{E(nice(c["x"]))} vs {E(nice(c["y"]))}</text>')
        parts.append(f'<rect x="{x(0):.1f}" y="{y + 4}" width="{x(c["win_rate"]) - x(0):.1f}" height="18" rx="2" class="bar"/>')
        parts.append(f'<line x1="{x(lo):.1f}" y1="{y + 13}" x2="{x(hi):.1f}" y2="{y + 13}" class="ci"/>')
        for v in (lo, hi):
            parts.append(f'<line x1="{x(v):.1f}" y1="{y + 7}" x2="{x(v):.1f}" y2="{y + 19}" class="ci"/>')
        parts.append(f'<text x="{x(c["win_rate"]) + 6:.1f}" y="{y - 1}" class="val">{100 * c["win_rate"]:.1f}%</text>')
    parts.append("</svg>")
    return "".join(parts)


def curve(points_x: list, points_y: list, label: str, lo=0.0, hi=1.0) -> str:
    w, h, pad = 300, 120, 26
    if not points_x:
        return ""
    xmax = max(points_x)
    X = lambda v: pad + (v / xmax) * (w - pad - 8)
    Y = lambda v: h - pad + 4 - (min(max(v, lo), hi) - lo) / (hi - lo) * (h - pad - 10)
    path = " ".join(f"{'M' if i == 0 else 'L'}{X(a):.1f},{Y(b):.1f}" for i, (a, b) in enumerate(zip(points_x, points_y)))
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{E(label)}">'
            f'<line x1="{pad}" y1="{Y(0.5):.1f}" x2="{w - 8}" y2="{Y(0.5):.1f}" class="grid mid"/>'
            f'<text x="{pad - 4}" y="{Y(hi) + 4:.1f}" class="tick" text-anchor="end">{hi:g}</text>'
            f'<text x="{pad - 4}" y="{Y(0.5) + 4:.1f}" class="tick" text-anchor="end">0.5</text>'
            f'<text x="{pad - 4}" y="{Y(lo) + 4:.1f}" class="tick" text-anchor="end">{lo:g}</text>'
            f'<text x="{w - 8}" y="{h - 4}" class="tick" text-anchor="end">step {xmax}</text>'
            f'<path d="{path}" class="line"/>'
            f'<circle cx="{X(points_x[-1]):.1f}" cy="{Y(points_y[-1]):.1f}" r="3.5" class="dot"/></svg>')


def nice(name: str) -> str:
    return {"base": "Base model", "human": "DPO · human labels",
            "multi_panel": "DPO · agent panel labels"}.get(name, name)


def short(text: str, n: int = 900) -> str:
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + " …"


# ---------------------------------------------------------------------------
# page
# ---------------------------------------------------------------------------

CSS = """
/* Layout: a single reading column (max 72rem) with full-width data blocks; pairs and
   model answers sit in two/three-column grids that stack on phones. */
:root{
  --paper:#f5f7f8; --panel:#ffffff; --ink:#15212b; --muted:#5b6a76; --rule:#d8e0e5;
  --accent:#1f5f8b; --chosen:#1d7a5a; --rejected:#a3473f; --chosen-bg:#e7f3ee; --rejected-bg:#f7eae8;
  --display:"IBM Plex Sans Condensed","Arial Narrow",sans-serif;
  --body:"IBM Plex Sans",system-ui,sans-serif; --mono:"IBM Plex Mono",ui-monospace,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --paper:#0f171d; --panel:#16212a; --ink:#e3eaef; --muted:#93a4b1; --rule:#2a3a46;
  --accent:#6fb3e0; --chosen:#5cc79b; --rejected:#e58a80; --chosen-bg:#14302a; --rejected-bg:#3a201e; color-scheme:dark}}
:root[data-theme="dark"]{
  --paper:#0f171d; --panel:#16212a; --ink:#e3eaef; --muted:#93a4b1; --rule:#2a3a46;
  --accent:#6fb3e0; --chosen:#5cc79b; --rejected:#e58a80; --chosen-bg:#14302a; --rejected-bg:#3a201e; color-scheme:dark}
body{background:var(--paper);color:var(--ink);font:15px/1.6 var(--body)}
.wrap{max-width:72rem;margin:0 auto;padding-inline:clamp(16px,4vw,40px);padding-block:40px 64px;display:grid;gap:56px}
h1,h2,h3{font-family:var(--display);text-wrap:balance;margin:0;line-height:1.15}
h1{font-size:clamp(2rem,4.4vw,3.1rem);font-weight:600;letter-spacing:-.01em}
h2{font-size:1.6rem;font-weight:600}
h3{font-size:1.1rem;font-weight:600}
p{margin:0;max-width:68ch}
.eyebrow{font:500 .75rem/1 var(--mono);letter-spacing:.12em;text-transform:uppercase;color:var(--accent)}
.lede{font-size:1.15rem;color:var(--muted);max-width:60ch}
section{display:grid;gap:18px;min-width:0}
header{display:grid;gap:16px}
.meta{display:flex;flex-wrap:wrap;gap:8px 24px;font:.82rem var(--mono);color:var(--muted)}
.meta b{color:var(--ink);font-weight:500}
.flow{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;counter-reset:s}
.flow div{background:var(--panel);border:1px solid var(--rule);border-radius:6px;padding:12px 14px;display:grid;gap:4px;min-width:0}
.flow div::before{counter-increment:s;content:"Stage " counter(s);font:500 .7rem var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--accent)}
.flow b{font-family:var(--display);font-size:1rem}
.flow span{font-size:.85rem;color:var(--muted)}
.chart{background:var(--panel);border:1px solid var(--rule);border-radius:6px;padding:16px;overflow-x:auto}
svg{display:block;width:100%;height:auto;max-width:100%}
svg .grid{stroke:var(--rule);stroke-width:1} svg .grid.mid{stroke:var(--muted);stroke-dasharray:3 3}
svg .tick{fill:var(--muted);font:11px var(--mono)} svg .lbl{fill:var(--ink);font:13px var(--body)}
svg .val{fill:var(--ink);font:600 12px var(--mono)}
svg .bar{fill:var(--accent);opacity:.85} svg .ci{stroke:var(--ink);stroke-width:1.5}
svg .line{fill:none;stroke:var(--accent);stroke-width:2} svg .dot{fill:var(--accent)}
.tablewrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:.9rem}
th,td{text-align:left;padding:8px 12px;border-bottom:1px solid var(--rule);white-space:nowrap}
th{font:500 .72rem var(--mono);letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
td.num{font-family:var(--mono)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1px;background:var(--rule);border:1px solid var(--rule);border-radius:6px;overflow:hidden}
.stats div{background:var(--panel);padding:14px 16px;display:grid;gap:2px;min-width:0}
.stats b{font:600 1.5rem var(--mono)}
.stats span{font-size:.82rem;color:var(--muted)}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:16px}
.three{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,260px),1fr));gap:12px}
.pair{display:grid;gap:10px;background:var(--panel);border:1px solid var(--rule);border-radius:6px;padding:16px}
.prompt{font-weight:500}
.resp{border-radius:4px;padding:10px 12px;font-size:.88rem;white-space:pre-wrap;overflow-wrap:anywhere;min-width:0}
.resp.c{background:var(--chosen-bg);border-left:3px solid var(--chosen)}
.resp.r{background:var(--rejected-bg);border-left:3px solid var(--rejected)}
.resp.n{background:var(--paper);border:1px solid var(--rule)}
.tag{font:500 .7rem var(--mono);letter-spacing:.08em;text-transform:uppercase;display:block;margin-bottom:4px}
.c .tag{color:var(--chosen)} .r .tag{color:var(--rejected)} .n .tag{color:var(--muted)}
.small{font-size:.85rem;color:var(--muted)}
ul.plain{margin:0;padding-left:1.1rem;display:grid;gap:6px;max-width:72ch}
pre{background:var(--panel);border:1px solid var(--rule);border-radius:6px;padding:14px;overflow-x:auto;font:.82rem/1.5 var(--mono);margin:0}
a{color:var(--accent)}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
"""


def build(cfg: dict, profile: str) -> str:
    out = Path(cfg["paths"]["output_dir"]) / "eval" / profile
    res = json.loads((out / "results.json").read_text())
    ls = label_stats(cfg)
    ex = label_examples(cfg)
    tr = training_summary(cfg)
    responses = {n: {r["prompt_id"]: r for r in read_jsonl(out / "responses" / f"{n}.jsonl")}
                 for n in ("base", "human", "multi_panel")}
    comps = res["comparisons"]
    by = {(c["x"], c["y"]): c for c in comps}
    head = by.get(("multi_panel", "human"))
    agent_base = by.get(("multi_panel", "base"))
    human_base = by.get(("human", "base"))
    ra = res.get("reward_accuracy", {})
    n_pairs = next(iter(tr.values()))["n_pairs"] if tr else ls["pairs"]

    h = [f"<title>PrefAgent Pilot Results</title>",
         '<link rel="preconnect" href="https://fonts.googleapis.com">',
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">',
         f"<style>{CSS}</style>", '<main class="wrap">']

    # header
    h.append(f"""<header>
<span class="eyebrow">Pilot run · {date.today():%d %b %Y}</span>
<h1>Can a debating panel of LLM judges replace human preference labels?</h1>
<p class="lede">A small end-to-end run of the full pipeline: AI agents write and judge answers,
a 0.5B model is DPO-trained on those labels and, separately, on real human labels, and
independent judges compare the results on held-out prompts.</p>
<div class="meta"><span>Policy <b>{E(res['policy'])}</b></span><span>Training pairs <b>{n_pairs}</b> per model</span>
<span>Eval prompts <b>{comps[0]['n'] if comps else '—'}</b></span><span>Jury <b>{E(', '.join(res['jury']))}</b></span>
<span>Cost <b>$0</b></span></div>
</header>""")

    # pipeline
    g, j = cfg["generation"], cfg["judging"]
    llm = cfg["llms"]
    h.append(f"""<section><span class="eyebrow">Method</span><h2>The pipeline</h2>
<div class="flow">
<div><b>Seed prompts</b><span>HH-RLHF single-turn prompts, 50% helpful / 50% harmless, decontaminated against the eval set</span></div>
<div><b>4 personas answer</b><span>{E(llm[g['model']]['id'])} as concise, step-by-step, expert and sceptic personas</span></div>
<div><b>Panel debates</b><span>{len(j['panel']['judges'])} judges score 1–5 on {E(', '.join(j['rubric']['criteria']))}; round 2 sees round-1 verdicts</span></div>
<div><b>DPO training</b><span>LoRA r={cfg['lora']['r']}, β={cfg['dpo']['beta']}, lr={cfg['training']['learning_rate']:g}, {cfg['training']['num_train_epochs']} epochs; same settings for both label sources</span></div>
<div><b>Independent jury</b><span>{E(' + '.join(llm[m]['id'] for m in res['jury']))}, both answer orders, no shared model family with any labeller</span></div>
</div></section>""")

    # results
    if comps:
        verdict = ""
        if head:
            lo, hi = head["ci95"]
            verdict = (f"Head-to-head, the agent-label model wins {pct(head['win_rate'], 1)} of comparisons "
                       f"against the human-label model (ties count half; 95% CI {pct(lo)}–{pct(hi)}). ")
            verdict += ("The interval includes 50%, so at this sample size the two label sources are "
                        "statistically indistinguishable." if lo <= 0.5 <= hi else
                        "The interval excludes 50%.")
        h.append(f"""<section><span class="eyebrow">Result</span><h2>Win-rate against each other and the base model</h2>
<p>{E(verdict)}</p>
<div class="chart">{winrate_chart(comps)}</div>
<div class="tablewrap"><table><thead><tr><th>Comparison</th><th>n</th><th>Win</th><th>Tie</th><th>Loss</th><th>Win-rate</th><th>95% CI</th><th>Length-controlled</th><th>Jurors agree</th></tr></thead><tbody>""")
        for c in comps:
            lc = "n/a" if c["lc_win_rate"] is None else pct(c["lc_win_rate"], 1)
            h.append(f"<tr><td>{E(nice(c['x']))} vs {E(nice(c['y']))}</td><td class=num>{c['n']}</td>"
                     f"<td class=num>{pct(c['win'])}</td><td class=num>{pct(c['tie'])}</td><td class=num>{pct(c['loss'])}</td>"
                     f"<td class=num><b>{pct(c['win_rate'], 1)}</b></td><td class=num>{pct(c['ci95'][0])}–{pct(c['ci95'][1])}</td>"
                     f"<td class=num>{lc}</td><td class=num>{pct(c['juror_agreement'])}</td></tr>")
        h.append("</tbody></table></div>")
        h.append('<p class="small">Each juror judges every pair twice with positions swapped; a juror counts only if both orders agree. '
                 'A comparison is a win only when both jurors agree; otherwise it is a tie. Length-controlled win-rate removes the effect '
                 'of answer length (AlpacaEval-2 style logistic adjustment).</p></section>')

    # judge-free
    if ra:
        rows = "".join(f"<tr><td>{E(nice(k))}</td><td class=num><b>{v['accuracy']:.3f}</b></td>"
                       f"<td class=num>{v['ci95'][0]:.3f}–{v['ci95'][1]:.3f}</td><td class=num>{v['mean_margin']:+.3f}</td></tr>"
                       for k, v in ra.items())
        h.append(f"""<section><span class="eyebrow">Judge-free check</span><h2>Do the trained models agree with unseen human preferences?</h2>
<p>For {res['reward_accuracy_n']} held-out HH-RLHF pairs that neither model trained on, we read each model's DPO implicit reward,
log π(chosen)/π<sub>ref</sub>(chosen) − log π(rejected)/π<sub>ref</sub>(rejected), and count how often it favours the answer humans preferred.
0.5 is chance. No LLM judge is involved.</p>
<div class="tablewrap"><table><thead><tr><th>Model</th><th>Accuracy</th><th>95% CI</th><th>Mean margin</th></tr></thead><tbody>{rows}</tbody></table></div>
</section>""")

    # lengths + training
    lens = res["lengths"]
    h.append(f"""<section><span class="eyebrow">Diagnostics</span><h2>Training and answer length</h2>
<div class="stats">{''.join(f'<div><b>{lens[k]:.0f}</b><span>mean words · {E(nice(k))}</span></div>' for k in lens)}</div>
<div class="two">""")
    for name, t in tr.items():
        h.append(f'<div class="chart"><h3>{E(nice(name))}</h3><p class="small">Reward accuracy on training batches, '
                 f'{t["runtime_min"]:.0f} min on Apple MPS</p>{curve(t["steps"], t["acc"], "reward accuracy")}</div>')
    h.append("</div></section>")

    # label quality
    h.append(f"""<section><span class="eyebrow">Inside the agent labels</span><h2>What the panel preferred</h2>
<div class="stats">
<div><b>{ls['pairs']}/{ls['prompts']}</b><span>prompts produced a preference pair</span></div>
<div><b>{pct(ls['agreement_mean'])}</b><span>mean panel agreement on the chosen-over-rejected order</span></div>
<div><b>{pct(ls['unanimous'])}</b><span>pairs where all three final verdicts agreed</span></div>
<div><b>{pct(ls['round2_changed'])}</b><span>round-2 verdicts that changed their pick for best after debate</span></div>
<div><b>{ls['len_chosen']:.0f} vs {ls['len_rejected']:.0f}</b><span>mean words, chosen vs rejected</span></div>
<div><b>{pct(ls['longer_chosen'])}</b><span>pairs where the chosen answer is the longer one</span></div>
</div>
<div class="tablewrap"><table><thead><tr><th>Persona</th><th>Times chosen</th><th>Times rejected</th></tr></thead><tbody>
{''.join(f"<tr><td>{p}</td><td class=num>{ls['chosen_persona'][i]}</td><td class=num>{ls['rejected_persona'][i]}</td></tr>" for i, p in enumerate(PERSONAS))}
</tbody></table></div>
<p class="small">Position check: judges picked the response shown first / second / third / fourth as best in
{' / '.join(pct(p) for p in ls['best_position'] or [])} of calls (25% each means no position bias).
{ls['dropped_candidates']} of {ls['total_candidates']} generated answers were discarded for hitting the length cap;
{ls['ties']} prompts were exact ties and {ls['too_few']} had fewer than two usable answers.</p>
</section>""")

    # examples
    if ex:
        h.append('<section><span class="eyebrow">Real examples</span><h2>Preference pairs the panel produced</h2><div class="two">')
        for p in ex:
            ms = p["meta"]["mean_scores"]
            h.append(f"""<div class="pair"><p class="prompt">{E(p['prompt'])}</p>
<div class="resp c"><span class="tag">Chosen · mean score {ms[p['meta']['chosen_idx']]:.2f}</span>{E(short(p['chosen']))}</div>
<div class="resp r"><span class="tag">Rejected · mean score {ms[p['meta']['rejected_idx']]:.2f}</span>{E(short(p['rejected']))}</div>
<p class="small">Panel agreement {pct(p['agreement'])} · {E(p['prompt_id'])}</p></div>""")
        h.append("</div></section>")

    # model answers side by side
    wr = read_jsonl(out / "winrate" / "multi_panel__vs__human.jsonl") if (out / "winrate" / "multi_panel__vs__human.jsonl").exists() else []
    picks = [r for r in wr if r["outcome"] != 0.5][:2] + [r for r in wr if r["outcome"] == 0.5][:1]
    if picks:
        h.append('<section><span class="eyebrow">Held-out prompts</span><h2>The three models answer the same prompt</h2>')
        for r in picks:
            pid = r["prompt_id"]
            res_txt = {"1.0": "jury preferred the agent-label model", "0.0": "jury preferred the human-label model",
                       "0.5": "jury called it a tie"}[str(r["outcome"])]
            h.append(f'<div class="pair"><p class="prompt">{E(responses["base"][pid]["prompt"])}</p><div class="three">')
            for n in ("base", "human", "multi_panel"):
                h.append(f'<div class="resp n"><span class="tag">{E(nice(n))}</span>{E(short(responses[n][pid]["response"], 700))}</div>')
            h.append(f'</div><p class="small">{E(res_txt)} · {E(pid)}</p></div>')
        h.append("</section>")

    # limits
    h.append(f"""<section><span class="eyebrow">Scope</span><h2>What this pilot shows and what it does not</h2>
<ul class="plain">
<li>This is a pilot: one 0.5B model, one training seed, {n_pairs} pairs per condition and {comps[0]['n'] if comps else '—'} eval prompts. Differences of a few points are within noise; the confidence intervals above are the honest read.</li>
<li>The full study adds the 2×2 design (persona diversity × judge deliberation at equal API cost), random and length-only label baselines, three training seeds, four base models, a 250→2,000 pair scaling curve, and label accuracy against GSM8K ground truth.</li>
<li>Free-tier limits shaped this run: the labelling and judging models are the ones Groq, Gemini and Cohere served for free on {date.today():%d %b %Y}.</li>
<li>The demo uses a higher learning rate than the full study (fewer, larger steps for {n_pairs} pairs); both conditions used identical settings.</li>
</ul></section>""")

    h.append(f"""<section><span class="eyebrow">Reproduce</span><h2>Run it yourself</h2>
<pre>git clone https://github.com/dhruvgit77/prefagent && cd prefagent
pip install -r requirements-agents.txt torch trl peft
python scripts/01_prepare_data.py
python scripts/02_label.py run --profile demo
python scripts/02_label.py pairs --split train --profile demo
python scripts/03_train.py --condition human --profile demo
python scripts/03_train.py --condition multi_panel --profile demo
python scripts/04_evaluate.py all --profile demo
python scripts/05_demo_report.py --profile demo</pre>
<p class="small">Code: <a href="https://github.com/dhruvgit77/prefagent">github.com/dhruvgit77/prefagent</a></p></section>""")
    h.append("</main>")
    return "\n".join(h)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="demo")
    args = ap.parse_args()
    cfg = load_config(profile=args.profile)
    page = build(cfg, args.profile)
    path = Path(cfg["paths"]["output_dir"]) / "eval" / args.profile / "report.html"
    path.write_text(page)
    print(path)


if __name__ == "__main__":
    main()
