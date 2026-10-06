"""Stage 4: evaluate the base model and every trained condition.

    python scripts/04_evaluate.py generate --profile demo   # answers on eval prompts (GPU/MPS)
    python scripts/04_evaluate.py reward   --profile demo   # judge-free human-pref accuracy
    python scripts/04_evaluate.py winrate  --profile demo   # jury win-rates (API)
    python scripts/04_evaluate.py report   --profile demo   # results.json + results.md
    python scripts/04_evaluate.py all      --profile demo

Outputs go to outputs/eval/<profile>/. Every step is cached on disk, so steps can be
re-run independently.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prefagent.config import load_config, save_snapshot  # noqa: E402
from prefagent.eval import stats  # noqa: E402
from prefagent.train.dpo import run_dir  # noqa: E402
from prefagent.utils.io import read_jsonl, write_json, write_jsonl  # noqa: E402
from prefagent.utils.seed import set_seed  # noqa: E402

log = logging.getLogger("evaluate")


def variants(cfg: dict, require: bool = True) -> dict[str, Path | None]:
    """name -> adapter path (None for the untuned base). With require=False, conditions
    not trained yet are left out instead of raising."""
    model = cfg["policies"]["primary"]
    seed = cfg["seeds"]["train"][0]
    out = {}
    for name, c in cfg["conditions"].items():
        if not c.get("train", True):
            out[name] = None
            continue
        runs = sorted(run_dir(cfg, name, model, seed, 0).parent.glob(
            f"{name}__{model.split('/')[-1]}__n*__s{seed}"))
        adapters = [r / "adapter" for r in runs if (r / "adapter").exists()]
        if not adapters:
            if require:
                raise SystemExit(f"No trained adapter for {name}; run scripts/03_train.py")
            log.warning("%s not trained yet — skipped", name)
            continue
        out[name] = adapters[-1]
    return out


def eval_prompts(cfg: dict) -> list[dict]:
    rows = read_jsonl(Path(cfg["paths"]["data_dir"]) / "prompts" / "eval.jsonl")
    return rows[: cfg["win_rate"].get("n_prompts") or len(rows)]


def step_generate(cfg: dict, out: Path) -> None:
    from prefagent.eval.policy import generate, load_policy

    prompts = eval_prompts(cfg)
    g = cfg["eval_generation"]
    for name, adapter in variants(cfg, require=False).items():
        path = out / "responses" / f"{name}.jsonl"
        if path.exists() and len(read_jsonl(path)) == len(prompts):
            log.info("%s: responses exist, skipping", name)
            continue
        set_seed(0, deterministic=True)
        model, tok = load_policy(cfg["policies"]["primary"], adapter)
        texts = generate(model, tok, [p["prompt"] for p in prompts], g["max_new_tokens"],
                         repetition_penalty=g["repetition_penalty"])
        write_jsonl(path, ({"prompt_id": p["id"], "prompt": p["prompt"], "response": t}
                           for p, t in zip(prompts, texts)))
        del model


def step_reward(cfg: dict, out: Path, n_pairs: int) -> None:
    from prefagent.eval.policy import implicit_reward_accuracy, load_policy

    pairs = read_jsonl(Path(cfg["paths"]["data_dir"]) / "prompts"
                       / "label_agreement.jsonl")[:n_pairs]
    results = {}
    for name, adapter in variants(cfg).items():
        if adapter is None:
            continue                       # base model: reference == policy, margin ≡ 0
        model, tok = load_policy(cfg["policies"]["primary"], adapter)
        r = implicit_reward_accuracy(model, tok, pairs)
        r["ci"] = stats.bootstrap_ci([float(m > 0) for m in r["per_pair"]])
        results[name] = r
        log.info("%s: human-preference accuracy %.3f", name, r["accuracy"])
        del model
    write_json(out / "reward_accuracy.json", {"n_pairs": len(pairs), "results": results})


def step_winrate(cfg: dict, out: Path) -> None:
    from prefagent.eval.winrate import compare
    from prefagent.llm.client import LLMClient

    client = LLMClient(cfg)
    resp = {name: {r["prompt_id"]: r for r in read_jsonl(out / "responses" / f"{name}.jsonl")}
            for name in variants(cfg)}
    for x, y in cfg["win_rate"]["comparisons"]:
        path = out / "winrate" / f"{x}__vs__{y}.jsonl"
        done = {r["prompt_id"] for r in read_jsonl(path)} if path.exists() else set()
        rows = [r for r in read_jsonl(path)] if path.exists() else []
        for pid, rx in resp[x].items():
            if pid in done:
                continue
            try:
                res = compare(client, cfg, rx["prompt"], rx["response"], resp[y][pid]["response"])
            except Exception as e:  # noqa: BLE001 — quota/network: leave for the next run
                log.error("%s vs %s, %s: %s (re-run to retry)", x, y, pid, e)
                continue
            rows.append({"prompt_id": pid, **res,
                         "len_x": len(rx["response"].split()),
                         "len_y": len(resp[y][pid]["response"].split())})
            write_jsonl(path, rows)
        log.info("%s vs %s: %d/%d judged", x, y, len(rows), len(resp[x]))


def step_report(cfg: dict, out: Path) -> dict:
    report = {"profile": cfg.get("_profile"), "policy": cfg["policies"]["primary"],
              "jury": cfg["win_rate"]["jury"], "comparisons": [], "lengths": {}}
    for name in variants(cfg):
        rows = read_jsonl(out / "responses" / f"{name}.jsonl")
        words = [len(r["response"].split()) for r in rows]
        report["lengths"][name] = sum(words) / len(words)
    for x, y in cfg["win_rate"]["comparisons"]:
        path = out / "winrate" / f"{x}__vs__{y}.jsonl"
        if not path.exists():
            continue
        rows = read_jsonl(path)
        o = [r["outcome"] for r in rows]
        report["comparisons"].append({
            "x": x, "y": y, "n": len(o),
            "win": sum(v == 1.0 for v in o) / len(o),
            "tie": sum(v == 0.5 for v in o) / len(o),
            "loss": sum(v == 0.0 for v in o) / len(o),
            "win_rate": sum(o) / len(o),                     # ties count half
            "ci95": stats.bootstrap_ci(o),
            "lc_win_rate": stats.length_controlled_win_rate(
                o, [r["len_x"] for r in rows], [r["len_y"] for r in rows]),
            "juror_agreement": _juror_agreement(rows),
        })
    ra = out / "reward_accuracy.json"
    if ra.exists():
        report["reward_accuracy"] = {k: {"accuracy": v["accuracy"], "ci95": v["ci"],
                                         "mean_margin": v["mean_margin"]}
                                     for k, v in json.loads(ra.read_text())["results"].items()}
        report["reward_accuracy_n"] = json.loads(ra.read_text())["n_pairs"]
    write_json(out / "results.json", report)
    (out / "results.md").write_text(render_md(report))
    print(render_md(report))
    return report


def _juror_agreement(rows: list[dict]) -> float | None:
    """Share of prompts where all jurors gave the same verdict (incl. tie)."""
    if not rows or len(rows[0]["jurors"]) < 2:
        return None
    same = [len({v["winner"] for v in r["jurors"].values()}) == 1 for r in rows]
    return sum(same) / len(same)


def render_md(r: dict) -> str:
    lines = [f"# Evaluation — {r['policy']}", "",
             f"Jury: {', '.join(r['jury'])} (both A/B orders per juror; strict majority)", "",
             "| X vs Y | n | win | tie | loss | win-rate (ties ½) | 95% CI | length-controlled |",
             "|---|---|---|---|---|---|---|---|"]
    for c in r["comparisons"]:
        lc = f"{c['lc_win_rate']:.3f}" if c["lc_win_rate"] is not None else "n/a"
        lines.append(f"| {c['x']} vs {c['y']} | {c['n']} | {c['win']:.2f} | {c['tie']:.2f} | "
                     f"{c['loss']:.2f} | **{c['win_rate']:.3f}** | "
                     f"[{c['ci95'][0]:.3f}, {c['ci95'][1]:.3f}] | {lc} |")
    if "reward_accuracy" in r:
        lines += ["", f"Agreement with held-out HUMAN preferences (n={r['reward_accuracy_n']}, "
                  "judge-free; 0.5 = chance):", "",
                  "| model | accuracy | 95% CI |", "|---|---|---|"]
        for k, v in r["reward_accuracy"].items():
            lines.append(f"| {k} | {v['accuracy']:.3f} | [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] |")
    lines += ["", "Mean answer length (words): "
              + ", ".join(f"{k} {v:.0f}" for k, v in r["lengths"].items())]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["generate", "reward", "winrate", "report", "all"])
    ap.add_argument("--profile", default=None)
    ap.add_argument("--reward-pairs", type=int, default=200)
    ap.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = ap.parse_args()

    cfg = load_config(overrides=args.overrides, profile=args.profile)
    cfg["_profile"] = args.profile
    logging.basicConfig(level=cfg["logging"]["level"], format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    out = Path(cfg["paths"]["output_dir"]) / "eval" / (args.profile or "main")
    save_snapshot(cfg, out)

    if args.step in ("generate", "all"):
        step_generate(cfg, out)
    if args.step in ("reward", "all"):
        step_reward(cfg, out, args.reward_pairs)
    if args.step in ("winrate", "all"):
        step_winrate(cfg, out)
    if args.step in ("report", "all"):
        step_report(cfg, out)


if __name__ == "__main__":
    main()
