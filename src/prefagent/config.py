"""Load, merge, override, validate and snapshot the YAML configs.

Every script calls `load_config(...)` once at start-up and passes the resulting dict
down. No module hard-codes a parameter, so the configs are the single source of truth,
and the snapshot written next to each run records exactly what produced it.
"""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
DEFAULT_FILES = ("base", "data", "agents", "train_dpo", "eval", "experiments")

# Model family of each policy we fine-tune, keyed by Hub name prefix. Used by the
# judge-independence rule: an eval juror must not share a family with the policy.
POLICY_FAMILIES = {
    "Qwen/": "qwen",
    "HuggingFaceTB/SmolLM": "smollm",
    "microsoft/Phi": "phi",
}


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into a copy of `base`; override wins on conflicts."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and value.get("_replace"):
            # `_replace: true` in an overlay replaces the whole section instead of merging
            # (e.g. the demo profile's smaller set of conditions).
            out[key] = {k: copy.deepcopy(v) for k, v in value.items() if k != "_replace"}
        elif isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply CLI overrides like `training.learning_rate=1e-5` (values parsed as YAML)."""
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        dotted, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"Override must look like key.path=value, got {item!r}")
        *parents, leaf = dotted.split(".")
        node = cfg
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = yaml.safe_load(raw)
    return cfg


def load_config(
    names: tuple[str, ...] = DEFAULT_FILES,
    overrides: list[str] | None = None,
    config_dir: Path = CONFIG_DIR,
    profile: str | None = None,
) -> dict[str, Any]:
    """Merge configs/<name>.yaml in order, then an optional profile overlay
    (configs/profiles/<profile>.yaml), then CLI overrides, and validate.

    Each base file owns distinct top-level sections (e.g. train_dpo.yaml owns `lora`,
    `dpo`, `training`); profiles and overrides change values on top of them.
    """
    names = tuple(names) + ((f"profiles/{profile}",) if profile else ())
    cfg: dict[str, Any] = {}
    for name in names:
        with (config_dir / f"{name}.yaml").open() as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    validate(cfg)
    return cfg


def policy_family(model_name: str) -> str:
    for prefix, family in POLICY_FAMILIES.items():
        if model_name.startswith(prefix):
            return family
    raise ValueError(f"Unknown policy family for {model_name!r}; add it to POLICY_FAMILIES")


def policy_models(cfg: dict) -> list[str]:
    policies = cfg["policies"]
    return [policies["primary"], *policies.get("generalization", [])]


def validate(cfg: dict) -> None:
    """Fail fast on settings that would silently invalidate the experiment."""
    if "judging" in cfg:
        # The 2×2 is only fair if both judging levels cost the same number of calls.
        judging = cfg["judging"]
        panel_calls = len(judging["panel"]["judges"]) * judging["panel"]["rounds"]
        if panel_calls != judging["single_judge"]["samples"]:
            raise ValueError(
                f"Judging is not compute-matched: panel makes {panel_calls} calls, "
                f"single judge {judging['single_judge']['samples']}"
            )

    if "generation" in cfg:
        gen = cfg["generation"]
        if len(gen["multi_persona"]) != gen["n_candidates"]:
            raise ValueError("multi_persona must list exactly n_candidates personas")

    if {"dpo", "generation", "hh_rlhf"} <= cfg.keys():
        # Longest possible pair must fit max_length, or DPO silently truncates answers.
        # Generator tokens ≠ policy tokens, so allow 30% tokenizer mismatch plus 64 tokens
        # of chat-template overhead.
        worst = (cfg["hh_rlhf"]["max_prompt_tokens"]
                 + int(1.3 * cfg["generation"]["max_tokens"]) + 64)
        if worst > cfg["dpo"]["max_length"]:
            raise ValueError(f"Longest prompt+answer (~{worst} tokens) exceeds "
                             f"dpo.max_length={cfg['dpo']['max_length']}")

    if {"win_rate", "judging", "generation", "policies"} <= cfg.keys():
        check_judge_independence(cfg)


def check_judge_independence(cfg: dict) -> None:
    """No eval juror may share a family with a policy or any data-generating model.

    Otherwise the evaluated model is graded by the same taste it was trained on
    (or by its own family, which LLM judges are known to favour).
    """
    llms = cfg["llms"]
    judging = cfg["judging"]
    data_side = {cfg["generation"]["model"], judging["single_judge"]["model"],
                 *judging["panel"]["judges"]}
    banned = {llms[name]["family"] for name in data_side}
    banned |= {policy_family(m) for m in policy_models(cfg)}

    clashes = {j: llms[j]["family"] for j in cfg["win_rate"]["jury"]
               if llms[j]["family"] in banned}
    if clashes:
        raise ValueError(f"Eval jurors share a family with training-side models: {clashes}")


def save_snapshot(cfg: dict, run_dir: Path) -> Path:
    """Write the resolved config and current git commit next to a run's outputs."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "config_snapshot.json"
    path.write_text(json.dumps({"config": cfg, "git_commit": _git_commit()},
                               indent=2, default=str))
    return path


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None

