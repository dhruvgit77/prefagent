"""DPO training of a LoRA adapter on one condition's preference pairs.

One function, two backends:
  unsloth  CUDA GPU (Colab/Kaggle T4): 4-bit QLoRA, Unsloth-patched kernels
  peft     Apple MPS / CPU / CUDA without Unsloth: plain LoRA in full precision

Everything that defines the experiment (LoRA shape, β, LR, schedule, epochs, data) comes
from the config and is identical across conditions; only the pairs file changes.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

from prefagent.config import save_snapshot
from prefagent.utils.io import read_jsonl
from prefagent.utils.seed import set_seed

log = logging.getLogger(__name__)


def device_kind() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_backend(cfg: dict) -> str:
    want = cfg["model"]["backend"]
    if want != "auto":
        return want
    if device_kind() == "cuda":
        try:
            import unsloth  # noqa: F401
            return "unsloth"
        except ImportError:
            pass
    return "peft"


def run_dir(cfg: dict, condition: str, model_name: str, seed: int, n_pairs: int) -> Path:
    short = model_name.split("/")[-1]
    return Path(cfg["paths"]["output_dir"]) / "runs" / f"{condition}__{short}__n{n_pairs}__s{seed}"


def load_pairs(cfg: dict, split: str, condition: str, n: int | None, scaling: bool = False):
    """Pairs in TRL's conversational format, so the policy's own chat template is applied
    exactly as at inference time."""
    from datasets import Dataset

    suffix = ".scaling" if scaling else ""
    path = Path(cfg["paths"]["data_dir"]) / "pairs" / split / f"{condition}{suffix}.jsonl"
    rows = read_jsonl(path)
    if n is not None:
        if len(rows) < n:
            raise ValueError(f"{path} has {len(rows)} pairs, {n} requested")
        rows = rows[:n]          # nested subsets: the first n rows, always the same ones
    return Dataset.from_list([{
        "prompt": [{"role": "user", "content": r["prompt"]}],
        "chosen": [{"role": "assistant", "content": r["chosen"]}],
        "rejected": [{"role": "assistant", "content": r["rejected"]}],
    } for r in rows])


def load_model(cfg: dict, model_name: str, backend: str):
    """Returns (model, tokenizer, peft_config). With unsloth the adapter is already
    attached, so peft_config is None; with peft, DPOTrainer attaches it."""
    lora = cfg["lora"]
    if backend == "unsloth":
        from unsloth import FastLanguageModel
        model, tok = FastLanguageModel.from_pretrained(
            model_name=model_name, max_seq_length=cfg["model"]["max_seq_length"],
            load_in_4bit=cfg["model"]["load_in_4bit"])
        model = FastLanguageModel.get_peft_model(
            model, r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
            target_modules=lora["target_modules"], bias=lora["bias"],
            use_rslora=lora["use_rslora"],
            use_gradient_checkpointing=lora["gradient_checkpointing"])
        return model, tok, None

    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = device_kind()
    # fp32 on MPS/CPU: half precision on MPS is not reliable for training.
    dtype = (torch.bfloat16 if dev == "cuda" and torch.cuda.is_bf16_supported()
             else torch.float16 if dev == "cuda" else torch.float32)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    peft_config = LoraConfig(
        r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"], bias=lora["bias"],
        use_rslora=lora["use_rslora"], task_type="CAUSAL_LM")
    return model, tok, peft_config


def train(cfg: dict, condition: str, model_name: str, seed: int, n_pairs: int,
          scaling: bool = False) -> Path:
    from trl import DPOConfig, DPOTrainer

    cond = cfg["conditions"][condition]
    if not cond.get("train", True):
        raise ValueError(f"{condition} is not a trained condition")
    if cond.get("dpo") == "disagreement_aware":
        raise NotImplementedError("disagreement-aware DPO loss is not implemented yet")

    set_seed(seed)
    out = run_dir(cfg, condition, model_name, seed, n_pairs)
    save_snapshot(cfg | {"run": {"condition": condition, "model": model_name, "seed": seed,
                                 "n_pairs": n_pairs}}, out)
    train_ds = load_pairs(cfg, "train", condition, n_pairs, scaling)
    dev_path = Path(cfg["paths"]["data_dir"]) / "pairs" / "dev" / f"{condition}.jsonl"
    eval_ds = load_pairs(cfg, "dev", condition, None) if dev_path.exists() else None

    backend = resolve_backend(cfg)
    if backend == "unsloth":
        from unsloth import PatchDPOTrainer
        PatchDPOTrainer()
    model, tok, peft_config = load_model(cfg, model_name, backend)

    t, d, ck, dg = cfg["training"], cfg["dpo"], cfg["checkpointing"], cfg["diagnostics"]
    dev = device_kind()
    steps = math.ceil(len(train_ds) / (t["per_device_train_batch_size"]
                                       * t["gradient_accumulation_steps"])) * t["num_train_epochs"]
    precision = {}
    if dev == "cuda":
        import torch
        bf16 = torch.cuda.is_bf16_supported()
        precision = {"bf16": bf16, "fp16": not bf16}
    else:
        precision = {"bf16": False, "fp16": False}
    optim = t["optim"] if dev == "cuda" else "adamw_torch"
    use_eval = eval_ds is not None and dg.get("eval_steps")

    args = DPOConfig(
        output_dir=str(out),
        seed=seed,
        beta=d["beta"],
        loss_type=[d["loss_type"]],
        label_smoothing=d["label_smoothing"],
        max_length=d["max_length"],
        truncation_mode=d["truncation_mode"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_steps=math.ceil(t["warmup_ratio"] * steps),
        weight_decay=t["weight_decay"],
        max_grad_norm=t["max_grad_norm"],
        optim=optim,
        gradient_checkpointing=True,
        logging_steps=dg["logging_steps"],
        save_steps=ck["save_steps"],
        save_total_limit=ck["save_total_limit"],
        eval_strategy="steps" if use_eval else "no",
        eval_steps=dg.get("eval_steps") if use_eval else None,
        report_to=cfg["logging"]["report_to"] if dev == "cuda" else "none",
        dataloader_pin_memory=dev == "cuda",
        **precision,
    )
    trainer = DPOTrainer(model=model, args=args, train_dataset=train_ds,
                         eval_dataset=eval_ds if use_eval else None,
                         processing_class=tok, peft_config=peft_config)
    resume = ck["resume_if_possible"] and any(out.glob("checkpoint-*"))
    log.info("training %s on %s (%d pairs, %d steps, backend=%s, device=%s)",
             model_name, condition, len(train_ds), steps, backend, dev)
    trainer.train(resume_from_checkpoint=True if resume else None)

    adapter = out / "adapter"
    trainer.model.save_pretrained(adapter)        # LoRA weights only (a few MB)
    tok.save_pretrained(adapter)
    (out / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=2))
    return adapter
