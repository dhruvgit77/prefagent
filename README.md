# Deliberation or Diversity? Multi-Agent AI Feedback for DPO on Small LMs

Research code for studying **what makes multi-agent AI feedback effective** as a
replacement for human preference labels when aligning small language models
with Direct Preference Optimization (DPO) — at $0 compute cost.

## Research questions

1. **Label quality** — How accurately do agent-generated preference labels match
   human labels (HH-RLHF) and verifiable ground truth (GSM8K), and what biases
   (length, position, hedging) do they carry?
2. **Mechanism** — Is the benefit of a multi-agent pipeline driven by *candidate
   diversity* (several personas answering) or by *deliberation* (a panel of
   judges debating), at matched compute? (2×2 factorial)
3. **Downstream effect** — How does DPO on agent labels compare with DPO on
   human labels, across model families, sizes and data scales?
4. **Method** — Does using panel disagreement as per-pair label uncertainty
   (disagreement-aware DPO) close the gap to human labels?

## Layout

```
configs/          All experiment parameters (YAML). Code never hard-codes them.
  base.yaml         paths, seeds, Hub settings
  data.yaml         seed-prompt sources, filters, splits
  agents.yaml       LLM providers, personas, judges, rubric
  train_dpo.yaml    LoRA + DPO hyperparameters
  eval.yaml         win-rate jury, benchmarks, statistics
  experiments.yaml  the experiment matrix (conditions × models × seeds)
src/prefagent/    Library code (config, LLM client, agents, data, train, eval, explain)
scripts/          Numbered CLI entry points, one per pipeline stage
notebooks/        Thin Colab notebooks that call scripts/ for GPU stages
tests/            Unit tests for parsing, schema and statistics
data/             Generated datasets (git-ignored)
outputs/          Checkpoints, logs, results (git-ignored)
docs/             Original project documents (research report, PRD, TRD, guide)
```

## Where things run

| Stage | Where | Why |
|---|---|---|
| Prompt prep, agent debate, labelling | Local machine (CPU) | Only API calls; no GPU needed |
| DPO training, generation, benchmarks | Colab / Kaggle free T4 | Needs CUDA (bitsandbytes, Unsloth) |
| Statistics, reward model, SHAP | Either | CPU is enough |

## Setup

Local (Python 3.11+):

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-agents.txt
cp .env.example .env   # then fill in your free API keys
```

Colab: see `notebooks/` (installs `requirements-train.txt`).
