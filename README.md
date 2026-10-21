# Deliberation or Diversity? Multi-Agent AI Feedback for DPO on Small LMs

Research code for studying **what makes multi-agent AI feedback effective** as a
replacement for human preference labels when aligning small language models
with Direct Preference Optimization (DPO) — at $0 compute cost.

> **Presenting this project?** See [PRESENTATION_GUIDE.md](PRESENTATION_GUIDE.md) for
> the demo script, every result with its source file, what is and isn't done yet,
> and likely questions.

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

## Pipeline stages

```bash
# 1. Prompt splits (CPU, minutes): data/prompts/*.jsonl + stats.json
python scripts/01_prepare_data.py

# 2. Agent labelling (API calls, resumable — re-run the same command after any stop)
python scripts/02_label.py budget              # calls and days per model, spends nothing
python scripts/02_label.py pilot --limit 20    # small run + printed spot-check
python scripts/02_label.py run                 # full labelling plan
python scripts/02_label.py pairs --split train # data/pairs/train/<condition>.jsonl
python scripts/02_label.py pairs --split dev

# 3. Reward model + SHAP explanation (CPU; no LLM API calls)
python scripts/06_explain_preferences.py --input data/prompts/train.jsonl   # ~30 s

# 4. Label quality vs. human labels (judge calls use the API; the report does not)
python scripts/02_label.py judge --split label_agreement --source human_pair --limit 50
python scripts/07_label_agreement.py
```

Run tests with `pytest` (no network or API keys needed).

## Evaluator demo (Windows)

This is a research pipeline, not a web application. The recommended live demo is
the CPU prompt-preparation stage: it downloads the public HH-RLHF dataset and
tokenizer/embedding models, filters and splits prompts, and writes inspectable
JSONL files. It does not call an LLM API or require API keys.

From PowerShell in the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-agents.txt
.\.venv\Scripts\python.exe scripts\01_prepare_data.py splits.train_prompts=20 splits.dev_pairs=10 splits.eval_prompts=20 splits.label_agreement_pairs=20 splits.gsm8k_label_problems=0 gsm8k.enabled=false
```

The small counts keep the live run manageable; the first run still downloads
the configured source data and models and therefore needs internet access. The
generated files are under the git-ignored `data/prompts/` directory:

| File | What to show |
|---|---|
| `train.jsonl` | Training prompts, with the source human-preferred and rejected responses |
| `dev.jsonl` | Separate development prompts |
| `eval.jsonl` | Held-out evaluation prompts |
| `label_agreement.jsonl` | Held-out prompts with human labels for later agreement checks |
| `stats.json` | Split counts and filtering/decontamination statistics |

With these settings the command writes 20 train, 10 dev, 20 eval, and 20
label-agreement prompts, each split balanced across the `helpful-base` and
`harmless-base` subsets. Each JSON record holds the prompt, subset, token count,
and the two human-labeled responses. These counts are a small pipeline check,
**not research results**.

To run the tests in the same environment:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

### Pipeline at a glance

```text
HH-RLHF (+ optional GSM8K)
          |
          v
Prompt preparation and decontamination
          |
          v
Candidate generation --> Judge / panel deliberation
          |                         |
          +----------> Preference pairs
                          |           |
                          |           +--> Reward model + SHAP explanations
                          v
                         DPO fine-tuning
                                |
                                v
                    Win-rate and statistical evaluation
```

### Added evaluator features: reward model and SHAP

The implementation guide calls for an auxiliary reward model and an explanation
of which response features influence its preference predictions. Run this after
prompt preparation:

```powershell
.\.venv\Scripts\python.exe scripts\06_explain_preferences.py --input data/prompts/train.jsonl
```

If `data/pairs/train/multi_panel.jsonl` exists, the script uses those agent-labeled
pairs; otherwise, it uses the prepared HH-RLHF human-labeled pairs in
`data/prompts/train.jsonl` (the report's `label_source` says which, or `mixed:…`).
Pass `--input` explicitly when you want a specific label source.

- **Model:** a pairwise Bradley–Terry reward model, i.e. logistic regression on the
  standardized feature *difference* φ(chosen) − φ(rejected), so it is trained on the
  same pairwise objective it is evaluated on.
- **Features:** named text signals (`log_word_count`, `hedge_count`,
  `refusal_count`, `question_count`, `list_item_count`) plus 8 PCA components of a
  MiniLM sentence embedding (`--embedding-components`, 0 for text-only). PCA is fit
  on training folds only. `embedding_pc_examples` in the report lists the responses
  at each end of every component so the axes can be named.
- **Evaluation:** 5-fold cross-validated pairwise accuracy with a bootstrap 95% CI,
  compared against a "longer response wins" baseline.
- **Outputs:** `outputs/explain/reward_model.json`, `shap_summary.png`, and
  `coefficients.png`.

When the input records carry a `subset` (HH-RLHF does), a model is also fitted per
subset and `subset_coefficients.png` compares their text-feature weights.

| Pairs (HH-RLHF human labels) | n | CV accuracy [95% CI] | "Longer wins" |
|---|---|---|---|
| All, text features only | 2000 | 52.7% [50.5, 54.8] | 51.8% |
| All, text + 8 PCs | 2000 | 56.5% [54.3, 58.7] | 51.8% |
| `helpful-base` | 1000 | 67.2% [64.3, 70.1] | 63.0% |
| `harmless-base` | 1000 | 62.9% [60.0, 65.8] | 40.6% |

The pooled model is weak because the subsets reward opposite things: longer
answers win and refusals lose in `helpful-base`, while shorter answers and refusals
win in `harmless-base`. SHAP and coefficients describe this model, not causal
effects on human preference.

### Label quality: agent judges vs. human labels

`scripts/07_label_agreement.py` reads the `label_agreement` judgements (each is an
HH-RLHF human pair shown to the judges in shuffled order) and reports agreement
with the human label with a bootstrap CI, by subset, by panel confidence
(unanimous vs. split), per judge and round (before vs. after deliberation), and the
rate at which judges pick the first-shown response (position bias). Output:
`outputs/label_agreement/agreement.json`.

### Interactive explorer

`scripts/08_build_explorer.py` turns the real data and outputs into one
self-contained page, `outputs/explorer/index.html`, that opens offline in any
browser. It has five views: an overview with pipeline status and the data funnel;
a label walkthrough showing each prompt's four persona answers and every judge's
round 1 and round 2 scores and reasons; agents vs. humans agreement with every pair
browsable; the reward-model results; and a live scorer where you type two answers
and see which one the text-feature reward model expects people to prefer, and why.

```powershell
.\.venv\Scripts\python.exe scripts\08_build_explorer.py
start outputs\explorer\index.html
```

### What is ready and what remains

- **Demonstrable now:** prompt preparation, saved split files and filtering
  statistics, plus the CPU reward-model and SHAP workflow; the unit tests run
  without network access or API keys.
- **Not included in the smoke test:** agent-generated responses, judge labels,
  DPO training, or win-rate results. Do not present the prompt counts or the
  exploratory reward-model output as evidence for the research hypotheses.
- **Agent labeling:** requires provider API credentials in a local `.env`.
  `python scripts/02_label.py budget` prints the configured call estimate
  without making calls. The `pilot` and `run` commands make API calls and may
  consume provider quota or incur charges.
- **Training and evaluation:** the training dependencies and a compatible GPU
  environment are needed for the full DPO pipeline; see
  `requirements-train.txt` and the project configuration before attempting it.
- **Reproducibility:** configuration is stored under `configs/`; generated
  datasets and outputs are intentionally git-ignored.

### Suggested evaluator presentation (5–7 minutes)

1. **Problem:** human preference labels are costly; the project studies whether
   multi-agent feedback can provide useful labels for aligning small language
   models.
2. **Research questions:** label quality, the roles of candidate diversity and
   judge deliberation, downstream DPO effects, and disagreement-aware training.
3. **Method:** walk through the pipeline diagram and explain that conditions
   are compared using shared prompt splits.
4. **Live demo:** run the small prompt-preparation command, show the split
   counts in `data/prompts/stats.json`, and open one record in `train.jsonl`.
5. **Status and limitations:** distinguish implemented pipeline stages from
   work not run in the demo; explain API-key and GPU requirements.
6. **Explainability demo:** run the reward-model command and show the
   cross-validated accuracy and CI against the length baseline, then the SHAP and
   coefficient plots.
7. **Next steps:** generate and judge a larger labeled pilot, manually inspect
   the preference pairs, then run training and evaluation in a GPU environment.
