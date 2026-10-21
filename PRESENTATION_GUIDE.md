# Presentation guide: PrefAgent progress review

What to show, what to say, and where every number comes from. Every figure in this
guide was recomputed from the repository's own files on 22 October 2026. The
"Where every number comes from" section says how to reproduce each one.

---

## 1. The project in two sentences

When AI models label preference data instead of people (RLAIF), labelling pipelines
mix several ingredients, and nobody knows which one makes the labels good. This
project separates two of them, **candidate diversity** (four personas answering
each prompt) and **deliberation** (a panel of three judges that scores, reads each
other's verdicts and scores again), in a 2×2 design at equal compute, and measures
the effect on a small model trained with DPO.

---

## 2. Before you present (10 minutes)

```bash
cd multiagent
source .venv/bin/activate            # Windows: .\.venv\Scripts\activate
python -m pytest -q                  # expect: 63 passed
python scripts/08_build_explorer.py  # rebuilds the explorer from the data
open outputs/explorer/index.html     # Windows: start outputs\explorer\index.html
```

- The explorer is one HTML file and needs no server. Without internet it still
  works; only the typeface falls back to a system font.
- Backup copy of the explorer (private; share it from its Share menu first if
  someone else must open it): https://claude.ai/artifact/P87STggtLGSWiuo3znJNko
- Paper: `paper/ieee_paper.pdf` (6 pages).
- Optional live command (about 30 seconds, CPU only, no API calls):
  `python scripts/06_explain_preferences.py --input data/prompts/train.jsonl`

---

## 3. Talk track (about 6 minutes), tab by tab in the explorer

### Overview tab (1 min)

> "Aligning a language model with DPO needs preference pairs: a prompt, a better
> answer and a worse one. Getting those from people is the most expensive part.
> AI judges can label them instead, but multi-agent labelling pipelines mix
> several ideas together. My project asks which ingredient actually matters:
> having more diverse answers to choose from, or having judges who debate."

Point at the pipeline strip:

> "Data preparation is done. Labelling has run as a pilot. Training and the final
> evaluation are written but not run yet; they need GPU time."

Point at the funnel:

> "I started from 91,038 raw HH-RLHF conversations. After keeping single-turn,
> valid, length-limited, deduplicated ones, 24,297 remained. I sampled 3,200 into
> balanced splits, plus 500 GSM8K math problems. A sentence-embedding check removed
> 45 prompts that were too similar to evaluation prompts, so test data can't leak
> into training."

### Label walkthrough tab (1.5 min)

It opens on **"Where is C.E. Byrd High School?"**

> "Here is one real prompt from the labelling pilot. One model answers in four
> personas: concise, step-by-step, domain expert, careful sceptic. Three judges
> score every answer for helpfulness, correctness and harmlessness."

Click **Round 1 · independent**, then **Round 2 · after debate**:

> "In round 1, GPT-OSS-120B preferred answer A. After reading the other judges'
> verdicts, it switched to D. The final verdict is unanimous: D gets 5 out of 5.
> But every one of these four addresses is made up. The real school is at 3201
> Line Avenue, Shreveport, Louisiana. So a unanimous panel is not the same as a
> correct panel, and debate can make judges converge on a wrong answer. That is
> why the study also uses GSM8K, where answers can be checked against the truth."

Then click **"What should you do if you see a bear?"**:

> "Here the three judges each named a different best answer, in both rounds.
> Panel agreement is 0.5. That disagreement is information, and I use it below."

### Agents vs. humans tab (1.5 min): the main result

> "To measure label quality, I gave the panel 50 pairs that humans had already
> labelled, in shuffled order, and checked whether it picks the same answer."

- Panel agrees with humans on **68.0%** of pairs (95% CI 56–80%).
- When the panel was **unanimous: 80.6%** (31 pairs).
- When the panel was **split: 46.2%** (13 pairs), about chance.
- Judges picked the first-shown answer 48.7% of the time (50% = no position bias).

> "The key finding: when the judges agree, the label is right four times out of
> five. When they disagree, it's a coin flip. That supports the second idea in my
> project, disagreement-aware DPO, which trusts a training pair less when the
> judges disagreed."

On the per-judge table:

> "Debate changed each judge only slightly: plus 4, minus 2 and plus 2 points. With
> 50 pairs I can't call that an effect yet. The full check uses 500 pairs."

Click **Disagreements** and open one pair to show both answers and the judges'
reasons.

### Reward model tab (1 min)

> "Separately, I trained a small interpretable reward model on 2,000 human-labelled
> pairs to see what human annotators actually reward."

- One model on all pairs: **56.5%** held-out accuracy, against **51.8%** for
  "the longer answer wins".
- One model per prompt type: **67.2%** on helpfulness prompts, **62.9%** on
  harmlessness prompts.

> "The single model is weak because the two prompt types reward opposite things.
> On helpfulness prompts, longer answers win and refusals lose. On harmlessness
> prompts, shorter answers and refusals win. 'The longer answer wins' is right
> only 40.6% of the time there. So AI judges have to be checked on each prompt
> type separately."

### Try the reward model tab (1 min, live)

Click **Harmful request**: the refusal (A) is preferred with P = 84.4%, entirely
because of the refusal feature (both answers are 32 words). Switch **Prompt type**
to *Helpfulness prompt* and the same refusal drops to P = 17.9%. Invite the audience to type their own answers.

> "This live version uses only five readable features, so treat it as a picture of
> what people tend to reward, not as a judge."

### Close (30 s)

> "Done: data pipeline, labelling pipeline with a pilot, a label-quality check
> against humans, and the reward-model analysis, with 63 automated tests. Next:
> label at full scale, run DPO training on a GPU, and evaluate with an independent
> jury of model families that never labelled any data."

---

## 4. What is done and what is not

Say this plainly if asked. Do not claim more.

| Part | Status | Evidence in the repo |
|---|---|---|
| Prompt preparation and decontamination | Done | `data/prompts/*.jsonl`, `data/prompts/stats.json` |
| Candidate generation (4 personas) | Pilot: 22 prompts | `data/candidates/train/multi_persona.jsonl` |
| Panel judging (3 judges × 2 rounds) | Pilot: 22 prompts → 15 pairs | `data/judgements/train/multi_persona__panel.jsonl` |
| Preference pair files | Built (15 agent pairs, 2,000 human pairs) | `data/pairs/train/` |
| Label agreement with humans | Pilot: 50 of 500 planned pairs | `outputs/label_agreement/agreement.json` |
| Reward model + SHAP | Done on 2,000 human pairs | `outputs/explain/reward_model.json` and plots |
| DPO training | **Code written, not run** (needs a GPU) | `scripts/03_train.py`; no `outputs/runs/` yet |
| Win-rate evaluation | **Code written, not run**; base-model responses generated for 100 prompts | `scripts/04_evaluate.py`, `outputs/eval/demo/responses/base.jsonl` |
| Disagreement-aware DPO loss | Designed (paper, Eq. 3), not yet wired into the trainer | `paper/ieee_paper.tex` |
| Tests | 63 pass, no network or API keys needed | `python -m pytest -q` |

---

## 5. Where every number comes from

| Number | Value | Source file | Reproduce with |
|---|---|---|---|
| Raw HH-RLHF conversations | 91,038 | `data/prompts/stats.json` (sum of `*:raw`) | `scripts/01_prepare_data.py` |
| After single-turn, length, dedup filters | 24,297 | `stats.json` (sum of `*:deduped`) | same |
| Near-duplicates of eval prompts removed | 45 of 6,600 | `stats.json` (`decontam:*`) | same |
| Final splits | train 2,000 · dev 200 · eval 500 · label agreement 500 · GSM8K 500 | `stats.json` (`counts`) | same |
| Panel vs. human agreement | 68.0% (CI 56–80%), n = 50, 6 ties | `outputs/label_agreement/agreement.json` | `scripts/07_label_agreement.py` |
| Unanimous / split agreement | 80.6% (n = 31) / 46.2% (n = 13) | same | same |
| First-shown pick rate | 48.7% of 300 judge calls | same | same |
| Per-judge round 1 → 2 | gpt-oss-120b 70→74%, gpt-oss-20b 70→68%, qwen3.8-27b 64→66% | same | same |
| By prompt type | harmless 69.6% (n = 23), helpful 66.7% (n = 27) | same | same |
| Reward model, all pairs | 56.5% (CI 54.3–58.7%) vs. 51.8% length baseline | `outputs/explain/reward_model.json` | `scripts/06_explain_preferences.py --input data/prompts/train.jsonl` |
| Reward model, text features only | 52.7% (CI 50.5–54.8%) | — | add `--embedding-components 0` |
| Per subset | helpful 67.2% (CI 64.3–70.1%), harmless 62.9% (CI 60.0–65.8%) | `reward_model.json` (`by_subset`) | same as above |
| "Longer wins" on harmless prompts | 40.6% | `reward_model.json` | same |
| Pilot pairs that were unanimous | 10 of 15 | pilot judgements file | explorer overview |
| Tests | 63 passed | — | `python -m pytest -q` |

The reward-model results are deterministic (fixed seed). A fresh run on
22 October 2026 reproduced `reward_model.json` exactly. The agreement numbers come
from real API calls to free-tier Groq models (openai/gpt-oss-120b,
openai/gpt-oss-20b, qwen/qwen3.8-27b). Re-running `07_label_agreement.py` re-reads
the stored judgements and makes no new calls.

The C.E. Byrd fact check: [Wikipedia](https://en.wikipedia.org/wiki/C._E._Byrd_High_School)
and the [NNLM directory](https://nnlm.gov/membership/directory/2112) both give
3201 Line Avenue, Shreveport, Louisiana.

---

## 6. Questions you may be asked

**Why only 50 agreement pairs?** The judges run on free-tier APIs capped at about
8,000 tokens per minute and 1,000 requests per day per model. Each pair takes 6
judge calls (2 per judge); the 50 pairs took about 25 minutes. 500 pairs means
about 1,000 calls per judge model, roughly one full day of free quota. The full check uses 500
pairs. The confidence interval (56–80%) shows how uncertain 50 pairs are.

**Is 68% good?** Humans don't fully agree with each other on HH-RLHF either; it is
known to be a noisy dataset. The more useful result is the gap between unanimous
(80.6%) and split (46.2%) verdicts. It tells us which AI labels to trust.

**Why not just use one big judge?** That is one cell of the 2×2 design: one judge
sampled six times, against three judges over two rounds. Both make six calls per
prompt, so a panel advantage can't come from extra compute.

**What is DPO?** Direct Preference Optimization trains a model directly on
(better, worse) answer pairs. Unlike RLHF it needs no separate reward model or
reinforcement-learning loop.

**What is a Bradley–Terry model?** It gives each answer a score and turns the score
difference between two answers into the probability that one is preferred. Here
the score is a weighted sum of readable features, so the weights show what is
rewarded.

**What is SHAP?** A standard way to split a model's prediction into each feature's
contribution. For a linear model like this one, it reduces to weight × feature
difference.

**What are the "embedding_pc" features?** The main directions of variation in a
sentence-embedding model's representation of each answer. They capture meaning,
not surface features. The report names each one by its most extreme example
answers. For example, PC8 runs from eager compliance ("Sure, here's a list of
profanities") to cautious answers, and the model penalises the compliant end.

**Why are some pilot prompts "no pair"?** Either too few answers survived (cut-off
answers are dropped) or the judges scored every answer the same, so there is no
preference to learn from. Both cases are excluded rather than guessed.

**Why hasn't the model been trained yet?** Training needs a GPU, and the labelled
data has to come first. Everything is designed to run on a free Colab T4.

**Could the judges be biased toward the first answer?** Each judge sees a shuffled
order, and the shuffle is recorded. The first-shown answer was picked 48.7% of the
time, which is close to the 50% you'd expect with no bias.

---

## 7. Things not to say

- Don't say the model has been trained or evaluated. It hasn't.
- Don't say deliberation helps or hurts. The 50-pair differences (−2 to +4
  points) are too small to tell.
- Don't call 68% "accuracy against ground truth". Human labels are noisy, so it is
  agreement with human annotators.
- Don't present the live scorer as a judge. It is a five-feature illustration.

---

## 8. Files to know

| File | What it is |
|---|---|
| `outputs/explorer/index.html` | The interactive explorer (rebuild: `scripts/08_build_explorer.py`) |
| `paper/ieee_paper.pdf` | Paper draft; Sections IX and X hold today's results |
| `outputs/explain/*.png` | Reward-model plots: SHAP summary, coefficients, subset comparison |
| `outputs/label_agreement/agreement.json` | Full agreement report |
| `configs/` | All settings: models, personas, judging, splits |
| `scripts/01`–`08` | The pipeline, one numbered script per stage |
