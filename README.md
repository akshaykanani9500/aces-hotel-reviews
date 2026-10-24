# Hotel Lab: how AI agents read hotel-room reviews

This repo is a research test bench. It checks whether an AI booking agent picks a hotel room because
of **what the reviews say** or because of **how the reviews are shown**: their order, emojis,
length, tone, the balance of praise and complaints, and whether the agent sees a summary or the
full reviews.

It is built **on top of ACES** (Agentic e-CommercE Simulator) by Allouah, Besbes, Figueroa, Kanoria and Kumar
([paper, arXiv:2508.02630](https://arxiv.org/abs/2508.02630) · [original code](https://github.com/mycustomai/ACES)).
The original ACES code is kept as it was, with its full git history (see `README_ACES.md`). Our work lives in
`hotel_lab/`, plus a small opt-in patch that lets the original ACES pipeline call open-weight
models through Ollama.

> Project: SPJIMR live project, Group 1, "How AI shopping agents read product reviews" (service version: hotel rooms).

---

## What it does

1. **A mock hotel booking site, run on your computer** (`StayCompare`, entirely fictional). It has a
   search results page with 4 comparable rooms, and a room page for each with amenities, policies,
   price, rating and guest reviews, or a platform-written review summary.
2. **Hidden fact sheets.** Each room has 3 hidden facts, for example cleanliness=good,
   noise=poor, wifi=average. The facts are never shown to the agent; all reviews are written from
   them. One room is best on the facts by a clear but moderate margin, so we can measure accuracy.
3. **One change at a time.** The planner changes one factor per block and randomises everything else:
   room positions are shuffled in every trial, and each trial gets its own seed.
4. **Any open-weight model.** Models run through Ollama, LM Studio, vLLM, OpenRouter or the Hugging Face
   router, with one OpenAI-compatible client. In **text mode** the agent gets the page content as JSON
   (the "headless" mode). In **screenshot mode** it gets real screenshots of the pages (vision models).
5. **Logs everything.** Each trial records the model, version or digest, quantisation, seed, full prompt,
   raw output, parsed choice, tokens and errors. Logs are append-only. A reply that can't be parsed is
   logged as a failure, never as a choice.
6. **Analysis.** It reports choice shares with 95% CIs, accuracy, and the surface-sensitivity index (the
   focal room's share minus its control share, with a cluster bootstrap). It also reports summary-vs-full
   agreement (Cohen's kappa) and a conditional logit with room fixed effects and position dummies.
   Standard errors are clustered by item, with Benjamini–Hochberg correction across tests.

## Quick start (about 5 minutes, no AI model needed)

Full step-by-step guide for VS Code on Windows or Mac: **[SETUP_VSCODE.md](SETUP_VSCODE.md)**

```bash
git clone https://github.com/akshaykanani9500/aces-hotel-reviews.git
cd aces-hotel-reviews
python -m venv .venv
# Windows:  .venv\Scripts\activate        Mac/Linux:  source .venv/bin/activate
pip install -r hotel_lab/requirements.txt

python -m hotel_lab validate      # check the stimulus files
python -m hotel_lab demo          # whole pipeline with a mock "model" (random picks)
python -m hotel_lab plan          # build the real trial plan
python -m hotel_lab serve         # open http://127.0.0.1:8765 and look at the pages
```

Then plug in a real open-weight model ([Ollama](https://ollama.com)):

```bash
ollama pull qwen2.5:7b
python -m hotel_lab hello --model ollama-qwen2.5-7b                 # week-1 "hello world" JSON test
python -m hotel_lab run --models ollama-qwen2.5-7b --limit 20       # small pilot
python -m hotel_lab export
python -m hotel_lab analyze                                         # results in runs/<experiment>/analysis/
```

## Commands

| Command | What it does |
|---|---|
| `validate` | Checks stimulus files: 4 rooms, ids, equal mean ratings across balance sets, every style variant present, best room strictly best |
| `checks` | Manipulation checks (word counts, emoji counts, readability) plus coder sheets for "same facts?" and summary fidelity |
| `plan` | Builds `runs/<experiment>/plan.jsonl` from `hotel_lab/config/experiment.yaml` and prints the call budget |
| `serve` | Runs the mock site locally. Open any trial to see exactly what the agent saw |
| `models` | Lists model configs in `hotel_lab/config/models/` |
| `hello --model X` | Tests whether a model returns valid JSON |
| `run --models A B [--limit N] [--blocks order] [--interface screenshot]` | Runs trials. Safe to stop and restart: finished trials are skipped |
| `export` | Raw logs → `trials.csv` (one row per trial) and `long.csv` (one row per room, for the logit) |
| `analyze` | Tables and figures in `runs/<experiment>/analysis/` |
| `summarize --model X` | Generates review summaries with one fixed summariser prompt, for coders to verify |
| `demo` | Plan, run (mock models), export and analyze in one go |

## How the brief maps to the code

| Brief | Where |
|---|---|
| Hidden fact sheet, one best product | `facts:` and `best_room_id` in `hotel_lab/stimuli/*.yaml` |
| 4 products × 8 reviews, several categories | 4 rooms × 8 reviews per **scenario** (one YAML file each). Two sample scenarios are included: Mumbai business and Goa couple |
| Order: newest / most helpful / random / most extreme | `order` block (also `best_first`, `worst_first`, `oldest`) |
| Style: emoji, length, tone | `style_*` blocks. Text variants live in each review's `text:` map. Combos like `emoji+long` are supported for the full 2×2×2 |
| Content balance at the same average rating | `review_sets: positive / mixed / complaints`. The validator enforces equal means |
| Full reviews vs platform-written summary | `format` block (`full`, `summary`, `both`) |
| Format × balance interaction | `format_x_balance` block (switch on in `experiment.yaml`) |
| Randomise positions, store the seed | Planner shuffles every trial. The seed comes from `global_seed` + trial id |
| Text main, screenshot for one block | `interface: text` or `--interface screenshot` |
| Temperature 1.0 main, 0.0 robustness | `temperature:` in each model YAML. Copy the file with `temperature: 0.0` for the check |
| One main prompt + two paraphrases | `hotel_lab/config/prompts/main.txt`, `para1.txt`, `para2.txt` → `prompt_ids` |
| Reasoning depth low/high | `reasoning_depth: low | high` (prompt-defined, token cap via `max_tokens`) |
| Parse failures < 5% | `analysis/1_failure_rates.csv` flags cells over 5% |
| Conditional logit, product FE, item clustering, BH | `hotel_lab/analysis.py` → `7_conditional_logit.csv` |
| Summary vs full: kappa + direction | `6_summary_vs_full.csv` |
| Manipulation checks, coder agreement | `python -m hotel_lab checks` → `runs/_checks/` |

### Two ways to assign a treatment

- **`whole_set`**: every room on the page gets the level, for example all reviews shown as summaries.
  Cells compare whole pages. Used for order and format.
- **`focal`**: one room (the focal room, rotated across items) gets the level and the others stay at control.
  We compare the focal room's share when treated with its share in the control cell. Used for style and
  balance. This is the "surface-sensitivity index".
- `per_room`: every room draws a level at random, as in the original ACES design.

## Folder layout

```
hotel_lab/
  stimuli/          scenario YAML files (rooms, hidden facts, review pool, variants, summaries) + SCHEMA.md
  config/
    experiment.yaml design: blocks, levels, repeats, prompts, interface
    models/         one YAML per model (Ollama, OpenRouter, HF router, mock)
    prompts/        prompt templates (main + 2 paraphrases, answer formats)
  site/             the mock booking website (FastAPI + Jinja templates)
  design.py         experiment.yaml -> reproducible trial plan
  conditions.py     applies a trial's levels -> exactly what the agent sees
  runner.py         calls models, parses replies, writes append-only logs
  analysis.py       shares, sensitivity, kappa, conditional logit, BH, figures
  tests/            python -m pytest hotel_lab
runs/               all outputs (git-ignored; archive them yourself)
agent/ sandbox/ experiments/ config/ run.py   original ACES code (see README_ACES.md)
docs/ACES_REVIEW.md  what the original repo does, what we reused, and issues we found
```

## Things to decide as a team (before pre-registration)

- **Fictional vs real hotels.** The samples use fictional hotels so brand knowledge can't steer the
  agent. The brief used real products. Pick one and justify it.
- **Room photos.** Every room uses the same neutral placeholder so photos can't bias the choice.
  Real photos would add realism but would also be a confound.
- **Sample stimuli are placeholders.** They show the format and pass the checks, but your team should
  write or edit the final reviews, run `checks` and do the two-coder pass.
- **Displayed rating is fixed per room**, and identical across rooms in the samples (4.2), so star ratings
  can't drive the choice. That is deliberate: the brief says "stars are a strong cue".
- **Freeze the model list in week 1.** Model names change quickly; check `ollama list` and record the digests.

## Credit and licence

ACES: Allouah, A., Besbes, O., Figueroa, J. D., Kanoria, Y., & Kumar, A. (2025). *What is your AI agent buying?
Evaluation, biases, model dependence, and emerging implications for agentic e-commerce.* arXiv:2508.02630.
The original README says ACES is MIT-licensed, but the repository has no LICENSE file. Please keep
this attribution and cite the paper in any write-up.
