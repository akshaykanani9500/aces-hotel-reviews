# Review of the original ACES repository

Source: https://github.com/mycustomai/ACES (commit `6c561d8`, "headless choice model", Dec 2025), linked from
section "Code and data availability" of Allouah et al. (2025), arXiv:2508.02630v3.

## What ACES does

| Part | Folder | How it works |
|---|---|---|
| Mock marketplace | `sandbox/` | FastAPI + Jinja page that looks like an Amazon search results grid: 8 products in a 2×4 grid, each with image, title, stars, rating count, price and optional tags (Sponsored, Overall Pick, Low stock …). **No product pages and no review text**, only star rating and review count. |
| Agent | `agent/` | LangChain wrapper (`LMMAgent`) for OpenAI, Anthropic, Gemini and AWS Bedrock. One tool, `add_to_cart(product_title, price, rating, number_of_reviews)`. The agent sees **one screenshot** of the grid and must call the tool, so it is a one-shot choice, not real browsing. |
| Experiments | `experiments/` | Datasets are downloaded from Hugging Face (ACE-BB, ACE-RS, ACE-SR) with **pre-captured screenshots**. Runtimes: `screenshot` (default), `batch` (provider batch APIs), `simple` (live Selenium + Chrome against the sandbox), `headless` (product list as JSON text). |
| Randomisation | `experiments/data_generation.py` | Per experiment: shuffle positions; perturb price (log-normal), rating (±0.4 toward 5) and rating count; random Sponsored / Overall Pick / Low-stock tags. Seeds = experiment number. |
| Analysis | `experiments/analysis/` | `market-share`, `rationality-suite`, and `choice-model`: a conditional logit (statsmodels `ConditionalLogit`) with position dummies, log price, log rating count, rating, tag dummies and product fixed effects `C(title)`, estimated per model. |

## What we reused (and where)

| ACES idea | Hotel Lab |
|---|---|
| FastAPI + Jinja mock store you fully control | `hotel_lab/site/` (results page **plus a room page with reviews**) |
| Randomise positions every trial, log the seed | `hotel_lab/design.py` (seed per trial, from `global_seed` + trial id) |
| Screenshot or headless JSON interface | `hotel_lab/runner.py` `--interface text|screenshot` |
| Generic, hypothesis-neutral task prompt ("helping someone find a good …") | `hotel_lab/config/prompts/main.txt` (same wording style) |
| Conditional logit with product fixed effects and position dummies, per model | `hotel_lab/analysis.py` (plus item-clustered SEs and BH) |
| Model configs as one YAML per model | `hotel_lab/config/models/*.yaml` |

## Why we did not just edit the ACES pipeline

1. **It has no reviews.** ACES manipulates price, rating, rating count and tags on a grid. Our study is about
   review *text*, so we needed product (room) pages, review lists, style variants, balance sets and summaries.
   That is a new stimulus model, not a tweak.
2. **Closed models only.** The agent supports OpenAI, Anthropic, Gemini and Bedrock. A `huggingface` engine type
   is declared but `LMMAgent` raises "Unsupported engine_type" for it. There is no local or open-weight path.
   Our brief requires open-weight or free models.
3. **Heavy cloud dependencies.** The dependencies include Google Cloud Storage, Vertex AI, AWS Bedrock and the
   provider batch APIs. The default flow downloads screenshots from Hugging Face. That is a lot to install on a
   student laptop, and none of it is needed for a local sandbox.
4. **Python ≥ 3.12 and `uv`** are required. Hotel Lab runs on Python 3.10–3.13 with plain `pip`.

So we kept ACES **intact with its full git history** (so you can still run and compare it) and built
`hotel_lab/` next to it. We made one small opt-in patch so ACES itself can also use open-weight models:

- `agent/src/core/mllm.py` and `experiments/runners/headless_runtime/base.py` now pass `base_url` to the
  OpenAI client. With `engine_type: openai` and `base_url: http://localhost:11434/v1`, ACES talks to Ollama.
  Example: `config/models/_ollama-qwen2.5-7b.yaml.example`.

## Issues found in the original code (worth knowing, some worth citing as limitations)

| # | Where | Issue | What Hotel Lab does instead |
|---|---|---|---|
| 1 | `headless_runtime/base.py::_parse_response` | If the reply cannot be parsed and no "Product N" is found, it **records product 1 as the choice** ("Parse failed, used first product"). That silently adds a position-1 bias, and the biggest effect in the paper is position. | Parse failures are logged as failures (`parse_ok: false`) and excluded from the choice model. Failure rates are reported per cell. |
| 2 | `headless_runtime/base.py::_create_prompt` | The headless prompt is hard-coded to include "Do not be influenced by the position of the product" and "The user is very price sensitive". These are robustness variants in the paper, not a neutral prompt. | Prompts are separate text files: one neutral main prompt and two paraphrases. |
| 3 | `analysis/choice_model*.py` | Products **never chosen by a model are dropped** before estimation, and the `usb_cable` category is dropped by name. These choices change estimates and should be reported. | Nothing is dropped by default. If a room is never chosen by a model, its fixed effect runs off to a very large negative number. The estimator caps the iterations and uses a pseudo-inverse, so the other coefficients are still estimated, but report such rooms. |
| 4 | `analysis/choice_model*.py` | Standard errors are not clustered by item or category. | Cluster-robust (sandwich) SEs by item, computed directly. |
| 5 | `sandbox/app.py` | Paths are relative to the working directory (`sandbox/templates`), so the server only works when started from the repo root. | Absolute paths from the package folder. |
| 6 | repo root | The README says "MIT License – see LICENSE", but **there is no LICENSE file**. | Keep attribution; cite the paper. If you plan to publish the code, ask the authors to add the file. |
| 7 | `experiments/runners/simple_runtime.py` | Live browsing uses Selenium + headless Chrome on port 5000, and one screenshot of the first viewport (1280×1080). Products below the fold may not be seen. | Full-page screenshots of every page, in display order, sent together. |

None of these make the paper wrong. Most matter for the headless and analysis code paths. They are good
examples for your limitations section and the "Dillion et al." discussion.

## Running the original ACES (optional)

```bash
pip install uv            # or see README_ACES.md
uv sync --all-groups --all-packages
cp .env.sample .env       # add API keys for the closed models you want
uv run run.py --hf-dataset rs --subset rating --include gpt-4o
```
For open-weight models via Ollama: rename `config/models/_ollama-qwen2.5-7b.yaml.example` to
`ollama-qwen2.5-7b.yaml` and use the headless runtime with a local dataset CSV
(see `experiments/data_generation.py`).
