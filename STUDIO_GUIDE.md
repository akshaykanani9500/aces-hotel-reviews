# Hotel Lab Studio: step-by-step guide

The Studio is a local website that runs the whole study:

- **Database:** the Google Sheet. You edit reviews, hypotheses and settings there.
- **Left panel:** pick a hypothesis (H1–H4, sub-hypotheses, checks). It shows H0/Ha, what changes and what is held constant.
- **Centre panel:** the hotel page exactly as the agent sees it. **Randomise** shows another real trial.
- **Right panel:** the AI models (local Ollama, OpenRouter, Groq, Hugging Face, OpenAI, Gemini, Anthropic, or a test agent).
- **Bottom panel:** run, watch progress, **Analyse all** → one verdict per hypothesis. Every trial is also written to the Sheet.

---

## Part A. One-time setup (about 20 minutes)

### A1. Get the code (VS Code)
Follow `SETUP_VSCODE.md` steps 1–4: clone, create `.venv`, run `pip install -r hotel_lab/requirements.txt`.
If you already cloned it, run `git pull` and the `pip install` line again.

### A2. Open the database Sheet
The Sheet was created in your Google Drive: **"Hotel Lab – Study Database (AI agents & hotel reviews)"**.
To make a fresh copy instead, start the Studio and click *Data source → Download starter workbook*, upload the file to
Drive, and open it with Google Sheets.

| Tab | What it holds | Who edits it |
|---|---|---|
| README | how the tabs work | – |
| Settings | experiment name, seed, sample size, equivalence margin, defaults, `api_token` | you |
| Hypotheses | one row per hypothesis; the website dropdown is built from this tab | you (carefully) |
| Scenarios | 6 trips; the room template (price, amenities, rating…) shared by all 4 rooms | you |
| Rooms | the hotel name of each room (names rotate during the experiment) | you |
| Facts | **hidden** ground truth: 3 attributes per room, good / average / poor | you |
| Reviews | 18 reviews per room; `text_base` plus `text_emoji`, `text_long`, `text_casual` variants | you |
| ReviewSets | which 8 reviews form base/mixed, positive, complaints | you |
| Summaries | platform-written summary per review set | you |
| Models | reference list only (the website reads models from `hotel_lab/config/models/`) | – |
| Results | one row per trial, **written by the website** | website |
| Verdicts | one row per hypothesis × model, **written by the website** | website |
| Runs | one row per run, **written by the website** | website |

### A3. Connect the Sheet to the website (Apps Script)
1. In the Sheet: **Extensions → Apps Script**.
2. Delete what is in `Code.gs`. Paste the whole file `hotel_lab/apps_script/Code.gs` from the repo. Click **Save** 💾.
3. **Deploy → New deployment →** gear icon **→ Web app**.
   - *Execute as:* **Me**
   - *Who has access:* **Anyone**
4. Click **Deploy**. Google asks for permission: choose your account → *Advanced* → *Go to … (unsafe)* → *Allow*.
   This screen appears because it's your own unpublished script.
5. Copy the **Web app URL**. It ends in `/exec`.
6. In the Sheet's **Settings** tab, copy the `api_token` value.

> **SPJIMR account note.** Your Sheet is in your SPJIMR Google Workspace. Some college workspaces don't allow
> "Anyone" for web apps, and offer only "Anyone within SPJIMR".
> If you can't pick **Anyone**, there are two ways round it:
> - **(a)** Make a copy of the Sheet in a personal Gmail account (File → Make a copy) and deploy the script there.
> - **(b)** Use the offline route in A5.

### A4. Start the Studio
```bash
python -m hotel_lab studio
```
Open **http://127.0.0.1:8800**. Click **Data source**, paste the URL and the token, then click **Save & test**.
The tab names and row counts should appear. Then click **Sync from Sheet**.

The top bar now shows `Data: Google Sheet snapshot <code>`. That code is a fingerprint of the exact data used, and it is stored
with every trial.

### A5. Offline alternative (no Apps Script)
In Google Sheets: **File → Download → Microsoft Excel (.xlsx)**. In the Studio: **Data source → Upload a downloaded .xlsx**.
Results are then saved locally only. To send them to the Sheet later, connect the Sheet and click **Send pending rows to Sheet**.

### A6. Add your models (right panel → ➕ Add a model)
- **Local and free:**
  1. Install Ollama.
  2. Run `ollama pull qwen2.5:7b`.
  3. Provider: *Ollama*. Model name: `qwen2.5:7b`. Short name: `qwen7b`.
- **Hosted open-weight models:** OpenRouter, Groq or Hugging Face. Paste your API key once. It is saved in your local `.env` file,
  **never** in the Sheet or on GitHub.
- **Closed models:** OpenAI, Gemini or Anthropic. Use these only if Prof. Jha allows them.
- **Test agent:** `mock-simulated` is a fake agent with planted preferences. Use it to check the pipeline before you spend money.

Click **Test** next to each model. A ✓ means it returned a valid JSON answer.

---

## Part B. Running the study (in this order)

The order follows the brief's gates. Don't move to the next step until the current one passes.

| Step | Pick in the dropdown | Mode | Pass condition |
|---|---|---|---|
| 1 | **V7 Placebo** with `mock` | Pilot 10/cell | PASS: the pipeline does not invent effects |
| 2 | **V1 Facts-only** | Pilot 10/cell, each real model | PASS (≥80%): the agent agrees with your "best room" |
| 3 | **V5 Dominance** | Pilot 10/cell | PASS (≥95%): the agent reads the page at all |
| 4 | **HB Baseline** | Pilot 20/cell | PASS: better than 25% chance, and it doesn't always pick one room |
| 5 | H1, H2a, H2b, H3a, H3b, H3c, H4 | Pilot 20/cell each (brief, Gate M2) | parse failures under 5% (verdicts say "Pilot") |
| 6 | **Pre-register** (OSF/AsPredicted) | – | lock Settings + Hypotheses; note the snapshot code |
| 7 | Each confirmatory hypothesis | **Full plan** | 192 trials per cell per model |
| 8 | H4b, H5 (secondary) | Full plan | – |
| 9 | **Analyse all** | – | verdict table + Verdicts tab |

**Robustness checks** (after the main run), done from the Settings tab:
- **Prompt paraphrases:** set `prompt_ids` to `main, para1, para2` and change `experiment_name`.
- **Temperature 0:** add a copy of a model with temperature 0.
- **Screenshots:** choose *Interface: Screenshots* with a vision model, for one block (e.g. H2b).

**Important:** change `experiment_name` in Settings for every new study (for example after editing reviews). Runs with
different names never mix.

---

## Part C. How each hypothesis isolates ONE effect

Everything that is not being tested is fixed or balanced, the same way for every hypothesis:

- **Identical rooms.** All 4 rooms show the same price, room type, amenities, policies, guest rating, review
  count and photo. Only the reviews differ.
- **Positions are balanced (Latin square).** Each room appears in each slot equally often, so "Option 1 bias" cancels out.
- **Names rotate.** Each hotel name is paired with every room profile, so a name can't carry an effect.
- **Same layout in every cell.** Treated and control trials share the same scenario, focal room, positions and
  review order, so they are compared like for like.
- **Other factors sit at their default:** order = random, style = base, review set = base, format = full.
- **Hidden facts never reach the agent.** Check the *Exact prompt* tab.

| Hypothesis | Only this changes | Compared against |
|---|---|---|
| H1 | every room shown as a summary instead of full reviews | full reviews, AND rerunning the identical page (the noise floor) |
| H2a | the sort rule for all rooms (newest / most helpful / most extreme) | random order |
| H2b | the focal room's reviews ordered best-first | worst-first (ordering: primacy + recency together) |
| H2c (secondary) | the focal room's best review first | random order (primacy alone) |
| H3a / b / c | the focal room's reviews with emoji / longer / casual (same facts, same stars) | plain reviews |
| H4 | focal room: 6 raves + 2 strong complaints | evenly mixed reviews, **same average rating** |
| H4b (secondary) | focal room: uniformly moderate praise | evenly mixed reviews |
| H5 (secondary) | complaint penalty under summaries | complaint penalty under full reviews |

## Part D. How to read the verdict table

- **Estimate.** In percentage points (pp). For example, "+8.3 pp" for H3a means the focal room is chosen 8.3 points more
  often when its reviews have emoji.
- **95% CI.** The range of effects consistent with the data. It comes from a *cluster bootstrap* over the 24 independent units
  (6 scenarios × 4 rooms).
- **p and q.** q is p corrected for testing 7 confirmatory hypotheses together (Benjamini–Hochberg).
- **Verdict.** Two questions decide it: is it significant, and is it inside the equivalence margin of ±5 pp (`sesoi_pp` in Settings)?

  | | 90% CI inside ±5 pp | 90% CI not inside ±5 pp |
  |---|---|---|
  | **Significant** | **Real but negligible** | **Effect** → reject H0 |
  | **Not significant** | **No meaningful effect** → an informative null | **Inconclusive** → need more data |

- **Pilot (descriptive only).** Fewer than 12 independent units or under 50 trials per cell. Use it to check the
  pipeline, never to conclude.
- **Precision.** Each sentence gives the half-width of the 90% CI. With the full plan (192 trials per cell) it is about ±7 pp.
  That means "No meaningful effect" can only appear if your margin (`sesoi_pp`) is about 8–10 pp, or if you run about
  850 trials per cell. **Decide the margin before you pre-register**, and write the reason down.
- **Warnings** at the end of a sentence flag:
  - a test agent;
  - a model that failed a gate or check (its results are descriptive only);
  - parse failures over 5%;
  - data from different stimulus versions mixed in one experiment.
- **Pooled row.** "(pooled: mean of real models)" averages only real models, never the test agents.
- **"Models differ? p = …"** This tests whether the effect is the same across your models. Never decide this by comparing which
  models came out significant.
- **Plain English** gives one sentence per row, which you can paste into notes.

**Can you trust a verdict on your own?** Yes, if three conditions hold:
1. V7, V1, V5 and HB all PASS for that model.
2. Parse failures stay under 5% (see `runs/<experiment>/analysis/1_failure_rates.csv`).
3. You ran the full plan, not just a pilot. Pilots are for checking the pipeline, not for conclusions.

**Coding the stated reasons (brief, secondary outcome).** "Analyse all" also writes
`runs/<experiment>__verdicts/reasons_coding_sheet.csv`, a random 10% of real-model reasons. Two of you label each one
facts / style / both / other. Then `python -m hotel_lab agreement <file>` gives your agreement and kappa.

**Where the data is:**
- **Sheet:** Results, Verdicts, Runs.
- **Locally:** `runs/<experiment>__<block>/raw/*.jsonl` holds every prompt and reply, `tables/` holds tidy CSVs, and
  `runs/<experiment>__verdicts/verdicts.csv` holds the verdicts.
- The detailed per-block analysis (logit, position shares, figures) can still be run from the terminal:
  `python -m hotel_lab analyze --experiment study_v1__style_emoji`.

## Troubleshooting

| Problem | Fix |
|---|---|
| "The Sheet link did not return JSON" | Deployment access must be **Anyone**, and use the `/exec` URL, not the editor URL |
| "bad or missing token" | Copy `api_token` from the Settings tab exactly |
| Synced but "NOT used, fix these errors" | The message lists the rows. Most often a review id in ReviewSets is missing from Reviews, or balance sets don't have equal mean stars |
| Changed `Code.gs` and nothing happened | Deploy → Manage deployments → ✏️ → Version: **New version** → Deploy |
| A model test fails with ConnectError | Ollama isn't running: open the Ollama app |
| Rows not in the Sheet | Click **Send pending rows to Sheet**. Rows wait locally if the internet or the link failed |
