# Hotel Lab Studio: step-by-step guide

The Studio is a local website that runs the whole study. The Google Sheet is its database: you edit reviews,
hypotheses and settings there.

**Layout.** Hover over (or tab to) any **?** or button to see what it means.

- **Top bar:**
  - The left pill shows where the data comes from: your Google Sheet, with the snapshot code, or the built-in sample files. Click it to connect the Sheet.
  - The right pill shows the current run's progress. Click it to open the Run tab.
- **Sidebar:** every hypothesis, grouped as main, secondary, checks and gate. Each one shows its status: not run, N trials, Pilot, or its verdict.
- **Tabs, in the order you use them:**
  1. **Plan:** H0 and Ha side by side, what changes, what is held constant, how big the experiment is, and the human benchmark.
  2. **What the agent sees:** the hotel page exactly as the agent sees it. *Show another trial* gives a new random trial. *Behind the scenes* shows the hidden facts, the best room and the focal room. *Exact prompt* shows the text sent to the model.
  3. **Models:** your three real models (qwen2.5 7B, llama3.1 8B, gemma3 12B, all free through Ollama) and one no-AI test agent. Click *Test*, then tick the ones you want. If a model appears twice, the copy is marked *Same as …*: remove it with ✕. *Add another model* is optional.
  4. **Run:** choose Pilot or Full, the interface and the reasoning level, then *Run*. Progress and the log appear here and in the top bar.
  5. **Results:** *Analyse all results* gives one verdict per hypothesis and model, each with a plain-English sentence. They are also written to the Sheet.

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
| Models | written by the website: the models it can use, without API keys. Updated on every Sync, Analyse, and when you add or remove a model. (The website reads models from `hotel_lab/config/models/`, not from this tab.) | ✏️ website |
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
> Some admins also block unverified scripts completely ("Access blocked").
> Note that "Anyone" plus the token gives read access to the whole Sheet, hidden Facts included, so never share the link.
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
Results are then saved locally only. You can send them to the Sheet in two ways:
- connect the Sheet later and click **Send pending rows to Sheet**;
- or in the Sheet use **File → Import** with `runs/<experiment>__<block>/tables/trials.csv` and
  `runs/<experiment>__verdicts/verdicts.csv`.

**Tip when adding rows in the Sheet.** Select the new rows and choose **Format → Number → Plain text** before you type
dates or ids. Dates are accepted as 2026-09-18 or 18/09/2026; the website converts them to 2026-09-18.

### A6. Your models

The study uses **three real models and one test agent**. All are already set up in the repo:

| In the Studio | What it is | Size | First time (in a terminal) |
|---|---|---|---|
| `ollama-qwen2.5-7b` | Alibaba's Qwen 2.5 | 7B, about 4.7 GB | `ollama pull qwen2.5:7b` |
| `ollama-llama3.1-8b` | Meta's Llama 3.1 | 8B, about 4.9 GB | `ollama pull llama3.1:8b` |
| `ollama-gemma3-12b` | Google's Gemma 3 (also reads images) | 12B, about 8 GB, needs 16 GB RAM | `ollama pull gemma3:12b` |
| `no-ai-test` | Test agent, **no AI**. It has planted preferences, so you can check the analysis finds them | – | nothing |

1. Install Ollama (https://ollama.com/download) and keep the Ollama app open while you run.
2. Pull the three models with the commands above (once).
3. In the Studio's **Models** tab, click **Test** on each model. *Works* means the model returned a valid answer.
4. Tick the models you want to use in the next run.

If you added models yourself earlier, they show up as `local-…`. A copy of one of the four is marked **Same as …**: remove it with ✕.
**Add another model (optional)** is only for trying an extra model later. A hosted model's API key is saved in your local `.env`, never
in the Sheet or on GitHub.

**Update the Apps Script once (November 2026 version).** The new script lets the website fill the Models tab. In the Sheet:
1. **Extensions → Apps Script**.
2. Replace `Code.gs` with the new `hotel_lab/apps_script/Code.gs`, then click **Save**.
3. **Deploy → Manage deployments → ✏️ → Version: New version → Deploy**.

The link and token stay the same. Until you do this, the website says "Models tab NOT updated", and everything else still works.

---

## Part B. Running the study (in this order)

The order follows the brief's gates. Don't move to the next step until the current one passes.

**Before the real study, try everything once with the test agents in a separate experiment.** In the Settings tab, set
`experiment_name` to `trial_run`, Sync, and run V7 and a couple of hypotheses with the test agent `no-ai-test`. It has planted
preferences (emoji, position 1, higher stars), so H3a should come out as **Effect**. That shows the analysis finds a real effect.
Then set it back to `study_v1` and Sync. Test agents are never averaged into the pooled row, but keeping them out of the real
experiment keeps the tables clean.

| Step | Pick in the dropdown | Mode | Pass condition |
|---|---|---|---|
| 1 | **V7 Placebo** with your real model(s) | Pilot 20/cell, then full | PASS: the pipeline does not invent effects (a pilot PASS is marked "weak") |
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

**Important:** if you edit reviews, hypotheses or settings after an experiment has results, the Studio **refuses to
run** and asks you to change `experiment_name`, for example to `study_v2`. Old and new data can never mix. A stopped run
continues where it stopped when you press Run again, and every run spreads its trials over all cells, so even a
stopped run is balanced.

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
| "Google asked for a sign-in" / your URL contains `/a/macros/spjimr.org/` | The web app only works for signed-in SPJIMR users. Deploy → Manage deployments → ✏️ → *Who has access* = **Anyone** → Version: New version → Deploy, then copy the new URL. If **Anyone** isn't offered, copy the Sheet to a personal Gmail account and deploy there, or use the offline .xlsx route |
| "The Sheet link did not return JSON" | Deployment access must be **Anyone**, and use the `/exec` URL, not the editor URL |
| "bad or missing token" | Copy `api_token` from the Settings tab exactly |
| Synced but "NOT used, fix these errors" | The message lists the rows. Most often a review id in ReviewSets is missing from Reviews, or balance sets don't have equal mean stars |
| Changed `Code.gs` and nothing happened | Deploy → Manage deployments → ✏️ → Version: **New version** → Deploy |
| A model test fails with ConnectError | Ollama isn't running: open the Ollama app |
| "Port 8800/8765 is already in use" | Another Studio or `serve` is still running. Close that terminal, or start with `--port 8820 --site-port 8785` |
| "The reviews, hypotheses or settings changed since …" | You edited the Sheet mid-study: change `experiment_name` in Settings, Sync, and run again |
| Closed the terminal mid-run | Start the Studio again and press Run: finished trials are skipped, and unsent Sheet rows are sent on the next run or with **Send pending rows** |
| Rows not in the Sheet | Click **Send pending rows to Sheet**. Rows wait locally if the internet or the link failed |
