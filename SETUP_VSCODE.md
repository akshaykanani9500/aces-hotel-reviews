# Set up Hotel Lab in VS Code (Windows or Mac)

You only need to do steps 1–6 once. Each step says what you should see if it worked.

## 1. Install the tools

| Tool | Get it | Check it worked (in a terminal) |
|---|---|---|
| Python 3.11, 3.12 or 3.13 | https://www.python.org/downloads/ (Windows: tick **"Add python.exe to PATH"**) | `python --version` (Mac: `python3 --version`) |
| Git | https://git-scm.com/downloads | `git --version` |
| VS Code | https://code.visualstudio.com | — |
| VS Code extension "Python" (Microsoft) | Extensions panel → search "Python" | — |
| Ollama (to run open-weight models locally) | https://ollama.com/download | `ollama --version` |

> On Mac, use `python3` wherever this guide says `python` until the virtual environment is active.

## 2. Clone the repo in VS Code

1. Open VS Code and press `Ctrl+Shift+P` (Mac: `Cmd+Shift+P`), then choose **Git: Clone**.
2. Paste `https://github.com/akshaykanani9500/aces-hotel-reviews.git`
3. Pick a folder (for example `Documents/projects`) and click **Open** when VS Code asks.

Or in a terminal:
```bash
git clone https://github.com/akshaykanani9500/aces-hotel-reviews.git
cd aces-hotel-reviews
code .
```

## 3. Create a virtual environment

Open the terminal in VS Code (**Terminal → New Terminal**). It opens in the repo folder.

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
If PowerShell says scripts are disabled, run this once, then activate again:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

**Mac / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

✅ The terminal prompt now starts with `(.venv)`.

Then tell VS Code to use it: `Ctrl+Shift+P` → **Python: Select Interpreter** → pick the one with `.venv`.

## 4. Install the packages

```bash
pip install -r hotel_lab/requirements.txt
```
This takes 1–3 minutes. ✅ It ends with "Successfully installed …".

Only needed for **screenshot mode** (vision models):
```bash
python -m playwright install chromium
```

## 5. Check everything works (no AI model needed)

```bash
python -m hotel_lab validate
python -m pytest hotel_lab
python -m hotel_lab demo
```
✅ `validate` ends with "Validation passed."  
✅ pytest shows "19 passed" (or more).  
✅ `demo` ends with a table where `mock-biased` picks position 1 about 60% of the time. That shows the
analysis picks up a position bias when one exists.

## 6. Look at the website

```bash
python -m hotel_lab plan
python -m hotel_lab serve
```
Open **http://127.0.0.1:8765** in your browser.
- Use the **Preview** form to see any scenario with any review order, style, review set or format.
- Click any trial in the table to see exactly what the agent sees in that trial.
- `/api/trial/<trial_id>` shows the exact JSON a text-mode agent receives.

Press `Ctrl+C` in the terminal to stop the server.

## 7. Connect an open-weight model (Ollama)

In a **second** terminal (Ollama runs as a background app after installing):
```bash
ollama pull qwen2.5:7b      # ~4.7 GB download, once
ollama list                 # shows the models you have
```
Back in the VS Code terminal (with `(.venv)` active):
```bash
python -m hotel_lab models                              # list model configs
python -m hotel_lab hello --model ollama-qwen2.5-7b     # ✅ "parse_ok": true
```

Other models already configured: `ollama-llama3.1-8b`, `ollama-mistral-7b`, `ollama-gemma3-12b` (vision),
`ollama-qwen2.5vl-7b` (vision). To add one, copy a YAML file in `hotel_lab/config/models/` and change
`model:` to the name shown by `ollama list`.

**Hardware:** 7–8B models need about 8 GB of free RAM (16 GB laptop recommended). 12B needs about 12 GB.
For 30B+ models, use OpenRouter or the Hugging Face router: copy `.env.sample` to `.env`, add your key,
then use `openrouter-llama3.3-70b` or `hf-router-example`.

## 8. Run a pilot

```bash
python -m hotel_lab run --models ollama-qwen2.5-7b --limit 20
python -m hotel_lab export
python -m hotel_lab analyze
```
Results are in `runs/pilot_v1/analysis/` (open the CSVs in Excel, or the PNG figures).
If the run stops (laptop sleeps, Ctrl+C), run the same command again. It continues where it stopped.

Screenshot mode with a vision model:
```bash
python -m hotel_lab run --models ollama-gemma3-12b --interface screenshot --blocks format --limit 10
```

## 9. Daily workflow

1. Edit stimuli in `hotel_lab/stimuli/*.yaml` (format: `hotel_lab/stimuli/SCHEMA.md`).
2. `python -m hotel_lab validate` and `python -m hotel_lab checks`
3. Edit the design in `hotel_lab/config/experiment.yaml`. **Change `experiment_name` for every new study**, so runs never mix.
4. `python -m hotel_lab plan`, then `run`, `export`, `analyze`.
5. Commit code and stimulus changes with Git. `runs/` is not committed; copy it to Google Drive or OneDrive daily, as the brief asks.

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` not found (Windows) | Reinstall Python and tick "Add to PATH", or use `py` instead of `python` |
| `No module named hotel_lab` | Run commands from the repo root folder (the one with `README.md`) |
| `No module named fastapi` | The venv is not active. Activate it (step 3) |
| `Connection refused` on `hello` | Ollama is not running. Open the Ollama app or run `ollama serve` |
| `model "qwen2.5:7b" not found` | `ollama pull qwen2.5:7b` |
| Many parse failures | Set `json_mode: true` in the model YAML, or keep `parse_retries: 1` in experiment.yaml |
| Screenshot mode: "Executable doesn't exist" | `python -m playwright install chromium` |
| Port 8765 busy | `python -m hotel_lab serve --port 8800` |
