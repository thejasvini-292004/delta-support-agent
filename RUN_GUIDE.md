# How to run this project (step by step)

## Why run it on your own Mac (not inside Claude)
The LLM (Gemini) API is **blocked by the sandbox network** Claude runs in, so the
LLM steps (reply drafting + LLM-judge) must run in **your own Terminal**, where your
internet is open. Everything else already ran and produced results in `reports/`.
Your Gemini key is already saved in `.env` (don't share that file).

## One-time setup
1. Open **Terminal** (Cmd+Space → type "Terminal").
2. Go to the project:
   ```
   cd ~/Desktop/hiver_delta_agent
   ```
3. (Recommended) make a clean Python environment and install dependencies:
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   python3 -m pip install -r requirements.txt
   ```
   (If you skip the venv, just run the last line.)

## Step 1 — check the Gemini key works
```
python3 src/llm.py
```
Expected: `reply: 'OK'`.
- You'll see a "google.generativeai is deprecated" warning — **ignore it**, it still works.
- If you see `API_KEY_INVALID`, your key is the wrong type. Get a fresh one at
  https://aistudio.google.com/apikey (it starts with `AIza…`), paste it into `.env`
  after `GEMINI_API_KEY=`, and re-run.

## Step 2 — run the agent on one message
```
python3 src/agent.py --msg "my flight got cancelled and I need to get home tonight"
```
Prints the intent, the auto/escalate decision + reason, and the grounded reply.

## Step 3 — run the full evaluation (this makes the LLM numbers)
Quick smoke test first (~3–5 min, small sample):
```
python3 src/evaluate.py --llm_n 20 --judge_n 8 --agree_n 15
```
Then the fuller run (~12–15 min; free Gemini tier is ~15 requests/min so it's slow):
```
python3 src/evaluate.py
```
Complete run on all 220 (slowest, best for the report):
```
python3 src/evaluate.py --full
```
Results are written to `reports/`:
- `intent_scores.csv` — majority vs TF-IDF vs **LLM** accuracy & macro-F1
- `escalation_scores.csv` — your rules vs baselines
- `reply_judge_summary.json` — LLM-judge avg scores + **judge–human agreement (kappa)**
- `sample_agent_outputs.json` — example agent replies

## Step 4 — read the write-up
Open `REPORT.md` (draft report) and drop the LLM numbers from `reports/` into the tables.

## If Gemini rate-limits you (error 429)
The code retries automatically. If it's still too slow, use smaller numbers, e.g.
`python3 src/evaluate.py --llm_n 30 --judge_n 10 --agree_n 20`.

## The chat UI (interactive)
```
cd ~/Desktop/hiver_delta_agent
source .venv/bin/activate
pip install streamlit
streamlit run app.py
```
This opens a chat page in your browser (http://localhost:8501). Type a customer
message or click an example in the sidebar. Each reply shows the predicted intent,
the auto-handle vs escalate decision + reason, and the historical Delta replies it
grounded on. Press Ctrl+C in the Terminal to stop it. Works with or without an LLM key.
