# ✈️ Delta Support Agent

An **evaluation-first AI customer-support agent** for Delta's Twitter mentions. It classifies each incoming message into an intent, grounds a reply in Delta's own historical resolutions, and decides — with a stated reason — whether to **auto-handle** or **escalate to a human**.

Built on the Kaggle *Customer Support on Twitter* dataset. Guiding principle: **the proof is worth more than the system** — the escalation policy and the evaluation harness got more care than the model itself.

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.9%2B-blue">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-TF--IDF%20%2B%20LogReg-orange">
  <img alt="Streamlit" src="https://img.shields.io/badge/UI-Streamlit-red">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
</p>

> **Live demo:** _add your Streamlit URL here after deploying (see [Deploy](#-deploy-on-streamlit) below)._
> **Full write-up:** see [`REPORT.md`](REPORT.md) and [`Delta_Support_Agent_Report.docx`](Delta_Support_Agent_Report.docx) for problem framing, baselines, failure analysis, and the research grounding.

---

## What it does

1. **Classify** — a TF-IDF + Logistic-Regression model (trained via weak supervision) sorts each message into one of **11 data-driven intents**.
2. **Ground** — a cosine retriever pulls the closest real *(customer → Delta reply)* pairs from **42k** historical resolutions; replies are grounded in those, never invented.
3. **Decide** — a transparent **5-rule policy** routes the message to auto-handle or human hand-off, and every response ships with its full reasoning trace (intent, confidence, grounding score, reason).

## 🏗️ Architecture

Four layers. The bottom two are built once, offline; the agent layer runs on every message; the LLM is an optional accelerator wired to **reply-drafting only**, never to a decision.

![End-to-end architecture](docs/diagram1_architecture.png)

*Raw tweets are prepared once into grounding pairs and clean messages; those bootstrap a TF-IDF intent classifier (via weak supervision) and a cosine retriever; `SupportAgent.handle()` chains classify → retrieve → route → draft on every request. The dashed amber box is the only place a language model touches the pipeline.*

## 🔀 How a message flows

A single request travels left to right. Classification and retrieval only gather evidence; the **router** is the decision point — five rules, any one of which forces a human hand-off.

![Message to response workflow](docs/diagram2_workflow.png)

**The five escalation rules — any one forces a hand-off:**

| # | Rule | Why |
|---|------|-----|
| R1 | **Negative sentiment** (negation-aware) | A dissatisfied customer goes to a human even on an auto-OK topic. |
| R2 | **Risk / financial language** | `refund`, `lawsuit`, `terrible`… trigger a hand-off regardless of confidence. |
| R3 | **Account / transaction intent** | Anything outside `AUTO_OK` needs a human with account access. |
| R4 | **Low confidence** (`< 0.40`) | The agent does not gamble on an uncertain label. |
| R5 | **Weak grounding** (`< 0.38`) | No close past case → hand off instead of guessing. |

## 📊 Key results

Measured on a **220-message hand-labeled golden set**, each model against ≥2 baselines. Intent uses 5-fold out-of-fold prediction; escalation uses oracle intents to isolate the policy.

**Intent classification**

| Model | Accuracy | Macro-F1 |
|---|--:|--:|
| Trivial — majority class | 0.164 | 0.026 |
| **TF-IDF + LogReg (ours)** | **0.491** | **0.426** |

**Escalation decision** — the part that got the most care

| Model | Acc | Esc. Precision | Esc. Recall |
|---|--:|--:|--:|
| Trivial — escalate all | 0.691 | 0.691 | 1.000 |
| Trivial — auto all | 0.309 | — | — |
| **Rule-based policy (ours)** | **0.786** | **0.843** | **0.849** |

The one number worth standing behind: **escalation recall 0.85 at 0.84 precision** — for a safety-regulated brand, catching the messages that need a human is the metric that maps to real cost. Full analysis, including *"what's misleading about my headline number"*, is in [`REPORT.md`](REPORT.md).

## 🚀 Quickstart (run locally)

```bash
git clone https://github.com/thejasvini-292004/delta-support-agent.git
cd delta-support-agent

python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
```

The app runs **fully offline** on TF-IDF + retrieval + templated replies — no API key required. To enable LLM-drafted replies, add an OpenAI (or Gemini) key:

```bash
cp .env.example .env      # then paste your key into .env
```

## 🧪 Reproduce the evaluation

```bash
python src/evaluate.py            # writes fresh metrics into reports/
```

Outputs land in [`reports/`](reports/): `intent_scores.csv`, `intent_classification_report.txt`, `escalation_scores.csv`. Task 2 (reply quality via LLM-as-judge + judge–human Cohen's κ) runs when an API key is present.

## 🗂️ Project structure

```
delta-support-agent/
├── app.py                     # Streamlit chat UI
├── requirements.txt
├── src/
│   ├── prepare_data.py        # Step 0: filter Delta, clean, rebuild threads → grounding pairs
│   ├── discover_intents.py    # Step 1: cluster messages to define intents
│   ├── taxonomy.py            # 11 canonical intents, AUTO_OK set, risk words, sentiment
│   ├── classifiers.py         # weak supervision + TF-IDF/LogReg production classifier
│   ├── retriever.py           # TF-IDF cosine retriever over 42k grounding pairs
│   ├── agent.py               # SupportAgent: classify → retrieve → route → draft
│   ├── llm.py                 # optional multi-provider LLM wrapper (drafting only)
│   ├── evaluate.py            # evaluation harness (3 tasks + baselines)
│   └── judge.py               # LLM-as-judge + Cohen's κ agreement
├── data/                      # derived grounding pairs, messages, golden labels
├── reports/                   # evaluation outputs (CSV/TXT/JSON)
├── docs/                      # architecture & workflow diagrams
├── REPORT.md                  # full technical report
└── Delta_Support_Agent_Report.docx
```

## 🧠 Design notes & research grounding

The design follows the consensus across six recent papers on production support agents — evaluation-driven development, policy adherence over cleverness, escalation as a measured action, a data flywheel from cheap signal, retrievable knowledge representation, and judging with a measured judge. How each maps to a concrete decision here is in [`REPORT.md` §2](REPORT.md).

## 📄 License

Released under the MIT License. Dataset: [Kaggle — Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) (not redistributed here beyond the derived Delta subset used for grounding).
