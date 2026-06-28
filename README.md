# JobAgent — Real Agentic AI (Groq Edition)

A genuine **ReAct agent** powered by **Groq + Llama-3.3-70b-versatile**.
The model autonomously decides which tools to call, in what order, based on
what it finds — not a pre-scripted pipeline.

## Why Groq?

- **Free tier** at console.groq.com — no credit card needed to start
- **Extremely fast** — Llama-3.3-70b runs at ~2000 tok/s on Groq hardware
- **Native tool/function calling** — same architecture as OpenAI, works cleanly
- **llama-3.3-70b-versatile** is capable enough for multi-step tool reasoning

## What makes it actually agentic

- **The model decides** which tools to call and in what order
- **ReAct loop**: Reason → Act → Observe → Reason → Act…
- **Retries**: if search results are weak, the model searches again with a different query
- **Stop condition**: model calls `summarize_findings` when it has enough data — your code doesn't decide this
- **Real tool implementations**: TF-IDF cosine scorer, BLS 2024 salary data, HuggingFace dataset

## Setup

### 1. Get a free Groq API key

Go to https://console.groq.com → sign up → API Keys → Create Key

### 2. Unzip and open in VS Code

```bash
unzip jobagent.zip && cd jobagent && code .
```

### 3. Virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure keys

```bash
cp .env.example .env
```

Edit `.env`:
```
GROQ_API_KEY=gsk_...            # from console.groq.com
HF_TOKEN=hf_...                 # from huggingface.co/settings/tokens (optional)
```

### 6. Run

```bash
uvicorn main:app --reload --port 8000
```

Open **http://localhost:8000**

---

## Architecture

```
main.py                   FastAPI server + SSE streaming
agent/
  react_loop.py           ReAct loop — Groq drives tool selection
  dataset.py              HuggingFace loader with field normalization
tools/
  definitions.py          6 real tool implementations + schemas
templates/
  index.html              Frontend — streams agent events in real time
```

## Tools the model can call

| Tool | What it does |
|------|-------------|
| `search_jobs` | TF-IDF semantic search over HF job postings |
| `score_match` | Cosine similarity + skill coverage + seniority fit scoring |
| `skill_demand` | Word-boundary frequency analysis across all postings |
| `salary_lookup` | BLS OES May 2024 data, location + role adjusted |
| `filter_by_location` | Filter results to a target metro |
| `summarize_findings` | Compile all data — signals ready for final report |

## Model

**llama-3.3-70b-versatile** on Groq
- Context: 128k tokens
- Tool calling: native (OpenAI-compatible format)
- Temperature: 0.1 (low for deterministic reasoning)
- Rate limits: 6000 req/day on free tier — more than enough

## Salary data

- Source: **BLS Occupational Employment & Wage Statistics, May 2024**
- Location multipliers from BLS metro area wage data
- Role premium: ML/AI (+15%), Data Science (+10%), MLOps (+8%)
- Seniority band matched to actual years of experience (not hardcoded)
- 0 years exp → Junior band ($82k–$118k national, adjusted for location)

## Dataset sources (tried in order)

1. `lukebarousse/data_jobs` — real job postings with structured fields
2. `jacob-hugging-face/job-postings` — alternative HF corpus
3. Synthetic fallback — seeded deterministic corpus if HF is rate-limited

Results cached for 24 hours in `.cache/`.
