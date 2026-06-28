# JobAgent

An agentic AI app that analyzes your profile against real job market data and generates a career intelligence report.

Built with a ReAct (Reason + Act) loop — the agent autonomously decides which tools to call, in what order, based on what it finds. It's not a pipeline.

## Stack

- **Groq** (llama-3.3-70b-versatile) — drives the agent loop
- **HuggingFace datasets** — real job postings (lukebarousse/data_jobs)
- **FastAPI** — backend + SSE streaming
- **TF-IDF cosine similarity** — deterministic job scoring, no randomness
- **BLS OES 2024** — salary data adjusted for location and seniority

## How it works

The agent gets your profile and a set of tools. It then decides what to search for, scores the results, analyzes skill demand, looks up salary data, and writes a grounded report — all without any hardcoded steps.

Tools: `search_jobs`, `score_match`, `skill_demand`, `salary_lookup`, `filter_by_location`, `summarize_findings`

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# add GROQ_API_KEY and HF_TOKEN to .env
mkdir static
uvicorn main:app --reload --port 8000
```

Open `http://localhost:8000`

## Keys needed

- Groq API key → console.groq.com (free)
- HuggingFace token → huggingface.co/settings/tokens (optional)
