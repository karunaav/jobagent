# JobAgent
![Python](https://img.shields.io/badge/python-3.10-blue?style=for-the-badge)
![Groq](https://img.shields.io/badge/Groq-LLM%20inference-F55036?style=for-the-badge)
![HuggingFace](https://img.shields.io/badge/HuggingFace-models-FFD21E?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688?style=for-the-badge)
![ReAct](https://img.shields.io/badge/agent-ReAct%20loop-blueviolet?style=for-the-badge)
![Status](https://img.shields.io/badge/status-active-brightgreen?style=for-the-badge)


An agentic AI app that analyzes your profile against real job market data and generates a career intelligence report.

Built with a ReAct (Reason + Act) loop — the agent autonomously decides which tools to call, in what order, based on what it finds. It's not a pipeline.

## Stack

- **Groq** (llama-3.3-70b-versatile) — drives the agent loop
- **HuggingFace datasets** — real job postings
  - [lukebarousse/data_jobs](https://huggingface.co/datasets/lukebarousse/data_jobs) — 786k LinkedIn job postings
  - [jacob-hugging-face/job-descriptions](https://huggingface.co/datasets/jacob-hugging-face/job-descriptions) — 853 job descriptions
- **FastAPI** — backend + SSE streaming
- **TF-IDF cosine similarity** — deterministic job scoring, no randomness
- **BLS OES 2024** — salary data adjusted for location and seniority

## How it works
<img width="1920" height="1032" alt="Screenshot 2026-06-27 233939" src="https://github.com/user-attachments/assets/6f619a71-3637-4ae1-82f8-935116bdc3ff" />


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

<img width="1920" height="1032" alt="Screenshot 2026-06-27 233201" src="https://github.com/user-attachments/assets/29ec4cad-f870-4a43-a661-67fda21dd1f1" />

---

<img width="1920" height="1032" alt="Screenshot 2026-06-27 233214" src="https://github.com/user-attachments/assets/72d072b3-8d44-4752-b679-c47655bba4c8" />

---

<img width="1920" height="1032" alt="Screenshot 2026-06-27 233225" src="https://github.com/user-attachments/assets/3326cc72-3cec-424b-a604-655390e29dcb" />

---

<img width="1920" height="1032" alt="Screenshot 2026-06-27 233229" src="https://github.com/user-attachments/assets/b9168d88-ef68-44ef-94f6-a9dd3b5d644e" />

---

<img width="1920" height="1032" alt="image" src="https://github.com/user-attachments/assets/70908204-b9cf-409c-877c-c33172109217" />

---

<img width="1920" height="1032" alt="image" src="https://github.com/user-attachments/assets/f56356a5-c9c6-4145-ae07-2714d625ae4b" />

---

<img width="1920" height="1032" alt="image" src="https://github.com/user-attachments/assets/7901310a-f735-4296-bf39-4e32428817dc" />




