"""
main.py — JobAgent server
FastAPI + SSE streaming
"""

import os
import json
import asyncio
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

from agent.dataset import load_dataset_from_hf
from agent.react_loop import run_agent

_dataset_rows: list[dict] = []
_dataset_source: str = ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _dataset_rows, _dataset_source
    print("Loading job dataset from HuggingFace...")
    hf_token = os.getenv("HF_TOKEN")
    _dataset_rows, _dataset_source = load_dataset_from_hf(hf_token=hf_token, max_rows=500)
    print(f"✓ Loaded {len(_dataset_rows)} jobs from: {_dataset_source}")
    yield


app = FastAPI(title="JobAgent", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


class ProfileRequest(BaseModel):
    role: str
    location: str
    years_exp: int
    skills: list[str]


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/index.html", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/api/dataset-info")
async def dataset_info():
    return {
        "source": _dataset_source,
        "total_rows": len(_dataset_rows),
        "sample_titles": [r["title"] for r in _dataset_rows[:5]],
    }


@app.post("/api/run")
async def run(req: ProfileRequest):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(500, "GROQ_API_KEY not set in .env")
    if not _dataset_rows:
        raise HTTPException(500, "Dataset not loaded")

    profile = {
        "role": req.role,
        "location": req.location,
        "years_exp": req.years_exp,
        "skills": req.skills,
    }

    async def event_stream():
        loop = asyncio.get_event_loop()

        def _run_sync():
            events = []
            for event in run_agent(profile, _dataset_rows, api_key):
                events.append(event)
            return events

        events = await loop.run_in_executor(None, _run_sync)
        for event in events:
            yield f"data: {json.dumps(event)}\n\n"
        yield 'data: {"type":"stream_end"}\n\n'

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/health")
async def health():
    return {"status": "ok", "dataset_rows": len(_dataset_rows), "source": _dataset_source}
