"""
agent/dataset.py
Load and cache real job postings from HuggingFace datasets.
Uses the datasets library directly — no REST API, no rate limits.
"""

import json
import hashlib
import time
from pathlib import Path

CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

# ── ADD YOUR DATASETS HERE ─────────────────────────────────────────────────
HF_DATASETS = [
    {"name": "lukebarousse/data_jobs",              "split": "train", "config": None},
    {"name": "jacob-hugging-face/job-descriptions", "split": "train", "config": None},
]
# ──────────────────────────────────────────────────────────────────────────

FIELD_MAP = {
    "title":        ["job_title_short", "job_title", "title", "position", "role", "job_name"],
    "company":      ["company_name", "company", "employer", "organization", "firm"],
    "location":     ["job_location", "location", "city", "place", "geo"],
    "description":  ["job_description", "description", "job_title", "summary", "details", "body", "text"],
    "skills":       ["job_skills", "skills", "required_skills", "tech_stack", "technologies"],
    "salary_min":   ["salary_year_avg", "salary_min", "min_salary", "salary_from", "pay_min"],
    "salary_max":   ["salary_year_avg", "salary_max", "max_salary", "salary_to", "pay_max"],
    "job_level":    ["job_schedule_type", "job_level", "seniority", "experience_level", "level"],
    "job_category": ["job_title_short", "job_category", "category", "department", "function"],
}


def _normalize_row(raw: dict) -> dict:
    out = {}
    for std_field, candidates in FIELD_MAP.items():
        for candidate in candidates:
            val = raw.get(candidate)
            if val is not None and str(val).strip():
                out[std_field] = str(val).strip()
                break
        if std_field not in out:
            out[std_field] = ""
    return out


def _fetch_dataset(ds: dict, hf_token: str | None, n: int) -> list[dict]:
    from datasets import load_dataset

    kwargs = {
        "split": ds["split"],
        "streaming": True,
        "trust_remote_code": False,
    }
    if ds.get("config"):
        kwargs["name"] = ds["config"]
    if hf_token:
        kwargs["token"] = hf_token

    dataset = load_dataset(ds["name"], **kwargs)

    rows = []
    for item in dataset:
        normalized = _normalize_row(item)
        # Accept row if it has at least a title
        # Build a description from available fields if missing
        if not normalized.get("description") and normalized.get("title"):
            parts = [normalized.get("title", "")]
            if normalized.get("company"):
                parts.append(f"at {normalized['company']}")
            if normalized.get("location"):
                parts.append(f"in {normalized['location']}")
            if normalized.get("skills"):
                parts.append(f"Skills: {normalized['skills']}")
            normalized["description"] = " ".join(parts)
        if normalized.get("title"):
            rows.append(normalized)
        if len(rows) >= n:
            break

    return rows


def load_dataset_from_hf(hf_token: str | None = None, max_rows: int = 500) -> tuple[list[dict], str]:
    cache_key = hashlib.md5(
        f"v4_{'_'.join(d['name'] for d in HF_DATASETS)}_{max_rows}".encode()
    ).hexdigest()[:8]
    cache_file = CACHE_DIR / f"jobs_{cache_key}.json"

    if cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < 86400:
            with open(cache_file) as f:
                cached = json.load(f)
            print(f"Using cached dataset: {cached['source']} ({len(cached['rows'])} rows)")
            return cached["rows"], cached["source"] + " (cached)"
        else:
            print("Cache expired — re-fetching...")

    per_ds = max(50, max_rows // len(HF_DATASETS))
    all_rows = []
    sources = []

    print(f"Loading {len(HF_DATASETS)} dataset(s), up to {per_ds} rows each...")

    for ds in HF_DATASETS:
        try:
            print(f"  {ds['name']}: loading...")
            rows = _fetch_dataset(ds, hf_token, per_ds)
            if rows:
                all_rows.extend(rows)
                sources.append(f"{ds['name']} ({len(rows)})")
                print(f"  ✓ {ds['name']}: {len(rows)} rows loaded")
            else:
                print(f"  ✗ {ds['name']}: no valid rows found")
        except Exception as e:
            print(f"  ✗ {ds['name']}: failed — {e}")

    if all_rows:
        seen = set()
        deduped = []
        for row in all_rows:
            key = (row.get("title", "").lower().strip(), row.get("company", "").lower().strip())
            if key not in seen:
                seen.add(key)
                deduped.append(row)

        removed = len(all_rows) - len(deduped)
        if removed:
            print(f"  Removed {removed} duplicates")

        source = " + ".join(sources)
        print(f"✓ Total: {len(deduped)} unique jobs loaded")

        with open(cache_file, "w") as f:
            json.dump({"rows": deduped, "source": source}, f)

        return deduped, source

    print("All HF datasets failed — using synthetic corpus")
    rows = _generate_synthetic(max_rows)
    source = "synthetic-corpus"
    with open(cache_file, "w") as f:
        json.dump({"rows": rows, "source": source}, f)
    return rows, source


def _generate_synthetic(n: int) -> list[dict]:
    import random
    random.seed(42)

    companies = [
        "Anthropic", "OpenAI", "Google DeepMind", "Meta AI", "Microsoft Research",
        "Databricks", "Snowflake", "Scale AI", "Palantir", "Stripe", "Figma", "Vercel",
        "Hugging Face", "Cohere", "Mistral AI", "Modal", "Weaviate", "Pinecone",
        "Airbnb", "DoorDash", "Netflix", "Spotify", "Lyft", "Instacart", "Notion",
        "Linear", "Retool", "Rippling", "Brex", "Ramp",
    ]
    locations = [
        "San Francisco, CA", "New York, NY", "Seattle, WA", "Austin, TX",
        "Boston, MA", "Chicago, IL", "Los Angeles, CA", "Remote", "Denver, CO",
    ]
    levels = ["Junior", "Mid-level", "Senior", "Staff", "Principal", "Lead"]
    skill_pools = {
        "ML/AI": ["Python", "PyTorch", "TensorFlow", "scikit-learn", "Hugging Face",
                  "LangChain", "MLflow", "Ray", "CUDA", "JAX", "LLMs", "RAG",
                  "Fine-tuning", "LoRA", "RLHF", "Embeddings", "Triton"],
        "Data":  ["Python", "SQL", "Spark", "Airflow", "dbt", "Kafka", "Snowflake",
                  "BigQuery", "Databricks", "Pandas", "NumPy", "Tableau"],
        "Infra": ["Kubernetes", "Docker", "AWS", "GCP", "Azure", "Terraform", "Helm",
                  "Argo", "Prometheus", "Grafana", "Linux"],
        "SWE":   ["Python", "TypeScript", "React", "FastAPI", "Node.js", "PostgreSQL",
                  "Redis", "GraphQL", "gRPC", "Go", "Rust"],
    }
    roles = {
        "ML/AI": ["Machine Learning Engineer", "AI Engineer", "Research Engineer",
                  "Applied Scientist", "MLOps Engineer", "LLM Engineer", "NLP Engineer"],
        "Data":  ["Data Scientist", "Data Engineer", "Analytics Engineer",
                  "Senior Data Scientist", "Staff Data Scientist"],
        "Infra": ["Platform Engineer", "Infrastructure Engineer", "Site Reliability Engineer",
                  "DevOps Engineer", "Cloud Engineer"],
        "SWE":   ["Software Engineer", "Backend Engineer", "Full Stack Engineer",
                  "Senior Software Engineer", "Staff Engineer"],
    }
    sal = {
        "Junior": (85, 120), "Mid-level": (120, 170), "Senior": (165, 230),
        "Staff": (215, 300), "Principal": (250, 380), "Lead": (190, 270),
    }

    rows = []
    cats = list(skill_pools.keys())
    for i in range(n):
        cat = cats[i % len(cats)]
        company = companies[i % len(companies)]
        location = locations[i % len(locations)]
        level = levels[i % len(levels)]
        role_title = roles[cat][i % len(roles[cat])]
        title = f"{level} {role_title}"
        pool = skill_pools[cat]
        required = random.sample(pool, min(5, len(pool)))
        bonus = random.sample(pool, min(3, len(pool)))
        lo, hi = sal[level]
        salary = random.randint(lo, hi)
        desc = (
            f"{company} is hiring a {title} to join our {cat} team in {location}. "
            f"Required: {', '.join(required)}. "
            f"Nice to have: {', '.join(bonus)}. "
            f"Compensation: ${salary}k-${salary + 30}k. "
            f"We offer remote-first culture, equity, and comprehensive benefits."
        )
        rows.append({
            "title": title, "company": company, "location": location,
            "description": desc, "skills": ", ".join(required),
            "salary_min": str(salary - 10), "salary_max": str(salary + 30),
            "job_level": level, "job_category": cat,
        })
    return rows
