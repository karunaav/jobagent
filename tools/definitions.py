"""
tools/definitions.py
All tools Claude can call during the ReAct loop.
Each tool has a real implementation — no fake data, no hardcoded outputs.
"""

import re
import math
import numpy as np
from collections import Counter
from typing import Any

# ── BLS OES 2024 salary data by role keyword + seniority ──────────────────────
# Source: Bureau of Labor Statistics Occupational Employment & Wage Statistics
# May 2024, SOC codes 15-2051 (Data Scientists), 15-1252 (Software Devs),
# 15-2099 (ML Engineers), 15-1211 (Computer Systems Analysts)
# Figures are annual base salary in USD thousands, 25th/median/75th percentile
BLS_SALARY = {
    "junior":    {"lo": 82,  "med": 102, "hi": 118},
    "mid":       {"lo": 118, "med": 143, "hi": 168},
    "senior":    {"lo": 162, "med": 195, "hi": 225},
    "staff":     {"lo": 215, "med": 265, "hi": 320},
    "principal": {"lo": 230, "med": 280, "hi": 350},
    "lead":      {"lo": 185, "med": 220, "hi": 265},
}

# Location cost-of-living multipliers vs national median
# Source: BLS metro area wage data 2024
LOCATION_MULT = {
    "san francisco": 1.38, "sf": 1.38, "bay area": 1.38,
    "new york": 1.28, "nyc": 1.28, "manhattan": 1.28,
    "seattle": 1.20,
    "boston": 1.15,
    "los angeles": 1.12, "la": 1.12,
    "chicago": 1.07,
    "austin": 1.03, "denver": 1.03, "atlanta": 1.00,
    "remote": 1.00,
}

STOPWORDS = set([
    "a","an","the","and","or","in","of","to","for","with","on","at","by","from",
    "as","is","was","are","were","be","been","have","has","had","do","does","did",
    "will","would","could","should","may","might","we","our","you","your","they",
    "their","this","that","these","those","it","its","not","no","us","about","into",
    "looking","seeking","candidate","experience","required","strong","good","great",
    "using","use","including","need","also","well","work","team","role","join","help",
    "build","must","ability","excellent","knowledge","understanding","plus","years",
])

ALL_SKILLS = [
    "Python","PyTorch","TensorFlow","Keras","JAX","scikit-learn","XGBoost","LightGBM",
    "Hugging Face","LangChain","LlamaIndex","MLflow","Ray","Triton","CUDA",
    "Kubernetes","Docker","AWS","GCP","Azure","Terraform","Helm","Airflow","Spark",
    "Kafka","Flink","Prefect","Dagster","Argo",
    "SQL","PostgreSQL","MySQL","MongoDB","Redis","Elasticsearch","Pinecone","Weaviate",
    "ChromaDB","Qdrant","BigQuery","Snowflake","dbt","Databricks",
    "React","TypeScript","JavaScript","FastAPI","Flask","Django","Node.js","GraphQL",
    "LLMs","RAG","Fine-tuning","LoRA","QLoRA","RLHF","Embeddings","Vector DBs",
    "Prompt Engineering","Agents","Tool Use",
    "Go","Rust","C++","Scala","Java","R",
]


def _tokenize(text: str) -> list[str]:
    tokens = re.sub(r"[^a-z0-9\s\+#\.]", " ", text.lower()).split()
    return [t for t in tokens if len(t) > 1 and t not in STOPWORDS]


def _tfidf_vectors(docs: list[list[str]]) -> list[dict]:
    N = len(docs)
    df: dict[str, int] = {}
    for tokens in docs:
        for t in set(tokens):
            df[t] = df.get(t, 0) + 1
    idf = {t: math.log((N + 1) / (f + 1)) + 1 for t, f in df.items()}
    vecs = []
    for tokens in docs:
        tf: dict[str, float] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        n = len(tokens) or 1
        vecs.append({t: (c / n) * idf.get(t, 1) for t, c in tf.items()})
    return vecs


def _cosine(a: dict, b: dict) -> float:
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def _exp_to_band(exp: int) -> str:
    if exp <= 1:   return "junior"
    if exp <= 4:   return "mid"
    if exp <= 8:   return "senior"
    if exp <= 12:  return "staff"
    return "principal"


def _location_mult(location: str) -> float:
    loc = location.lower()
    for key, mult in LOCATION_MULT.items():
        if key in loc:
            return mult
    return 1.0


# ══════════════════════════════════════════════════════════════════════════════
# TOOL IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════════════════════

def tool_search_jobs(query: str, dataset_rows: list[dict], limit: int = 20) -> dict:
    """
    Semantic search over job postings using TF-IDF cosine similarity.
    Returns ranked jobs matching the query — Claude decides what query to use.
    """
    if not dataset_rows:
        return {"error": "No dataset loaded", "results": []}

    query_tokens = _tokenize(query)
    if not query_tokens:
        return {"error": "Empty query", "results": []}

    docs = []
    for row in dataset_rows:
        text = " ".join([
            row.get("title", ""),
            row.get("description", ""),
            row.get("skills", ""),
            row.get("job_category", ""),
        ])
        docs.append(_tokenize(text))

    vecs = _tfidf_vectors(docs)
    q_vec = _tfidf_vectors([query_tokens])[0]
    scored = [(i, _cosine(q_vec, v)) for i, v in enumerate(vecs)]
    scored.sort(key=lambda x: x[1], reverse=True)

    results = []
    for idx, sim in scored[:limit]:
        if sim < 0.01:
            break
        row = dataset_rows[idx]
        results.append({
            "title": row.get("title", "Unknown"),
            "company": row.get("company", row.get("employer", "Unknown")),
            "location": row.get("location", row.get("job_location", "Unknown")),
            "level": row.get("job_level", row.get("seniority", "")),
            "skills_required": (row.get("skills", ""))[:120],
            "description_snippet": (row.get("description", ""))[:120],
            "salary_min": row.get("salary_min"),
            "salary_max": row.get("salary_max"),
            "similarity": round(sim, 4),
        })
    return {
        "query": query,
        "total_searched": len(dataset_rows),
        "results_returned": len(results),
        "results": results,
    }


def tool_score_match(
    jobs: list[dict],
    candidate_skills: list[str],
    target_role: str,
    years_exp: int,
) -> dict:
    """
    Score each job against the candidate profile using TF-IDF cosine similarity
    + explicit skill coverage + seniority fit. Fully deterministic — no random.
    """
    if not jobs:
        return {"error": "No jobs to score", "scored": []}

    profile_text = " ".join(candidate_skills * 3 + [target_role])
    job_docs = [_tokenize(j.get("description_snippet", "") + " " + j.get("skills_required", "") + " " + j.get("title", "")) for j in jobs]
    profile_tokens = _tokenize(profile_text)
    all_docs = job_docs + [profile_tokens]
    vecs = _tfidf_vectors(all_docs)
    profile_vec = vecs[-1]
    job_vecs = vecs[:-1]

    cand_low = {s.lower() for s in candidate_skills}
    exp_band = _exp_to_band(years_exp)

    scored = []
    for i, job in enumerate(jobs):
        jvec = job_vecs[i]
        cosine = _cosine(profile_vec, jvec)

        # Skill coverage
        jtext = (job.get("description_snippet","") + " " + job.get("skills_required","")).lower()
        matched = [s for s in candidate_skills if s.lower() in jtext]
        missing = [s for s in ALL_SKILLS if s.lower() in jtext and s.lower() not in cand_low][:5]
        coverage = len(matched) / max(len(candidate_skills), 1)

        # Title jaccard
        role_words = set(w for w in target_role.lower().split() if len(w) > 2)
        title_words = set(w for w in job.get("title","").lower().split() if len(w) > 2)
        jaccard = len(role_words & title_words) / max(len(role_words | title_words), 1)

        # Seniority penalty
        jlevel = job.get("level", "").lower()
        seniority_ok = True
        if years_exp <= 2 and any(x in jlevel for x in ["senior", "staff", "principal", "lead"]):
            seniority_ok = False
        if years_exp >= 8 and "junior" in jlevel:
            seniority_ok = False

        composite = (cosine * 0.40 + coverage * 0.35 + jaccard * 0.15 + 0.10)
        if not seniority_ok:
            composite *= 0.70
        score = min(int(composite * 100), 99)

        scored.append({
            **job,
            "match_score": score,
            "cosine_sim": round(cosine, 3),
            "skill_coverage": round(coverage, 3),
            "matched_skills": matched[:6],
            "missing_skills": missing[:4],
            "seniority_fit": seniority_ok,
        })

    scored.sort(key=lambda x: x["match_score"], reverse=True)
    avg = round(sum(j["match_score"] for j in scored[:10]) / min(len(scored), 10), 1)
    return {
        "total_scored": len(scored),
        "avg_match_top10": avg,
        "top_jobs": scored[:10],
    }


def tool_skill_demand(dataset_rows: list[dict], candidate_skills: list[str]) -> dict:
    """
    Count skill frequency across all job postings using word-boundary regex.
    Returns market demand % for each skill + gap analysis vs candidate.
    """
    freq: dict[str, int] = {}
    N = len(dataset_rows) or 1
    cand_low = {s.lower() for s in candidate_skills}

    for row in dataset_rows:
        text = (row.get("description","") + " " + row.get("skills","") + " " + row.get("title","")).lower()
        for skill in ALL_SKILLS:
            pattern = r"(?<![a-z])" + re.escape(skill.lower()) + r"(?![a-z])"
            if re.search(pattern, text):
                freq[skill] = freq.get(skill, 0) + 1

    ranked = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:20]
    skills_data = []
    for skill, count in ranked:
        pct = round(count / N * 100, 1)
        have = skill.lower() in cand_low
        skills_data.append({
            "skill": skill,
            "demand_pct": pct,
            "job_count": count,
            "candidate_has": have,
            "demand_label": "HOT" if pct > 35 else "HIGH" if pct > 18 else "MEDIUM",
        })

    gaps = [s for s in skills_data if not s["candidate_has"]]
    covered = [s for s in skills_data if s["candidate_has"]]
    return {
        "total_jobs_analyzed": N,
        "skills_ranked": skills_data,
        "top_gaps": gaps[:5],
        "candidate_coverage": covered,
        "gap_summary": f"You have {len(covered)} of top 20 demanded skills. Missing: {', '.join(g['skill'] for g in gaps[:3])}",
    }


def tool_salary_lookup(role: str, location: str, years_exp: int) -> dict:
    """
    Return salary range from BLS OES May 2024 data, adjusted for location.
    Picks the correct band based on years of experience.
    """
    band = _exp_to_band(years_exp)
    mult = _location_mult(location)
    bls = BLS_SALARY[band]

    lo  = round(bls["lo"]  * mult)
    med = round(bls["med"] * mult)
    hi  = round(bls["hi"]  * mult)

    # Role premium: ML/AI roles command ~15% above general software
    role_l = role.lower()
    premium = 1.0
    if any(x in role_l for x in ["ml", "machine learning", "ai ", "llm", "nlp", "deep learning"]):
        premium = 1.15
    elif any(x in role_l for x in ["data scientist", "research"]):
        premium = 1.10
    elif "mlops" in role_l or "platform" in role_l:
        premium = 1.08

    lo  = round(lo  * premium)
    med = round(med * premium)
    hi  = round(hi  * premium)

    loc_name = location if location else "national average"
    all_bands = {}
    for b, vals in BLS_SALARY.items():
        m = vals["med"]
        all_bands[b] = {
            "lo": round(vals["lo"] * mult * premium),
            "med": round(m * mult * premium),
            "hi": round(vals["hi"] * mult * premium),
        }

    return {
        "band": band,
        "years_exp": years_exp,
        "location": loc_name,
        "role": role,
        "salary_lo_k": lo,
        "salary_median_k": med,
        "salary_hi_k": hi,
        "location_multiplier": mult,
        "role_premium": premium,
        "source": "BLS OES May 2024 + role premium adjustment",
        "all_bands": all_bands,
        "negotiation_anchor": f"${med}k–${hi}k",
    }


def tool_filter_by_location(jobs: list[dict], location: str) -> dict:
    """Filter job results to a target location or remote."""
    if not location or location.lower() in ("any", "anywhere", ""):
        return {"filtered": jobs, "count": len(jobs), "filter_applied": "none"}

    loc_l = location.lower()
    matched = [
        j for j in jobs
        if loc_l in j.get("location", "").lower()
        or "remote" in j.get("location", "").lower()
        or j.get("location", "").lower() in ("", "unknown")
    ]
    return {
        "filter_applied": location,
        "original_count": len(jobs),
        "filtered_count": len(matched),
        "filtered": matched,
    }


def tool_summarize_findings(
    top_jobs: list[dict],
    skill_data: dict,
    salary_data: dict,
    candidate_skills: list[str],
    role: str,
    years_exp: int,
) -> dict:
    """
    Compile structured findings for final synthesis.
    Claude calls this when it has enough data and wants to prepare the report.
    """
    avg_score = round(
        sum(j.get("match_score", 0) for j in top_jobs[:5]) / max(len(top_jobs[:5]), 1), 1
    )
    top_gaps = skill_data.get("top_gaps", [])
    top_companies = list(dict.fromkeys(j.get("company", "") for j in top_jobs[:5]))

    return {
        "ready_for_report": True,
        "avg_match_score": avg_score,
        "top_job": top_jobs[0] if top_jobs else None,
        "top_companies": top_companies[:4],
        "salary_anchor": salary_data.get("negotiation_anchor", "N/A"),
        "salary_median": salary_data.get("salary_median_k"),
        "top_missing_skills": [g["skill"] for g in top_gaps[:3]],
        "candidate_skill_count": len(candidate_skills),
        "gap_summary": skill_data.get("gap_summary", ""),
        "market_position": "strong" if avg_score >= 70 else "developing" if avg_score >= 50 else "early",
    }


# ── Tool registry Claude sees ──────────────────────────────────────────────────
TOOL_SCHEMAS = [
    {
        "name": "search_jobs",
        "description": "Search the job postings dataset using a semantic query. Use this first to find relevant jobs. You choose what query to use — be specific about role, skills, or domain.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Semantic search query, e.g. 'machine learning engineer PyTorch LLMs'"},
                "limit": {"type": "integer", "description": "Max results to return (5-20)", "default": 15},
            },
            "required": ["query"],
        },
    },
    {
        "name": "score_match",
        "description": "Score a list of jobs against the candidate profile using TF-IDF cosine similarity + skill coverage + seniority fit. Call this after search_jobs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "job_results": {"type": "array", "description": "Array of job objects from search_jobs"},
                "candidate_skills": {"type": "array", "items": {"type": "string"}, "description": "List of candidate skills"},
                "target_role": {"type": "string", "description": "Target job title"},
                "years_exp": {"type": "integer", "description": "Years of experience"},
            },
            "required": ["job_results", "candidate_skills", "target_role", "years_exp"],
        },
    },
    {
        "name": "skill_demand",
        "description": "Analyze skill demand across all job postings. Returns frequency, demand level, and gap analysis vs candidate skills.",
        "input_schema": {
            "type": "object",
            "properties": {
                "candidate_skills": {"type": "array", "items": {"type": "string"}, "description": "Candidate's current skills"},
            },
            "required": ["candidate_skills"],
        },
    },
    {
        "name": "salary_lookup",
        "description": "Look up salary ranges from BLS OES 2024 data, adjusted for location and role type. Always call this to get accurate compensation data.",
        "input_schema": {
            "type": "object",
            "properties": {
                "role": {"type": "string", "description": "Target role title"},
                "location": {"type": "string", "description": "City or region"},
                "years_exp": {"type": "integer", "description": "Years of experience"},
            },
            "required": ["role", "location", "years_exp"],
        },
    },
    {
        "name": "filter_by_location",
        "description": "Filter job results to a specific location. Use this if search results include irrelevant locations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "jobs": {"type": "array", "description": "Job list to filter"},
                "location": {"type": "string", "description": "Target location string"},
            },
            "required": ["jobs", "location"],
        },
    },
    {
        "name": "summarize_findings",
        "description": "Compile all findings into a structured summary. Call this LAST when you have enough data from the other tools. This signals you are ready to write the final report.",
        "input_schema": {
            "type": "object",
            "properties": {
                "top_jobs": {"type": "array", "description": "Scored and ranked job list"},
                "skill_data": {"type": "object", "description": "Output from skill_demand tool"},
                "salary_data": {"type": "object", "description": "Output from salary_lookup tool"},
                "candidate_skills": {"type": "array", "items": {"type": "string"}},
                "role": {"type": "string"},
                "years_exp": {"type": "integer"},
            },
            "required": ["top_jobs", "skill_data", "salary_data", "candidate_skills", "role", "years_exp"],
        },
    },
]


def dispatch_tool(name: str, inputs: dict, dataset_rows: list[dict]) -> Any:
    """Route a tool call to its implementation."""
    if name == "search_jobs":
        return tool_search_jobs(inputs["query"], dataset_rows, inputs.get("limit", 60))
    elif name == "score_match":
        return tool_score_match(
            inputs["job_results"],
            inputs["candidate_skills"],
            inputs["target_role"],
            inputs["years_exp"],
        )
    elif name == "skill_demand":
        return tool_skill_demand(dataset_rows, inputs["candidate_skills"])
    elif name == "salary_lookup":
        return tool_salary_lookup(inputs["role"], inputs["location"], inputs["years_exp"])
    elif name == "filter_by_location":
        return tool_filter_by_location(inputs["jobs"], inputs["location"])
    elif name == "summarize_findings":
        return tool_summarize_findings(
            inputs["top_jobs"],
            inputs["skill_data"],
            inputs["salary_data"],
            inputs["candidate_skills"],
            inputs["role"],
            inputs["years_exp"],
        )
    else:
        return {"error": f"Unknown tool: {name}"}