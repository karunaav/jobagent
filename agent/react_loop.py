"""
agent/react_loop.py
ReAct agent loop powered by Groq (llama-3.3-70b-versatile).
"""

import json
import time
from groq import Groq
from typing import Generator
from tools.definitions import TOOL_SCHEMAS, dispatch_tool


def _to_groq_tools(schemas: list[dict]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s["description"],
                "parameters": s["input_schema"],
            },
        }
        for s in schemas
    ]


SYSTEM_PROMPT = """You are JobAgent — an expert career intelligence analyst with access to real job market tools.

Your job: analyze a candidate's profile against real job postings and produce a grounded career intelligence report.

TOOLS AVAILABLE:
- search_jobs: semantic search over real HuggingFace job postings
- score_match: TF-IDF cosine similarity scoring of jobs vs candidate profile
- skill_demand: frequency analysis of skills across all postings
- salary_lookup: BLS OES 2024 salary data, location + seniority adjusted
- filter_by_location: filter results to a target city
- summarize_findings: compile all data — call this LAST when ready

RULES:
1. Always start with search_jobs AND skill_demand.
2. After search_jobs, always call score_match on the results.
3. Always call salary_lookup with the candidate's exact role, location, and years_exp.
4. If search results are weak (fewer than 10 results), retry search_jobs with a different query.
5. Only call summarize_findings after you have data from: search_jobs, score_match, skill_demand, salary_lookup.
6. After summarize_findings, write the final report immediately.

REPORT FORMAT (write this after summarize_findings):
## Market Position
## Top Job Matches
## Skill Gap Analysis
## Salary Intelligence
## 30-Day Action Plan

Every sentence must contain a specific number or data point. No filler."""


def run_agent(
    profile: dict,
    dataset_rows: list[dict],
    api_key: str,
) -> Generator[dict, None, None]:
    client = Groq(api_key=api_key)
    groq_tools = _to_groq_tools(TOOL_SCHEMAS)

    candidate_str = (
        f"Candidate profile:\n"
        f"  Role target:        {profile['role']}\n"
        f"  Location:           {profile['location']}\n"
        f"  Years experience:   {profile['years_exp']}\n"
        f"  Skills:             {', '.join(profile['skills'])}\n\n"
        f"Dataset: {len(dataset_rows)} real job postings loaded from HuggingFace.\n\n"
        f"Analyze this candidate's market position and produce a career intelligence report. "
        f"Use your tools to gather real data — do not guess or make up numbers."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": candidate_str},
    ]

    iteration = 0
    max_iterations = 14

    while iteration < max_iterations:
        iteration += 1
        yield {"type": "iteration", "n": iteration}

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=groq_tools,
            tool_choice="auto",
            max_tokens=4096,
            temperature=0.1,
        )

        msg = response.choices[0].message
        finish = response.choices[0].finish_reason

        if msg.content and msg.content.strip():
            yield {"type": "thought", "text": msg.content.strip()}

        if finish == "stop" or not msg.tool_calls:
            final_text = msg.content or "Analysis complete."
            yield {"type": "final_report", "text": final_text}
            break

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ],
        })

        for tc in msg.tool_calls:
            tool_name = tc.function.name
            try:
                tool_inputs = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                tool_inputs = {}

            yield {"type": "tool_call", "name": tool_name, "inputs": tool_inputs}

            start = time.time()
            try:
                result = dispatch_tool(tool_name, tool_inputs, dataset_rows)
                elapsed = round(time.time() - start, 2)
                summary = _summarize_result(tool_name, result)
                yield {
                    "type": "tool_result",
                    "name": tool_name,
                    "summary": summary,
                    "elapsed": elapsed,
                    "full_result": result,
                }
            except Exception as e:
                result = {"error": str(e)}
                elapsed = round(time.time() - start, 2)
                yield {
                    "type": "tool_result",
                    "name": tool_name,
                    "summary": f"Error: {e}",
                    "elapsed": elapsed,
                    "full_result": result,
                }

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result),
            })

            if tool_name == "summarize_findings" and result.get("ready_for_report"):
                yield {"type": "summary_ready", "data": result}

    yield {"type": "done", "iterations": iteration}


def _summarize_result(tool_name: str, result: dict) -> str:
    if "error" in result:
        return f"Error: {result['error']}"
    if tool_name == "search_jobs":
        n = result.get("results_returned", 0)
        total = result.get("total_searched", 0)
        return f"Found {n} relevant jobs from {total} postings · query: '{result.get('query', '')}'"
    if tool_name == "score_match":
        n = result.get("total_scored", 0)
        avg = result.get("avg_match_top10", 0)
        top = (result.get("top_jobs") or [{}])[0]
        return (
            f"Scored {n} jobs · top: '{top.get('title', '')}' "
            f"@ {top.get('company', '')} ({top.get('match_score', 0)}%) · avg top-10: {avg}%"
        )
    if tool_name == "skill_demand":
        return result.get("gap_summary", "Skill demand analyzed.")
    if tool_name == "salary_lookup":
        med = result.get("salary_median_k", "N/A")
        band = result.get("band", "").title()
        anchor = result.get("negotiation_anchor", "")
        loc = result.get("location", "")
        src = result.get("source", "")
        return f"{band} median: ${med}k in {loc} · anchor: {anchor} · {src}"
    if tool_name == "filter_by_location":
        return f"Filtered to {result.get('filtered_count', 0)} jobs in {result.get('filter_applied', '')}"
    if tool_name == "summarize_findings":
        avg = result.get("avg_match_score", 0)
        pos = result.get("market_position", "")
        return f"Findings compiled · market position: {pos} · avg match: {avg}% · ready for report"
    return str(result)[:120]
