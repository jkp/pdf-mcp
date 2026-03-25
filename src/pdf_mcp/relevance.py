"""LLM-based relevance scoring for PDF search results.

Same approach as email MCP: single LLM call scores all results
against the query on a 1-5 scale, filters out noise.
"""

from __future__ import annotations

import httpx
import structlog

logger = structlog.get_logger(__name__)

_MODEL = "mistralai/Mistral-Small-24B-Instruct-2501"
_API_URL = "https://api.together.xyz/v1/chat/completions"
_RELEVANCE_THRESHOLD = 3

_SYSTEM_PROMPT = """\
You are a search relevance judge. Given a search query and a list of PDF \
search results with filenames and text snippets, score each result's \
relevance to the query.

Use this scale:
1 = completely unrelated, no connection to the query
2 = tangentially related, shares a broad topic but doesn't address the query
3 = somewhat relevant, related to the query but not a direct match
4 = clearly relevant, directly addresses the query topic
5 = highly relevant, exactly what the searcher is looking for

Output ONLY a comma-separated list of scores, one per result, in the same \
order as the input. Example: 5,3,1,4,2

Do not explain your reasoning. Just output the scores."""


async def score_relevance(
    query: str,
    results: list[dict],
    api_key: str,
    threshold: int = _RELEVANCE_THRESHOLD,
) -> list[dict]:
    """Score and filter search results by relevance.

    Returns only results scoring >= threshold, with relevance_score added.
    Falls back to all results on failure.
    """
    if not api_key or not results:
        return results

    lines = []
    for i, r in enumerate(results, 1):
        snippet = r.get("snippet", "")[:150]
        lines.append(f'{i}. File: {r["filename"]} | Page {r["page_num"]} | {snippet}')

    user_prompt = f"Query: {query}\n\nResults:\n" + "\n".join(lines)

    try:
        scores = await _llm_score(user_prompt, api_key, len(results))
    except Exception as e:
        logger.warning("relevance.score_failed", error=str(e))
        return results

    if not scores or len(scores) != len(results):
        logger.warning(
            "relevance.score_mismatch",
            expected=len(results),
            got=len(scores) if scores else 0,
        )
        return results

    filtered = []
    for r, score in zip(results, scores):
        if score >= threshold:
            r["relevance_score"] = score
            filtered.append(r)

    logger.info(
        "relevance.filtered",
        query=query,
        before=len(results),
        after=len(filtered),
        scores=",".join(str(s) for s in scores),
    )

    return filtered if filtered else results[:3]


async def _llm_score(prompt: str, api_key: str, count: int) -> list[int] | None:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            _API_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": _MODEL,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 100,
                "temperature": 0.0,
            },
            timeout=10,
        )

    if resp.status_code != 200:
        logger.warning("relevance.api_error", status=resp.status_code)
        return None

    data = resp.json()
    text = data["choices"][0]["message"]["content"].strip()

    try:
        scores = [int(s.strip()) for s in text.split(",")]
        return [max(1, min(5, s)) for s in scores]
    except (ValueError, AttributeError):
        logger.warning("relevance.parse_failed", raw=text[:100])
        return None
