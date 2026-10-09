"""Tests for LLM relevance scoring."""

from unittest.mock import AsyncMock, MagicMock, patch


def _mock_client(content: str) -> MagicMock:
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    client = AsyncMock()
    client.post.return_value = resp
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client


class TestLlmScore:
    async def test_uses_gpt_oss_with_low_reasoning_and_headroom(self) -> None:
        """Mistral-Small-24B stopped being served serverless (HTTP 400 since
        2026-10-07), which silently disabled the filter. gpt-oss-120b reasons
        before answering, so the budget must scale with the result count plus
        room for that reasoning."""
        from pdf_mcp.relevance import _llm_score

        client = _mock_client("5,3,1")
        with patch("pdf_mcp.relevance.httpx.AsyncClient", return_value=client):
            assert await _llm_score("prompt", "key", 3) == [5, 3, 1]

        payload = client.post.call_args.kwargs["json"]
        assert payload["model"] == "openai/gpt-oss-120b"
        assert payload["reasoning_effort"] == "low"
        assert payload["max_tokens"] >= 3 * 8 + 1000

    async def test_budget_scales_with_result_count(self) -> None:
        from pdf_mcp.relevance import _llm_score

        client = _mock_client(",".join(["3"] * 40))
        with patch("pdf_mcp.relevance.httpx.AsyncClient", return_value=client):
            await _llm_score("prompt", "key", 40)

        assert client.post.call_args.kwargs["json"]["max_tokens"] >= 40 * 8 + 1000
