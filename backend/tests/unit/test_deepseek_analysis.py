import json

import httpx
import pytest
from openai import AsyncOpenAI
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.deepseek import DeepSeekProvider

from src import ai
from src.api.routes.admin import SETTING_METADATA
from src.config import Config
from src.runtime_settings import RUNTIME_SETTING_KEYS


@pytest.fixture
def deepseek_config(monkeypatch):
    monkeypatch.setenv("LLM", "deepseek:deepseek-flash")
    monkeypatch.setenv("RERANK_LLM", "deepseek:deepseek-flash")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    return Config()


def test_deepseek_model_uses_official_endpoint(deepseek_config):
    model = ai._build_transcript_model(deepseek_config)

    assert isinstance(model, OpenAIChatModel)
    assert model.model_name == "deepseek-flash"
    assert model.provider.name == "deepseek"
    assert model.provider.base_url == "https://api.deepseek.com"


def test_rerank_agent_uses_deepseek_when_configured(deepseek_config, monkeypatch):
    assert deepseek_config.rerank_llm == "deepseek:deepseek-flash"
    monkeypatch.setattr(ai, "get_config", lambda: deepseek_config)
    monkeypatch.setattr(ai, "_rerank_agent", None)
    monkeypatch.setattr(ai, "_rerank_agent_signature", None)
    monkeypatch.setattr(ai, "apply_settings_to_process_env", lambda settings: None)

    agent = ai.get_rerank_agent()

    assert isinstance(agent.model, OpenAIChatModel)
    assert agent.model.model_name == "deepseek-flash"


def test_missing_deepseek_key_fails_before_request(deepseek_config):
    deepseek_config.deepseek_api_key = None

    error = ai._get_missing_llm_key_error(deepseek_config.llm, deepseek_config)

    assert "DEEPSEEK_API_KEY" in error


def test_every_runtime_setting_has_admin_metadata():
    assert set(RUNTIME_SETTING_KEYS) <= set(SETTING_METADATA)


async def test_deepseek_structured_output_uses_tool_choice_auto(deepseek_config, monkeypatch):
    """The SDK request must not force tool_choice, which thinking mode rejects."""
    captured = []
    analysis = ai.TranscriptAnalysis(
        most_relevant_segments=[
            ai.TranscriptSegment(
                start_time="00:00",
                end_time="00:30",
                text="A complete explanation with a practical takeaway.",
            )
        ],
        summary="Practical advice",
        key_topics=["advice"],
    )

    def handler(request):
        assert request.url.host == "api.deepseek.com"
        assert request.headers["Authorization"] == "Bearer test-deepseek-key"
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "test-completion",
                "object": "chat.completion",
                "created": 1,
                "model": "deepseek-flash",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "final_result",
                                        "arguments": analysis.model_dump_json(),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 100, "total_tokens": 200},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        sdk = AsyncOpenAI(
            api_key="test-deepseek-key",
            base_url="https://api.deepseek.com",
            http_client=client,
        )
        real_provider = DeepSeekProvider
        monkeypatch.setattr(
            ai,
            "DeepSeekProvider",
            type(
                "TestProvider",
                (),
                {
                    "model_profile": staticmethod(real_provider.model_profile),
                    "__new__": lambda cls, **kwargs: real_provider(openai_client=sdk),
                },
            ),
        )
        model = ai._build_transcript_model(deepseek_config)
        agent = Agent(model, output_type=ai.TranscriptAnalysis)
        result = await agent.run("Select a clip from the transcript.")

    assert result.output == analysis
    body = captured[0]
    assert body["model"] == "deepseek-flash"
    assert body["tool_choice"] == "auto"
    assert body["max_completion_tokens"] == ai.DEEPSEEK_MAX_TOKENS
    assert body["tools"][0]["function"]["name"] == "final_result"
