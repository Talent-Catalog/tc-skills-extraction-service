from __future__ import annotations

import json
from collections.abc import Callable, Generator

import httpx
import pytest

from app.services.llm_client import (
  LlmClient,
  LlmServiceUnavailableError,
  MalformedLlmResponseError,
)


@pytest.fixture
def create_client() -> Generator[
    Callable[[httpx.MockTransport], LlmClient],
    None,
    None,
]:
  """Create an LLM client backed by an in-memory HTTP transport."""
  clients: list[httpx.Client] = []

  def factory(transport: httpx.MockTransport) -> LlmClient:
    http_client = httpx.Client(transport=transport)
    clients.append(http_client)
    return LlmClient(
      base_url="http://llm.test/v1/",
      model_name="test-model",
      timeout=12.0,
      http_client=http_client,
    )

  yield factory

  for client in clients:
    client.close()


def test_extracts_assistant_message_content(
    create_client: Callable[[httpx.MockTransport], LlmClient],
) -> None:
  def handler(request: httpx.Request) -> httpx.Response:
    assert request.url == "http://llm.test/v1/chat/completions"
    payload = json.loads(request.content)
    assert payload["model"] == "test-model"
    assert payload["messages"] == [
      {"role": "system", "content": "system"},
      {"role": "user", "content": "evidence"},
    ]
    assert payload["chat_template_kwargs"] == {
      "enable_thinking": False,
    }
    assert payload["temperature"] == 0.1
    return httpx.Response(
      200,
      json={
        "choices": [
          {"message": {"content": "generated content"}}
        ]
      },
    )

  result = create_client(
    httpx.MockTransport(handler)
  ).generate("system", "evidence")

  assert result == "generated content"


def test_unavailable_endpoint_raises_clear_error(
    create_client: Callable[[httpx.MockTransport], LlmClient],
) -> None:
  def handler(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("Connection refused", request=request)

  with pytest.raises(
      LlmServiceUnavailableError,
      match="LLM service is unavailable",
  ):
    create_client(
      httpx.MockTransport(handler)
    ).generate("system", "evidence")


def test_malformed_chat_completion_raises_clear_error(
    create_client: Callable[[httpx.MockTransport], LlmClient],
) -> None:
  def handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"choices": []})

  with pytest.raises(
      MalformedLlmResponseError,
      match="malformed chat-completion response",
  ):
    create_client(
      httpx.MockTransport(handler)
    ).generate("system", "evidence")


def test_empty_assistant_content_is_malformed(
    create_client: Callable[[httpx.MockTransport], LlmClient],
) -> None:
  def handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
      200,
      json={
        "choices": [
          {"message": {"content": "   "}}
        ]
      },
    )

  with pytest.raises(
      MalformedLlmResponseError,
      match="empty or non-text",
  ):
    create_client(
      httpx.MockTransport(handler)
    ).generate("system", "evidence")
