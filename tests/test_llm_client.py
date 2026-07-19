from __future__ import annotations

import json
from collections.abc import Generator
from typing import Protocol

import httpx
import pytest

from app.services.llm_client import (
  LlmClient,
  LlmServiceUnavailableError,
  MalformedLlmResponseError,
)


class CreateLlmClient(Protocol):
  """Describes the configurable LLM client factory used by tests."""

  def __call__(
      self,
      transport: httpx.MockTransport,
      *,
      base_url: str = "http://llm.test/v1/",
      model_name: str = "test-model",
      api_key: str | None = None,
  ) -> LlmClient:
    """Create an LLM client using the supplied test transport."""


@pytest.fixture
def create_client() -> Generator[
    CreateLlmClient,
    None,
    None,
]:
  """Create an LLM client backed by an in-memory HTTP transport."""
  clients: list[httpx.Client] = []

  def factory(
      transport: httpx.MockTransport,
      *,
      base_url: str = "http://llm.test/v1/",
      model_name: str = "test-model",
      api_key: str | None = None,
  ) -> LlmClient:
    http_client = httpx.Client(transport=transport)
    clients.append(http_client)
    return LlmClient(
      base_url=base_url,
      model_name=model_name,
      timeout=12.0,
      http_client=http_client,
      api_key=api_key,
    )

  yield factory

  for client in clients:
    client.close()


def test_extracts_assistant_message_content(
    create_client: CreateLlmClient,
) -> None:
  def handler(request: httpx.Request) -> httpx.Response:
    assert request.url == "https://provider.test/openai/v1/chat/completions"
    assert "Authorization" not in request.headers
    payload = json.loads(request.content)
    assert payload == {
      "model": "configured-model",
      "messages": [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "evidence"},
      ],
      "temperature": 0.1,
    }
    assert "chat_template_kwargs" not in payload
    return httpx.Response(
      200,
      json={
        "choices": [
          {"message": {"content": "generated content"}}
        ]
      },
    )

  result = create_client(
    httpx.MockTransport(handler),
    base_url="https://provider.test/openai/v1/",
    model_name="configured-model",
  ).generate("system", "evidence")

  assert result == "generated content"


def test_configured_api_key_sends_bearer_authorization(
    create_client: CreateLlmClient,
) -> None:
  def handler(request: httpx.Request) -> httpx.Response:
    assert request.headers["Authorization"] == "Bearer secret-api-key"
    return httpx.Response(
      200,
      json={
        "choices": [
          {"message": {"content": "generated content"}}
        ]
      },
    )

  result = create_client(
    httpx.MockTransport(handler),
    api_key="secret-api-key",
  ).generate("system", "evidence")

  assert result == "generated content"


@pytest.mark.parametrize("api_key", [None, "", "   "])
def test_missing_or_empty_api_key_sends_no_authorization(
    create_client: CreateLlmClient,
    api_key: str | None,
) -> None:
  def handler(request: httpx.Request) -> httpx.Response:
    assert "Authorization" not in request.headers
    return httpx.Response(
      200,
      json={
        "choices": [
          {"message": {"content": "generated content"}}
        ]
      },
    )

  result = create_client(
    httpx.MockTransport(handler),
    api_key=api_key,
  ).generate("system", "evidence")

  assert result == "generated content"


def test_http_failure_raises_service_unavailable_without_exposing_key(
    create_client: CreateLlmClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
  def handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
      503,
      json={"detail": "Provider unavailable"},
    )

  with pytest.raises(
      LlmServiceUnavailableError,
      match="LLM service is unavailable",
  ) as exception_info:
    create_client(
      httpx.MockTransport(handler),
      api_key="secret-api-key",
    ).generate("system", "evidence")

  assert "secret-api-key" not in str(exception_info.value)
  assert "secret-api-key" not in caplog.text


def test_invalid_json_raises_malformed_response(
    create_client: CreateLlmClient,
) -> None:
  def handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=b"not JSON")

  with pytest.raises(
      MalformedLlmResponseError,
      match="malformed chat-completion response",
  ):
    create_client(
      httpx.MockTransport(handler)
    ).generate("system", "evidence")


def test_malformed_chat_completion_raises_clear_error(
    create_client: CreateLlmClient,
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
    create_client: CreateLlmClient,
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
