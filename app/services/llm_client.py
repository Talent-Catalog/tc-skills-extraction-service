from __future__ import annotations

from typing import Any

import httpx


class LlmServiceUnavailableError(RuntimeError):
  """Raised when the configured LLM service cannot complete a request."""


class MalformedLlmResponseError(RuntimeError):
  """Raised when an LLM response is not a valid chat completion."""


class LlmClient:
  """
  Communicates with any OpenAI-compatible LLM endpoint.

  This provider-independent boundary contains only chat transport concerns,
  allowing the inference server or model to change without changing domain
  prompt logic.
  """

  def __init__(
      self,
      base_url: str,
      model_name: str,
      timeout: float,
      http_client: httpx.Client,
  ) -> None:
    self._base_url = base_url.rstrip("/")
    self._model_name = model_name
    self._timeout = timeout
    self._http_client = http_client

  def generate(
      self,
      system_prompt: str,
      user_prompt: str,
  ) -> str:
    """Generate assistant message content from the supplied prompts."""
    try:
      response = self._http_client.post(
        f"{self._base_url}/chat/completions",
        json={
          "model": self._model_name,
          "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
          ],
          "chat_template_kwargs": {
            "enable_thinking": False,
          },
          "temperature": 0.1,
        },
        timeout=self._timeout,
      )
      response.raise_for_status()
    except httpx.HTTPError as exception:
      raise LlmServiceUnavailableError(
        "The LLM service is unavailable"
      ) from exception

    try:
      payload: Any = response.json()
      content = payload["choices"][0]["message"]["content"]
    except (
        ValueError,
        UnicodeError,
        KeyError,
        IndexError,
        TypeError,
    ) as exception:
      raise MalformedLlmResponseError(
        "The LLM service returned a malformed chat-completion response"
      ) from exception

    if not isinstance(content, str) or not content.strip():
      raise MalformedLlmResponseError(
        "The LLM service returned empty or non-text assistant content"
      )

    return content
