from __future__ import annotations

import pytest
from pydantic import AnyHttpUrl

from app.services.llm_client import LlmAuthentication
from app.settings import Settings

_SKILLS_BASE_URL = AnyHttpUrl("http://skills.test")


@pytest.fixture(autouse=True)
def _clear_llm_environment(monkeypatch: pytest.MonkeyPatch) -> None:
  """
  These tests assert Settings' own defaults, so they must not be affected
  by LLM-specific environment variables the developer happens to have set.
  """
  for name in [
    "LLM_BASE_URL",
    "LLM_AWS_REGION",
    "LLM_MODEL_NAME",
    "LLM_AUTHENTICATION",
    "LLM_API_KEY",
    "LLM_REQUEST_TIMEOUT_SECONDS",
  ]:
    monkeypatch.delenv(name, raising=False)


def test_aws_sigv4_is_the_default_authentication_mode() -> None:
  settings = Settings(SKILLS_BASE_URL=_SKILLS_BASE_URL)

  assert settings.llm_authentication == LlmAuthentication.AWS_SIGV4


def test_no_llm_specific_configuration_is_required_by_default() -> None:
  """
  The point of AWS_SIGV4 as the default is that an appropriately authorised
  developer needs no LLM-specific environment variables at all.
  """
  settings = Settings(SKILLS_BASE_URL=_SKILLS_BASE_URL)

  assert settings.llm_base_url is None
  assert settings.llm_api_key is None
  assert settings.llm_aws_region == "eu-west-2"


def _resolve_llm_base_url(settings: Settings) -> str:
  """Mirrors the base-URL construction in main.py's lifespan function."""
  return settings.llm_base_url or (
    f"https://bedrock-runtime.{settings.llm_aws_region}.amazonaws.com/openai/v1"
  )


def test_default_bedrock_url_is_constructed_from_the_configured_region() -> None:
  settings = Settings(
    SKILLS_BASE_URL=_SKILLS_BASE_URL,
    llm_aws_region="us-east-1",
  )

  assert _resolve_llm_base_url(settings) == (
    "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1"
  )


def test_explicit_llm_base_url_overrides_the_default_bedrock_url() -> None:
  settings = Settings(
    SKILLS_BASE_URL=_SKILLS_BASE_URL,
    llm_base_url="http://vllm.internal/v1",
  )

  assert _resolve_llm_base_url(settings) == "http://vllm.internal/v1"
