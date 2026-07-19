from __future__ import annotations

from typing import cast
from unittest.mock import Mock

import httpx
import pytest

from app.models.explanation_models import (
  CandidateExperience,
  GenerateExplanationRequest,
)
from app.services.explanation_service import (
  ExplanationGenerationError,
  ExplanationService,
)
from app.services.llm_client import LlmClient


class FakeLlmClient(LlmClient):
  """Return predefined model content without making an HTTP request."""

  def __init__(self, content: str) -> None:
    super().__init__(
      base_url="http://llm.test/v1",
      model_name="test-model",
      timeout=1.0,
      http_client=cast(
        httpx.Client,
        Mock(spec=httpx.Client),
      ),
    )
    self.content = content
    self.system_prompt: str | None = None
    self.user_prompt: str | None = None

  def generate(self, system_prompt: str, user_prompt: str) -> str:
    self.system_prompt = system_prompt
    self.user_prompt = user_prompt
    return self.content


@pytest.fixture
def explanation_request() -> GenerateExplanationRequest:
  """Return candidate and opportunity text shared by service tests."""
  return GenerateExplanationRequest(
    candidate_id="candidate-1",
    opportunity_description="Seeking an accountant.",
    experiences=[
      CandidateExperience(
        experience_id="experience-1",
        job_title="Accountant",
        description="Prepared monthly financial reports.",
      )
    ],
  )


def test_validates_successful_generated_json(
    explanation_request: GenerateExplanationRequest,
) -> None:
  llm_client = FakeLlmClient(
    """{
      "candidate_id": "candidate-1",
      "summary": "The supplied experience includes relevant work.",
      "experience_explanations": [
        {
          "experience_id": "experience-1",
          "explanation": "Financial reporting relates to the opportunity."
        }
      ],
      "limitations": ["The supplied text does not cover every requirement."]
    }"""
  )

  response = ExplanationService(llm_client).generate_explanation(
    explanation_request
  )

  assert response.candidate_id == "candidate-1"
  assert response.experience_explanations[0].experience_id == "experience-1"
  assert "candidate-1" in llm_client.user_prompt
  assert "Do not invent" in llm_client.system_prompt
  assert "directly" in llm_client.system_prompt
  assert "search rankings" in llm_client.system_prompt
  assert "candidate_score" not in llm_client.user_prompt
  assert "similarity" not in llm_client.user_prompt


def test_invalid_generated_json_raises_error(
    explanation_request: GenerateExplanationRequest,
) -> None:
  with pytest.raises(ExplanationGenerationError):
    ExplanationService(
      FakeLlmClient("not JSON")
    ).generate_explanation(explanation_request)


def test_invalid_generated_schema_raises_error(
    explanation_request: GenerateExplanationRequest,
) -> None:
  with pytest.raises(ExplanationGenerationError):
    ExplanationService(
      FakeLlmClient('{"candidate_id": "candidate-1"}')
    ).generate_explanation(explanation_request)


def test_generated_candidate_id_must_match_request(
    explanation_request: GenerateExplanationRequest,
) -> None:
  content = """{
    "candidate_id": "invented-candidate",
    "summary": "Summary",
    "experience_explanations": [
      {"experience_id": "experience-1", "explanation": "Explanation"}
    ],
    "limitations": []
  }"""

  with pytest.raises(
      ExplanationGenerationError,
      match="IDs that do not match",
  ):
    ExplanationService(
      FakeLlmClient(content)
    ).generate_explanation(explanation_request)


def test_generated_experience_ids_must_match_request(
    explanation_request: GenerateExplanationRequest,
) -> None:
  content = """{
    "candidate_id": "candidate-1",
    "summary": "Summary",
    "experience_explanations": [
      {"experience_id": "invented-experience", "explanation": "Explanation"}
    ],
    "limitations": []
  }"""

  with pytest.raises(
      ExplanationGenerationError,
      match="IDs that do not match",
  ):
    ExplanationService(
      FakeLlmClient(content)
    ).generate_explanation(explanation_request)
