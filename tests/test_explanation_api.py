from __future__ import annotations

import httpx
from fastapi.testclient import TestClient

from app.dependencies import get_explanation_service
from app.main import app
from app.models.explanation_models import (
  ExperienceExplanation,
  GenerateExplanationRequest,
  GenerateExplanationResponse,
)
from app.services.explanation_service import ExplanationGenerationError
from app.services.llm_client import LlmServiceUnavailableError


VALID_REQUEST = {
  "candidate_id": "candidate-1",
  "rank": 1,
  "candidate_score": 0.9,
  "opportunity_description": "Seeking an accountant.",
  "matching_experiences": [
    {
      "experience_id": "experience-1",
      "job_title": "Accountant",
      "description": "Prepared financial reports.",
      "similarity": 0.8,
    }
  ],
}


class FakeExplanationService:
  """Return a configured result or error without calling an LLM."""

  def __init__(
      self,
      result: GenerateExplanationResponse | None = None,
      error: Exception | None = None,
  ) -> None:
    self._result = result
    self._error = error

  def generate_explanation(
      self,
      request: GenerateExplanationRequest,
  ) -> GenerateExplanationResponse:
    if self._error is not None:
      raise self._error

    if self._result is None:
      raise AssertionError("Fake explanation result was not configured")

    return self._result


def post_with_service(
    service: FakeExplanationService,
) -> httpx.Response:
  """Call the endpoint with its singleton dependency replaced by a fake."""
  app.dependency_overrides[get_explanation_service] = lambda: service
  client = TestClient(app, raise_server_exceptions=False)
  try:
    return client.post("/explanations", json=VALID_REQUEST)
  finally:
    client.close()
    app.dependency_overrides.clear()


def test_post_explanations_returns_generated_explanation() -> None:
  response_model = GenerateExplanationResponse(
    candidate_id="candidate-1",
    summary="Relevant supplied experience.",
    ranking_basis="The supplied evidence supports the existing rank.",
    experience_explanations=[
      ExperienceExplanation(
        experience_id="experience-1",
        explanation="The description mentions financial reporting.",
      )
    ],
    limitations=["Semantic similarity is not proof of all requirements."],
  )

  response = post_with_service(FakeExplanationService(response_model))

  assert response.status_code == 200
  assert response.json() == response_model.model_dump()


def test_post_explanations_maps_unavailable_llm_to_503() -> None:
  response = post_with_service(
    FakeExplanationService(
      error=LlmServiceUnavailableError("LLM unavailable")
    )
  )

  assert response.status_code == 503
  assert response.json() == {"detail": "LLM unavailable"}


def test_post_explanations_maps_invalid_output_to_502() -> None:
  response = post_with_service(
    FakeExplanationService(
      error=ExplanationGenerationError("Invalid model output")
    )
  )

  assert response.status_code == 502
  assert response.json() == {"detail": "Invalid model output"}
