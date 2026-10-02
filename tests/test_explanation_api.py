from __future__ import annotations

import httpx
from fastapi.testclient import TestClient

from app.dependencies import get_explanation_service
from app.main import app
from app.models.explanation_models import (
  CandidateExplanationResult,
  ExperienceExplanation,
  ExplanationError,
  ExplanationErrorCode,
  ExplanationsRequest,
  ExplanationsResponse,
)


VALID_REQUEST = {
  "opportunity_description": "Seeking an accountant.",
  "candidates": [
    {
      "candidate_id": "candidate-1",
      "experiences": [
        {
          "experience_id": "experience-1",
          "job_title": "Accountant",
          "description": "Prepared financial reports.",
        }
      ],
    }
  ],
}


class FakeExplanationService:
  """Return a configured result or error without calling an LLM."""

  def __init__(
      self,
      result: ExplanationsResponse | None = None,
      error: Exception | None = None,
  ) -> None:
    self._result = result
    self._error = error

  def generate_explanations(
      self,
      request: ExplanationsRequest,
  ) -> ExplanationsResponse:
    if self._error is not None:
      raise self._error

    if self._result is None:
      raise AssertionError("Fake explanation result was not configured")

    return self._result


def post_with_service(
    service: FakeExplanationService,
    payload: dict | None = None,
) -> httpx.Response:
  """Call the endpoint with its singleton dependency replaced by a fake."""
  app.dependency_overrides[get_explanation_service] = lambda: service
  client = TestClient(app, raise_server_exceptions=False)
  try:
    return client.post(
      "/explanations",
      json=payload if payload is not None else VALID_REQUEST,
    )
  finally:
    client.close()
    app.dependency_overrides.clear()


def test_post_explanations_returns_results_for_multiple_candidates() -> None:
  result = ExplanationsResponse(
    requested=2,
    succeeded=2,
    failed=0,
    results=[
      CandidateExplanationResult(
        candidate_id="candidate-1",
        summary="Relevant supplied experience.",
        experience_explanations=[
          ExperienceExplanation(
            experience_id="experience-1",
            explanation="The description mentions financial reporting.",
          )
        ],
        limitations=[],
      ),
      CandidateExplanationResult(
        candidate_id="candidate-2",
        summary="Also relevant supplied experience.",
        experience_explanations=[
          ExperienceExplanation(
            experience_id="experience-2",
            explanation="The description mentions bookkeeping.",
          )
        ],
        limitations=[],
      ),
    ],
  )

  response = post_with_service(
    FakeExplanationService(result),
    payload={
      "opportunity_description": "Seeking an accountant.",
      "candidates": [
        {
          "candidate_id": "candidate-1",
          "experiences": [
            {"experience_id": "experience-1", "description": "..."}
          ],
        },
        {
          "candidate_id": "candidate-2",
          "experiences": [
            {"experience_id": "experience-2", "description": "..."}
          ],
        },
      ],
    },
  )

  assert response.status_code == 200
  assert response.json() == result.model_dump()


def test_post_explanations_preserves_item_level_failure() -> None:
  """
  An HTTP 200 response can contain both successful and failed candidate
  results, mirroring the embeddings API's approach to item failures.
  """
  result = ExplanationsResponse(
    requested=2,
    succeeded=1,
    failed=1,
    results=[
      CandidateExplanationResult(
        candidate_id="candidate-1",
        summary="Relevant supplied experience.",
        experience_explanations=[
          ExperienceExplanation(
            experience_id="experience-1",
            explanation="The description mentions financial reporting.",
          )
        ],
        limitations=[],
      ),
      CandidateExplanationResult(
        candidate_id="candidate-2",
        error=ExplanationError(
          code=ExplanationErrorCode.LLM_SERVICE_UNAVAILABLE,
          message="The LLM service is unavailable",
        ),
      ),
    ],
  )

  response = post_with_service(FakeExplanationService(result))

  assert response.status_code == 200
  body = response.json()
  assert body == result.model_dump()
  assert body["results"][1]["candidate_id"] == "candidate-2"
  assert body["results"][1]["error"]["code"] == "LLM_SERVICE_UNAVAILABLE"


def test_post_explanations_rejects_empty_candidate_list() -> None:
  response = post_with_service(
    FakeExplanationService(),
    payload={
      "opportunity_description": "Seeking an accountant.",
      "candidates": [],
    },
  )

  assert response.status_code == 422


def test_post_explanations_rejects_duplicate_candidate_ids() -> None:
  response = post_with_service(
    FakeExplanationService(),
    payload={
      "opportunity_description": "Seeking an accountant.",
      "candidates": [
        {
          "candidate_id": "candidate-1",
          "experiences": [
            {"experience_id": "experience-1", "description": "..."}
          ],
        },
        {
          "candidate_id": "candidate-1",
          "experiences": [
            {"experience_id": "experience-2", "description": "..."}
          ],
        },
      ],
    },
  )

  assert response.status_code == 422
