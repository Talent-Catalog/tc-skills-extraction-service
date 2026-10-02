from __future__ import annotations

import json
import logging
from typing import cast
from unittest.mock import Mock

import httpx
import pytest

from app.models.explanation_models import (
  BatchCandidateExplanation,
  CandidateExperience,
  ExplanationErrorCode,
  ExplanationRequest,
  ExplanationsRequest,
)
from app.services.explanation_service import (
  ExplanationGenerationError,
  ExplanationService,
)
from app.services.llm_client import (
  LlmClient,
  LlmResult,
  LlmServiceUnavailableError,
  LlmUsage,
)


class FakeLlmClient(LlmClient):
  """Return predefined model content without making an HTTP request."""

  def __init__(
      self,
      content: str,
      usage: LlmUsage = LlmUsage(
        prompt_tokens=11,
        completion_tokens=22,
        total_tokens=33,
      ),
  ) -> None:
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
    self.usage = usage
    self.system_prompt: str | None = None
    self.user_prompt: str | None = None

  def generate(self, system_prompt: str, user_prompt: str) -> LlmResult:
    self.system_prompt = system_prompt
    self.user_prompt = user_prompt
    return LlmResult(content=self.content, usage=self.usage)


VALID_GENERATED_CONTENT = """{
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


def _valid_content(candidate_id: str, experience_id: str) -> str:
  """Build generated JSON that echoes back the given IDs, as the LLM would."""
  return json.dumps({
    "candidate_id": candidate_id,
    "summary": f"Summary for {candidate_id}.",
    "experience_explanations": [
      {
        "experience_id": experience_id,
        "explanation": f"Explanation for {experience_id}.",
      }
    ],
    "limitations": [],
  })


class ScriptedLlmClient(LlmClient):
  """
  Returns a scripted sequence of results (or raises a scripted exception)
  from generate(), one per call, in order - so each candidate in a batch
  can be given its own outcome.
  """

  def __init__(self, outcomes: list[LlmResult | Exception]) -> None:
    super().__init__(
      base_url="http://llm.test/v1",
      model_name="test-model",
      timeout=1.0,
      http_client=cast(
        httpx.Client,
        Mock(spec=httpx.Client),
      ),
    )
    self._outcomes = list(outcomes)
    self.user_prompts: list[str] = []

  def generate(self, system_prompt: str, user_prompt: str) -> LlmResult:
    self.user_prompts.append(user_prompt)
    outcome = self._outcomes.pop(0)

    if isinstance(outcome, Exception):
      raise outcome

    return outcome


@pytest.fixture
def explanation_request() -> ExplanationRequest:
  """Return candidate and opportunity text shared by service tests."""
  return ExplanationRequest(
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
    explanation_request: ExplanationRequest,
) -> None:
  llm_client = FakeLlmClient(VALID_GENERATED_CONTENT)

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


def test_token_usage_is_logged(
    explanation_request: ExplanationRequest,
    caplog: pytest.LogCaptureFixture,
) -> None:
  llm_client = FakeLlmClient(
    VALID_GENERATED_CONTENT,
    usage=LlmUsage(
      prompt_tokens=101,
      completion_tokens=202,
      total_tokens=303,
    ),
  )

  with caplog.at_level(logging.INFO):
    ExplanationService(llm_client).generate_explanation(explanation_request)

  assert "candidate-1" in caplog.text
  assert "101" in caplog.text
  assert "202" in caplog.text
  assert "303" in caplog.text


def test_token_usage_is_logged_even_when_the_response_is_invalid(
    explanation_request: ExplanationRequest,
    caplog: pytest.LogCaptureFixture,
) -> None:
  """The tokens were spent whether or not the generated JSON validates."""
  llm_client = FakeLlmClient(
    "not JSON",
    usage=LlmUsage(
      prompt_tokens=101,
      completion_tokens=202,
      total_tokens=303,
    ),
  )

  with caplog.at_level(logging.INFO):
    with pytest.raises(ExplanationGenerationError):
      ExplanationService(llm_client).generate_explanation(explanation_request)

  assert "303" in caplog.text


def test_invalid_generated_json_raises_error(
    explanation_request: ExplanationRequest,
) -> None:
  with pytest.raises(ExplanationGenerationError):
    ExplanationService(
      FakeLlmClient("not JSON")
    ).generate_explanation(explanation_request)


def test_invalid_generated_schema_raises_error(
    explanation_request: ExplanationRequest,
) -> None:
  with pytest.raises(ExplanationGenerationError):
    ExplanationService(
      FakeLlmClient('{"candidate_id": "candidate-1"}')
    ).generate_explanation(explanation_request)


def test_generated_candidate_id_must_match_request(
    explanation_request: ExplanationRequest,
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


def test_html_is_stripped_from_the_llm_prompt() -> None:
  """
  HTML in opportunity_description, description and job_title must be
  converted to readable text before it reaches the LLM prompt.
  """
  request = ExplanationRequest(
    candidate_id="candidate-1",
    opportunity_description=(
      "<p>Java developer</p><ul><li>Spring Boot</li><li>PostgreSQL</li></ul>"
    ),
    experiences=[
      CandidateExperience(
        experience_id="experience-1",
        job_title="<strong>Senior</strong> Accountant",
        description="<p>Prepared <em>monthly</em> financial reports.</p>",
      )
    ],
  )

  llm_client = FakeLlmClient(VALID_GENERATED_CONTENT)

  ExplanationService(llm_client).generate_explanation(request)

  assert "<p>" not in llm_client.user_prompt
  assert "<li>" not in llm_client.user_prompt
  assert "<strong>" not in llm_client.user_prompt
  assert "<em>" not in llm_client.user_prompt
  assert "Java developer" in llm_client.user_prompt
  assert "Spring Boot" in llm_client.user_prompt
  assert "PostgreSQL" in llm_client.user_prompt
  assert "Senior Accountant" in llm_client.user_prompt
  assert "Prepared monthly financial reports." in llm_client.user_prompt
  # IDs are untouched by cleaning.
  assert "candidate-1" in llm_client.user_prompt
  assert "experience-1" in llm_client.user_prompt


def test_html_cleaning_does_not_mutate_the_original_request() -> None:
  """The caller's Pydantic request object must not be modified in place."""
  original_description = "<p>Prepared <em>monthly</em> financial reports.</p>"
  original_job_title = "<strong>Senior</strong> Accountant"
  original_opportunity_description = "<p>Java developer</p>"

  request = ExplanationRequest(
    candidate_id="candidate-1",
    opportunity_description=original_opportunity_description,
    experiences=[
      CandidateExperience(
        experience_id="experience-1",
        job_title=original_job_title,
        description=original_description,
      )
    ],
  )

  ExplanationService(FakeLlmClient(VALID_GENERATED_CONTENT)).generate_explanation(
    request
  )

  assert request.opportunity_description == original_opportunity_description
  assert request.experiences[0].description == original_description
  assert request.experiences[0].job_title == original_job_title


def _batch_candidate(candidate_id: str, experience_id: str) -> BatchCandidateExplanation:
  return BatchCandidateExplanation(
    candidate_id=candidate_id,
    experiences=[
      CandidateExperience(
        experience_id=experience_id,
        job_title="Accountant",
        description="Prepared monthly financial reports.",
      )
    ],
  )


def test_generate_explanations_returns_a_result_per_candidate_in_order(
) -> None:
  llm_client = ScriptedLlmClient([
    LlmResult(
      content=_valid_content("candidate-1", "experience-1"),
      usage=LlmUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    ),
    LlmResult(
      content=_valid_content("candidate-2", "experience-2"),
      usage=LlmUsage(prompt_tokens=3, completion_tokens=3, total_tokens=6),
    ),
  ])

  request = ExplanationsRequest(
    opportunity_description="Seeking an accountant.",
    candidates=[
      _batch_candidate("candidate-1", "experience-1"),
      _batch_candidate("candidate-2", "experience-2"),
    ],
  )

  response = ExplanationService(llm_client).generate_explanations(request)

  assert response.requested == 2
  assert response.succeeded == 2
  assert response.failed == 0

  assert [result.candidate_id for result in response.results] == [
    "candidate-1",
    "candidate-2",
  ]
  assert response.results[0].error is None
  assert response.results[0].summary == "Summary for candidate-1."
  assert (
    response.results[0].experience_explanations[0].experience_id
    == "experience-1"
  )
  assert response.results[1].summary == "Summary for candidate-2."


def test_generate_explanations_strips_html_for_every_candidate() -> None:
  """
  HTML must be cleaned independently for every candidate's own experiences
  before its own (separate) LLM call - not just the first one in the batch.
  """
  llm_client = ScriptedLlmClient([
    LlmResult(
      content=_valid_content("candidate-1", "experience-1"),
      usage=LlmUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    ),
    LlmResult(
      content=_valid_content("candidate-2", "experience-2"),
      usage=LlmUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    ),
  ])

  request = ExplanationsRequest(
    opportunity_description="<p>Java developer</p><ul><li>Spring Boot</li></ul>",
    candidates=[
      BatchCandidateExplanation(
        candidate_id="candidate-1",
        experiences=[
          CandidateExperience(
            experience_id="experience-1",
            job_title="<strong>Senior</strong> Accountant",
            description="<p>Prepared <em>monthly</em> reports.</p>",
          )
        ],
      ),
      BatchCandidateExplanation(
        candidate_id="candidate-2",
        experiences=[
          CandidateExperience(
            experience_id="experience-2",
            job_title=None,
            description="<ul><li>Managed payroll</li><li>Ran audits</li></ul>",
          )
        ],
      ),
    ],
  )

  ExplanationService(llm_client).generate_explanations(request)

  assert len(llm_client.user_prompts) == 2

  first_prompt, second_prompt = llm_client.user_prompts

  assert "<p>" not in first_prompt and "<p>" not in second_prompt
  assert "<li>" not in first_prompt and "<li>" not in second_prompt

  assert "Java developer" in first_prompt
  assert "Spring Boot" in first_prompt
  assert "Senior Accountant" in first_prompt
  assert "Prepared monthly reports." in first_prompt

  assert "Java developer" in second_prompt
  assert "Managed payroll" in second_prompt
  assert "Ran audits" in second_prompt


def test_generate_explanations_one_candidate_failure_does_not_fail_others(
) -> None:
  """
  A candidate whose LLM call fails entirely (LlmServiceUnavailableError) is
  reported as an item-level LLM_SERVICE_UNAVAILABLE error, without
  preventing the surrounding candidates from succeeding.
  """
  llm_client = ScriptedLlmClient([
    LlmResult(
      content=_valid_content("candidate-1", "experience-1"),
      usage=LlmUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    ),
    LlmServiceUnavailableError("The LLM service is unavailable"),
    LlmResult(
      content=_valid_content("candidate-3", "experience-3"),
      usage=LlmUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    ),
  ])

  request = ExplanationsRequest(
    opportunity_description="Seeking an accountant.",
    candidates=[
      _batch_candidate("candidate-1", "experience-1"),
      _batch_candidate("candidate-2", "experience-2"),
      _batch_candidate("candidate-3", "experience-3"),
    ],
  )

  response = ExplanationService(llm_client).generate_explanations(request)

  assert response.requested == 3
  assert response.succeeded == 2
  assert response.failed == 1

  assert [result.candidate_id for result in response.results] == [
    "candidate-1",
    "candidate-2",
    "candidate-3",
  ]

  failed_result = response.results[1]
  assert failed_result.error.code == ExplanationErrorCode.LLM_SERVICE_UNAVAILABLE
  assert "unavailable" in failed_result.error.message
  assert failed_result.summary is None

  assert response.results[0].summary == "Summary for candidate-1."
  assert response.results[2].summary == "Summary for candidate-3."


def test_generate_explanations_invalid_llm_output_is_an_item_level_error(
) -> None:
  """
  A candidate whose LLM call succeeds but returns unusable content is
  reported as an item-level INVALID_LLM_OUTPUT error - distinct from a
  wholesale LLM service outage.
  """
  llm_client = ScriptedLlmClient([
    LlmResult(
      content="not JSON",
      usage=LlmUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
    ),
  ])

  request = ExplanationsRequest(
    opportunity_description="Seeking an accountant.",
    candidates=[_batch_candidate("candidate-1", "experience-1")],
  )

  response = ExplanationService(llm_client).generate_explanations(request)

  assert response.failed == 1
  assert (
    response.results[0].error.code == ExplanationErrorCode.INVALID_LLM_OUTPUT
  )


def test_generate_explanations_logs_token_usage_for_every_candidate(
    caplog: pytest.LogCaptureFixture,
) -> None:
  llm_client = ScriptedLlmClient([
    LlmResult(
      content=_valid_content("candidate-1", "experience-1"),
      usage=LlmUsage(prompt_tokens=101, completion_tokens=201, total_tokens=302),
    ),
    LlmResult(
      content=_valid_content("candidate-2", "experience-2"),
      usage=LlmUsage(prompt_tokens=103, completion_tokens=203, total_tokens=306),
    ),
  ])

  request = ExplanationsRequest(
    opportunity_description="Seeking an accountant.",
    candidates=[
      _batch_candidate("candidate-1", "experience-1"),
      _batch_candidate("candidate-2", "experience-2"),
    ],
  )

  with caplog.at_level(logging.INFO):
    ExplanationService(llm_client).generate_explanations(request)

  assert "candidate-1" in caplog.text
  assert "302" in caplog.text
  assert "candidate-2" in caplog.text
  assert "306" in caplog.text


def test_generate_explanations_empty_candidate_list_is_rejected() -> None:
  with pytest.raises(ValueError):
    ExplanationsRequest(
      opportunity_description="Seeking an accountant.",
      candidates=[],
    )


def test_generate_explanations_duplicate_candidate_ids_are_rejected() -> None:
  with pytest.raises(ValueError, match="unique"):
    ExplanationsRequest(
      opportunity_description="Seeking an accountant.",
      candidates=[
        _batch_candidate("candidate-1", "experience-1"),
        _batch_candidate("candidate-1", "experience-2"),
      ],
    )


def test_generated_experience_ids_must_match_request(
    explanation_request: ExplanationRequest,
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
