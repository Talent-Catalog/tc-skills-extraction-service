from __future__ import annotations

from collections import Counter
import json
import logging

from pydantic import ValidationError

from app.models.explanation_models import (
  BatchCandidateExplanation,
  CandidateExplanationResult,
  ExplanationError,
  ExplanationErrorCode,
  ExplanationRequest,
  ExplanationResponse,
  ExplanationsRequest,
  ExplanationsResponse,
)
from app.services.llm_client import (
  LlmClient,
  LlmServiceUnavailableError,
  MalformedLlmResponseError,
)

logger = logging.getLogger(__name__)


class ExplanationGenerationError(RuntimeError):
  """Raised when generated explanation JSON cannot be validated."""


class ExplanationService:
  """
  Compares candidate experience text with an opportunity description.

  Explanations are grounded only in the supplied texts and do not use hybrid
  search ranks, scores, or vector similarities.

  LLM inference is provided by a separately hosted OpenAI-compatible
  service, keeping this API process lightweight and independently scalable.
  """

  SYSTEM_PROMPT = """
Compare the supplied candidate experience text directly with the supplied
opportunity description. Base every statement only on those texts. Do not use,
infer, or mention search rankings, scores, vector similarity, or other matching
engine output. Do not invent skills, qualifications, dates, experience,
proficiency, opportunity requirements, or any other facts. Distinguish between
requirements supported by the experience text and requirements for which the
supplied text provides no evidence. State relevant limitations clearly.

Return JSON only, without Markdown fences, matching this exact structure:
{
  "candidate_id": "string",
  "summary": "string",
  "experience_explanations": [
    {"experience_id": "string", "explanation": "string"}
  ],
  "limitations": ["string"]
}
Preserve the supplied candidate_id and experience_id values exactly.
""".strip()

  def __init__(self, llm_client: LlmClient) -> None:
    self._llm_client = llm_client

  def generate_explanation(
      self,
      request: ExplanationRequest,
  ) -> ExplanationResponse:
    """Compare the supplied texts and validate the generated explanation."""
    user_prompt = (
      "Compare these candidate experiences with the opportunity using only "
      "the supplied text:\n"
      f"{request.model_dump_json()}"
    )

    result = self._llm_client.generate(
      system_prompt=self.SYSTEM_PROMPT,
      user_prompt=user_prompt,
    )

    # Logged before validation because the tokens were spent either way.
    logger.info(
      "Explanation generated for candidate %s using %d prompt tokens, "
      "%d completion tokens, %d tokens in total.",
      request.candidate_id,
      result.usage.prompt_tokens,
      result.usage.completion_tokens,
      result.usage.total_tokens,
    )

    try:
      parsed = json.loads(result.content)
      response = ExplanationResponse.model_validate(parsed)
    except (json.JSONDecodeError, ValidationError, TypeError) as exception:
      raise ExplanationGenerationError(
        "The LLM generated an invalid explanation response"
      ) from exception

    expected_experience_ids = Counter(
      experience.experience_id
      for experience in request.experiences
    )
    generated_experience_ids = Counter(
      explanation.experience_id
      for explanation in response.experience_explanations
    )

    if (
        response.candidate_id != request.candidate_id
        or generated_experience_ids != expected_experience_ids
    ):
      raise ExplanationGenerationError(
        "The LLM generated explanation IDs that do not match the request"
      )

    return response

  def generate_explanations(
      self,
      request: ExplanationsRequest,
  ) -> ExplanationsResponse:
    """
    Generate one explanation for every candidate in the batch.

    Each candidate is explained independently - one LLM call per candidate,
    reusing generate_explanation() - so a failure for one candidate is
    reported as an item-level error without preventing the others from
    succeeding.
    """
    results = [
      self._generate_candidate_result(
        opportunity_description=request.opportunity_description,
        candidate=candidate,
      )
      for candidate in request.candidates
    ]

    succeeded = sum(result.error is None for result in results)
    failed = len(results) - succeeded

    logger.info(
      "Explanation batch completed: %d requested, %d succeeded, %d failed.",
      len(request.candidates),
      succeeded,
      failed,
    )

    return ExplanationsResponse(
      requested=len(request.candidates),
      succeeded=succeeded,
      failed=failed,
      results=results,
    )

  def _generate_candidate_result(
      self,
      opportunity_description: str,
      candidate: BatchCandidateExplanation,
  ) -> CandidateExplanationResult:
    """Generate one candidate's result, without raising on item-level failures."""
    try:
      response = self.generate_explanation(
        ExplanationRequest(
          candidate_id=candidate.candidate_id,
          opportunity_description=opportunity_description,
          experiences=candidate.experiences,
        )
      )

    except LlmServiceUnavailableError as exception:
      return self._failure(
        candidate_id=candidate.candidate_id,
        code=ExplanationErrorCode.LLM_SERVICE_UNAVAILABLE,
        message=str(exception),
      )

    except (
        MalformedLlmResponseError,
        ExplanationGenerationError,
    ) as exception:
      return self._failure(
        candidate_id=candidate.candidate_id,
        code=ExplanationErrorCode.INVALID_LLM_OUTPUT,
        message=str(exception),
      )

    return CandidateExplanationResult(
      candidate_id=response.candidate_id,
      summary=response.summary,
      experience_explanations=response.experience_explanations,
      limitations=response.limitations,
    )

  @staticmethod
  def _failure(
      candidate_id: str,
      code: ExplanationErrorCode,
      message: str,
  ) -> CandidateExplanationResult:
    """Create a consistent item-level failure result."""
    return CandidateExplanationResult(
      candidate_id=candidate_id,
      error=ExplanationError(
        code=code,
        message=message,
      ),
    )
