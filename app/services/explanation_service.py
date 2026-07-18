from __future__ import annotations

from collections import Counter
import json

from pydantic import ValidationError

from app.models.explanation_models import (
  GenerateExplanationRequest,
  GenerateExplanationResponse,
)
from app.services.llm_client import LlmClient


class ExplanationGenerationError(RuntimeError):
  """Raised when generated explanation JSON cannot be validated."""


class ExplanationService:
  """
  Generates explanations grounded only in supplied ranking evidence.

  The separately hosted Qwen inference process owns model resources, keeping
  this API process lightweight and independently scalable.
  """

  SYSTEM_PROMPT = """
You explain an existing candidate ranking using only the evidence supplied by
the user. Do not calculate, alter, or second-guess the rank or candidate score.
Do not invent skills, qualifications, dates, experience, proficiency, or any
other candidate facts. Vector similarity indicates semantic alignment, not
proof that the candidate meets all opportunity requirements. State relevant
limitations or missing evidence clearly.

Return JSON only, without Markdown fences, matching this exact structure:
{
  "candidate_id": "string",
  "summary": "string",
  "ranking_basis": "string",
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
      request: GenerateExplanationRequest,
  ) -> GenerateExplanationResponse:
    """Explain supplied ranking evidence and validate the generated JSON."""
    user_prompt = (
      "Explain this supplied ranking evidence only:\n"
      f"{request.model_dump_json()}"
    )

    content = self._llm_client.generate(
      system_prompt=self.SYSTEM_PROMPT,
      user_prompt=user_prompt,
    )

    try:
      parsed = json.loads(content)
      response = GenerateExplanationResponse.model_validate(parsed)
    except (json.JSONDecodeError, ValidationError, TypeError) as exception:
      raise ExplanationGenerationError(
        "The LLM generated an invalid explanation response"
      ) from exception

    expected_experience_ids = Counter(
      experience.experience_id
      for experience in request.matching_experiences
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
