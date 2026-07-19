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
  Compares candidate experience text with an opportunity description.

  Explanations are grounded only in the supplied texts and do not use hybrid
  search ranks, scores, or vector similarities.

  The separately hosted Qwen inference process owns model resources, keeping
  this API process lightweight and independently scalable.
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
      request: GenerateExplanationRequest,
  ) -> GenerateExplanationResponse:
    """Compare the supplied texts and validate the generated explanation."""
    user_prompt = (
      "Compare these candidate experiences with the opportunity using only "
      "the supplied text:\n"
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
