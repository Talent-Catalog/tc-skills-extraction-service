from __future__ import annotations

from pydantic import BaseModel


class CandidateExperience(BaseModel):
  """Contains candidate experience text to compare with an opportunity."""

  experience_id: str
  job_title: str | None = None
  description: str


class ExplanationRequest(BaseModel):
  """
  Requests a comparison of candidate experience and an opportunity.

  The LLM bases its explanation only on the supplied opportunity description
  and candidate experience text. Search ranks, scores, and similarities are
  not inputs to the explanation.
  """

  candidate_id: str
  opportunity_description: str
  experiences: list[CandidateExperience]


class ExperienceExplanation(BaseModel):
  """Explains how one supplied experience relates to the opportunity."""

  experience_id: str
  explanation: str


class ExplanationResponse(BaseModel):
  """Explains a text-only comparison with the supplied opportunity."""

  candidate_id: str
  summary: str
  experience_explanations: list[ExperienceExplanation]
  limitations: list[str]
