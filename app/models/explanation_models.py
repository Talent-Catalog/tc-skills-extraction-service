from __future__ import annotations

from pydantic import BaseModel, Field


class MatchingExperienceEvidence(BaseModel):
  """Contains supplied evidence for one semantically matching experience."""

  experience_id: str
  job_title: str | None = None
  description: str
  similarity: float = Field(ge=-1.0, le=1.0)


class GenerateExplanationRequest(BaseModel):
  """
  Requests an explanation of an existing candidate ranking.

  The LLM explains only the supplied ranking evidence. It does not calculate
  or change the candidate's rank or score.
  """

  candidate_id: str
  rank: int = Field(ge=1)
  candidate_score: float = Field(ge=-1.0, le=1.0)
  opportunity_description: str
  matching_experiences: list[MatchingExperienceEvidence]


class ExperienceExplanation(BaseModel):
  """Explains how one supplied experience relates to the opportunity."""

  experience_id: str
  explanation: str


class GenerateExplanationResponse(BaseModel):
  """Explains supplied ranking evidence without recalculating the ranking."""

  candidate_id: str
  summary: str
  ranking_basis: str
  experience_explanations: list[ExperienceExplanation]
  limitations: list[str]
