from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


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


class BatchCandidateExplanation(BaseModel):
  """One candidate within a batch explanation request."""

  candidate_id: str = Field(min_length=1)
  experiences: list[CandidateExperience]


class ExplanationsRequest(BaseModel):
  """
  Requests explanations for multiple candidates against one opportunity
  description shared by all of them.

  Each candidate is explained independently by the LLM - one call per
  candidate. Batching is for the API between the caller and this service,
  not for comparing candidates with each other.
  """

  opportunity_description: str

  candidates: list[BatchCandidateExplanation] = Field(
    min_length=1,
    # Generation is currently sequential (one LLM call per candidate), so
    # this is kept modest to bound total request time; raise this once
    # LLM-call concurrency is introduced.
    max_length=50,
  )

  @model_validator(mode="after")
  def validate_unique_candidate_ids(self) -> ExplanationsRequest:
    """
    Reject duplicate candidate IDs because each result must map
    unambiguously back to one candidate.
    """
    candidate_ids = [candidate.candidate_id for candidate in self.candidates]

    if len(candidate_ids) != len(set(candidate_ids)):
      raise ValueError(
        "Candidate IDs must be unique within the request"
      )

    return self


class ExplanationErrorCode(StrEnum):
  """
  Stable item-level error codes returned to the batch caller.
  """

  LLM_SERVICE_UNAVAILABLE = "LLM_SERVICE_UNAVAILABLE"
  INVALID_LLM_OUTPUT = "INVALID_LLM_OUTPUT"


class ExplanationError(BaseModel):
  """Describes why an individual candidate's explanation could not be generated."""

  code: ExplanationErrorCode
  message: str = Field(min_length=1)


class CandidateExplanationResult(BaseModel):
  """
  Contains the explanation result for one candidate.

  The caller-supplied candidate_id is returned unchanged so that the result
  can be correlated with the corresponding candidate in the request.

  A successful result contains summary/experience_explanations/limitations
  and no error. A failed result contains an error and no explanation content.
  """

  candidate_id: str = Field(min_length=1)

  summary: str | None = Field(
    default=None,
    description="Null when explanation generation failed for this candidate.",
  )

  experience_explanations: list[ExperienceExplanation] | None = Field(
    default=None,
    description="Null when explanation generation failed for this candidate.",
  )

  limitations: list[str] | None = Field(
    default=None,
    description="Null when explanation generation failed for this candidate.",
  )

  error: ExplanationError | None = Field(
    default=None,
    description=(
      "Details of an item-level failure. Null when the explanation was "
      "generated successfully."
    ),
  )

  @model_validator(mode="after")
  def validate_outcome(self) -> CandidateExplanationResult:
    """
    Ensure that each result contains exactly one outcome.
    """
    has_explanation = self.summary is not None
    has_error = self.error is not None

    if has_explanation == has_error:
      raise ValueError(
        "Exactly one of explanation content or error must be supplied"
      )

    return self


class ExplanationsResponse(BaseModel):
  """Reports one result for every requested candidate."""

  requested: int = Field(ge=0)
  succeeded: int = Field(ge=0)
  failed: int = Field(ge=0)

  results: list[CandidateExplanationResult]

  @model_validator(mode="after")
  def validate_counts(self) -> ExplanationsResponse:
    """
    Ensure that the summary counts agree with the item-level results.
    """
    if self.requested != len(self.results):
      raise ValueError(
        "Requested count must equal the number of results"
      )

    actual_succeeded = sum(
      result.error is None
      for result in self.results
    )

    actual_failed = sum(
      result.error is not None
      for result in self.results
    )

    if self.succeeded != actual_succeeded:
      raise ValueError(
        "Succeeded count does not match successful results"
      )

    if self.failed != actual_failed:
      raise ValueError(
        "Failed count does not match failed results"
      )

    if self.requested != self.succeeded + self.failed:
      raise ValueError(
        "Requested count must equal succeeded plus failed"
      )

    return self
