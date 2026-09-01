from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

Status = Literal["ACTIVE", "INACTIVE", "DELETED"]


class CamelCaseModel(BaseModel):
  """
  Base for models whose JSON wire format must match tc-api-spec's camelCase
  schemas exactly (CandidateJobExperience.yaml etc. - see
  app/services/cv_extraction/schemas/), while still allowing normal
  snake_case construction/attribute access from Python.

  This is a deliberate departure from this repo's other models (e.g.
  ExplanationRequest, SkillName), which use plain snake_case field names as
  their own wire format - those aren't mirroring a published external
  schema. These models exist specifically to produce JSON "corresponding to
  the schemas" in tc-api-spec, so matching that casing exactly takes
  priority here.
  """

  model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class Country(CamelCaseModel):
  """Mirrors Country.yaml in tc-api-spec."""

  iso_code: str
  name: str
  status: Status


class Occupation(CamelCaseModel):
  """Mirrors Occupation.yaml in tc-api-spec."""

  isco08_code: str | None = None
  name: str | None = None
  status: Status | None = None


class CandidateJobExperience(CamelCaseModel):
  """Mirrors CandidateJobExperience.yaml in tc-api-spec."""

  country: Country | None = None
  company_name: str | None = None
  role: str | None = None
  start_date: date | None = None
  end_date: date | None = None
  full_time: bool | None = None
  paid: bool | None = None
  description: str | None = None


class CandidateOccupation(CamelCaseModel):
  """Mirrors CandidateOccupation.yaml in tc-api-spec."""

  occupation: Occupation
  years_experience: int | None = None
  candidate_job_experiences: list[CandidateJobExperience] = Field(default_factory=list)


class CvExtractionResponse(CamelCaseModel):
  """
  Response of POST /extract_candidate_occupations.

  candidate_occupations is empty whenever is_cv is False - the endpoint
  does not attempt to extract anything from a document that isn't
  recognizably a CV/résumé.
  """

  is_cv: bool
  is_cv_confidence: float
  candidate_occupations: list[CandidateOccupation] = Field(default_factory=list)
