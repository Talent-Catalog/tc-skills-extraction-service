from __future__ import annotations

import httpx
from fastapi.testclient import TestClient

from app.dependencies import get_cv_extraction_service
from app.main import app
from app.models.candidate_occupation_models import (
  CandidateJobExperience,
  CandidateOccupation,
  Country,
  CvExtractionResponse,
  Occupation,
)
from app.services.cv_extraction.pdf_to_doctags import PdfConversionError
from app.services.llm_client import LlmServiceUnavailableError, MalformedLlmResponseError


FAKE_PDF_BYTES = b"%PDF-1.4 fake resume content"


class FakeCvExtractionService:
  """Return a configured result or error without calling Docling/Anthropic."""

  def __init__(
      self,
      result: CvExtractionResponse | None = None,
      error: Exception | None = None,
  ) -> None:
    self._result = result
    self._error = error

  def extract(self, pdf_bytes: bytes) -> CvExtractionResponse:
    if self._error is not None:
      raise self._error

    if self._result is None:
      raise AssertionError("Fake extraction result was not configured")

    return self._result


def post_with_service(service: FakeCvExtractionService) -> httpx.Response:
  """Call the endpoint with its singleton dependency replaced by a fake."""
  app.dependency_overrides[get_cv_extraction_service] = lambda: service
  client = TestClient(app, raise_server_exceptions=False)
  try:
    return client.post(
      "/extract_candidate_occupations",
      files={"file": ("resume.pdf", FAKE_PDF_BYTES, "application/pdf")},
    )
  finally:
    client.close()
    app.dependency_overrides.clear()


def test_extract_returns_candidate_occupations() -> None:
  response_model = CvExtractionResponse(
    is_cv=True,
    is_cv_confidence=0.98,
    candidate_occupations=[
      CandidateOccupation(
        occupation=Occupation(name="Mechanical Engineer", status="ACTIVE", isco08_code="2144"),
        years_experience=9,
        candidate_job_experiences=[
          CandidateJobExperience(
            country=Country(iso_code="LB", name="Lebanon", status="ACTIVE"),
            company_name="Acme Corp",
            role="Engineer",
            description="Led the platform team.",
          )
        ],
      )
    ],
  )

  response = post_with_service(FakeCvExtractionService(response_model))

  assert response.status_code == 200
  body = response.json()
  assert body["isCv"] is True
  assert body["candidateOccupations"][0]["occupation"]["name"] == "Mechanical Engineer"
  assert body["candidateOccupations"][0]["candidateJobExperiences"][0]["companyName"] == "Acme Corp"
  assert body["candidateOccupations"][0]["candidateJobExperiences"][0]["country"]["isoCode"] == "LB"


def test_extract_returns_empty_result_for_non_cv() -> None:
  response_model = CvExtractionResponse(is_cv=False, is_cv_confidence=0.95)

  response = post_with_service(FakeCvExtractionService(response_model))

  assert response.status_code == 200
  body = response.json()
  assert body["isCv"] is False
  assert body["candidateOccupations"] == []


def test_extract_maps_pdf_conversion_error_to_422() -> None:
  response = post_with_service(
    FakeCvExtractionService(error=PdfConversionError("Not a valid PDF"))
  )

  assert response.status_code == 422
  assert response.json() == {"detail": "Not a valid PDF"}


def test_extract_maps_unavailable_llm_to_503() -> None:
  response = post_with_service(
    FakeCvExtractionService(error=LlmServiceUnavailableError("LLM unavailable"))
  )

  assert response.status_code == 503
  assert response.json() == {"detail": "LLM unavailable"}


def test_extract_maps_malformed_llm_output_to_502() -> None:
  response = post_with_service(
    FakeCvExtractionService(error=MalformedLlmResponseError("Invalid model output"))
  )

  assert response.status_code == 502
  assert response.json() == {"detail": "Invalid model output"}


def test_extract_rejects_non_pdf_content_type() -> None:
  app.dependency_overrides[get_cv_extraction_service] = lambda: FakeCvExtractionService(
    CvExtractionResponse(is_cv=False, is_cv_confidence=0.0)
  )
  client = TestClient(app, raise_server_exceptions=False)
  try:
    response = client.post(
      "/extract_candidate_occupations",
      files={"file": ("resume.txt", b"not a pdf", "text/plain")},
    )
  finally:
    client.close()
    app.dependency_overrides.clear()

  assert response.status_code == 415
