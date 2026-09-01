import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.dependencies import get_cv_extraction_service
from app.models.candidate_occupation_models import CvExtractionResponse
from app.services.cv_extraction.cv_extraction_service import CvExtractionService
from app.services.cv_extraction.pdf_to_doctags import PdfConversionError
from app.services.llm_client import LlmServiceUnavailableError, MalformedLlmResponseError

logger = logging.getLogger(__name__)

router = APIRouter(
  tags=[
    "cv-extraction",
  ],
)


@router.post(
  "/extract_candidate_occupations",
  response_model=CvExtractionResponse,
  status_code=status.HTTP_200_OK,
  summary="Extract candidate occupations and job experience from a CV PDF",
)
async def extract_candidate_occupations(
    file: UploadFile = File(...),
    cv_extraction_service: CvExtractionService = Depends(
      get_cv_extraction_service
    ),
) -> CvExtractionResponse:
  """
  Accepts an uploaded PDF and, if it looks like a CV/résumé, returns the
  candidate occupations and job experience extracted from it - JSON shaped
  to match CandidateOccupation.yaml / CandidateJobExperience.yaml in
  tc-api-spec. isCv is false (with an empty candidateOccupations) when the
  document doesn't look like a CV at all.

  :param file: the uploaded PDF
  :param cv_extraction_service: the singleton service created at startup
  """
  if file.content_type not in ("application/pdf", "application/x-pdf"):
    raise HTTPException(
      status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
      detail=f"Expected a PDF file, got content type '{file.content_type}'",
    )

  pdf_bytes = await file.read()

  try:
    return cv_extraction_service.extract(pdf_bytes)
  except PdfConversionError as exception:
    logger.warning("PDF conversion failed: %s", exception)
    raise HTTPException(
      status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
      detail=str(exception),
    ) from exception
  except LlmServiceUnavailableError as exception:
    logger.error("LLM service unavailable: %s", exception)
    raise HTTPException(
      status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
      detail=str(exception),
    ) from exception
  except MalformedLlmResponseError as exception:
    logger.error("Invalid LLM output during CV extraction: %s", exception)
    raise HTTPException(
      status_code=status.HTTP_502_BAD_GATEWAY,
      detail=str(exception),
    ) from exception
