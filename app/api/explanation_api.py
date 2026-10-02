from fastapi import APIRouter, Depends, status

from app.dependencies import get_explanation_service
from app.models.explanation_models import (
  ExplanationsRequest,
  ExplanationsResponse,
)
from app.services.explanation_service import ExplanationService

router = APIRouter(
  prefix="/explanations",
  tags=[
    "explanations",
  ],
)


@router.post(
  "",
  response_model=ExplanationsResponse,
  status_code=status.HTTP_200_OK,
  summary="Compare multiple candidates' experience with one opportunity",
)
def generate_explanations(
    request: ExplanationsRequest,
    explanation_service: ExplanationService = Depends(
      get_explanation_service
    ),
) -> ExplanationsResponse:
  """
  Explain each candidate independently against one shared opportunity.

  An HTTP 200 response can contain both successful and failed candidate
  results: a failure generating one candidate's explanation is reported as
  an item-level error and does not fail the others.
  """
  return explanation_service.generate_explanations(request)
