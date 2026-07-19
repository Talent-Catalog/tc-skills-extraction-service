import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import get_explanation_service
from app.models.explanation_models import (
  GenerateExplanationRequest,
  GenerateExplanationResponse,
)
from app.services.explanation_service import (
  ExplanationGenerationError,
  ExplanationService,
)
from app.services.llm_client import (
  LlmServiceUnavailableError,
  MalformedLlmResponseError,
)

logger = logging.getLogger(__name__)

router = APIRouter(
  prefix="/explanations",
  tags=[
    "explanations",
  ],
)


@router.post(
  "",
  response_model=GenerateExplanationResponse,
  status_code=status.HTTP_200_OK,
  summary="Compare candidate experience with an opportunity",
)
def generate_explanation(
    request: GenerateExplanationRequest,
    explanation_service: ExplanationService = Depends(
      get_explanation_service
    ),
) -> GenerateExplanationResponse:
  """Explain a direct comparison using only the supplied text."""
  try:
    return explanation_service.generate_explanation(request)
  except LlmServiceUnavailableError as exception:
    logger.error("LLM service unavailable: %s", exception)
    raise HTTPException(
      status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
      detail=str(exception),
    ) from exception
  except (
      MalformedLlmResponseError,
      ExplanationGenerationError,
  ) as exception:
    logger.error("Invalid LLM explanation output: %s", exception)
    raise HTTPException(
      status_code=status.HTTP_502_BAD_GATEWAY,
      detail=str(exception),
    ) from exception
