import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import get_embedding_service
from app.models.embedding_models import (
  EmbeddingsRequest,
  EmbeddingsResponse,
)
from app.services.embedding_service import (
  EmbeddingModelConfigurationError,
  EmbeddingModelNotAvailableError,
  EmbeddingService,
)

logger = logging.getLogger(__name__)

router = APIRouter(
  prefix="/embeddings",
  tags=[
    "embeddings",
  ],
)

@router.post(
  "",
  response_model=EmbeddingsResponse,
  status_code=status.HTTP_200_OK,
  summary="Generate multiple embeddings",
)
def generate_embeddings(
    request: EmbeddingsRequest,
    embedding_service: EmbeddingService = Depends(
      get_embedding_service
    ),
) -> EmbeddingsResponse:
  """
  Generate multiple embeddings using one fixed model configuration.

  An HTTP 200 response can contain both successful and failed item results.
  Batch-level model failures are returned as HTTP errors.
  """
  try:
    return embedding_service.generate_embeddings(
      request
    )

  except EmbeddingModelConfigurationError as exception:
    logger.warning(
      "Embedding model configuration error: %s",
      exception,
    )

    raise HTTPException(
      status_code=status.HTTP_409_CONFLICT,
      detail=str(exception),
    ) from exception

  except EmbeddingModelNotAvailableError as exception:
    logger.error(
      "Embedding model unavailable: %s",
      exception,
    )

    raise HTTPException(
      status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
      detail=str(exception),
    ) from exception
