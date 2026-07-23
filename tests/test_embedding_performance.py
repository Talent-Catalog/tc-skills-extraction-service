import time

import pytest

from app.models.embedding_models import (
  EmbeddingInput,
  EmbeddingModelDetails,
  GenerateEmbeddingsRequest, EmbeddingConfigurationVersion,
)
from app.services.embedding_service import EmbeddingService, \
  SentenceTransformerModelProvider
from app.services.text_preprocessor import SpacyTextPreprocessor


@pytest.fixture
def embedding_service() -> EmbeddingService:
  """
  Create an embedding service using the real Sentence Transformer model
  provider and spaCy text preprocessor.

  This fixture is intended for integration and performance tests because it
  loads and executes the real embedding model.
  """
  return EmbeddingService(
    model_provider=SentenceTransformerModelProvider(),
    text_preprocessor=SpacyTextPreprocessor(),
  )

@pytest.fixture
def real_embedding_model_details() -> EmbeddingModelDetails:
  """
  Return details for the real Sentence Transformer model used by
  performance tests.
  """
  return EmbeddingModelDetails(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    configuration_version=(
      EmbeddingConfigurationVersion.SPACY_PREPROCESSING_V3
    ),
    dimensions=384,
  )

@pytest.mark.performance
def test_embeddings_per_second(
    embedding_service: EmbeddingService,
    real_embedding_model_details: EmbeddingModelDetails,
) -> None:
  """
  Measure how many successful embeddings the service generates per second.

  A warm-up request is made before timing so that model loading and initial
  model setup are not included in the throughput measurement.
  """
  number_of_inputs = 500

  inputs = [
    EmbeddingInput(
      id=str(index),
      text=(
        "Managed customer accounts, prepared financial reports, "
        "performed bookkeeping, and worked with business stakeholders."
        "Managed customer accounts, prepared financial reports, "
        "performed bookkeeping, and worked with business stakeholders."
        "Managed customer accounts, prepared financial reports, "
        "performed bookkeeping, and worked with business stakeholders."
        "Managed customer accounts, prepared financial reports, "
        "performed bookkeeping, and worked with business stakeholders."
        "Managed customer accounts, prepared financial reports, "
        "performed bookkeeping, and worked with business stakeholders."
      ),
    )
    for index in range(number_of_inputs)
  ]

  request = GenerateEmbeddingsRequest(
    model=real_embedding_model_details,
    inputs=inputs,
  )

  # Warm up the model and ensure it has already been loaded before timing.
  warm_up_request = GenerateEmbeddingsRequest(
    model=real_embedding_model_details,
    inputs=inputs[:10],
  )

  warm_up_response = embedding_service.generate_embeddings(
    warm_up_request
  )

  assert warm_up_response.succeeded == 10
  assert warm_up_response.failed == 0

  start_time = time.perf_counter()

  response = embedding_service.generate_embeddings(request)

  elapsed_seconds = time.perf_counter() - start_time

  embeddings_per_second = (
      response.succeeded / elapsed_seconds
  )

  print(
    f"\nModel:                 "
    f"{real_embedding_model_details.model_name}"
  )
  print(
    f"Configuration:         "
    f"{real_embedding_model_details.configuration_version}"
  )
  print(f"Requested:             {response.requested}")
  print(f"Succeeded:             {response.succeeded}")
  print(f"Failed:                {response.failed}")
  print(f"Elapsed time:          {elapsed_seconds:.3f} seconds")
  print(
    f"Embeddings per second: "
    f"{embeddings_per_second:.2f}"
  )

  assert response.requested == number_of_inputs
  assert response.succeeded == number_of_inputs
  assert response.failed == 0
  assert len(response.results) == number_of_inputs
  assert embeddings_per_second > 0
