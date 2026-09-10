from __future__ import annotations

import numpy as np
import pytest

from app.models.embedding_models import (
  EmbeddingErrorCode,
  EmbeddingInput,
  EmbeddingInputType,
  EmbeddingModelDetails,
  EmbeddingsRequest, EmbeddingConfigurationVersion,
)
from app.services.embedding_service import (
  EmbeddingModelConfigurationError,
  EmbeddingService,
  SentenceTransformerModelProvider,
)
from app.services.text_preprocessor import SpacyTextPreprocessor


class FakeTextPreprocessor(SpacyTextPreprocessor):
  """
  Return predictable preprocessed text without loading spaCy.
  """

  def __init__(
      self,
      failure_texts: set[str] | None = None,
      empty_texts: set[str] | None = None,
  ) -> None:
    self._failure_texts = failure_texts or set()
    self._empty_texts = empty_texts or set()

  def preprocess(
      self,
      text: str,
      configuration_version: EmbeddingConfigurationVersion,
  ) -> str:
    """
    Normalize test text or fail for configured values.
    """
    if text in self._failure_texts:
      raise ValueError(
        "Preprocessing test failure"
      )

    if text in self._empty_texts:
      return ""

    return text.strip().lower()


class FakeEmbeddingModel:
  """
  Provide deterministic embeddings without loading a real model.
  """

  def __init__(
      self,
      dimensions: int,
      *,
      fail_batch: bool = False,
      failing_texts: set[str] | None = None,
  ) -> None:
    self._dimensions = dimensions
    self._fail_batch = fail_batch
    self._failing_texts = failing_texts or set()
    self.calls: list[list[str]] = []

  def get_embedding_dimension(self) -> int:
    """
    Return the configured test dimensions.
    """
    return self._dimensions

  def encode(
      self,
      sentences: list[str],
      *,
      batch_size: int,
      show_progress_bar: bool,
      convert_to_numpy: bool,
      normalize_embeddings: bool,
  ) -> np.ndarray:
    """
    Generate deterministic vectors for each supplied text.
    """
    texts = list(sentences)
    self.calls.append(texts)

    if self._fail_batch and len(texts) > 1:
      raise RuntimeError(
        "Batch encoding test failure"
      )

    for text in texts:
      if text in self._failing_texts:
        raise RuntimeError(
          f"Unable to encode '{text}'"
        )

    return np.asarray(
      [
        [
          float(input_index + dimension_index)
          for dimension_index in range(
          self._dimensions
        )
        ]
        for input_index, _ in enumerate(texts)
      ],
      dtype=np.float32,
    )


class FakeModelProvider(
  SentenceTransformerModelProvider
):
  """
  Return a predefined embedding model without loading anything.
  """

  def __init__(
      self,
      model: FakeEmbeddingModel,
  ) -> None:
    super().__init__()
    self._model = model

  def get_model(
      self,
      model_details: EmbeddingModelDetails,
  ):
    """
    Return the configured test model.
    """
    return self._model


@pytest.fixture
def model_details() -> EmbeddingModelDetails:
  """
  Return fixed model details used by the tests.
  """
  return EmbeddingModelDetails(
    model_name="test-model",
    configuration_version=EmbeddingConfigurationVersion.SPACY_PREPROCESSING_V3,
    dimensions=3,
  )


def create_service(
    model: FakeEmbeddingModel,
    preprocessor: FakeTextPreprocessor | None = None,
) -> EmbeddingService:
  """
  Create the embedding service with test doubles.
  """
  return EmbeddingService(
    model_provider=FakeModelProvider(model),
    text_preprocessor=(
        preprocessor or FakeTextPreprocessor()
    ),
    encoder_batch_size=16,
    normalize_embeddings=True,
  )


def test_generates_multiple_embeddings_in_one_batch(
    model_details: EmbeddingModelDetails,
) -> None:
  model = FakeEmbeddingModel(
    dimensions=3
  )

  service = create_service(model)

  response = service.generate_embeddings(
    EmbeddingsRequest(
      model=model_details,
      type=EmbeddingInputType.DOCUMENT,
      inputs=[
        EmbeddingInput(
          id="101",
          text="First text",
        ),
        EmbeddingInput(
          id="102",
          text="Second text",
        ),
      ],
    )
  )

  assert response.model == model_details
  assert response.requested == 2
  assert response.succeeded == 2
  assert response.failed == 0

  assert response.results[0].id == "101"
  assert response.results[0].embedding == [
    0.0,
    1.0,
    2.0,
  ]

  assert response.results[1].id == "102"
  assert response.results[1].embedding == [
    1.0,
    2.0,
    3.0,
  ]

  assert model.calls == [
    [
      "first text",
      "second text",
    ]
  ]


def test_blank_text_is_an_item_failure(
    model_details: EmbeddingModelDetails,
) -> None:
  model = FakeEmbeddingModel(
    dimensions=3
  )

  service = create_service(model)

  response = service.generate_embeddings(
    EmbeddingsRequest(
      model=model_details,
      type=EmbeddingInputType.DOCUMENT,
      inputs=[
        EmbeddingInput(
          id="101",
          text="Valid text",
        ),
        EmbeddingInput(
          id="102",
          text=" ",
        ),
        EmbeddingInput(
          id="103",
          text="Other text",
        ),
      ],
    )
  )

  assert response.requested == 3
  assert response.succeeded == 2
  assert response.failed == 1

  failed_result = response.results[1]

  assert failed_result.id == "102"
  assert failed_result.embedding is None
  assert failed_result.error is not None
  assert (
      failed_result.error.code
      == EmbeddingErrorCode.INVALID_TEXT
  )

  assert model.calls == [
    [
      "valid text",
      "other text",
    ]
  ]


def test_preprocessing_failure_does_not_stop_batch(
    model_details: EmbeddingModelDetails,
) -> None:
  model = FakeEmbeddingModel(
    dimensions=3
  )

  service = create_service(
    model=model,
    preprocessor=FakeTextPreprocessor(
      failure_texts={
        "Broken text",
      }
    ),
  )

  response = service.generate_embeddings(
    EmbeddingsRequest(
      model=model_details,
      type=EmbeddingInputType.DOCUMENT,
      inputs=[
        EmbeddingInput(
          id="101",
          text="Good text",
        ),
        EmbeddingInput(
          id="102",
          text="Broken text",
        ),
      ],
    )
  )

  assert response.succeeded == 1
  assert response.failed == 1

  failed_result = response.results[1]

  assert failed_result.error is not None
  assert (
      failed_result.error.code
      == EmbeddingErrorCode.PREPROCESSING_FAILED
  )


def test_failed_batch_is_retried_individually(
    model_details: EmbeddingModelDetails,
) -> None:
  model = FakeEmbeddingModel(
    dimensions=3,
    fail_batch=True,
    failing_texts={
      "bad text",
    },
  )

  service = create_service(model)

  response = service.generate_embeddings(
    EmbeddingsRequest(
      model=model_details,
      type=EmbeddingInputType.DOCUMENT,
      inputs=[
        EmbeddingInput(
          id="101",
          text="Good text",
        ),
        EmbeddingInput(
          id="102",
          text="Bad text",
        ),
        EmbeddingInput(
          id="103",
          text="Other text",
        ),
      ],
    )
  )

  assert response.requested == 3
  assert response.succeeded == 2
  assert response.failed == 1

  assert response.results[0].embedding is not None

  assert response.results[1].embedding is None
  assert response.results[1].error is not None
  assert (
      response.results[1].error.code
      == EmbeddingErrorCode.EMBEDDING_FAILED
  )

  assert response.results[2].embedding is not None

  assert model.calls == [
    [
      "good text",
      "bad text",
      "other text",
    ],
    [
      "good text",
    ],
    [
      "bad text",
    ],
    [
      "other text",
    ],
  ]


def test_context_only_input_succeeds_without_text(
    model_details: EmbeddingModelDetails,
) -> None:
  model = FakeEmbeddingModel(
    dimensions=3
  )

  service = create_service(model)

  response = service.generate_embeddings(
    EmbeddingsRequest(
      model=model_details,
      type=EmbeddingInputType.DOCUMENT,
      inputs=[
        EmbeddingInput(
          id="101",
          context="Some context",
        ),
      ],
    )
  )

  assert response.succeeded == 1
  assert response.failed == 0

  assert response.results[0].embedding is not None

  assert model.calls == [
    [
      "Some context",
    ]
  ]


def test_whitespace_only_context_is_treated_as_missing(
    model_details: EmbeddingModelDetails,
) -> None:
  model = FakeEmbeddingModel(
    dimensions=3
  )

  service = create_service(model)

  response = service.generate_embeddings(
    EmbeddingsRequest(
      model=model_details,
      type=EmbeddingInputType.DOCUMENT,
      inputs=[
        EmbeddingInput(
          id="101",
          text="Valid text",
          context="   ",
        ),
      ],
    )
  )

  assert response.succeeded == 1
  assert response.failed == 0

  assert model.calls == [
    [
      "valid text",
    ]
  ]


def test_blank_text_and_blank_context_is_item_failure(
    model_details: EmbeddingModelDetails,
) -> None:
  model = FakeEmbeddingModel(
    dimensions=3
  )

  service = create_service(model)

  response = service.generate_embeddings(
    EmbeddingsRequest(
      model=model_details,
      type=EmbeddingInputType.DOCUMENT,
      inputs=[
        EmbeddingInput(
          id="101",
          text=" ",
          context=" ",
        ),
      ],
    )
  )

  assert response.succeeded == 0
  assert response.failed == 1

  failed_result = response.results[0]

  assert failed_result.embedding is None
  assert failed_result.error is not None
  assert (
      failed_result.error.code
      == EmbeddingErrorCode.INVALID_TEXT
  )
  assert (
      failed_result.error.message
      == "Text and context must not both be empty"
  )

  assert model.calls == []


def test_context_used_when_preprocessing_empties_text(
    model_details: EmbeddingModelDetails,
) -> None:
  model = FakeEmbeddingModel(
    dimensions=3
  )

  service = create_service(
    model=model,
    preprocessor=FakeTextPreprocessor(
      empty_texts={
        "Stopwords only",
      }
    ),
  )

  response = service.generate_embeddings(
    EmbeddingsRequest(
      model=model_details,
      type=EmbeddingInputType.DOCUMENT,
      inputs=[
        EmbeddingInput(
          id="101",
          text="Stopwords only",
          context="Some context",
        ),
      ],
    )
  )

  assert response.succeeded == 1
  assert response.failed == 0

  assert model.calls == [
    [
      "Some context",
    ]
  ]


def test_text_empty_after_preprocessing_with_no_context_is_item_failure(
    model_details: EmbeddingModelDetails,
) -> None:
  model = FakeEmbeddingModel(
    dimensions=3
  )

  service = create_service(
    model=model,
    preprocessor=FakeTextPreprocessor(
      empty_texts={
        "Stopwords only",
      }
    ),
  )

  response = service.generate_embeddings(
    EmbeddingsRequest(
      model=model_details,
      type=EmbeddingInputType.DOCUMENT,
      inputs=[
        EmbeddingInput(
          id="101",
          text="Stopwords only",
        ),
      ],
    )
  )

  assert response.succeeded == 0
  assert response.failed == 1

  failed_result = response.results[0]

  assert failed_result.embedding is None
  assert failed_result.error is not None
  assert (
      failed_result.error.code
      == EmbeddingErrorCode.INVALID_TEXT
  )
  assert (
      failed_result.error.message
      == "Text and context were both empty after preprocessing"
  )

  assert model.calls == []


def test_context_and_text_are_combined_for_embedding(
    model_details: EmbeddingModelDetails,
) -> None:
  model = FakeEmbeddingModel(
    dimensions=3
  )

  service = create_service(model)

  response = service.generate_embeddings(
    EmbeddingsRequest(
      model=model_details,
      type=EmbeddingInputType.DOCUMENT,
      inputs=[
        EmbeddingInput(
          id="101",
          text="Some text",
          context="Extra context",
        ),
      ],
    )
  )

  assert response.succeeded == 1
  assert response.failed == 0

  assert model.calls == [
    [
      "Extra context\nsome text",
    ]
  ]


def test_model_dimension_mismatch_fails_entire_batch(
    model_details: EmbeddingModelDetails,
) -> None:
  model = FakeEmbeddingModel(
    dimensions=4
  )

  service = create_service(model)

  with pytest.raises(
      EmbeddingModelConfigurationError,
      match="produces 4 dimensions",
  ):
    service.generate_embeddings(
      EmbeddingsRequest(
        model=model_details,
        type=EmbeddingInputType.DOCUMENT,
        inputs=[
          EmbeddingInput(
            id="101",
            text="Some text",
          ),
        ],
      )
    )
