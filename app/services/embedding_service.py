from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from app.models.embedding_models import (
  EmbeddingError,
  EmbeddingErrorCode,
  EmbeddingInput,
  EmbeddingModelDetails,
  EmbeddingResult,
  GenerateEmbeddingsRequest,
  GenerateEmbeddingsResponse,
)
from app.services.text_preprocessor import SpacyTextPreprocessor

logger = logging.getLogger(__name__)


class EmbeddingModelNotAvailableError(RuntimeError):
  """
  Raised when the requested Sentence Transformer model cannot be loaded.
  """


class EmbeddingModelConfigurationError(RuntimeError):
  """
  Raised when fixed database model details do not match the loaded model.
  """


@dataclass(frozen=True)
class PreparedEmbeddingInput:
  """
  Holds one input after successful validation and preprocessing.
  """

  original: EmbeddingInput
  prepared_text: str


class SentenceTransformerModelProvider:
  """
  Loads and caches Sentence Transformer models by model name.

  Loading a Sentence Transformer is expensive, so each model is loaded at
  most once for the lifetime of this provider.
  """

  def __init__(self) -> None:
    self._models: dict[str, SentenceTransformer] = {}

  def get_model(
      self,
      model_details: EmbeddingModelDetails,
  ) -> SentenceTransformer:
    """
    Return a cached model or load it on first use.
    """
    existing_model = self._models.get(
      model_details.model_name
    )

    if existing_model is not None:
      return existing_model

    try:
      logger.info(
        "Loading embedding model '%s'.",
        model_details.model_name,
      )

      model = SentenceTransformer(
        model_details.model_name
      )

      self._models[model_details.model_name] = model

      logger.info(
        "Loaded embedding model '%s'.",
        model_details.model_name,
      )

      return model

    except Exception as exception:
      logger.exception(
        "Unable to load embedding model '%s'.",
        model_details.model_name,
      )

      raise EmbeddingModelNotAvailableError(
        "Unable to load embedding model "
        f"'{model_details.model_name}'"
      ) from exception


class EmbeddingService:
  """
  Generates multiple embeddings using one fixed model configuration.

  Input-level failures are reported in the response and do not prevent other
  inputs from succeeding. Model loading and model configuration failures abort
  the entire request because they affect every item.
  """

  def __init__(
      self,
      model_provider: SentenceTransformerModelProvider,
      text_preprocessor: SpacyTextPreprocessor,
      *,
      encoder_batch_size: int = 32,
      normalize_embeddings: bool = True,
  ) -> None:
    self._model_provider = model_provider
    self._text_preprocessor = text_preprocessor
    self._encoder_batch_size = encoder_batch_size
    self._normalize_embeddings = normalize_embeddings

  def generate_embeddings(
      self,
      request: GenerateEmbeddingsRequest,
  ) -> GenerateEmbeddingsResponse:
    """
    Generate one item-level outcome for every input in the request.
    """
    model = self._model_provider.get_model(
      request.model
    )

    self._validate_model_dimensions(
      model=model,
      model_details=request.model,
    )

    prepared_inputs: list[PreparedEmbeddingInput] = []
    results_by_id: dict[int, EmbeddingResult] = {}

    for input_item in request.inputs:
      prepared_input, failure = self._prepare_input(
        model_details=request.model,
        input_item=input_item,
      )

      if failure is not None:
        results_by_id[input_item.id] = failure
      elif prepared_input is not None:
        prepared_inputs.append(prepared_input)

    if prepared_inputs:
      generated_results = self._encode_prepared_inputs(
        model=model,
        model_details=request.model,
        prepared_inputs=prepared_inputs,
      )

      for generated_result in generated_results:
        results_by_id[generated_result.id] = generated_result

    # Return results in the same order as the inputs.
    ordered_results = [
      results_by_id[input_item.id]
      for input_item in request.inputs
    ]

    succeeded = sum(
      result.embedding is not None
      for result in ordered_results
    )

    failed = len(ordered_results) - succeeded

    logger.info(
      "Embedding batch completed using model '%s', configuration '%s': "
      "%d requested, %d succeeded, %d failed.",
      request.model.model_name,
      request.model.configuration_version,
      len(request.inputs),
      succeeded,
      failed,
    )

    return GenerateEmbeddingsResponse(
      model=request.model,
      requested=len(request.inputs),
      succeeded=succeeded,
      failed=failed,
      results=ordered_results,
    )

  def _prepare_input(
      self,
      model_details: EmbeddingModelDetails,
      input_item: EmbeddingInput,
  ) -> tuple[
    PreparedEmbeddingInput | None,
    EmbeddingResult | None,
  ]:
    """
    Validate and preprocess one input without affecting other batch items.
    """
    if not input_item.text or not input_item.text.strip():
      return None, self._failure(
        input_id=input_item.id,
        code=EmbeddingErrorCode.INVALID_TEXT,
        message="Text must not be empty",
      )

    try:
      prepared_text = self._text_preprocessor.preprocess(
        text=input_item.text,
        configuration_version=(
          model_details.configuration_version
        ),
      )

    except Exception as exception:
      logger.exception(
        "Preprocessing failed for input ID %d.",
        input_item.id,
      )

      return None, self._failure(
        input_id=input_item.id,
        code=EmbeddingErrorCode.PREPROCESSING_FAILED,
        message=self._safe_error_message(
          exception=exception,
          fallback="Text preprocessing failed",
        ),
      )

    if not prepared_text or not prepared_text.strip():
      return None, self._failure(
        input_id=input_item.id,
        code=EmbeddingErrorCode.INVALID_TEXT,
        message="Text was empty after preprocessing",
      )

    return (
      PreparedEmbeddingInput(
        original=input_item,
        prepared_text=prepared_text,
      ),
      None,
    )

  def _encode_prepared_inputs(
      self,
      model: SentenceTransformer,
      model_details: EmbeddingModelDetails,
      prepared_inputs: Sequence[PreparedEmbeddingInput],
  ) -> list[EmbeddingResult]:
    """
    Try efficient batch encoding first.

    If the batch fails, retry each input individually so that one problematic
    input does not prevent all other inputs from succeeding.
    """
    texts = [
      prepared_input.prepared_text
      for prepared_input in prepared_inputs
    ]

    try:
      vectors = model.encode(
        texts,
        batch_size=self._encoder_batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=self._normalize_embeddings,
      )

      return self._build_batch_results(
        model_details=model_details,
        prepared_inputs=prepared_inputs,
        vectors=vectors,
      )

    except Exception:
      logger.exception(
        "Batch embedding failed for %d inputs. "
        "Retrying each input individually.",
        len(prepared_inputs),
      )

      return [
        self._encode_single_input(
          model=model,
          model_details=model_details,
          prepared_input=prepared_input,
        )
        for prepared_input in prepared_inputs
      ]

  def _build_batch_results(
      self,
      model_details: EmbeddingModelDetails,
      prepared_inputs: Sequence[PreparedEmbeddingInput],
      vectors: np.ndarray,
  ) -> list[EmbeddingResult]:
    """
    Convert a successful model batch response into item-level results.
    """
    if len(vectors) != len(prepared_inputs):
      raise ValueError(
        "The model returned a different number of embeddings "
        "than requested"
      )

    return [
      self._build_success_or_failure(
        input_id=prepared_input.original.id,
        vector=vector,
        expected_dimensions=model_details.dimensions,
      )
      for prepared_input, vector in zip(
        prepared_inputs,
        vectors,
        strict=True,
      )
    ]

  def _encode_single_input(
      self,
      model: SentenceTransformer,
      model_details: EmbeddingModelDetails,
      prepared_input: PreparedEmbeddingInput,
  ) -> EmbeddingResult:
    """
    Generate one embedding after the original batch call failed.
    """
    try:
      vectors = model.encode(
        [prepared_input.prepared_text],
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=self._normalize_embeddings,
      )

      if len(vectors) != 1:
        return self._failure(
          input_id=prepared_input.original.id,
          code=EmbeddingErrorCode.EMBEDDING_FAILED,
          message=(
            "The model did not return exactly one embedding"
          ),
        )

      return self._build_success_or_failure(
        input_id=prepared_input.original.id,
        vector=vectors[0],
        expected_dimensions=model_details.dimensions,
      )

    except Exception as exception:
      logger.exception(
        "Embedding failed for input ID %d.",
        prepared_input.original.id,
      )

      return self._failure(
        input_id=prepared_input.original.id,
        code=EmbeddingErrorCode.EMBEDDING_FAILED,
        message=self._safe_error_message(
          exception=exception,
          fallback="Embedding generation failed",
        ),
      )

  def _build_success_or_failure(
      self,
      input_id: int,
      vector: np.ndarray | Sequence[float],
      expected_dimensions: int,
  ) -> EmbeddingResult:
    """
    Validate one generated vector before returning it.
    """
    embedding = np.asarray(
      vector,
      dtype=np.float32,
    )

    if embedding.ndim != 1:
      return self._failure(
        input_id=input_id,
        code=EmbeddingErrorCode.INVALID_EMBEDDING,
        message=(
          "The generated embedding was not a "
          "one-dimensional vector"
        ),
      )

    if len(embedding) != expected_dimensions:
      return self._failure(
        input_id=input_id,
        code=EmbeddingErrorCode.INVALID_DIMENSIONS,
        message=(
          f"Expected {expected_dimensions} dimensions but "
          f"the model returned {len(embedding)}"
        ),
      )

    if not np.all(np.isfinite(embedding)):
      return self._failure(
        input_id=input_id,
        code=EmbeddingErrorCode.INVALID_EMBEDDING,
        message=(
          "The generated embedding contained a "
          "non-finite value"
        ),
      )

    return EmbeddingResult(
      id=input_id,
      embedding=embedding.tolist(),
      error=None,
    )

  @staticmethod
  def _validate_model_dimensions(
      model: SentenceTransformer,
      model_details: EmbeddingModelDetails,
  ) -> None:
    """
    Ensure that the fixed database dimensions match the loaded model.
    """
    actual_dimensions = (
      model.get_embedding_dimension()
    )

    if actual_dimensions is None:
      raise EmbeddingModelConfigurationError(
        "The loaded model did not report its embedding dimensions"
      )

    if actual_dimensions != model_details.dimensions:
      raise EmbeddingModelConfigurationError(
        f"Model '{model_details.model_name}' produces "
        f"{actual_dimensions} dimensions, but configuration "
        f"'{model_details.configuration_version}' expects "
        f"{model_details.dimensions}"
      )

  @staticmethod
  def _failure(
      input_id: int,
      code: EmbeddingErrorCode,
      message: str,
  ) -> EmbeddingResult:
    """
    Create a consistent item-level failure result.
    """
    return EmbeddingResult(
      id=input_id,
      embedding=None,
      error=EmbeddingError(
        code=code,
        message=message,
      ),
    )

  @staticmethod
  def _safe_error_message(
      exception: Exception,
      fallback: str,
  ) -> str:
    """
    Return a useful error message without returning a stack trace.
    """
    message = str(exception).strip()

    return message if message else fallback
