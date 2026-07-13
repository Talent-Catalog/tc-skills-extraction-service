from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.embedding_models import (
  EmbeddingConfigurationVersion,
  EmbeddingModelDetails,
)
from app.services.text_preprocessor import SpacyTextPreprocessor


class EmbeddingService:
  """
  Generates embeddings using Sentence Transformers.

  Depending on the configuration version, the original text may first be
  processed by spaCy.
  """

  def __init__(
      self,
      text_preprocessor: SpacyTextPreprocessor | None = None,
  ) -> None:
    """
    Create the embedding service.

    Args:
        text_preprocessor: Optional preprocessor, primarily useful for
            dependency injection in tests.
    """
    self._text_preprocessor = (
        text_preprocessor or SpacyTextPreprocessor()
    )

  def generate_embedding(
      self,
      text: str,
      model_details: EmbeddingModelDetails,
  ) -> list[float]:
    """
    Generate a normalized embedding for the supplied text.

    Normalized embeddings have a vector length of approximately 1.0.
    This means cosine similarity can also be calculated using the dot
    product of two generated embeddings.

    Args:
        text: Text to convert into an embedding.
        model_details: Details of the Sentence Transformers model.

    Returns:
        The generated embedding as a normal Python list.

    Raises:
        ValueError: If the text is empty or the model produces an
            unexpected number of dimensions.
    """
    if not text.strip():
      raise ValueError("Text must not be empty")

    prepared_text = self.prepare_text(
      text=text,
      configuration_version=(
        model_details.configuration_version
      ),
    )

    model = self._load_model(model_details.model_name)

    embedding = model.encode(
      prepared_text,
      convert_to_numpy=True,
      normalize_embeddings=True,
    )

    vector = np.asarray(
      embedding,
      dtype=np.float32,
    )

    actual_dimensions = len(vector)

    if actual_dimensions != model_details.dimensions:
      raise ValueError(
        f"Model '{model_details.model_name}' generated "
        f"{actual_dimensions} dimensions, but "
        f"{model_details.dimensions} were expected"
      )

    return vector.tolist()

  def prepare_text(
      self,
      text: str,
      configuration_version: EmbeddingConfigurationVersion,
  ) -> str:
    """
    Prepare text according to the selected configuration version.

    This method is public so preprocessing can be tested independently
    from embedding generation.
    """
    if (
        configuration_version
        == EmbeddingConfigurationVersion.SBERT_RAW_V1
    ):
      return text

    if (
        configuration_version
        == EmbeddingConfigurationVersion.SPACY_PREPROCESSING_V1
    ):
      return self._text_preprocessor.preprocess_v1(text)

    raise ValueError(
      f"Unsupported embedding configuration version: "
      f"{configuration_version}"
    )

  @staticmethod
  @lru_cache(maxsize=4)
  def _load_model(model_name: str) -> SentenceTransformer:
    """
    Load and cache each Sentence Transformers model.
    """
    return SentenceTransformer(model_name)
