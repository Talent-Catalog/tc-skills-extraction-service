from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.embedding_models import EmbeddingModelDetails


class EmbeddingService:
  """
  Generates text embeddings using Sentence Transformers.
  """

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

    model = self._load_model(model_details.model_name)

    embedding = model.encode(
      text,
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

  @staticmethod
  @lru_cache(maxsize=4)
  def _load_model(model_name: str) -> SentenceTransformer:
    """
    Load and cache a Sentence Transformers model.

    Loading a model is expensive, so each model is loaded only once
    for the lifetime of the Python process.
    """
    return SentenceTransformer(model_name)
