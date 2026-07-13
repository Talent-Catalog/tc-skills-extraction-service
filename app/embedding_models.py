from enum import Enum

from pydantic import BaseModel, Field


class EmbeddingConfigurationVersion(str, Enum):
  """
  Identifies the preprocessing algorithm used before embedding generation.

  A change to preprocessing creates a different embedding configuration,
  even when the same Sentence Transformer model is used.
  """

  SBERT_RAW_V1 = "SBERT_RAW_V1"

  SPACY_PREPROCESSING_V1 = "SPACY_PREPROCESSING_V1"

  SPACY_PREPROCESSING_V2 = "SPACY_PREPROCESSING_V2"

  SPACY_PREPROCESSING_V3 = "SPACY_PREPROCESSING_V3"

class EmbeddingModelDetails(BaseModel):
  """
  Details supplied from the Postgres EmbeddingModel table.
  """

  model_name: str = Field(
    ...,
    min_length=1,
    description="Sentence Transformers model name",
  )

  dimensions: int = Field(
    ...,
    gt=0,
    description="Expected embedding dimensions",
  )

  configuration_version: EmbeddingConfigurationVersion = Field(
    ...,
    description="Version of the embedding and preprocessing algorithm",
  )


class GenerateEmbeddingRequest(BaseModel):
  """
  Request for generating an embedding.
  """

  text: str = Field(
    ...,
    min_length=1,
    description="Text from which to generate an embedding",
  )

  model: EmbeddingModelDetails


class GenerateEmbeddingResponse(BaseModel):
  """
  Response containing the generated embedding.
  """

  dimensions: int
  embedding: list[float]
