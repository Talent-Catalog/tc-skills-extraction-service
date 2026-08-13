"""
Embedding quality regression tests for the real all-MiniLM-L6-v2 model.

These tests are deliberately separate from the unit tests for batching,
validation and item-level failures. They exercise the complete FastAPI
embedding endpoint and therefore load the real SentenceTransformer and spaCy
models.

Run with:

    pytest -s tests/test_embedding_quality.py

The ``-s`` option displays the similarity measurements printed by the test.
"""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass

from numpy import dot
from numpy.linalg import norm

from app.models.embedding_models import (
  EmbeddingConfigurationVersion,
  EmbeddingInputType,
)

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


MODEL_NAME = "all-MiniLM-L6-v2"
MODEL_DIMENSIONS = 384
GENERATE_EMBEDDINGS_PATH = "/embeddings"

QUERY_TEXT = """
We are seeking an experienced accountant to prepare financial statements,
manage budgets, reconcile accounts, and support financial reporting.
"""

RELATED_TEXT = """
I worked as an accountant. I prepared monthly financial reports, managed
budgets, reconciled ledger accounts, and supported annual audits.
"""

UNRELATED_TEXT = """
I worked as a chef in a busy restaurant, preparing meals, planning menus,
maintaining food safety standards, and supervising kitchen staff.
"""


@dataclass(frozen=True)
class SimilarityMeasurement:
  """Contains the quality measurements for one embedding configuration."""

  configuration: EmbeddingConfigurationVersion
  related_similarity: float
  unrelated_similarity: float

  @property
  def ranking_margin(self) -> float:
    """Return the separation between the related and unrelated results."""
    return self.related_similarity - self.unrelated_similarity


@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
  """
  Start the FastAPI application without calling the Spring Boot skills API.

  The patch must be active before entering the TestClient context because
  FastAPI lifespan startup runs when TestClient starts.
  """
  mock_skills = [
    "accounting",
    "bookkeeping",
    "financial reporting",
    "budget management",
  ]

  with patch(
      "app.services.skills_service.SkillsService.get_skills",
      return_value=mock_skills,
  ):
    with TestClient(app) as test_client:
      yield test_client

def cosine_similarity(left: list[float], right: list[float]) -> float:
  """Calculate cosine similarity between two embedding vectors."""
  left_norm = norm(left)
  right_norm = norm(right)

  assert left_norm > 0
  assert right_norm > 0

  return float(dot(left, right) / (left_norm * right_norm))


def request_embeddings(
    client: TestClient,
    configuration: EmbeddingConfigurationVersion,
    input_type: EmbeddingInputType,
    inputs: list[dict[str, str]],
) -> dict[str, list[float]]:
  """Call the public FastAPI endpoint for one batch of one input type."""
  response = client.post(
    GENERATE_EMBEDDINGS_PATH,
    json={
      "model": {
        "model_name": MODEL_NAME,
        "configuration_version": configuration.value,
        "dimensions": MODEL_DIMENSIONS,
      },
      "type": input_type.value,
      "inputs": inputs,
    },
  )

  assert response.status_code == 200, response.text

  body = response.json()

  assert body["requested"] == len(inputs)
  assert body["succeeded"] == len(inputs)
  assert body["failed"] == 0

  embeddings: dict[str, list[float]] = {}

  for result in body["results"]:
    assert result["error"] is None
    assert result["embedding"] is not None
    assert len(result["embedding"]) == MODEL_DIMENSIONS

    embeddings[result["id"]] = result["embedding"]

  return embeddings


def generate_embeddings(
    client: TestClient,
    configuration: EmbeddingConfigurationVersion,
) -> dict[str, list[float]]:
  """
  Generate embeddings for the query and the candidate documents.

  The query and documents are requested separately because
  ``EmbeddingsRequest.type`` applies to the whole batch, and a query is
  embedded using a different input type than the documents being searched.
  """
  embeddings = request_embeddings(
    client,
    configuration,
    EmbeddingInputType.QUERY,
    [{"id": "query", "text": QUERY_TEXT}],
  )

  embeddings.update(
    request_embeddings(
      client,
      configuration,
      EmbeddingInputType.DOCUMENT,
      [
        {"id": "related", "text": RELATED_TEXT},
        {"id": "unrelated", "text": UNRELATED_TEXT},
      ],
    )
  )

  assert set(embeddings) == {"query", "related", "unrelated"}

  return embeddings


def measure_configuration(
    client: TestClient,
    configuration: EmbeddingConfigurationVersion,
) -> SimilarityMeasurement:
  """Generate and compare embeddings for one configuration."""
  embeddings = generate_embeddings(client, configuration)

  return SimilarityMeasurement(
    configuration=configuration,
    related_similarity=cosine_similarity(
      embeddings["query"], embeddings["related"]
    ),
    unrelated_similarity=cosine_similarity(
      embeddings["query"], embeddings["unrelated"]
    ),
  )


@pytest.mark.parametrize(
  "configuration",
  list(EmbeddingConfigurationVersion),
)
def test_related_experience_ranks_above_unrelated_experience(
    client: TestClient,
    configuration: EmbeddingConfigurationVersion,
) -> None:
  """Verify the fundamental ranking behaviour for every configuration."""
  measurement = measure_configuration(client, configuration)

  print()
  print(f"Configuration:       {measurement.configuration.value}")
  print(f"Related similarity:  {measurement.related_similarity:.4f}")
  print(f"Unrelated similarity:{measurement.unrelated_similarity:.4f}")
  print(f"Ranking margin:       {measurement.ranking_margin:.4f}")

  assert measurement.related_similarity > measurement.unrelated_similarity
  assert measurement.ranking_margin > 0


def test_preprocessing_versions_improve_ranking_separation(
    client: TestClient,
) -> None:
  """
  Compare all configurations against the raw SBERT baseline.

  V1 is reported but is not required to improve on raw SBERT because your
  earlier measurements showed identical results for those configurations.
  """
  measurements = {
    configuration: measure_configuration(client, configuration)
    for configuration in EmbeddingConfigurationVersion
  }

  print()
  print("Embedding quality comparison")
  print("----------------------------")

  for configuration, measurement in measurements.items():
    print(
      f"{configuration.value:28} "
      f"related={measurement.related_similarity:.4f} "
      f"unrelated={measurement.unrelated_similarity:.4f} "
      f"margin={measurement.ranking_margin:.4f}"
    )

  raw_margin = measurements[
    EmbeddingConfigurationVersion.SBERT_RAW_V1
  ].ranking_margin

  v2_margin = measurements[
    EmbeddingConfigurationVersion.SPACY_PREPROCESSING_V2
  ].ranking_margin

  v3_margin = measurements[
    EmbeddingConfigurationVersion.SPACY_PREPROCESSING_V3
  ].ranking_margin

  assert v2_margin > raw_margin
  assert v3_margin > raw_margin
  assert v3_margin >= v2_margin
