import numpy as np
import pytest

from unittest.mock import Mock

from app.embedding_models import (
  EmbeddingConfigurationVersion,
  EmbeddingModelDetails
)

from app.services.embedding_service import EmbeddingService

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_DIMENSIONS = 384


@pytest.fixture(scope="module")
def embedding_service() -> EmbeddingService:
  """
  Create one embedding service for all tests in this module.

  The underlying Sentence Transformer model is also cached by the service,
  so it is not reloaded for each test.
  """
  return EmbeddingService()


@pytest.fixture(scope="module")
def raw_model_details() -> EmbeddingModelDetails:
  """
  Model configuration without preprocessing.
  """
  return EmbeddingModelDetails(
    model_name=MODEL_NAME,
    dimensions=MODEL_DIMENSIONS,
    configuration_version=(
      EmbeddingConfigurationVersion.SBERT_RAW_V1
    ),
  )

@pytest.fixture(scope="module")
def preprocessed_model_details() -> EmbeddingModelDetails:
  """
  Model configuration using spaCy preprocessing V1.
  """
  return EmbeddingModelDetails(
    model_name=MODEL_NAME,
    dimensions=MODEL_DIMENSIONS,
    configuration_version=(
      EmbeddingConfigurationVersion.SPACY_PREPROCESSING_V1
    ),
  )

def cosine_similarity(
    first_embedding: list[float],
    second_embedding: list[float],
) -> float:
  """
  Calculate cosine similarity between two embedding vectors.

  A score closer to 1 means that the texts are more semantically similar.
  A score closer to 0 means that they are less similar.

  The service returns normalized embeddings, but this implementation still
  performs the full cosine calculation so it also works with non-normalized
  vectors.
  """
  first_vector = np.asarray(
    first_embedding,
    dtype=np.float32,
  )
  second_vector = np.asarray(
    second_embedding,
    dtype=np.float32,
  )

  denominator = (
      np.linalg.norm(first_vector)
      * np.linalg.norm(second_vector)
  )

  if denominator == 0:
    raise ValueError(
      "Cosine similarity cannot be calculated for a zero vector"
    )

  return float(
    np.dot(first_vector, second_vector) / denominator
  )


def test_generate_embedding_returns_expected_dimensions(
    embedding_service: EmbeddingService,
    raw_model_details: EmbeddingModelDetails,
) -> None:
  """
  Verify that the model generates the configured number of dimensions.
  """
  embedding = embedding_service.generate_embedding(
    text=(
      "Experienced accountant with bookkeeping and "
      "financial reporting skills."
    ),
    model_details=raw_model_details,
  )

  assert len(embedding) == MODEL_DIMENSIONS


def test_generate_embedding_returns_floats(
    embedding_service: EmbeddingService,
    raw_model_details: EmbeddingModelDetails,
) -> None:
  """
  Verify that the returned embedding can be serialized as JSON numbers.
  """
  embedding = embedding_service.generate_embedding(
    text="Experienced software engineer",
    model_details=raw_model_details,
  )

  assert embedding
  assert all(
    isinstance(value, float)
    for value in embedding
  )


def test_generated_embedding_is_normalized(
    embedding_service: EmbeddingService,
    raw_model_details: EmbeddingModelDetails,
) -> None:
  """
  Verify that normalize_embeddings=True produces a unit-length vector.
  """
  embedding = embedding_service.generate_embedding(
    text="Experienced accountant and financial analyst",
    model_details=raw_model_details,
  )

  vector_length = np.linalg.norm(
    np.asarray(
      embedding,
      dtype=np.float32,
    )
  )

  assert vector_length == pytest.approx(
    1.0,
    abs=0.0001,
  )


def test_same_text_generates_same_embedding(
    embedding_service: EmbeddingService,
    raw_model_details: EmbeddingModelDetails,
) -> None:
  """
  Verify that embedding generation is deterministic for the same model
  and input text.
  """
  text = "Accountant experienced in financial reporting"

  first_embedding = embedding_service.generate_embedding(
    text=text,
    model_details=raw_model_details,
  )

  second_embedding = embedding_service.generate_embedding(
    text=text,
    model_details=raw_model_details,
  )

  assert first_embedding == pytest.approx(
    second_embedding,
    abs=0.000001,
  )


def test_empty_text_is_rejected(
    embedding_service: EmbeddingService,
    raw_model_details: EmbeddingModelDetails,
) -> None:
  """
  Verify that empty or whitespace-only input is rejected.
  """
  with pytest.raises(
      ValueError,
      match="Text must not be empty",
  ):
    embedding_service.generate_embedding(
      text="   ",
      model_details=raw_model_details,
    )


def test_unexpected_dimensions_are_rejected(
    embedding_service: EmbeddingService,
) -> None:
  """
  Verify that a mismatch between the configured and actual dimensions
  is reported.
  """
  incorrect_model_details = EmbeddingModelDetails(
    model_name=MODEL_NAME,
    dimensions=768,
    configuration_version=(
      EmbeddingConfigurationVersion.SBERT_RAW_V1
    ),
  )

  with pytest.raises(
      ValueError,
      match="generated 384 dimensions",
  ):
    embedding_service.generate_embedding(
      text="Experienced accountant",
      model_details=incorrect_model_details,
    )


def test_similar_texts_have_high_similarity(
    embedding_service: EmbeddingService,
    raw_model_details: EmbeddingModelDetails,
) -> None:
  """
  Demonstrate cosine similarity between two texts with similar meanings.
  """
  first_text = (
    "Experienced accountant skilled in bookkeeping, budgeting, "
    "and preparing financial reports."
  )

  second_text = (
    "Financial professional with experience managing accounts, "
    "maintaining ledgers, and producing financial statements."
  )

  first_embedding = embedding_service.generate_embedding(
    text=first_text,
    model_details=raw_model_details,
  )

  second_embedding = embedding_service.generate_embedding(
    text=second_text,
    model_details=raw_model_details,
  )

  similarity = cosine_similarity(
    first_embedding,
    second_embedding,
  )

  print(
    f"\nSimilarity between related accounting texts: "
    f"{similarity:.4f}"
  )

  assert similarity > 0.50


def test_related_texts_are_more_similar_than_unrelated_texts(
    embedding_service: EmbeddingService,
    raw_model_details: EmbeddingModelDetails,
) -> None:
  """
  Demonstrate that a job description is closer to a related candidate
  experience than to an unrelated candidate experience.
  """
  opportunity_text = (
    "We are seeking an accountant with experience in bookkeeping, "
    "financial reporting, budgeting, and account reconciliation."
  )

  related_candidate_text = (
    "Worked as an accountant preparing monthly financial reports, "
    "reconciling accounts, maintaining ledgers, and managing budgets."
  )

  unrelated_candidate_text = (
    "Worked as a commercial chef preparing meals, designing menus, "
    "ordering ingredients, and supervising kitchen staff."
  )

  opportunity_embedding = embedding_service.generate_embedding(
    text=opportunity_text,
    model_details=raw_model_details,
  )

  related_embedding = embedding_service.generate_embedding(
    text=related_candidate_text,
    model_details=raw_model_details,
  )

  unrelated_embedding = embedding_service.generate_embedding(
    text=unrelated_candidate_text,
    model_details=raw_model_details,
  )

  related_similarity = cosine_similarity(
    opportunity_embedding,
    related_embedding,
  )

  unrelated_similarity = cosine_similarity(
    opportunity_embedding,
    unrelated_embedding,
  )

  print(
    f"\nOpportunity to related candidate:   "
    f"{related_similarity:.4f}"
  )
  print(
    f"Opportunity to unrelated candidate: "
    f"{unrelated_similarity:.4f}"
  )

  assert related_similarity > unrelated_similarity

def test_raw_configuration_does_not_preprocess_text() -> None:
  """
  Verify that SBERT_RAW_V1 returns the original text unchanged.
  """
  preprocessor = Mock()
  service = EmbeddingService(
    text_preprocessor=preprocessor,
  )

  original_text = "The accountants prepared reports."

  prepared_text = service.prepare_text(
    text=original_text,
    configuration_version=(
      EmbeddingConfigurationVersion.SBERT_RAW_V1
    ),
  )

  assert prepared_text == original_text
  preprocessor.preprocess_v1.assert_not_called()


def test_spacy_configuration_preprocesses_text() -> None:
  """
  Verify that SPACY_PREPROCESSING_V1 invokes the spaCy preprocessor.
  """
  preprocessor = Mock()
  preprocessor.preprocess_v1.return_value = (
    "accountant prepare report"
  )

  service = EmbeddingService(
    text_preprocessor=preprocessor,
  )

  prepared_text = service.prepare_text(
    text="The accountants prepared reports.",
    configuration_version=(
      EmbeddingConfigurationVersion.SPACY_PREPROCESSING_V1
    ),
  )

  assert prepared_text == "accountant prepare report"

  preprocessor.preprocess_v1.assert_called_once_with(
    "The accountants prepared reports."
  )

def test_preprocessing_reduces_difference_between_word_forms(
    embedding_service: EmbeddingService,
    raw_model_details: EmbeddingModelDetails,
    preprocessed_model_details: EmbeddingModelDetails,
) -> None:
  """
  Compare texts that use different grammatical forms of the same concepts.

  This demonstrates the intended benefit of lemmatization. The printed
  values should be treated as experimental observations rather than as
  universal similarity thresholds.
  """
  first_text = (
    "Accountants managed budgets, reconciled accounts, "
    "and prepared financial reports."
  )

  second_text = (
    "An accountant manages budgeting, reconciles account balances, "
    "and prepares financial reporting."
  )

  raw_first = embedding_service.generate_embedding(
    text=first_text,
    model_details=raw_model_details,
  )

  raw_second = embedding_service.generate_embedding(
    text=second_text,
    model_details=raw_model_details,
  )

  preprocessed_first = embedding_service.generate_embedding(
    text=first_text,
    model_details=preprocessed_model_details,
  )

  preprocessed_second = embedding_service.generate_embedding(
    text=second_text,
    model_details=preprocessed_model_details,
  )

  raw_similarity = cosine_similarity(
    raw_first,
    raw_second,
  )

  preprocessed_similarity = cosine_similarity(
    preprocessed_first,
    preprocessed_second,
  )

  print(
    f"\nRaw similarity:          {raw_similarity:.4f}"
  )
  print(
    f"Preprocessed similarity: {preprocessed_similarity:.4f}"
  )

  assert preprocessed_similarity > raw_similarity

def test_preprocessing_improves_similarity_for_noisy_text(
    embedding_service: EmbeddingService,
    raw_model_details: EmbeddingModelDetails,
    preprocessed_model_details: EmbeddingModelDetails,
) -> None:
  """
  Demonstrate preprocessing on text containing punctuation, boilerplate,
  casing differences and repeated function words.
  """
  opportunity_text = (
    "Accountant required for bookkeeping, account reconciliation, "
    "budget management and financial reporting."
  )

  noisy_candidate_text = (
    "THE candidate has been responsible for the MANAGEMENT of budgets; "
    "they were RECONCILING the accounts, and they have also PREPARED "
    "the monthly financial reports!!!"
  )

  raw_opportunity = embedding_service.generate_embedding(
    text=opportunity_text,
    model_details=raw_model_details,
  )

  raw_candidate = embedding_service.generate_embedding(
    text=noisy_candidate_text,
    model_details=raw_model_details,
  )

  preprocessed_opportunity = (
    embedding_service.generate_embedding(
      text=opportunity_text,
      model_details=preprocessed_model_details,
    )
  )

  preprocessed_candidate = (
    embedding_service.generate_embedding(
      text=noisy_candidate_text,
      model_details=preprocessed_model_details,
    )
  )

  raw_similarity = cosine_similarity(
    raw_opportunity,
    raw_candidate,
  )

  preprocessed_similarity = cosine_similarity(
    preprocessed_opportunity,
    preprocessed_candidate,
  )

  print(
    f"\nRaw noisy-text similarity:          "
    f"{raw_similarity:.4f}"
  )
  print(
    f"Preprocessed noisy-text similarity: "
    f"{preprocessed_similarity:.4f}"
  )

  assert preprocessed_similarity > raw_similarity
