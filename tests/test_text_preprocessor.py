from unittest.mock import patch

import pytest

from app.models.embedding_models import EmbeddingConfigurationVersion
from app.services.text_preprocessor import SpacyTextPreprocessor


@pytest.fixture
def preprocessor() -> SpacyTextPreprocessor:
  """
  Create the text preprocessor used by dispatcher tests.
  """
  return SpacyTextPreprocessor()


def test_preprocess_returns_raw_text_for_sbert_raw_v1(
    preprocessor: SpacyTextPreprocessor,
) -> None:
  """
  Raw SBERT preprocessing should leave the source text unchanged.
  """
  text = "  Accountants &amp; bookkeepers.\n"

  result = preprocessor.preprocess(
    text=text,
    configuration_version=(
      EmbeddingConfigurationVersion.SBERT_RAW_V1
    ),
  )

  assert result == text


def test_preprocess_delegates_to_v1(
    preprocessor: SpacyTextPreprocessor,
) -> None:
  """
  The V1 configuration should delegate to preprocess_v1.
  """
  with patch.object(
      preprocessor,
      "preprocess_v1",
      return_value="v1 result",
  ) as preprocess_v1:
    result = preprocessor.preprocess(
      text="source text",
      configuration_version=(
        EmbeddingConfigurationVersion.SPACY_PREPROCESSING_V1
      ),
    )

  assert result == "v1 result"
  preprocess_v1.assert_called_once_with("source text")


def test_preprocess_delegates_to_v2(
    preprocessor: SpacyTextPreprocessor,
) -> None:
  """
  The V2 configuration should delegate to preprocess_v2.
  """
  with patch.object(
      preprocessor,
      "preprocess_v2",
      return_value="v2 result",
  ) as preprocess_v2:
    result = preprocessor.preprocess(
      text="source text",
      configuration_version=(
        EmbeddingConfigurationVersion.SPACY_PREPROCESSING_V2
      ),
    )

  assert result == "v2 result"
  preprocess_v2.assert_called_once_with("source text")


def test_preprocess_delegates_to_v3(
    preprocessor: SpacyTextPreprocessor,
) -> None:
  """
  The V3 configuration should delegate to preprocess_v3.
  """
  with patch.object(
      preprocessor,
      "preprocess_v3",
      return_value="v3 result",
  ) as preprocess_v3:
    result = preprocessor.preprocess(
      text="source text",
      configuration_version=(
        EmbeddingConfigurationVersion.SPACY_PREPROCESSING_V3
      ),
    )

  assert result == "v3 result"
  preprocess_v3.assert_called_once_with("source text")


def test_v1_performs_basic_cleanup(
    preprocessor: SpacyTextPreprocessor,
) -> None:
  """
  V1 should clean encoding and whitespace without rewriting words.
  """
  result = preprocessor.preprocess_v1(
    "  The   accountants\nwere preparing "
    "financial reports &amp; budgets.  "
  )

  assert result == (
    "The accountants were preparing "
    "financial reports & budgets."
  )


def test_v1_preserves_original_word_forms(
    preprocessor: SpacyTextPreprocessor,
) -> None:
  """
  V1 should not lemmatise words.
  """
  result = preprocessor.preprocess_v1(
    "Accountants managed accounts."
  )

  assert result == "Accountants managed accounts."


def test_v2_lemmatises_words(
    preprocessor: SpacyTextPreprocessor,
) -> None:
  """
  V2 should reduce inflected words to their base forms.
  """
  result = preprocessor.preprocess_v2(
    "Accountants managed accounts and they also prepared reports"
  )

  """
  Demonstrate that the lemmatisation is working as expected.
  """
  doc = preprocessor._process_with_spacy(result)
  for token in doc:
    print(
      token.text,
      token.lemma_,
      token.pos_,
    )

  tokens = result.split()

  assert "accountant" in tokens
  assert "manage" in tokens
  assert "account" in tokens
  assert "prepare" in tokens
  assert "report" in tokens

  assert "accountants" not in tokens
  assert "managed" not in tokens
  assert "prepared" not in tokens


def test_v2_preserves_stop_words(
    preprocessor: SpacyTextPreprocessor,
) -> None:
  """
  V2 lemmatises but does not remove stop words.
  """
  result = preprocessor.preprocess_v2(
    "The accountant prepared the report."
  )

  assert result == "the accountant prepare the report."


def test_v3_removes_stop_words(
    preprocessor: SpacyTextPreprocessor,
) -> None:
  """
  V3 should retain the important terms and remove common stop words.
  """
  result = preprocessor.preprocess_v3(
    "The accountants were preparing the financial reports."
  )

  assert result == (
    "accountant prepare financial report"
  )


def test_v3_preserves_negation(
    preprocessor: SpacyTextPreprocessor,
) -> None:
  """
  Stop-word removal must not remove words that reverse meaning.
  """
  result = preprocessor.preprocess_v3(
    "The candidate does not have accounting experience."
  )

  assert "not" in result
  assert "accounting" in result
  assert "experience" in result


def test_v2_normalizes_different_word_forms(
    preprocessor: SpacyTextPreprocessor,
) -> None:
  """
  Demonstrate the direct advantage of lemmatisation.

  Different grammatical forms should produce approximately the same
  normalized concepts.
  """
  first_result = preprocessor.preprocess_v2(
    "Accountants managed budgets and prepared reports."
  )

  second_result = preprocessor.preprocess_v2(
    "An accountant manages budgets and prepares reports."
  )

  assert "accountant" in first_result
  assert "accountant" in second_result

  assert "manage" in first_result
  assert "manage" in second_result

  assert "prepare" in first_result
  assert "prepare" in second_result


def test_empty_text_is_rejected(
    preprocessor: SpacyTextPreprocessor,
) -> None:
  """
  Empty input should not be sent to SBERT.
  """
  with pytest.raises(
      ValueError,
      match="Text must not be empty",
  ):
    preprocessor.preprocess_v1("   ")
