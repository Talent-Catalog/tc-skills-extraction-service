import pytest

from app.services.text_preprocessor import SpacyTextPreprocessor


@pytest.fixture(scope="module")
def preprocessor() -> SpacyTextPreprocessor:
  """
  Reuse one preprocessor and its cached spaCy model.
  """
  return SpacyTextPreprocessor()


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
    "Accountants managed accounts and prepared reports."
  )

  assert "accountant" in result
  assert "manage" in result
  assert "account" in result
  assert "prepare" in result
  assert "report" in result

  assert "accountants" not in result
  assert "managed" not in result
  assert "prepared" not in result


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
