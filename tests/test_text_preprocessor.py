import pytest

from app.services.text_preprocessor import SpacyTextPreprocessor


@pytest.fixture(scope="module")
def preprocessor() -> SpacyTextPreprocessor:
  """
  Create one preprocessor for all tests in this module.
  """
  return SpacyTextPreprocessor()


def test_preprocessing_removes_punctuation_and_stop_words(
    preprocessor: SpacyTextPreprocessor,
) -> None:
  """
  Verify the basic V1 preprocessing behaviour.
  """
  processed_text = preprocessor.preprocess_v1(
    "The accountants are preparing the financial reports!"
  )

  assert processed_text == "accountant prepare financial report"


def test_preprocessing_normalizes_case(
    preprocessor: SpacyTextPreprocessor,
) -> None:
  """
  Verify that equivalent text with different casing is normalized.
  """
  first_text = preprocessor.preprocess_v1(
    "FINANCIAL REPORTING AND BOOKKEEPING"
  )

  second_text = preprocessor.preprocess_v1(
    "Financial reporting and bookkeeping"
  )

  assert first_text == second_text


def test_preprocessing_uses_word_lemmas(
    preprocessor: SpacyTextPreprocessor,
) -> None:
  """
  Verify that inflected words are reduced to their lemmas.
  """
  processed_text = preprocessor.preprocess_v1(
    "Accountants managed accounts and prepared reports."
  )

  assert "accountant" in processed_text
  assert "manage" in processed_text
  assert "prepare" in processed_text
  assert "report" in processed_text


def test_preprocessing_decodes_html_entities(
    preprocessor: SpacyTextPreprocessor,
) -> None:
  """
  Verify that encoded HTML characters do not remain in the text.
  """
  processed_text = preprocessor.preprocess_v1(
    "Accounting &amp; financial reporting"
  )

  assert "&amp;" not in processed_text
  assert "accounting" in processed_text
  assert "financial" in processed_text
  assert "reporting" in processed_text


def test_preprocessing_rejects_text_with_no_meaningful_tokens(
    preprocessor: SpacyTextPreprocessor,
) -> None:
  """
  Verify that punctuation-only input does not reach SBERT.
  """
  with pytest.raises(
      ValueError,
      match="removed all meaningful text",
  ):
    preprocessor.preprocess_v1("... !!! ???")

def test_preprocessing_preserves_negation(
    preprocessor: SpacyTextPreprocessor,
) -> None:
  """
  Verify that preprocessing does not reverse the meaning of negative text.
  """
  processed_text = preprocessor.preprocess_v1(
    "The candidate does not have accounting experience."
  )

  assert "not" in processed_text
  assert "accounting" in processed_text
  assert "experience" in processed_text
