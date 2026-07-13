from functools import lru_cache
from html import unescape
import unicodedata

import spacy
from spacy.language import Language

class SpacyTextPreprocessor:
  """
  Applies versioned text preprocessing before SBERT generation.
  """

  SPACY_MODEL_NAME = "en_core_web_sm"

  NEGATION_WORDS = {
    "no",
    "not",
    "never",
    "neither",
    "nor",
    "without",
  }

  def preprocess_v1(self, text: str) -> str:
    """
    Apply spaCy preprocessing version 1.

    V1 performs the following operations:

    1. Decode HTML entities.
    2. Apply Unicode normalization.
    3. Tokenize using spaCy.
    4. Remove whitespace and punctuation tokens.
    5. Remove stop words.
    6. Convert words to lowercase lemmas.
    7. Retain numeric tokens.

    Args:
        text: Original text.

    Returns:
        Text prepared for SBERT.

    Raises:
        ValueError: If preprocessing removes all meaningful content.
    """
    normalized_text = self._normalize_text(text)

    nlp = self._load_spacy_model()
    document = nlp(normalized_text)

    processed_tokens: list[str] = []

    for token in document:
      if token.is_space or token.is_punct:
        continue

      """
      Retain negation words even if they are stop words (the, a, an, etc), 
      since they can change the meaning of a sentence.
      
      For example, "I have experience" vs "I do not have experience" are very 
      different statements.
      """
      if (
          token.is_stop
          and token.lower_ not in self.NEGATION_WORDS
      ):
        continue

      processed_token = self._get_processed_token(token)

      if processed_token:
        processed_tokens.append(processed_token)

    processed_text = " ".join(processed_tokens)

    if not processed_text:
      raise ValueError(
        "spaCy preprocessing removed all meaningful text"
      )

    return processed_text

  @staticmethod
  def _normalize_text(text: str) -> str:
    """
    Decode HTML entities and normalize equivalent Unicode characters.
    """
    decoded_text = unescape(text)

    return unicodedata.normalize(
      "NFKC",
      decoded_text,
    ).strip()

  @staticmethod
  def _get_processed_token(token) -> str:
    """
    Return the normalized representation of one spaCy token.
    """
    if token.like_num:
      return token.text.lower()

    # A pronoun lemma may be represented as "-PRON-" by some pipelines.
    # In that case, retain the original token text.
    lemma = token.lemma_.strip()

    if not lemma or lemma == "-PRON-":
      lemma = token.text

    return lemma.lower()

  @staticmethod
  @lru_cache(maxsize=1)
  def _load_spacy_model() -> Language:
    """
    Load the spaCy pipeline once and reuse it for subsequent requests.

    Parser and named-entity recognition are not needed for V1
    preprocessing, so they are disabled to reduce processing work.
    """
    return spacy.load(
      SpacyTextPreprocessor.SPACY_MODEL_NAME,
      disable=[
        "parser",
        "ner",
      ],
    )
