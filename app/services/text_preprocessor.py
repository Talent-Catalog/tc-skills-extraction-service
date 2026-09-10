import re
import unicodedata
from functools import lru_cache
from html import unescape

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Token

from app.models.embedding_models import EmbeddingConfigurationVersion


class SpacyTextPreprocessor:
  """
  Applies versioned preprocessing to text before SBERT generates an
  embedding.
  """

  SPACY_MODEL_NAME = "en_core_web_sm"

  # These words are normally spaCy stop words, but removing them can reverse
  # or seriously alter the meaning of a sentence.
  NEGATION_WORDS = {
    "no",
    "not",
    "never",
    "neither",
    "nor",
    "without",
  }

  def preprocess(
      self,
      text: str,
      configuration_version: EmbeddingConfigurationVersion,
  ) -> str:
    """
    Apply the preprocessing associated with an embedding configuration.

    The raw SBERT configuration performs no preprocessing. Each spaCy
    configuration delegates to its corresponding versioned implementation.
    """
    match configuration_version:
      case EmbeddingConfigurationVersion.SBERT_RAW_V1:
        return text

      case EmbeddingConfigurationVersion.SPACY_PREPROCESSING_V1:
        return self.preprocess_v1(text)

      case EmbeddingConfigurationVersion.SPACY_PREPROCESSING_V2:
        return self.preprocess_v2(text)

      case EmbeddingConfigurationVersion.SPACY_PREPROCESSING_V3:
        return self.preprocess_v3(text)

    # Fail explicitly if a new configuration is added but preprocessing
    # support has not been implemented.
    raise ValueError(
      "Unsupported embedding configuration version: "
      f"{configuration_version}"
    )

  def preprocess_v1(self, text: str) -> str:
    """
    Apply conservative basic cleanup.

    V1:
    - decodes HTML entities;
    - applies Unicode normalization;
    - replaces repeated whitespace with a single space;
    - preserves natural-language wording;
    - preserves punctuation;
    - preserves stop words;
    - does not lemmatise words.
    """
    return self._basic_cleanup(text)

  def preprocess_v2(self, text: str) -> str:
    """
    Apply basic cleanup and lemmatisation.

    V2:
    - includes all V1 cleanup;
    - converts normal words to their dictionary base form;
    - preserves punctuation;
    - preserves stop words.
    """
    cleaned_text = self._basic_cleanup(text)

    document = self._process_with_spacy(cleaned_text)

    processed_text = self._lemmatise_document(
      document=document,
      remove_stop_words=False,
    )

    return self._validate_result(processed_text)

  def preprocess_v3(self, text: str) -> str:
    """
    Apply basic cleanup, lemmatisation and stop-word removal.

    V3:
    - includes all V1 cleanup;
    - converts words to their dictionary base form;
    - removes punctuation;
    - removes stop words;
    - preserves important negation words.
    """
    cleaned_text = self._basic_cleanup(text)

    document = self._process_with_spacy(cleaned_text)

    processed_text = self._lemmatise_document(
      document=document,
      remove_stop_words=True,
    )

    return self._validate_result(processed_text)

  @staticmethod
  def _basic_cleanup(text: str) -> str:
    """
    Perform cleanup that does not change the words or sentence meaning.
    """
    if not text or not text.strip():
      raise ValueError("Text must not be empty")

    # Convert values such as '&amp;' into '&'.
    decoded_text = unescape(text)

    # Normalize equivalent Unicode representations.
    normalized_text = unicodedata.normalize(
      "NFKC",
      decoded_text,
    )

    # Convert newlines, tabs and repeated spaces into one normal space.
    cleaned_text = re.sub(
      r"\s+",
      " ",
      normalized_text,
    ).strip()

    if not cleaned_text:
      raise ValueError("Text must not be empty")

    return cleaned_text

  def _lemmatise_document(
      self,
      document: Doc,
      remove_stop_words: bool,
  ) -> str:
    """
    Convert spaCy tokens into their normalized forms.
    """
    processed_tokens: list[str] = []

    for token in document:
      if token.is_space:
        continue

      if remove_stop_words and token.is_punct:
        continue

      if self._should_remove_stop_word(
          token=token,
          remove_stop_words=remove_stop_words,
      ):
        continue

      processed_token = self._process_token(
        token=token,
        preserve_punctuation=not remove_stop_words,
      )

      if processed_token:
        processed_tokens.append(processed_token)

    if remove_stop_words:
      # V3 is intentionally converted into a simple list of meaningful
      # normalized terms.
      return " ".join(processed_tokens)

    # V2 retains punctuation while avoiding spaces before punctuation.
    return self._join_tokens_with_punctuation(processed_tokens)

  def _should_remove_stop_word(
      self,
      token: Token,
      remove_stop_words: bool,
  ) -> bool:
    """
    Determine whether a stop word should be removed.

    Negation words are retained because removing them could reverse the
    meaning of the source text.
    """
    if not remove_stop_words:
      return False

    if not token.is_stop:
      return False

    return token.lower_ not in self.NEGATION_WORDS

  @staticmethod
  def _process_token(
      token: Token,
      preserve_punctuation: bool,
  ) -> str:
    """
    Return the desired representation of one token.
    """
    if token.is_punct:
      return token.text if preserve_punctuation else ""

    # Preserve numbers exactly as they appeared.
    if token.like_num:
      return token.text.lower()

    lemma = token.lemma_.strip()

    # Fall back to the original token if spaCy did not provide a useful
    # lemma.
    if not lemma or lemma == "-PRON-":
      lemma = token.text

    return lemma.lower()

  @staticmethod
  def _join_tokens_with_punctuation(
      tokens: list[str],
  ) -> str:
    """
    Join tokens while avoiding output such as 'report .'.
    """
    result = ""

    for token in tokens:
      if not result:
        result = token
      elif token in {".", ",", "!", "?", ";", ":", "%"}:
        result += token
      elif token in {"'", "’"}:
        result += token
      else:
        result += f" {token}"

    return result

  @staticmethod
  def _validate_result(text: str) -> str:
    """
    Checks result of preprocessing.
    """
    result = text.strip()

    # We used to reject preprocessing that removed all usable content.
    # This can happen, for example, if the text is all in Arabic or some other
    # language that Spacy is not configured for.
    # It can also happen if the text is all whitespace or filler words like
    # "the" or "and".
    # However, we now can also get useful data from the context such as the
    # occupation associated with an experience.
    # So we don't reject preprocessing that removes all usable content.
    # The original code was:
    # if not result:
    #   raise ValueError(
    #     "spaCy preprocessing removed all meaningful text"
    #   )

    return result

  @classmethod
  def _process_with_spacy(
      cls,
      text: str,
  ) -> Doc:
    """
    Process text through the cached spaCy pipeline.
    """
    return cls._load_spacy_model()(text)

  @classmethod
  @lru_cache(maxsize=1)
  def _load_spacy_model(cls) -> Language:
    """
    Load and cache the spaCy model.

    Named-entity recognition is unnecessary for these preprocessing
    versions. The parser is retained because grammatical analysis can
    contribute to accurate lemmatisation.
    """
    return spacy.load(
      cls.SPACY_MODEL_NAME,
      disable=[
        "ner",
      ],
    )
