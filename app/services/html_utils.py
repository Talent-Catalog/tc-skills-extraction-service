import re

from bs4 import BeautifulSoup


def remove_html(text: str) -> str:
  """
  Remove HTML markup while preserving its human-readable text.
  """
  # A space separator prevents adjacent tags (e.g. "<li>a</li><li>b</li>")
  # from merging into one word, but it also adds an extra space next to
  # any tag that already had surrounding whitespace in the markup.
  text_without_html = BeautifulSoup(
    text,
    "html.parser",
  ).get_text(separator=" ")

  # Collapse that possible double whitespace back to a single space.
  # Otherwise, downstream spaCy tokenization would turn a double space
  # into its own token, splitting a multi-word skill like
  # "project management" and silently breaking PhraseMatcher on it.
  return re.sub(r"\s+", " ", text_without_html).strip()
