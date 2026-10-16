import re

from bs4 import BeautifulSoup

# Tags whose content should be separated by a line break from whatever comes
# next, so that e.g. adjacent list items or paragraphs don't run together.
# Inline tags (<strong>, <em>, <a>, ...) are deliberately not included here -
# their content should flow inline with the surrounding text, exactly as a
# person reading the rendered HTML would see it.
_BLOCK_LEVEL_TAGS = frozenset({
  "p", "div", "li", "ul", "ol", "br",
  "h1", "h2", "h3", "h4", "h5", "h6",
  "tr", "table", "thead", "tbody",
  "blockquote", "section", "article",
  "header", "footer", "pre",
})


def html_to_readable_text(text: str) -> str:
  """
  Convert HTML to human-readable plain text suitable for an LLM prompt.

  Unlike remove_html(), this preserves line-level separation between block
  elements (paragraphs, list items, headings, etc.) instead of flattening
  everything onto one line, while leaving inline formatting flowing inline
  with the surrounding text.
  """
  soup = BeautifulSoup(text, "html.parser")

  for block_tag in soup.find_all(_BLOCK_LEVEL_TAGS):
    block_tag.append("\n")

  text_without_html = soup.get_text()

  lines = (
    re.sub(r"[ \t]+", " ", line).strip()
    for line in text_without_html.splitlines()
  )

  return "\n".join(line for line in lines if line)


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
