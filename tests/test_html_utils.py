from app.services.html_utils import html_to_readable_text, remove_html


def test_removes_tags():
  result = remove_html("<p>The <b>accountant</b> prepared reports.</p>")

  assert result == "The accountant prepared reports."


def test_does_not_merge_adjacent_words():
  """
  Adjacent tags with no whitespace between them must not merge the words
  either side of the boundary into one word.
  """
  result = remove_html("<li>Python</li><li>accounting</li>")

  assert result == "Python accounting"


def test_collapses_whitespace_introduced_around_tags():
  """
  The space separator used to avoid merging adjacent tags can introduce
  extra whitespace around tags that already had surrounding whitespace;
  that must be collapsed so multi-word phrase matching stays intact.
  """
  result = remove_html("<p>project <b>management</b> experience</p>")

  assert result == "project management experience"


def test_leaves_plain_text_unchanged():
  result = remove_html("Accountants managed accounts.")

  assert result == "Accountants managed accounts."


def test_html_to_readable_text_leaves_plain_text_unchanged():
  result = html_to_readable_text("Accountants managed accounts.")

  assert result == "Accountants managed accounts."


def test_html_to_readable_text_separates_paragraphs_onto_their_own_lines():
  result = html_to_readable_text(
    "<p>Line one.</p><p>Line two.</p>"
  )

  assert result == "Line one.\nLine two."


def test_html_to_readable_text_separates_list_items_onto_their_own_lines():
  result = html_to_readable_text(
    "<p>Java developer</p><ul><li>Spring Boot</li><li>PostgreSQL</li></ul>"
  )

  assert result == "Java developer\nSpring Boot\nPostgreSQL"


def test_html_to_readable_text_keeps_inline_formatting_on_one_line():
  """
  Inline tags such as <strong>/<em> must not fragment a sentence onto
  separate lines - only block-level elements should introduce line breaks.
  """
  result = html_to_readable_text(
    "Skilled in <strong>Python</strong> and <em>SQL</em>."
  )

  assert result == "Skilled in Python and SQL."


def test_html_to_readable_text_does_not_merge_adjacent_elements():
  """
  Adjacent block elements with no whitespace between them in the source
  must not merge the words either side of the boundary together.
  """
  result = html_to_readable_text("<li>Python</li><li>accounting</li>")

  assert result == "Python\naccounting"


def test_html_to_readable_text_normalizes_excess_whitespace():
  result = html_to_readable_text(
    "<div>   <p>  Nested   and    spaced  </p>   </div>"
  )

  assert result == "Nested and spaced"
