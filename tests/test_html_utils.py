from app.services.html_utils import remove_html


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
