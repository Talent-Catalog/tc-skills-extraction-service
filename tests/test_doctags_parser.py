from __future__ import annotations

from app.services.cv_extraction.doctags_parser import build_candidate

# Mirrors the header-row layout the pipeline was designed and tested
# against in tc-api-spec (see fixture_noureddine.doctags there): a company
# and quoted role on one row, location on a second chunk, date on a third,
# then either bullets or plain paragraphs underneath.
DOCTAGS_WITH_BULLETS = """<doctag>
<text><loc_36><loc_100><loc_150><loc_106>Acme Corp 'Engineer'</text>
<section_header_level_1><loc_155><loc_100><loc_250><loc_106>Beirut - Lebanon</section_header_level_1>
<text><loc_385><loc_100><loc_461><loc_106>Jan 2019 - Dec 2021</text>
<unordered_list><list_item><loc_51><loc_115><loc_460><loc_121>Led the platform team.</list_item>
</unordered_list>
</doctag>"""

DOCTAGS_WITH_PARAGRAPHS = """<doctag>
<text><loc_36><loc_100><loc_150><loc_106>Acme Corp 'Engineer'</text>
<section_header_level_1><loc_155><loc_100><loc_250><loc_106>Beirut - Lebanon</section_header_level_1>
<text><loc_385><loc_100><loc_461><loc_106>Jan 2019 - Dec 2021</text>
<text><loc_51><loc_115><loc_460><loc_130>Led the platform team through a major replatforming effort.</text>
<text><loc_51><loc_132><loc_460><loc_147>Mentored three junior engineers.</text>
</doctag>"""


def test_build_candidate_extracts_bulleted_job() -> None:
  candidate = build_candidate(DOCTAGS_WITH_BULLETS)
  entries = candidate["entries"]

  acme = next(e for e in entries if e.get("companyName") == "Acme Corp")
  assert acme["role"] == "Engineer"
  assert acme["startDate"] == "2019-01-01"
  assert acme["endDate"] == "2021-12-01"
  assert acme["country"] == {"isoCode": "LB", "name": "Lebanon", "status": "ACTIVE"}
  assert acme["description"] == "Led the platform team."


def test_build_candidate_splits_unbulleted_paragraphs_into_separate_entries() -> None:
  """
  Documents a known Stage 1 limitation (see doctags_parser.py's
  propose_entries docstring): a job described in plain paragraphs, not
  bullets, is NOT attached to that job's entry automatically. Stage 2/3
  (see test_cv_extraction_service.py) is what reassembles it, using the
  element ids this test also checks are exposed correctly.
  """
  candidate = build_candidate(DOCTAGS_WITH_PARAGRAPHS)
  entries = candidate["entries"]

  acme = next(e for e in entries if e.get("companyName") == "Acme Corp")
  assert acme["description"] == "", (
    "paragraphs aren't bullets, so Stage 1 should NOT attach them automatically"
  )

  paragraph_entries = [
    e for e in entries
    if e.get("companyName") is None
    and ("Led the platform team" in (e.get("role") or "") or (e.get("role") or "").startswith("Mentored"))
  ]
  assert len(paragraph_entries) == 2, (
    "each unbulleted paragraph should be proposed as its own separate entry"
  )
