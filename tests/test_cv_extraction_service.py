from __future__ import annotations

import pytest

from app.services.cv_extraction.cv_extraction_service import PipelineError, apply_corrections
from app.services.cv_extraction.doctags_parser import build_candidate
from app.services.cv_extraction.pipeline_schemas import EntryClassification, LabelResult, ReassignElement

DOCTAGS_WITH_PARAGRAPHS = """<doctag>
<text><loc_36><loc_100><loc_150><loc_106>Acme Corp 'Engineer'</text>
<section_header_level_1><loc_155><loc_100><loc_250><loc_106>Beirut - Lebanon</section_header_level_1>
<text><loc_385><loc_100><loc_461><loc_106>Jan 2019 - Dec 2021</text>
<text><loc_51><loc_115><loc_460><loc_130>Led the platform team through a major replatforming effort.</text>
<text><loc_51><loc_132><loc_460><loc_147>Mentored three junior engineers.</text>
<text><loc_36><loc_160><loc_150><loc_166>Beta LLC 'Consultant'</text>
<section_header_level_1><loc_155><loc_160><loc_250><loc_166>Saida - Lebanon</section_header_level_1>
<text><loc_385><loc_160><loc_461><loc_166>Jan 2022 - Present</text>
<unordered_list><list_item><loc_51><loc_175><loc_460><loc_181>Advised clients on cloud migration strategy.</list_item>
</unordered_list>
</doctag>"""


def _classify(entries: list[dict], job_experience_indices: set[int]) -> list[EntryClassification]:
  return [
    EntryClassification(
      entryIndex=i,
      targetSchema="CandidateJobExperience" if i in job_experience_indices else "none",
      confidence=0.9,
      reason="test",
    )
    for i in range(len(entries))
  ]


def test_apply_corrections_reassembles_unbulleted_paragraph_job() -> None:
  """
  Proves the fix for the "not every job is bulleted" gap: Stage 1 alone
  splits a two-paragraph job into 3 entries (see test_doctags_parser.py);
  this test proves Stage 3 correctly reassembles it into one job - with
  role/dates intact and both paragraphs appended, in order, to
  description - once given role-tagged reassign_element corrections.
  """
  candidate = build_candidate(DOCTAGS_WITH_PARAGRAPHS)
  entries = candidate["entries"]

  acme_idx = next(i for i, e in enumerate(entries) if e.get("companyName") == "Acme Corp")
  beta_idx = next(i for i, e in enumerate(entries) if e.get("companyName") == "Beta LLC")
  paragraph_ids = [
    eid
    for e in entries
    if e.get("companyName") is None
    for eid in e["elementIds"]["header"]
    if "Led the platform team" in e.get("role", "") or e.get("role", "").startswith("Mentored")
  ]
  assert len(paragraph_ids) == 2

  label = LabelResult(
    isCv=True,
    isCvConfidence=0.99,
    isCvReason="test",
    classifications=_classify(entries, {acme_idx, beta_idx}),
    corrections=[
      ReassignElement(
        op="reassign_element", elementId=eid, fromEntryIndex=None,
        toEntryIndex=acme_idx, role="description",
        reason="Unbulleted paragraph describing the Acme job.",
      )
      for eid in paragraph_ids
    ],
  )

  result = apply_corrections(candidate, label)

  acme = next(e for e in result if e["companyName"] == "Acme Corp")
  assert acme["role"] == "Engineer"
  assert acme["startDate"] == "2019-01-01"
  assert acme["endDate"] == "2021-12-01"
  assert "Led the platform team" in acme["description"]
  assert "Mentored three junior engineers" in acme["description"]
  assert acme["description"].index("Led the platform team") < acme["description"].index("Mentored")
  assert "Led the platform team" not in acme["role"], "paragraph text leaked into the role field"

  beta = next(e for e in result if e["companyName"] == "Beta LLC")
  assert beta["description"] == "Advised clients on cloud migration strategy."


def test_apply_corrections_rejects_hallucinated_element_id() -> None:
  candidate = build_candidate(DOCTAGS_WITH_PARAGRAPHS)
  entries = candidate["entries"]

  label = LabelResult(
    isCv=True,
    isCvConfidence=0.9,
    isCvReason="test",
    classifications=_classify(entries, set(range(len(entries)))),
    corrections=[
      ReassignElement(
        op="reassign_element", elementId=999_999, fromEntryIndex=None,
        toEntryIndex=0, role="header", reason="hallucinated",
      )
    ],
  )

  with pytest.raises(PipelineError):
    apply_corrections(candidate, label)


def test_apply_corrections_rejects_incomplete_classification() -> None:
  candidate = build_candidate(DOCTAGS_WITH_PARAGRAPHS)
  entries = candidate["entries"]

  label = LabelResult(
    isCv=True,
    isCvConfidence=0.9,
    isCvReason="test",
    classifications=_classify(entries, set(range(len(entries))))[:-1],
    corrections=[],
  )

  with pytest.raises(PipelineError):
    apply_corrections(candidate, label)
