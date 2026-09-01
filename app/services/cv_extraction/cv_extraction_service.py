"""
Stage 2 (LLM label), Stage 3 (deterministic apply+validate), and Stage 4
(LLM occupation grouping) on top of doctags_parser.py's Stage 1 output.

Ported from tc-api-spec/tools/doctags2schema/llm_pipeline.py. Design rules
carried over unchanged:
  - Stage 2 NEVER retypes content. It only classifies existing element ids
    and issues a closed set of correction ops that reference them.
  - Stage 3 re-runs the SAME deterministic extractors from doctags_parser.py
    on the corrected element groupings. The LLM's own field guesses (if it
    supplied any) are discarded - only its *grouping/classification*
    decisions are trusted.
  - Stage 4 is free-form clustering, but only over Stage 3's already-locked,
    already-traceable entries - no raw doctags content passes through it.

Differences from the tc-api-spec original:
  - Restructured as a class (CvExtractionService) taking its dependencies
    via constructor injection, matching this repo's service convention
    (EmbeddingService, ExplanationService, SkillsExtractor).
  - Uses the real `anthropic` SDK directly rather than an --offline stub -
    tests use dependency_overrides + a fake CvExtractionService instead
    (see tests/test_cv_extraction_api.py), matching this repo's existing
    testing convention rather than the CLI tool's own --offline flag.
  - Reuses LlmServiceUnavailableError/MalformedLlmResponseError from
    app.services.llm_client rather than inventing new exception types, so
    the router's error-mapping convention stays uniform across both the
    OpenAI-compatible Qwen client and this Anthropic-based one.
"""
from __future__ import annotations

import json
import logging

import anthropic
import pydantic

from app.models.candidate_occupation_models import (
  CandidateJobExperience,
  CandidateOccupation,
  Country,
  CvExtractionResponse,
  Occupation,
)
from app.services.cv_extraction.doctags_parser import Element, build_candidate, extract_fields_from_chunks
from app.services.cv_extraction.pdf_to_doctags import PdfToDoctagsConverter
from app.services.cv_extraction.pipeline_schemas import (
  DiscardElement,
  DiscardEntry,
  EntryClassification,
  LabelResult,
  NewEntry,
  OccupationGroupResult,
  ReassignElement,
)
from app.services.cv_extraction.schema_docs import all_target_schema_docs
from app.services.llm_client import LlmServiceUnavailableError, MalformedLlmResponseError

logger = logging.getLogger(__name__)


LABEL_SYSTEM_PROMPT = """\
You are reviewing a deterministic parser's proposed entries, extracted \
from a résumé's doctags markup by bounding-box geometry alone - grouped \
into header-shaped rows with bullets, with NO judgment about what each row \
means. Many will be job experience; many won't (education, certifications, \
skills, a lone section title with nothing under it, a free-flowing summary \
paragraph, ...). Do not assume anything about section-heading vocabulary - \
different résumés title the same content differently ("Career History", \
"Professional Experience", "Employment", or no heading at all).

Your job has two parts:

1. Decide whether this document is a CV/résumé at all, and classify EVERY \
proposed entry by which of these real API schemas its content matches - \
these ARE the categories, not a vocabulary of your own:

""" + all_target_schema_docs() + """

Use targetSchema="none" for genuine CV content that isn't one of the above \
(a summary paragraph, a references line, a lone section title with no \
distinct entries of its own). Use "unrecognized" when you can't tell - \
prefer this over guessing.

2. Optionally correct the parser's element grouping using ONLY the closed \
set of operations below. You may NEVER invent, retype, paraphrase, or \
summarize text - every operation references element ids that already \
exist in the input. If content is missing from every entry but you can see \
it in the raw `elements` list (e.g. a date sitting in the wrong entry due \
to column-layout drift), use reassign_element to move it - do not describe \
what it should say.

IMPORTANT - not every entry's content is bulleted. The parser only knows \
how to attach literal <list_item> bullets to a header row; it has NO way \
to notice that a paragraph of plain prose below a job title is that job's \
description rather than a new entry. So a job described in paragraphs, not \
bullets, comes out of Stage 1 as SEVERAL separate proposed entries: one for \
the real header row, and one MORE per paragraph of prose that follows it \
(each looking like its own header-less, bulletless, dateless entry). Watch \
for this pattern and reassign each paragraph's elements into the job entry \
they actually belong to, with role="description" - then classify the \
now-empty leftover entries as targetSchema="none" (or discard_entry them).

Operations:
- discard_entry: this whole proposed entry isn't a real, distinct record of any kind.
- discard_element: one element leaked into an entry and doesn't belong there.
- reassign_element: move an element (by id) from one entry to another, tagged \
role="header" (company/role/date/location text to parse) or \
role="description" (prose/bullet text to append verbatim) - use this both \
for geometry mistakes (a date on the wrong row/column) and for the \
unbulleted-paragraph case above.
- new_entry: the parser merged two distinct entries into one, or missed one \
entirely, but you can point at the existing header/description element ids \
that should form a separate entry.

Be conservative: only correct what you're confident is wrong. Leave \
ambiguous cases as "unrecognized" rather than guessing.
"""

OCCUPATION_SYSTEM_PROMPT = """\
You are given a validated list of a candidate's job experiences (already \
fact-checked - company, role, dates, description). Group them into \
occupations: which jobs represent the same real-world occupation, what \
that occupation should be called, and - if you can determine one - its \
ISCO-08 code. Also decide yearsExperience per occupation group: prefer the \
candidate's own stated claim if one appears in a job description, \
otherwise compute it from the date ranges, and say which you did and why. \
Every job experience index must appear in exactly one group.
"""


class PipelineError(RuntimeError):
  """
  Raised when the LLM's label/correction data is internally inconsistent
  (a hallucinated id, a missing classification, ...). Caught and re-wrapped
  as MalformedLlmResponseError by CvExtractionService so the router only
  ever needs to handle the two generic LLM-failure exception types.
  """


def apply_corrections(candidate: dict, label: LabelResult) -> list[dict]:
  """Stage 3: validates every id/index the LLM referenced actually exists,
  applies the corrections, keeps only targetSchema == "CandidateJobExperience"
  entries, then re-runs the Stage 1 extractors on the corrected element
  groups."""
  entries = [dict(e) for e in candidate["entries"]]
  elements_by_id = {el["id"]: el for el in candidate["elements"]}
  n = len(entries)

  def require_entry(idx: int, ctx: str):
    if not (0 <= idx < n):
      raise PipelineError(f"{ctx}: entryIndex {idx} out of range (0..{n - 1})")

  def require_element(eid: int, ctx: str):
    if eid not in elements_by_id:
      raise PipelineError(f"{ctx}: elementId {eid} does not exist in Stage 1 output")

  classification_by_idx = {c.entryIndex: c for c in label.classifications}
  for i in range(n):
    if i not in classification_by_idx:
      raise PipelineError(f"LLM label omitted a classification for entryIndex {i}")

  discarded_entries: set[int] = set()
  # Per entry, per element id -> "header" | "description". Seeded from
  # Stage 1's OWN partition, but tracked explicitly from here on rather
  # than re-derived from the element's raw `kind` - that's what lets a
  # corrected-in paragraph-text element be appended to `description`
  # instead of being mis-parsed as a header field.
  element_roles: dict[int, dict[int, str]] = {
    i: {eid: "header" for eid in entries[i]["elementIds"]["header"]}
       | {eid: "description" for eid in entries[i]["elementIds"]["bullets"]}
    for i in range(n)
  }
  new_entries: list[dict] = []

  for corr in label.corrections:
    if isinstance(corr, DiscardEntry):
      require_entry(corr.entryIndex, "discard_entry")
      discarded_entries.add(corr.entryIndex)

    elif isinstance(corr, DiscardElement):
      require_entry(corr.entryIndex, "discard_element")
      require_element(corr.elementId, "discard_element")
      element_roles[corr.entryIndex].pop(corr.elementId, None)

    elif isinstance(corr, ReassignElement):
      require_element(corr.elementId, "reassign_element")
      require_entry(corr.toEntryIndex, "reassign_element.toEntryIndex")
      if corr.fromEntryIndex is not None:
        require_entry(corr.fromEntryIndex, "reassign_element.fromEntryIndex")
        element_roles[corr.fromEntryIndex].pop(corr.elementId, None)
      else:
        for roles in element_roles.values():
          roles.pop(corr.elementId, None)
      element_roles[corr.toEntryIndex][corr.elementId] = corr.role

    elif isinstance(corr, NewEntry):
      for eid in corr.headerElementIds + corr.descriptionElementIds:
        require_element(eid, "new_entry")
      new_idx = n + len(new_entries)
      element_roles[new_idx] = (
        {eid: "header" for eid in corr.headerElementIds}
        | {eid: "description" for eid in corr.descriptionElementIds}
      )
      new_entries.append({
        "elementIds": {"header": corr.headerElementIds, "bullets": corr.descriptionElementIds},
        "_flags": [f"created_by_llm_correction:{corr.reason}"],
      })
      classification_by_idx[new_idx] = EntryClassification(
        entryIndex=new_idx, targetSchema="CandidateJobExperience", confidence=1.0,
        reason="LLM-created entry from reassigned elements.",
      )
    else:
      raise PipelineError(f"Unknown correction op: {corr}")

  all_entries = entries + new_entries
  result: list[dict] = []
  for i, entry in enumerate(all_entries):
    cls = classification_by_idx[i]
    if i in discarded_entries or cls.targetSchema != "CandidateJobExperience":
      continue

    roles = element_roles[i]
    header_elements = [elements_by_id[eid] for eid, role in roles.items() if role == "header"]
    description_elements = sorted(
      (elements_by_id[eid] for eid, role in roles.items() if role == "description"),
      key=lambda el: (el["page"], el["box"][1]),
    )

    fake_header_els = [Element(el["id"], el["kind"], el["page"], tuple(el["box"]), el["text"]) for el in header_elements]
    fields = extract_fields_from_chunks(fake_header_els) if fake_header_els else {"_flags": ["no_header_elements"]}

    result.append({
      "companyName": fields.get("companyName"),
      "role": fields.get("role", ""),
      "startDate": fields.get("startDate"),
      "endDate": fields.get("endDate"),
      "country": fields.get("country"),
      "description": "; ".join(el["text"].strip() for el in description_elements if el["text"].strip()),
    })
  return result


class CvExtractionService:
  """
  Extracts CandidateOccupation-shaped data from an uploaded CV PDF.

  Runs the four-stage pipeline described in this module's docstring. See
  app/services/cv_extraction/schemas/README.md for how the target schemas
  used in Stage 2's prompt get into this repo.
  """

  def __init__(
      self,
      pdf_converter: PdfToDoctagsConverter,
      anthropic_client: anthropic.Anthropic,
      model_name: str,
  ) -> None:
    self._pdf_converter = pdf_converter
    self._client = anthropic_client
    self._model_name = model_name

  def extract(self, pdf_bytes: bytes) -> CvExtractionResponse:
    """PdfConversionError propagates from self._pdf_converter.convert() -
    it is not caught here; the router maps it to a 4xx, distinct from the
    LLM-failure exceptions this method itself raises."""
    doctags = self._pdf_converter.convert(pdf_bytes)
    candidate = build_candidate(doctags)

    label = self._label(candidate)
    if not label.isCv:
      return CvExtractionResponse(is_cv=False, is_cv_confidence=label.isCvConfidence)

    try:
      entries = apply_corrections(candidate, label)
    except PipelineError as exception:
      raise MalformedLlmResponseError(
        "The LLM's proposed corrections were internally inconsistent"
      ) from exception

    if not entries:
      return CvExtractionResponse(is_cv=True, is_cv_confidence=label.isCvConfidence)

    grouping = self._group(entries)
    occupations = _to_candidate_occupations(entries, grouping)
    return CvExtractionResponse(
      is_cv=True, is_cv_confidence=label.isCvConfidence, candidate_occupations=occupations,
    )

  def _label(self, candidate: dict) -> LabelResult:
    return self._parse(
      system=LABEL_SYSTEM_PROMPT,
      user_content="Stage 1 candidate proposal (elements + entries), as JSON:\n\n"
                   + json.dumps(candidate, indent=2),
      output_format=LabelResult,
    )

  def _group(self, entries: list[dict]) -> OccupationGroupResult:
    return self._parse(
      system=OCCUPATION_SYSTEM_PROMPT,
      user_content="Validated job experiences:\n\n" + json.dumps(
        [{"index": i, **e} for i, e in enumerate(entries)], indent=2,
      ),
      output_format=OccupationGroupResult,
    )

  def _parse(self, system: str, user_content: str, output_format):
    try:
      response = self._client.messages.parse(
        model=self._model_name,
        max_tokens=16000,
        system=system,
        messages=[{"role": "user", "content": user_content}],
        output_format=output_format,
      )
    except anthropic.APIError as exception:
      logger.error("Anthropic API error during CV extraction: %s", exception)
      raise LlmServiceUnavailableError("The LLM service is unavailable") from exception

    if response.stop_reason == "refusal":
      raise MalformedLlmResponseError("The LLM declined to process this document")

    try:
      parsed = response.parsed_output
      if parsed is None:
        raise MalformedLlmResponseError("The LLM returned no structured output")
      return parsed
    except pydantic.ValidationError as exception:
      logger.error("LLM structured output failed schema validation: %s", exception)
      raise MalformedLlmResponseError(
        "The LLM's response did not match the expected structure"
      ) from exception


def _to_candidate_occupations(
    entries: list[dict],
    grouping: OccupationGroupResult,
) -> list[CandidateOccupation]:
  occupations = []
  for g in grouping.groups:
    job_experiences = [
      _to_candidate_job_experience(entries[i]) for i in g.entryIndices
    ]
    occupations.append(CandidateOccupation(
      occupation=Occupation(name=g.occupationName, status="ACTIVE", isco08_code=g.isco08Code),
      years_experience=g.yearsExperience,
      candidate_job_experiences=job_experiences,
    ))
  return occupations


def _to_candidate_job_experience(entry: dict) -> CandidateJobExperience:
  country = entry.get("country")
  return CandidateJobExperience(
    country=Country(**{
      "iso_code": country["isoCode"], "name": country["name"], "status": country["status"],
    }) if country else None,
    company_name=entry.get("companyName"),
    role=entry.get("role") or None,
    start_date=entry.get("startDate"),
    end_date=entry.get("endDate"),
    description=entry.get("description") or None,
  )
