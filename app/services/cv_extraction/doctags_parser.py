"""
Deterministic Docling Doctags -> candidate proposal parser (Stage 1 of the
CV-extraction pipeline).

Ported from tc-api-spec/tools/doctags2schema/parse_doctags.py, where the
full pipeline was designed and tested against a real CV before being moved
here. See app/services/cv_extraction/README.md for the pipeline overview.

Scope / honesty note:
This handles the *layout reconstruction* that is genuinely mechanical:
tag/bbox parsing, row clustering by y-coordinate, date-range regex
extraction, company/role/location string splitting, country -> ISO code
lookup, and bullet/paragraph concatenation. It deliberately does NOT decide
whether the document is a CV, which entries are job experience vs.
education/certifications/etc., or how job experiences group into an
Occupation - all of that is Stage 2/3/4 in llm_pipeline.py, against the
real target schemas in schema_docs.py.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# --------------------------------------------------------------------------
# 1. Tag / bbox tokenizer - fully deterministic
# --------------------------------------------------------------------------

@dataclass
class Element:
  id: int
  kind: str  # 'section_header', 'text', 'bullet', 'picture'
  page: int
  box: tuple  # (x1, y1, x2, y2)
  text: str = ""
  list_id: Optional[int] = None  # for kind == 'bullet': id of the owning <unordered_list>


_LOC_RE = re.compile(r"<loc_(\d+)>")

_LEAF_TAG_RE = re.compile(
  r"<(?P<tag>picture|section_header_level_1|text)>(?P<body>.*?)</(?P=tag)>",
  re.S,
)
_LIST_ITEM_RE = re.compile(r"<list_item>(?P<body>.*?)</list_item>", re.S)

# Any top-level construct, in document order: picture / header / text / list / page_break
_TOP_RE = re.compile(
  r"(?P<picture><picture>.*?</picture>)"
  r"|(?P<header><section_header_level_1>.*?</section_header_level_1>)"
  r"|(?P<text><text>.*?</text>)"
  r"|(?P<list><unordered_list>.*?</unordered_list>)"
  r"|(?P<pagebreak><page_break>)",
  re.S,
)

_TAG_KIND = {
  "picture": "picture",
  "section_header_level_1": "section_header",
  "text": "text",
}


def _split_leading_boxes(body: str) -> tuple[list[tuple[int, int, int, int]], str]:
  """A leaf body is N groups of 4 <loc_NNN> tags followed by plain text."""
  nums = [int(n) for n in _LOC_RE.findall(body)]
  text = _LOC_RE.sub("", body).strip()
  boxes = [tuple(nums[i:i + 4]) for i in range(0, len(nums) - len(nums) % 4, 4)]
  return boxes, text


def parse_doctags(raw: str) -> list[Element]:
  elements: list[Element] = []
  next_id = 0
  page = 1

  def alloc_id() -> int:
    nonlocal next_id
    eid = next_id
    next_id += 1
    return eid

  for m in _TOP_RE.finditer(raw):
    if m.group("pagebreak"):
      page += 1
      continue
    if m.group("picture"):
      body = re.search(r"<picture>(.*?)</picture>", m.group(0), re.S).group(1)
      boxes, _ = _split_leading_boxes(body)
      elements.append(Element(alloc_id(), "picture", page, boxes[0] if boxes else (0, 0, 0, 0)))
      continue
    if m.group("list"):
      list_id = alloc_id()
      for li in _LIST_ITEM_RE.finditer(m.group("list")):
        boxes, text = _split_leading_boxes(li.group("body"))
        elements.append(Element(
          alloc_id(), "bullet", page, boxes[0] if boxes else (0, 0, 0, 0), text, list_id=list_id,
        ))
      continue
    # header or text
    leaf = _LEAF_TAG_RE.match(m.group(0))
    tag = leaf.group("tag")
    boxes, text = _split_leading_boxes(leaf.group("body"))
    elements.append(Element(alloc_id(), _TAG_KIND[tag], page, boxes[0] if boxes else (0, 0, 0, 0), text))
  return elements


# --------------------------------------------------------------------------
# 2. NO section-title vocabulary here, deliberately.
#
# Stage 1 does not try to identify which sections are "work experience".
# That assumes every CV titles its sections the same way, which doesn't
# hold - and the correct authority for "what counts as work experience" is
# the target OpenAPI schema (CandidateJobExperience.yaml and its siblings -
# see schema_docs.py), not a wordlist of English section headings.
#
# Step 3 below proposes a candidate entry for EVERY header-shaped row + its
# bullets/paragraphs, across the WHOLE document - job experience, training,
# education, certifications, whatever. It is Stage 2 (llm_pipeline.py),
# reading the real schema definitions, that decides which proposed entries
# actually match CandidateJobExperience and which match something else.
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# 3. Row clustering + field extraction, across the whole document
# --------------------------------------------------------------------------

MONTHS = {
  m: i + 1
  for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
  )
}
_DATE_RE = re.compile(
  r"(?P<sm>[A-Za-z]{3,9})\.?\s+(?P<sy>\d{4})\s*[-–—]\s*"
  r"(?:(?P<em>[A-Za-z]{3,9})\.?\s+(?P<ey>\d{4})|(?P<present>Present))",
)

COUNTRY_ISO = {
  "LEBANON": "LB",
  "NIGERIA": "NG",
  # extend as needed / replace with a full ISO-3166 gazetteer - see the
  # README's "Where an LLM is actually necessary" notes ported alongside
  # this file for why this is a known, deliberate limitation.
}

# A regex alone cannot tell "Port Harcourt - Nigeria" from "...Engineer'
# Port Harcourt - Nigeria" without knowing "Port Harcourt" is one city
# name, not a role word plus a city. This gazetteer is a placeholder,
# standing in for a real one (or an LLM) - see the pipeline README.
KNOWN_CITIES = ["Beirut", "Saida", "Port Harcourt", "Ondo", "Lagos"]

_LOCATION_RE = re.compile(
  r"(?P<prefix>.*?)\b(?P<city>" + "|".join(re.escape(c) for c in KNOWN_CITIES) + r")"
  r"\s*-\s*(?P<country>" + "|".join(re.escape(c.title()) for c in COUNTRY_ISO) + r")\b\s*$"
)


def _parse_date(month_str: str, year_str: str) -> str:
  mon = MONTHS[month_str.strip()[:3].lower()]
  return f"{int(year_str):04d}-{mon:02d}-01"


def _extract_date(text: str) -> tuple[Optional[dict], str]:
  """Returns (date_fields, remaining_text_with_date_removed)."""
  m = _DATE_RE.search(text)
  if not m:
    return None, text
  dates = {"startDate": _parse_date(m.group("sm"), m.group("sy"))}
  if not m.group("present"):
    dates["endDate"] = _parse_date(m.group("em"), m.group("ey"))
  remaining = (text[: m.start()] + " " + text[m.end():]).strip()
  return dates, remaining


def _extract_location(text: str) -> tuple[Optional[dict], str]:
  m = _LOCATION_RE.search(text)
  if not m:
    return None, text
  country_name = m.group("country").upper()
  loc = {"isoCode": COUNTRY_ISO[country_name], "name": country_name.title(), "status": "ACTIVE"}
  remaining = m.group("prefix").strip()
  return loc, remaining


def _split_company_role(text: str) -> tuple[Optional[str], str, str]:
  """Recognizes the `Company 'Role'` convention this pipeline was built and
  tested against. Falls back to treating the whole string as the role with
  company=None, flagged - a different CV's own convention (or none at all)
  needs Stage 2 (an LLM) to resolve, not a smarter regex."""
  m = re.search(r"^(?P<company>.*?)\s*['‘’](?P<role>.*?)['‘’]\s*$", text)
  if m and m.group("company").strip():
    return m.group("company").strip(" .'"), m.group("role").strip(), "quote-pattern"
  return None, text.strip(), "no-pattern:left-as-role"


def _row_key(el: Element, tol: int = 2) -> tuple:
  return (el.page, round(el.box[1] / (tol * 2)))


def extract_fields_from_chunks(chunks: list[Element]) -> dict:
  """Given the Elements making up one header row, run the same
  deterministic extractors used at parse time. Shared by Stage 1 and by
  Stage 3 (llm_pipeline.py) so corrected groupings are re-derived the same
  way, never trusted verbatim from the LLM."""
  texts = [c.text for c in sorted(chunks, key=lambda c: c.box[0])]
  combined = " ".join(texts)
  fields: dict = {"_flags": []}

  dates, combined = _extract_date(combined)
  location, combined = _extract_location(combined)
  company, role, split_confidence = _split_company_role(combined)

  fields["companyName"] = company
  fields["role"] = role
  if split_confidence != "quote-pattern":
    fields["_flags"].append(f"company_role_split:{split_confidence}")
  if dates:
    fields.update(dates)
  else:
    fields["_flags"].append("no_date_found")
  if location:
    fields["country"] = location
  else:
    fields["_flags"].append("no_location_found")
  return fields


def propose_entries(elements: list[Element]) -> list[dict]:
  """elements is the WHOLE document's content (all sections, all pages),
  already in document order (NOT necessarily reading order - bbox y is
  authoritative). Proposes one candidate entry per header-shaped row + its
  bullets, with NO judgment about whether that row is a job, a training
  course, an education entry, or a section title with nothing under it -
  that classification is Stage 2's job (see schema_docs.py).

  Known limitation: only literal <list_item> bullets are attached to an
  entry's description automatically. A job described in plain paragraphs
  (no bullets) comes out as several SEPARATE proposed entries instead - one
  per paragraph. Stage 2/3 can reassemble this (ReassignElement carries a
  `role` of "header" or "description" for exactly this reason - see
  pipeline_schemas.py) but Stage 1 itself does not attempt to guess it,
  since misjudging "is this a continuation or a genuinely new entry" risks
  silently merging two distinct records, which is a worse failure than
  over-segmenting.

  Returns entries carrying `elementIds`: {"header": [...], "bullets": [...]}
  so Stage 3 can apply LLM corrections by id rather than by re-parsing text.
  """
  row_candidates = [e for e in elements if e.kind in ("section_header", "text")]
  bullet_elements = [e for e in elements if e.kind == "bullet"]

  rows_by_key: dict[tuple, list[Element]] = {}
  for e in row_candidates:
    rows_by_key.setdefault(_row_key(e), []).append(e)
  rows = [
    {"page": k[0], "y": min(c.box[1] for c in chunks), "chunks": sorted(chunks, key=lambda c: c.box[0])}
    for k, chunks in rows_by_key.items()
  ]

  # IMPORTANT: bullets are placed individually by their OWN y, not by the y
  # of the first item in their <unordered_list>. Docling sometimes emits
  # two visually-distinct bullet lists (belonging to two different jobs) as
  # a single <unordered_list> element, so anchoring placement to the list's
  # first item mis-attributes every later bullet to the wrong job.
  bullets = [("bullet", el.page, el.box[1], el) for el in bullet_elements]

  blocks = [("row", r["page"], r["y"], r) for r in rows] + bullets
  blocks.sort(key=lambda b: (b[1], b[2]))

  entries: list[dict] = []
  current: Optional[dict] = None

  for kind, page, y, payload in blocks:
    if kind == "bullet":
      el = payload
      if current is None:
        continue  # bullets with no preceding header - shouldn't happen; drop rather than guess
      current["_bullet_elements"].append(el)
      continue

    row = payload
    texts = [c.text for c in row["chunks"]]
    is_lone_lowercase_fragment = (
      len(texts) == 1
      and current is not None
      and current["_bullet_elements"]
      and texts[0][:1].islower()
    )
    if is_lone_lowercase_fragment:
      # Heuristic only: a short lowercase-leading fragment right after a
      # page break, with no sibling row chunk, is treated as the tail of
      # the previous bullet rather than a new entry header.
      last = current["_bullet_elements"][-1]
      last.text = last.text.rstrip() + " " + texts[0]
      # The fragment's own element id is deliberately NOT added to
      # elementIds["header"] - its content is already absorbed into
      # `last`'s text, and `last`'s id is already tracked via
      # elementIds["bullets"]. Adding it to "header" too would let Stage 3
      # re-select it as a separate header/date/location-contributing chunk
      # and duplicate it into the role/company text.
      current.setdefault("_mergedFragmentIds", []).extend(c.id for c in row["chunks"])
      current["_flags"].append("merged_fragment_across_page_break:heuristic")
      continue

    # New entry-header row.
    if current is not None:
      entries.append(current)
    current = {
      "_bullet_elements": [],
      "elementIds": {"header": [c.id for c in row["chunks"]], "bullets": []},
      "_flags": [],
    }
    current.update(extract_fields_from_chunks(row["chunks"]))

  if current is not None:
    entries.append(current)

  for e in entries:
    e["elementIds"]["bullets"] = [b.id for b in e["_bullet_elements"]]
    e["elementIds"]["mergedFragments"] = e.pop("_mergedFragmentIds", [])
    e["description"] = "; ".join(b.text.strip() for b in e["_bullet_elements"] if b.text.strip())
    e.pop("_bullet_elements", None)
  return entries


# --------------------------------------------------------------------------
# 4. Candidate-proposal builder - the artifact llm_pipeline.py consumes
# --------------------------------------------------------------------------

def build_candidate(raw_doctags: str) -> dict:
  """The full Stage 1 output: every parsed element (for id-based joins)
  plus a document-wide entry proposal - no section filtering, no title
  vocabulary. Many proposed entries will NOT be job experience (some will
  be education, certifications, a lone section title with no body, a
  free-flowing summary paragraph, ...) - Stage 2 sorts that out against the
  real target schemas, not against anything decided here."""
  elements = parse_doctags(raw_doctags)
  content_elements = [el for el in elements if el.kind != "picture"]
  entries = propose_entries(content_elements)

  return {
    "elements": [
      {"id": el.id, "kind": el.kind, "page": el.page, "box": list(el.box), "text": el.text}
      for el in content_elements
    ],
    "entries": entries,
  }
