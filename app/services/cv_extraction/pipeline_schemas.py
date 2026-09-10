"""
Pydantic schemas for the INTERNAL Stage 2/3/4 pipeline plumbing (LLM
correction ops, entry classification, occupation grouping). These are
distinct from app/models/candidate_occupation_models.py, which are the
PUBLIC response models this service actually returns - these never appear
in an HTTP response body.

Kept in their own module, imported by both llm_pipeline.py (the real
pipeline) and any test doubles, so nothing needs an import cycle.
"""
from __future__ import annotations

from typing import Optional, Literal

from pydantic import BaseModel, Field

from app.services.cv_extraction.schema_docs import TARGET_SCHEMAS

# EntryClassification.targetSchema's allowed values ARE the real schema
# names from schema_docs.py (plus two sentinels), not an invented
# vocabulary - this is what makes classification map onto the actual API
# contract instead of a category set someone made up. Built dynamically so
# adding/removing a schema in TARGET_SCHEMAS doesn't require touching this
# Literal by hand.
TargetSchemaName = Literal[*TARGET_SCHEMAS, "none", "unrecognized"]

# What role an element plays WITHIN an entry. Stage 1 can only ever propose
# "header" (the row: company/role/date/location text) and "description"
# (literal <list_item> bullets) - it has no way to notice a job described
# by plain paragraphs rather than bullets, so those paragraphs come out as
# their own separate proposed entries. "role" is what lets Stage 2 correct
# that: reassign a paragraph-text element into the job entry it actually
# belongs to, explicitly as "description" rather than "header".
ElementRole = Literal["header", "description"]


# --------------------------------------------------------------------------
# Stage 2 - what the LLM is allowed to say. No free-text content fields:
# every op references an existing element id or entry index.
# --------------------------------------------------------------------------

class DiscardEntry(BaseModel):
  op: Literal["discard_entry"]
  entryIndex: int
  reason: str


class DiscardElement(BaseModel):
  op: Literal["discard_element"]
  entryIndex: int
  elementId: int
  reason: str


class ReassignElement(BaseModel):
  op: Literal["reassign_element"]
  elementId: int
  fromEntryIndex: Optional[int] = Field(
    None, description="null if the element wasn't in any Stage 1 entry"
  )
  toEntryIndex: int
  role: ElementRole = Field(
    description="'header' if this element's text is company/role/date/location "
                "content to parse; 'description' if it's prose/bullet content to "
                "append verbatim to the description field - e.g. a paragraph "
                "describing the job that wasn't bulleted, so Stage 1 proposed it "
                "as a separate entry instead of attaching it to this one."
  )
  reason: str


class NewEntry(BaseModel):
  op: Literal["new_entry"]
  headerElementIds: list[int] = Field(description="existing element ids only - never invent text")
  descriptionElementIds: list[int] = Field(
    default_factory=list,
    description="existing element ids for description content - bullets AND/OR "
                "unbulleted paragraph text, in either case appended verbatim, "
                "never retyped or summarized",
  )
  reason: str


Correction = DiscardEntry | DiscardElement | ReassignElement | NewEntry


class EntryClassification(BaseModel):
  entryIndex: int
  targetSchema: TargetSchemaName = Field(
    description="Which real API schema this proposed entry's content matches - "
                "'none' if it's genuine CV content but not one of these record "
                "types (e.g. a summary paragraph, references), 'unrecognized' if "
                "you can't tell."
  )
  confidence: float = Field(ge=0, le=1)
  reason: str


class LabelResult(BaseModel):
  isCv: bool
  isCvConfidence: float = Field(ge=0, le=1)
  isCvReason: str
  classifications: list[EntryClassification]
  corrections: list[Correction] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Stage 4 - free-form occupation clustering, over already-locked entries only
# --------------------------------------------------------------------------

class OccupationGroup(BaseModel):
  entryIndices: list[int]
  occupationName: str
  isco08Code: Optional[str] = None
  yearsExperience: Optional[int] = None
  yearsExperienceReason: str = Field(description="why this number, or why omitted")


class OccupationGroupResult(BaseModel):
  groups: list[OccupationGroup]
