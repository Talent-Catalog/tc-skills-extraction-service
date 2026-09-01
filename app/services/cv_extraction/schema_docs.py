"""
Reads the vendored OpenAPI schema files to build Stage 2's classification
target list - instead of a hardcoded, invented category vocabulary.

The API spec already defines the categories a CV block can map to: it's
the set of Candidate*.yaml schemas that describe one CV-shaped record -
CandidateJobExperience, CandidateEducation, CandidateCertification,
CandidateLanguage, CandidateSkill, CandidateExam. Rendering their real
`description` and `properties` text into the Stage 2 prompt means the
LLM's classification criteria ARE the schema contract, not a paraphrase of
it that can drift out of sync when the schema changes.

These files are a vendored COPY of tc-api-spec's schemas (see
app/services/cv_extraction/schemas/README.md) - there is no live link back
to that repo the way tc-api-spec/tools/doctags2schema's original version of
this file had (it read the schemas directly out of the same repo). If those
schemas change, someone has to re-copy the affected files here by hand.

(CandidateOccupation itself isn't a target here - an individual doctags
block maps to one CandidateJobExperience; grouping job experiences into an
Occupation is Stage 4's job, over already-classified entries.)
"""
from __future__ import annotations

from pathlib import Path

import yaml

SCHEMAS_DIR = Path(__file__).resolve().parent / "schemas" / "v1"

# The subset of Candidate*.yaml schemas that describe a single CV-block-shaped
# record (as opposed to e.g. CandidateVisaCheck, CandidateRegistration,
# CandidateCoupon, which aren't things a résumé section maps to).
TARGET_SCHEMAS = [
  "CandidateJobExperience",
  "CandidateEducation",
  "CandidateCertification",
  "CandidateLanguage",
  "CandidateSkill",
  "CandidateExam",
]


def _ref_name(node) -> str | None:
  if isinstance(node, dict):
    if "$ref" in node:
      return Path(node["$ref"]).stem
    for sub in node.get("allOf", []):
      name = _ref_name(sub)
      if name:
        return name
  return None


def render_schema_summary(name: str) -> str:
  path = SCHEMAS_DIR / f"{name}.yaml"
  spec = yaml.safe_load(path.read_text())
  lines = [f"### {name}", spec.get("description", "").strip()]
  for prop_name, prop_spec in (spec.get("properties") or {}).items():
    desc = (prop_spec or {}).get("description", "").strip()
    ref = _ref_name(prop_spec)
    suffix = f" (see {ref}.yaml)" if ref else ""
    lines.append(f"- `{prop_name}`{suffix}: {desc}")
  required = spec.get("required")
  if required:
    lines.append(f"Required: {', '.join(required)}")
  return "\n".join(lines)


def all_target_schema_docs() -> str:
  return "\n\n".join(render_schema_summary(name) for name in TARGET_SCHEMAS)
