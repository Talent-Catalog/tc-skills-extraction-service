from __future__ import annotations

from app.services.cv_extraction.schema_docs import TARGET_SCHEMAS, all_target_schema_docs, render_schema_summary


def test_all_target_schemas_load_and_render() -> None:
  """
  Guards against the vendored YAML in app/services/cv_extraction/schemas/
  drifting out of sync with TARGET_SCHEMAS (e.g. a file renamed/removed
  without updating the list, or vice versa) - this would otherwise only
  surface as a FileNotFoundError deep inside a live Stage 2 API call.
  """
  for name in TARGET_SCHEMAS:
    summary = render_schema_summary(name)
    assert name in summary
    assert len(summary) > 0


def test_combined_docs_mention_every_target_schema() -> None:
  combined = all_target_schema_docs()
  for name in TARGET_SCHEMAS:
    assert f"### {name}" in combined
