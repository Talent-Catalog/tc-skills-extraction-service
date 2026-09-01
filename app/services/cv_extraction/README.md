# CV extraction pipeline

Ported from `tc-api-spec/tools/doctags2schema`, where this pipeline was
designed and tested against a real CV before being moved here for
`POST /extract_candidate_occupations`. That repo's git history has the
full design conversation; this file is the condensed version.

## The four-stage pipeline

```
PDF ──▶ [pdf_to_doctags.py, Docling] ──▶ Doctags markup
                                              │
                                              ▼
        [1: doctags_parser.py] ──▶ candidate proposal: EVERY header-shaped
                                    row + its bullets, whole document, no
                                    section-title vocabulary (elements +
                                    entries, all id-tagged)
                                              │
                                              ▼
        [2: LLM] isCv? + per-entry targetSchema classification against the
                 REAL schemas (schema_docs.py reads the vendored
                 CandidateJobExperience.yaml + siblings - the schema
                 defines the categories, not a wordlist) + a CLOSED set of
                 correction ops, every op referencing existing ids - never
                 free text
                                              │
                                              ▼
        [3: apply_corrections(), deterministic] validates every id/index
        the LLM referenced actually exists, applies the corrections, keeps
        only targetSchema == "CandidateJobExperience" entries, then
        RE-RUNS Stage 1's own extractors on the corrected element groups.
        The LLM's classification is trusted; its field guesses (if any)
        are discarded.
                                              │
                                              ▼
             [4: LLM] free-form clustering of the now-locked, fully
             traceable entries into Occupation groups
                                              │
                                              ▼
                       final CvExtractionResponse
```

**Why two different trust levels for stage 2 vs. stage 4:** stage 2 touches
raw doctags content, so it's constrained to editing Stage 1's proposal by
id - this is what makes every output field traceable back to a specific
bbox, and stops the LLM from quietly retyping/paraphrasing text. Stage 4
only sees already-validated, already-traceable entries - there's no raw
content left to mangle, so free-form clustering (deciding occupation
names/counts/ISCO codes) is safe there.

## Known, deliberate limitations (not silently swept under the rug)

- **Company/role splitting** only recognizes the `Company 'Role'`
  convention this pipeline was tested against (`doctags_parser.py`,
  `_split_company_role`). A CV with no such delimiter gets its whole header
  text left in `role` with `companyName: null` - flagged internally, but
  nothing currently routes that flag back to Stage 2 for a second look.
- **Location parsing** uses a small hardcoded city gazetteer
  (`KNOWN_CITIES` in `doctags_parser.py`), not a real one. Any city not in
  that list won't resolve to a `country`.
- **`fullTime`/`paid`** on `CandidateJobExperience` are never populated -
  there's no reliable textual signal to derive them from, for a model or a
  regex, so guessing was deliberately avoided rather than attempted.
- **Section-header vocabulary** is NOT hardcoded (this was the original
  version's biggest flaw, fixed before porting here) - Stage 1 proposes
  entries for the whole document regardless of what any section is titled,
  and Stage 2 classifies each one against the real target schemas.
- **Unbulleted paragraph descriptions** are handled, but only via Stage 2
  noticing the pattern and reassigning elements with `role="description"`
  - Stage 1 itself does not try to auto-detect "this paragraph continues
    the job above it", since misjudging that risks silently merging two
    distinct jobs into one, a worse failure than over-segmenting. See
    `pipeline_schemas.py`'s `ElementRole` docstring and
    `tests/test_cv_extraction_service.py` for the mechanism and a worked
    example.

## Differences from the tc-api-spec original

- No `--offline` CLI stub. Tests use `app.dependency_overrides` +
  hand-written fakes (`FakeCvExtractionService` in
  `tests/test_cv_extraction_api.py`), matching this repo's existing test
  convention, instead of a bespoke offline-mode flag.
- The real Anthropic SDK (`anthropic.Anthropic`, via `client.messages.parse`)
  is called directly here - the original CLI tool's version had the same
  code, just without a service class wrapping it.
- Target schemas are a vendored copy (`schemas/`), not a live read out of a
  sibling `tc-api-spec` checkout - see `schemas/README.md`.
