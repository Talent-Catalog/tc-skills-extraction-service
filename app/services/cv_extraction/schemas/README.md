# Vendored schemas

These files are a **copy** of the relevant schemas from
[`tc-api-spec`](https://github.com/Talent-Catalog/tc-api-spec)
(`components/schemas/v1/` and `components/schemas/common/enums/`), copied on
2026-09-01. There is no automated link back to that repo — if those schemas
change, someone needs to re-copy the affected files here by hand.

`schema_docs.py` reads `v1/CandidateJobExperience.yaml` and five siblings
(`CandidateEducation`, `CandidateCertification`, `CandidateLanguage`,
`CandidateSkill`, `CandidateExam`) at runtime to build the classification
prompt for Stage 2 of the extraction pipeline — see
`app/services/cv_extraction/llm_pipeline.py`. The remaining files
(`CandidateOccupation`, `Occupation`, `Country`, `EducationMajor`,
`Language`, `LanguageLevel`, and the enum files) aren't read by any code —
they're vendored for reference, so anyone maintaining
`app/models/candidate_occupation_models.py` (the hand-authored pydantic
mirror of these schemas) can diff against the real contract without
switching repos.

**When `tc-api-spec` changes these schemas:** re-copy the affected file(s)
here, then check whether `app/models/candidate_occupation_models.py` and
`schema_docs.py`'s `TARGET_SCHEMAS` list still match.
