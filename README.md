# tc-skills-extraction-service
FastAPI Python service for extracting skills from text, doing vector embeddings, providing
LLM based explanations for job/candidate matchings.

### `todo` - Rename this service  

## 🐳 Local Docker build & run

To build and run the service locally:

### Build the image
```bash
docker build -t tc-skills:local .
```

### Run the container (connecting to local Talent Catalog API)
```bash
docker run --rm -p 8000:8000 \
  -e SKILLS_BASE_URL=http://host.docker.internal:8080/api/public/skill/names \
  tc-skills:local
```
## Running in IntelliJ

See comments in `main.py` for instructions on how to run the service in IntelliJ or manually from
a terminal.

## LLM configuration

The explanation API uses an OpenAI-compatible Chat Completions endpoint,
talking to Amazon Bedrock's OpenAI-compatible endpoint by default. `LlmClient`
itself (`app/services/llm_client.py`) is provider-independent and supports
three authentication modes (`LlmAuthentication`): `AWS_SIGV4` (the default),
`BEARER`, and `NONE`.

### Local development: no configuration needed

If you have an AWS user or role authorised to invoke the configured Bedrock
model, **you don't need to set any `LLM_*` environment variables at all.**
Leave every setting at its default and the service will:

- authenticate with AWS Signature Version 4 (`LLM_AUTHENTICATION` defaults to
  `AWS_SIGV4`), using your normal AWS credentials from the standard AWS
  credential provider chain (environment variables, `~/.aws/credentials`,
  SSO login, etc. - whatever `aws sts get-caller-identity` already resolves
  for you);
- call Amazon Bedrock's OpenAI-compatible endpoint in `eu-west-2`
  (`LLM_AWS_REGION` defaults to `eu-west-2`, and `LLM_BASE_URL` defaults to
  `None`, which `main.py` turns into
  `https://bedrock-runtime.eu-west-2.amazonaws.com/openai/v1`);
  and use the default `LLM_MODEL_NAME`.

All you need to do is make sure your AWS user/role has been granted access
to that model in the Bedrock console (Bedrock model access is per-region and
per-model, separate from general AWS permissions) and has an IAM policy
allowing `bedrock:InvokeModel`/`bedrock:InvokeModelWithResponseStream` (or
equivalent) for it. No API key, and no local `.env` entries, are required.

If your AWS credentials are for a region other than `eu-west-2`, or your
Bedrock model access was granted in a different region, override just the
region and both the signing region and the default endpoint URL follow it:

```dotenv
LLM_AWS_REGION=us-east-1
```

### Alternative: BEARER (a short-term Bedrock API key)

Useful when you don't have (or don't want to use) local AWS credentials, or
for diagnosing whether a problem is SigV4-specific:

```dotenv
LLM_AUTHENTICATION=BEARER
LLM_API_KEY=<short-term Bedrock API key>
```

Generate a short-term key from the Bedrock console, e.g.
https://eu-west-2.console.aws.amazon.com/bedrock/home?region=eu-west-2#/api-keys?tab=short-term.
`LLM_API_KEY` must be supplied through a secret-management mechanism (or a
local, untracked `.env`) rather than committed to source control.

### Alternative: NONE (an unauthenticated OpenAI-compatible server)

For a locally hosted inference server such as vLLM that requires no
authentication:

```dotenv
LLM_BASE_URL=http://localhost:8001/v1
LLM_AUTHENTICATION=NONE
LLM_MODEL_NAME=mlx-community/Qwen3-8B-4bit
LLM_REQUEST_TIMEOUT_SECONDS=120
```

`LLM_AWS_REGION` is irrelevant here (AWS signing isn't used) and can be left
at its default.

### Production (ECS/Fargate)

Production uses the same `AWS_SIGV4` path, with credentials coming from the
ECS task role via the same standard AWS credential provider chain - no
`LLM_API_KEY` is configured in production. `LLM_AWS_REGION` only needs
setting there if the deployment region differs from the `eu-west-2` default.

## CV extraction (POST /extract_candidate_occupations)

Accepts an uploaded CV PDF and returns `CandidateOccupation`/
`CandidateJobExperience`-shaped JSON (matching the schemas in
[tc-api-spec](https://github.com/Talent-Catalog/tc-api-spec)), or
`{"isCv": false, ...}` if the upload doesn't look like a CV. See
`app/services/cv_extraction/README.md` for the pipeline design (it was
built and tested in `tc-api-spec/tools/doctags2schema` before being ported
here) and `app/services/cv_extraction/schemas/README.md` for how the
target schemas get into this repo.

Uses the Anthropic API (Claude) directly, unlike the OpenAI-compatible
`LLM_*` configuration above:

```dotenv
ANTHROPIC_API_KEY=<your key>
CV_EXTRACTION_MODEL_NAME=claude-opus-5
```

`ANTHROPIC_API_KEY` is optional in `Settings` only in the sense that the
Anthropic SDK falls back to resolving it from the environment itself if
unset here - one of the two still needs to be set for this endpoint to work.

**Dependencies:** `docling` (added for PDF→doctags conversion) pulls its own
`torch` dependency by default - a full CUDA build several GB heavier than
the `+cpu` build pinned in `requirements.txt` for `sentence-transformers`.
This is resolved for the deployed (Linux/Fargate) image - see the comment
above the `docling` line in `requirements.txt` for how it was verified,
given this project's prior OOM-on-Fargate incident from an unpinned torch
change.

For local development on macOS, use `requirements-dev.txt` instead of
`requirements.txt` - the `+cpu` wheel pin only exists for Linux/Windows, so
`requirements.txt` fails to resolve on a Mac. See that file's header for
details.
