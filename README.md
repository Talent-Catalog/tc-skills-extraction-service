# tc-skills-extraction-service
FastAPI Python service for extracting skills from text

## 🐳 Local Docker build & run

To build and run the Skills Extraction Service locally:

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

The explanation API uses an OpenAI-compatible Chat Completions endpoint. A
local inference server such as vLLM can be configured with:

```dotenv
LLM_BASE_URL=http://localhost:8001/v1
LLM_MODEL_NAME=mlx-community/Qwen3-8B-4bit
LLM_API_KEY=
LLM_REQUEST_TIMEOUT_SECONDS=120
```

Amazon Bedrock's OpenAI-compatible endpoint can be selected using only
environment configuration:

```dotenv
LLM_BASE_URL=https://bedrock-mantle.<aws-region>.api.aws/v1
LLM_MODEL_NAME=<configured-bedrock-model-id>
LLM_API_KEY=<bedrock-api-key>
LLM_REQUEST_TIMEOUT_SECONDS=120
```

`LLM_API_KEY` is optional. When configured, it is sent as a bearer token and
must be supplied through a secret-management mechanism rather than committed
to source control.
