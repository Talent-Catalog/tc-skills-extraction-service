from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import AnyHttpUrl, Field

from app.services.llm_client import LlmAuthentication

# Settings for the application. These are loaded from environment variables or
# a .env file.
class Settings(BaseSettings):
  SKILLS_BASE_URL: AnyHttpUrl

  # llm_base_url=None means "use the default Amazon Bedrock OpenAI-compatible
  # endpoint for llm_aws_region" (constructed in main.py). Override it to
  # point at another OpenAI-compatible endpoint instead, such as a locally
  # hosted vLLM server.
  llm_base_url: str | None = None
  llm_aws_region: str = "eu-west-2"
  llm_model_name: str = "qwen.qwen3-235b-a22b-2507-v1:0"

  # AWS_SIGV4 is the default: an appropriately authorised developer's normal
  # AWS credentials (or, in ECS, the task role) are enough to run this
  # service with no LLM-specific environment variables. BEARER (a short-term
  # Bedrock API key in llm_api_key) and NONE (no authentication, e.g. a
  # local vLLM server) remain available - see LlmAuthentication.
  llm_authentication: LlmAuthentication = LlmAuthentication.AWS_SIGV4

  # Only needed when llm_authentication is BEARER.
  llm_api_key: str | None = None
  llm_request_timeout_seconds: float = Field(default=60.0, gt=0)

  # anthropic_api_key is optional here purely for parity with the settings
  # above - if unset, anthropic.Anthropic() falls back to the standard
  # ANTHROPIC_API_KEY environment variable / `ant auth login` profile on
  # its own, same as passing api_key=None explicitly would.
  anthropic_api_key: str | None = None
  cv_extraction_model_name: str = "claude-opus-5"

  model_config = SettingsConfigDict(env_file=str(Path(__file__).resolve().parent.parent / ".env"))

# noinspection PyArgumentList
settings = Settings()
