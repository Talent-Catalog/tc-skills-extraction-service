from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import AnyHttpUrl, Field

from app.services.llm_client import LlmAuthentication

# Settings for the application. These are loaded from environment variables or
# a .env file.
class Settings(BaseSettings):
  SKILLS_BASE_URL: AnyHttpUrl

  # These default values can be overridden by env variables: LLM_BASE_URL etc.
  # However, these default values will work for local dev purposes if you
  # generate a Short-Term API key on AWS Amazon Bedrock:
  # see https://eu-west-2.console.aws.amazon.com/bedrock/home?region=eu-west-2#/api-keys?tab=short-term
  # Specify that API key in the LLM_API_KEY environment variable.
  llm_base_url: str = "https://bedrock-runtime.eu-west-2.amazonaws.com/openai/v1"
  llm_model_name: str = "qwen.qwen3-235b-a22b-2507-v1:0"
  llm_authentication: LlmAuthentication = LlmAuthentication.BEARER

  # Set this to an Amazon Bedrock short-term API key - see comments above
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
