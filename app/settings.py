from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import AnyHttpUrl, Field

class Settings(BaseSettings):
  SKILLS_BASE_URL: AnyHttpUrl

  llm_base_url: str = "http://localhost:8001/v1"
  llm_model_name: str = "Qwen/Qwen3-8B"
  llm_request_timeout_seconds: float = Field(default=60.0, gt=0)

  model_config = SettingsConfigDict(env_file=str(Path(__file__).resolve().parent.parent / ".env"))

# noinspection PyArgumentList
settings = Settings()
