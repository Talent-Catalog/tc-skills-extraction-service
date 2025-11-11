from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import AnyHttpUrl

class Settings(BaseSettings):
  SKILLS_BASE_URL: AnyHttpUrl

  model_config = SettingsConfigDict(env_file=str(Path(__file__).resolve().parent.parent / ".env"))

settings = Settings()
