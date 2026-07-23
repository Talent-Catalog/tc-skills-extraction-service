from fastapi import Request
from dataclasses import dataclass

from app.services.embedding_service import EmbeddingService
from app.services.skills_extractor import SkillsExtractor

@dataclass(frozen=True)
class ApplicationServices:
  """Contains singleton services created during application startup."""
  embedding_service: EmbeddingService
  skills_extractor: SkillsExtractor


def get_embedding_service(request: Request) -> EmbeddingService:
  """
  Returns the singleton embedding service created during application startup.
  """
  return request.app.state.services.embedding_service


def get_skills_extractor(request: Request) -> SkillsExtractor:
  """
  Returns the singleton skills extractor created during application startup.
  """
  return request.app.state.services.skills_extractor  # created once in lifespan()

