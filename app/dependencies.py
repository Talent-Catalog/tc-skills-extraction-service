from fastapi import Request

from app.services.embedding_service import EmbeddingService
from app.services.skills_extractor import SkillsExtractor


def get_embedding_service(request: Request) -> EmbeddingService:
  """
  Returns the singleton embedding service created during application startup.
  """
  return request.app.state.embedding_service


def get_skills_extractor(request: Request) -> SkillsExtractor:
  return request.app.state.skills_extractor  # created once in lifespan()
