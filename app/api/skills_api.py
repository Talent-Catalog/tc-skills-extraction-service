from fastapi import APIRouter, Depends

from app.dependencies import get_skills_extractor
from app.models.skills_models import ExtractSkillsRequest, SkillName
from app.services.skills_extractor import SkillsExtractor


router = APIRouter(
  tags=[
    "skills",
  ],
)

@router.post(
  "/extract_skills",
  response_model=list[SkillName],
)
def extract_skills(
    payload: ExtractSkillsRequest,
    skills_extractor: SkillsExtractor = Depends(
      get_skills_extractor
    ),
) -> list[SkillName]:
  """
  Extract skills from the given text
  :param payload: the text to extract skills from
  :param skills_extractor: the skills extractor to use
  :return: a List of extracted skills

  The SkillsExtractor is created during application startup and stored in
  application state. FastAPI injects it through get_skills_extractor().
  """
  return skills_extractor.extract_skills(payload.text)
