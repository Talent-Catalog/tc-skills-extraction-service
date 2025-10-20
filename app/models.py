from typing import List
from pydantic import BaseModel, Field

class ExtractSkillsResponse(BaseModel):
  skills: List[str]

class ExtractSkillsRequest(BaseModel):
  text: str = Field(..., description="The text to extract skills from")

class SkillName(BaseModel):
  """
  This matches the SkillName model in the TC skills-api
  """
  lang: str = Field(..., description="Language - eg 'en' for English")
  name: str = Field(..., description="Name of the skill")
