from typing import List
from pydantic import BaseModel, Field

class ExtractSkillsResponse(BaseModel):
  skills: List[str]

class ExtractSkillsRequest(BaseModel):
  text: str = Field(..., description="The text to extract skills from")
