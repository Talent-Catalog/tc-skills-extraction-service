from typing import List
from pydantic import BaseModel, Field

class ExtractedSkills(BaseModel):
  skills: List[str]

class TextToProcess(BaseModel):
  text: str = Field(..., description="The text to extract skills from")
