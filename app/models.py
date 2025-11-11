from pydantic import BaseModel, Field

class ExtractSkillsRequest(BaseModel):
  lang: str = Field("en", description="Language - eg 'en' for English")
  text: str = Field(..., description="The text to extract skills from")

class SkillName(BaseModel):
  """
  This matches the SkillName model in the TC skills-api
  """
  lang: str = Field(..., description="Language - eg 'en' for English")
  name: str = Field(..., description="Name of the skill")

  def __lt__(self, other):
    if not isinstance(other, SkillName):
      return NotImplemented
    return self.name.lower() < other.name.lower()
