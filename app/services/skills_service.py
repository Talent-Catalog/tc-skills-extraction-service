from typing import List

class SkillsService:
  def __init__(self):
    print("hello")

  def get_skills(self) -> List[str]:
    #Example: this should be populated from skills on our Postgres database
    # populated from ESCO.
    # Could be around 20,000 of these
    return ["Java", "Spring Boot", "Python", "FastAPI", "Docker", "C++", "Kubernetes", "PostgreSQL",
            "MapStruct", "Angular", "AWS", "Terraform", "Natural language processing","Spring"]
