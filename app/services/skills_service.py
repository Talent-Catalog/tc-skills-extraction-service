from typing import Dict, Iterable, Optional
import requests

from typing import List

class SkillsService:
  def __init__(self):
    self.base_url = "http://localhost:8080"

  def get_skills(self) -> List[str]:
    #Example: this should be populated from skills on our Postgres database
    # populated from ESCO.
    # Could be around 20,000 of these
    items = self.__load_all_items()
    skills = List[str]()
    for item in items:
      skills.append(item["label"])

    return skills
    # return ["Java", "Spring Boot", "Python", "FastAPI", "Docker", "C++", "Kubernetes", "PostgreSQL",
    #         "MapStruct", "Angular", "AWS", "Terraform", "Natural language processing","Spring"]

  def __load_all_items(self, size=100, token=None) -> Iterable[Dict]:
    """
    Private method (note the __ prefix) that loads all items from the API.
    :param size: Requested page size
    :param token: Optional authorization token
    :return: Iterable of items
    """
    headers = {}
    if token:
      headers["Authorization"] = f"Bearer {token}"

    params = dict()
    params["page"] = 0      # Spring Boot pages are 0-based
    params["size"] = size

    all_items = []

    while True:
      response = requests.get(self.base_url, headers=headers, params=params, timeout=15)
      response.raise_for_status()
      data = response.json()

      items = data.get("content", [])
      all_items.extend(items)

      # Determine if we reached the last page
      last = data.get("last")
      number = data.get("number")
      total_pages = data.get("totalPages")

      if last or number + 1 >= total_pages:
        break

      # Move to next page
      params["page"] = number + 1

    return all_items
