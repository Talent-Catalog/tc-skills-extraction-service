import requests
from app.models import SkillName
from app.settings import settings
from typing import List

class SkillsService:
  """
  Service which loads all known skills from the skills-api of the TC server
  """
  def __init__(self):
    self.base_url = str(settings.SKILLS_BASE_URL)

  def get_skills(self) -> List[str]:
    items: List[SkillName] = self.__load_all_items()
    skills = []
    for item in items:
      skills.append(item.name)

    return skills

  def __load_all_items(self, size=100, token=None) -> List[SkillName]:
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

    all_items: List[SkillName] = []

    while True:
      response = requests.get(self.base_url, headers=headers, params=params, timeout=15)
      response.raise_for_status()

      # Converts the json response to a dict
      data = response.json()

      # The TC (Spring) paging API returns the page of data in the "content" field
      content = data.get("content", [])
      # Convert the list of dicts to a list of SkillName objects
      items = [SkillName(**item) for item in content]
      all_items.extend(items)

      # Determine if we reached the last page
      # These are standard TC (Spring) paging fields
      last = data.get("last")
      number = data.get("number")
      total_pages = data.get("totalPages")

      if last or number + 1 >= total_pages:
        break

      # Move to the next page
      params["page"] = number + 1

    return all_items
