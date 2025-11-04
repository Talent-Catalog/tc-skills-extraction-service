import spacy
from spacy.matcher import PhraseMatcher
from typing import List
from app.models import SkillName
import logging

logger = logging.getLogger(__name__)

class SkillsExtractor:
  def __init__(self, nlp: spacy.language.Language, matcher: PhraseMatcher):
    self._nlp = nlp
    self._matcher = matcher

  def extract_skills(self, text: str) -> List[SkillName]:

    doc = self._nlp(text)

    # Run matcher on the doc to find all the skills mentioned.
    skill_matches = []
    for match_id, start, end in self._matcher(doc):
      match = doc[start:end]
      skill_matches.append(match.text)

    # De-dup skills
    skills = set([s.lower() for s in skill_matches])

    skill_names = [SkillName(name=skill, lang=self._nlp.lang) for skill in skills]
    skill_names.sort()

    # Log the extracted skills
    if skill_names:
      logged_skills = ', '.join(s.name for s in skill_names)
      logger.info(f"Extracted {len(skill_names)} skills: {logged_skills}")
    else:
      logger.info("No skills extracted.")

    return skill_names

