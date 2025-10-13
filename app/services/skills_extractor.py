import spacy
from spacy.matcher import PhraseMatcher
from typing import List

class SkillsExtractor:
  def __init__(self, nlp: spacy.language.Language, matcher: PhraseMatcher):
    self._nlp = nlp
    self._matcher = matcher

  def extract_skills(self, text: str) -> List[str]:

    doc = self._nlp(text)  #todo skip full pipeline with nlp.make_doc unless you need POS/NER

    # Run matcher on the doc to find all the skills mentioned.
    skill_matches = []
    for match_id, start, end in self._matcher(doc):
      match = doc[start:end]
      skill_matches.append(match.text)

    # De-dup skills
    skills = sorted(set([s.lower() for s in skill_matches]))
    return skills

