"""
Dependencies:

#see https://realpython.com/get-started-with-fastapi/
# Note also Intellij's support for FastAPI development
# https://www.jetbrains.com/guide/tags/fastapi/
python -m pip install "fastapi[standard]"

"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, Request
import spacy
from spacy.matcher import PhraseMatcher
from app.models import ExtractSkillsResponse, ExtractSkillsRequest
from app.services.skills_extractor import SkillsExtractor


@asynccontextmanager
async def lifespan(app_: FastAPI):
  # Load heavy resources ONCE, without blocking the event loop.
  # asyncio.to_thread runs blocking calls in a threadpool.
  nlp = spacy.load("en_core_web_sm")

  #Example: this should be populated from skills on our Postgres database
  # populated from ESCO.
  # Could be around 20,000 of these
  esco_skill_labels = [
    "Java", "Spring Boot", "Python", "FastAPI", "Docker", "Kubernetes", "PostgreSQL",
    "MapStruct", "Angular", "AWS", "Terraform", "Natural language processing","Spring"
  ]

  # Matching is case-insensitive - see https://spacy.io/api/phrasematcher#init
  matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

  # Convert text skills into an array of NLP docs.
  patterns = [nlp.make_doc(s) for s in esco_skill_labels]

  # Configure the matcher to recognize the skills as a special kind of match.
  # See https://spacy.io/api/phrasematcher#add
  matcher.add("SKILL", patterns)

  # The above matcher only needs to be created once at start-up (20,000 skill
  # names - so only want to do once).

  app_.state.extractor = SkillsExtractor(nlp=nlp, matcher=matcher) # type: ignore[attr-defined] (disable checking, state is there only at runtime)

  # Everything before the yield runs once at startup.
  # FastAPI won't start listening on the port until the pre-yield code is done.
  yield
  # Everything from here runs once at shutdown.

  # spaCy doesn't need an explicit teardown.
  # If you opened sockets/files, close them here.
  app.state.extractor = None # type: ignore[attr-defined] (disable checking, state is there only at runtime)


def get_extractor(request: Request) -> SkillsExtractor:
  return request.app.state.extractor  # created once in lifespan()

app = FastAPI(title="Skills Extractor API", lifespan=lifespan)

@app.post("/extract_skills", response_model=ExtractSkillsResponse)
def extract_skills(payload: ExtractSkillsRequest,
    extractor: SkillsExtractor = Depends(get_extractor)):
  """
  Extract skills from the given text
  :param payload: the text to extract skills from
  :param extractor: the SkillsExtractor instance
  :return: a List of extracted skills
  """
  result = extractor.extract_skills(payload.text)
  return ExtractSkillsResponse(skills=result)

