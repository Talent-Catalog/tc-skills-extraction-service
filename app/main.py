"""
Dependencies:

#see https://realpython.com/get-started-with-fastapi/
# Note also Intellij's support for FastAPI development
# https://www.jetbrains.com/guide/tags/fastapi/
python -m pip install "fastapi[standard]"

"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, Request
from typing import List
import spacy
from spacy.matcher import PhraseMatcher
from app.models import ExtractSkillsResponse, ExtractSkillsRequest
from app.services.skills_extractor import SkillsExtractor
from app.services.skills_service import SkillsService

def build_matcher(nlp: spacy.language.Language, skills: List[str]) -> PhraseMatcher:
  """
  The matcher only needs to be created once at start-up (20,000 skill
  # names - so only want to do once)
  :param nlp: Spacy model
  :param skills: List of skills
  :return: PhraseMatcher
  """

  # Convert text skills into an array of NLP docs.
  patterns = [nlp.make_doc(s) for s in skills]

  # Matching is case-insensitive - see https://spacy.io/api/phrasematcher#init
  matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

  # Configure the matcher to recognize the skills as a special kind of match.
  # See https://spacy.io/api/phrasematcher#add
  matcher.add("SKILL", patterns)

  return matcher

@asynccontextmanager
async def lifespan(app_: FastAPI):
  """
  This runs once at startup up to the yield statement.
  :param app_: Reference to the FastAPI app

  Sets up the app state including global services and resources which are
  attached to the app instance through the app.state attribute which can
  be accessed in other parts of the code through Request dependency injection.
  Globals are accessible through Request.app.state - whose values are set
  in this set-up function.
  """

  # Load heavy resources ONCE, without blocking the event loop.
  # asyncio.to_thread runs blocking calls in a threadpool.
  nlp = spacy.load("en_core_web_sm")

  skills_service = SkillsService()
  # This can take a while. Skills are retrieved from an external service and
  # there could be around 20,000 of them.
  esco_skill_labels = skills_service.get_skills()

  # Build matcher once at startup. All skills have to be added to the matcher.
  matcher = build_matcher(nlp, esco_skill_labels)

  # The extractor is configured with all the heavy resources.
  app_.state.extractor = SkillsExtractor(nlp=nlp, matcher=matcher) # type: ignore[attr-defined] (disable checking, state is there only at runtime)
  app_.state.ready = True # type: ignore[attr-defined] (disable checking, state is there only at runtime)

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
def extract_skills(
    payload: ExtractSkillsRequest,
    extractor: SkillsExtractor = Depends(get_extractor)) -> ExtractSkillsResponse:
  """
  Extract skills from the given text
  :param payload: the text to extract skills from
  :param extractor: the SkillsExtractor instance
  :return: a List of extracted skills
  """
  result = extractor.extract_skills(payload.text)
  return ExtractSkillsResponse(skills=result)

@app.get("/readyz")
def readyz(request: Request):
  """
  This is a commonly used liveness check endpoint.
  :param request: contains the app instance where the state is stored
  :return: a JSON response with the ready status
  """
  ready = bool(getattr(request.app.state, "ready", False))
  return {"ready": ready}
