"""
Dependencies:

#see https://realpython.com/get-started-with-fastapi/
# Note also Intellij's support for FastAPI development
# https://www.jetbrains.com/guide/tags/fastapi/

# Note also that we should be running Python version 3.12 - because we are not
# sure that the libraries that we use have been converted to later versions.

# Run the following command in your Intellij terminal (which will be running
# in the same virtual environment (venv) as your Intellij project.
# There will be a .venv subdirectory in your project directory.
 
python -m pip install "fastapi[standard]"

pip install pydantic-settings

pip install spacy

python -m spacy download en_core_web_sm  

# Run the app with:

uvicorn app.main:app --reload
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from typing import List
import spacy
from spacy.matcher import PhraseMatcher
from app.models import SkillName, ExtractSkillsRequest
from app.services.skills_extractor import SkillsExtractor
from app.services.skills_service import SkillsService

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
  skill_labels = skills_service.get_skills()

  # Build matcher once at startup. All skills have to be added to the matcher.
  matcher = build_matcher(nlp, skill_labels)

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
  app_.state.ready = False # type: ignore[attr-defined] (disable checking, state is there only at runtime)

app = FastAPI(title="Skills Extractor API", lifespan=lifespan)

@app.post("/extract_skills", response_model=List[SkillName])
def extract_skills(
    payload: ExtractSkillsRequest, request: Request,) -> List[SkillName]:
  """
  Extract skills from the given text
  :param payload: the text to extract skills from
  :param request: the request object (which contains the app instance)
  :return: a List of extracted skills
  """
  extractor = get_extractor(request)
  result = extractor.extract_skills(payload.text)
  return result

@app.get("/readyz")
def readyz(request: Request):
  """
  This is a commonly used liveness check endpoint.
  :param request: contains the app instance where the state is stored
  :return: a JSON response with the ready status
  """
  ready = bool(getattr(request.app.state, "ready", False))
  return {"ready": ready}


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


def get_extractor(request: Request) -> SkillsExtractor:
  return request.app.state.extractor  # created once in lifespan()
