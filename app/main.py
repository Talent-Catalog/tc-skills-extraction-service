"""
Dependencies:

#see https://realpython.com/get-started-with-fastapi/
# Note also Intellij's support for FastAPI development
# https://www.jetbrains.com/guide/tags/fastapi/

# Note also that we should be running Python version 3.12 - because we are not
# sure that the libraries that we use have been converted to later versions.
# See https://pypi.org/project/spacy/

# Run the following command in your Intellij terminal (which will be running
# in the same virtual environment (venv) as your Intellij project.
# There will be a .venv subdirectory in your project directory.

python -m pip install "fastapi[standard]"

(Note that fastapi[standard] includes uvicorn, pydantic, and other dependencies)

pip install pydantic-settings

pip install spacy

python -m spacy download en_core_web_sm

# Run the app with:

uvicorn app.main:app --reload --log-config=log_conf.yaml

OPTIONALLY...
# Run the app inside Intellij

You need to have installed all the above pip and python commands so that all
the libraries are entered into the virtual environment (venv) of your Intellij project.

Create a new Intellij run configuration. Press + and select Python.
Select "module" (not "script") and enter "uvicorn" in the module field.
In the parameters field, enter "app.main:app --log-config=log_conf.yaml"
Select the Python interpreter that is the same as your project interpreter (the venv).

"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import spacy
from fastapi import FastAPI
from spacy.matcher import PhraseMatcher

from app.api.embedding_api import router as embedding_router
from app.api.health_api import router as health_router
from app.api.skills_api import router as skills_router
from app.services.embedding_service import (
  EmbeddingService,
  SentenceTransformerModelProvider,
)
from app.services.skills_service import SkillsService
from app.services.skills_extractor import SkillsExtractor
from app.services.text_preprocessor import SpacyTextPreprocessor


@asynccontextmanager
async def lifespan(app_: FastAPI) -> AsyncIterator[None]:
  """
  This runs once at startup up to the yield statement.
  :param app_: Reference to the FastAPI app

  Sets up the app state including global services and resources which are
  attached to the app instance through the app.state attribute which can
  be accessed in other parts of the code through Request dependency injection.
  Globals are accessible through Request.app.state - whose values are set
  in this set-up function.
  """

  """
  Creates expensive application services once before requests are accepted.
  """
  app_.state.embedding_service = EmbeddingService(
    model_provider=SentenceTransformerModelProvider(),
    text_preprocessor=SpacyTextPreprocessor(),
    encoder_batch_size=32,
    normalize_embeddings=True,
  )

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
  app_.state.skills_extractor = SkillsExtractor(nlp=nlp, matcher=matcher)
  app_.state.ready = True

  # Everything before the yield runs once at startup.
  # FastAPI won't start listening on the port until the pre-yield code is done.
  yield
  # Everything from here runs once at shutdown.

  # spaCy doesn't need an explicit teardown.
  # If you opened sockets/files, close them here.
  app_.state.skills_extractor = None
  app_.state.embedding_service = None
  app_.state.ready = False

app = FastAPI(
  title="Embedding Service",
  lifespan=lifespan,
)

app.include_router(embedding_router)
app.include_router(skills_router)
app.include_router(health_router)


def build_matcher(nlp: spacy.language.Language, skills: list[str]) -> PhraseMatcher:
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

