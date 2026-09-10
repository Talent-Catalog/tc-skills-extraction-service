# Local development setup - see README.md for the full instructions,
# including why local development uses requirements-dev.txt while the
# deployed image uses requirements.txt.
#
# Quick reference (run in your IntelliJ terminal, which uses the same
# Python interpreter/venv as your IntelliJ project):
#
#   pip install -r requirements-dev.txt
#   python -m spacy download en_core_web_sm
#   uvicorn app.main:app --reload --log-config=log_conf.yaml
#
# This project targets Python 3.12, matching the Dockerfile's base image.
# It won't run on Python 3.14: spacy and torch (see requirements.txt) have
# no 3.14 wheels at the versions pinned here, so pip install fails outright
# on a venv built against 3.14. Neither is a permanent restriction - spacy's
# own latest release (3.8.16) already ships 3.14 wheels, and torch's latest
# (2.13.0) does too - but bumping either here needs its own verification
# (torch in particular, given the Fargate OOM history in requirements.txt).
#
#
# To run inside IntelliJ instead of the terminal: create a new Python run
# configuration (+ > Python), select "Module" (not "Script") and enter
# "uvicorn" as the module, "app.main:app --log-config=log_conf.yaml" as the
# parameters, and select your project's Python interpreter.
#
# See also https://realpython.com/get-started-with-fastapi/ and IntelliJ's
# FastAPI support: https://www.jetbrains.com/guide/tags/fastapi/
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import anthropic
import httpx
import spacy
from fastapi import FastAPI
from spacy.matcher import PhraseMatcher

from app.api.cv_extraction_api import router as cv_extraction_router
from app.api.embedding_api import router as embedding_router
from app.api.explanation_api import router as explanation_router
from app.api.health_api import router as health_router
from app.api.skills_api import router as skills_router
from app.dependencies import ApplicationServices
from app.services.cv_extraction.cv_extraction_service import CvExtractionService
from app.services.cv_extraction.pdf_to_doctags import PdfToDoctagsConverter
from app.services.embedding_service import (
  EmbeddingService,
  SentenceTransformerModelProvider,
)
from app.services.explanation_service import ExplanationService
from app.services.llm_client import LlmClient
from app.services.skills_service import SkillsService
from app.services.skills_extractor import SkillsExtractor
from app.services.text_preprocessor import SpacyTextPreprocessor
from app.settings import Settings


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

  Creates expensive application services once before requests are accepted.
  """
  embedding_service = EmbeddingService(
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
  skills_extractor = SkillsExtractor(nlp=nlp, matcher=matcher)

  # SKILLS_BASE_URL is populated by Settings from the environment variables or
  # .env file. It is used by SkillsService to retrieve the skills from the TC server.

  # noinspection PyArgumentList
  settings = Settings()
  # Qwen runs behind a separate OpenAI-compatible inference server so model
  # resources are not loaded into this FastAPI process.
  llm_http_client = httpx.Client()
  llm_client = LlmClient(
    base_url=settings.llm_base_url,
    model_name=settings.llm_model_name,
    timeout=settings.llm_request_timeout_seconds,
    http_client=llm_http_client,
    api_key=settings.llm_api_key,
  )
  explanation_service = ExplanationService(llm_client)

  # DocumentConverter loads Docling's layout/OCR models once, like
  # spaCy's nlp.load() above - so it's built here, not per request.
  cv_extraction_service = CvExtractionService(
    pdf_converter=PdfToDoctagsConverter(),
    anthropic_client=anthropic.Anthropic(api_key=settings.anthropic_api_key),
    model_name=settings.cv_extraction_model_name,
  )

  # Store the services in the app state so that they can be accessed in the dependencies
  app_.state.services = ApplicationServices(
    embedding_service=embedding_service,
    skills_extractor=skills_extractor,
    explanation_service=explanation_service,
    cv_extraction_service=cv_extraction_service,
  )

  app_.state.ready = True

  # Everything before the yield runs once at startup.
  # FastAPI won't start listening on the port until the pre-yield code is done.
  try:
    yield
  finally:
    # Everything from here runs once at shutdown.
    try:
      llm_http_client.close()
    finally:
      # spaCy doesn't need an explicit teardown.
      app_.state.services = None
      app_.state.ready = False

app = FastAPI(
  title="Talent Catalog AI Service",
  lifespan=lifespan,
)

app.include_router(embedding_router)
app.include_router(explanation_router)
app.include_router(skills_router)
app.include_router(health_router)
app.include_router(cv_extraction_router)


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
