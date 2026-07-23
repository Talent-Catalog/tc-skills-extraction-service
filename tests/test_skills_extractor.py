import spacy
import pytest
from app.main import build_matcher
from app.services.skills_extractor import (
  SkillName,
  SkillsExtractor,
)


@pytest.fixture
def nlp():
  """
  A blank English pipeline is sufficient because PhraseMatcher with
  attr="LOWER" only needs tokenization.
  """
  return spacy.blank("en")


@pytest.fixture
def extractor(nlp):
  skills = [
    "accounting",
    "bookkeeping",
    "project management",
    "Python",
  ]

  matcher = build_matcher(nlp, skills)
  return SkillsExtractor(nlp, matcher)


def test_extracts_single_skill(extractor):
  result = extractor.extract_skills(
    "I have several years of accounting experience."
  )

  assert result == [
    SkillName(name="accounting", lang="en"),
  ]


def test_matching_is_case_insensitive(extractor):
  result = extractor.extract_skills(
    "My main skills are ACCOUNTING and Bookkeeping."
  )

  assert result == [
    SkillName(name="accounting", lang="en"),
    SkillName(name="bookkeeping", lang="en"),
  ]


def test_extracts_multi_word_skill(extractor):
  result = extractor.extract_skills(
    "She has extensive project management experience."
  )

  assert result == [
    SkillName(name="project management", lang="en"),
  ]


def test_extracts_multiple_skills_in_sorted_order(extractor):
  result = extractor.extract_skills(
    "I use Python for bookkeeping and accounting."
  )

  assert result == [
    SkillName(name="accounting", lang="en"),
    SkillName(name="bookkeeping", lang="en"),
    SkillName(name="python", lang="en"),
  ]


def test_removes_duplicate_skills(extractor):
  result = extractor.extract_skills(
    "Accounting was my first role. I later returned to ACCOUNTING."
  )

  assert result == [
    SkillName(name="accounting", lang="en"),
  ]


def test_matches_skill_next_to_punctuation(extractor):
  result = extractor.extract_skills(
    'Her skills include "accounting", bookkeeping, and Python.'
  )

  assert result == [
    SkillName(name="accounting", lang="en"),
    SkillName(name="bookkeeping", lang="en"),
    SkillName(name="python", lang="en"),
  ]


def test_does_not_match_skill_inside_larger_word(extractor):
  result = extractor.extract_skills(
    "The application uses Pythonic coding conventions."
  )

  assert result == []


def test_returns_empty_list_when_no_skills_match(extractor):
  result = extractor.extract_skills(
    "I enjoy walking and reading."
  )

  assert result == []


def test_related_word_does_not_currently_match(extractor):
  """
  Documents the current limitation: PhraseMatcher performs phrase matching,
  not semantic or derivational matching.
  """
  result = extractor.extract_skills(
    "I worked as an accountant."
  )

  assert result == []
