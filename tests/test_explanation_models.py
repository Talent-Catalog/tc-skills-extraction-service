import pytest
from pydantic import ValidationError

from app.models.explanation_models import GenerateExplanationRequest


def valid_request_data() -> dict[str, object]:
  """Return mutable valid evidence for request validation tests."""
  return {
    "candidate_id": "candidate-1",
    "rank": 1,
    "candidate_score": 0.7,
    "opportunity_description": "Seeking an accountant.",
    "matching_experiences": [
      {
        "experience_id": "experience-1",
        "job_title": None,
        "description": "Prepared financial reports.",
        "similarity": 0.8,
      }
    ],
  }


@pytest.mark.parametrize("rank", [0, -1])
def test_request_rejects_invalid_rank(rank: int) -> None:
  data = valid_request_data()
  data["rank"] = rank

  with pytest.raises(ValidationError):
    GenerateExplanationRequest.model_validate(data)


@pytest.mark.parametrize("candidate_score", [-1.01, 1.01])
def test_request_rejects_invalid_candidate_similarity(
    candidate_score: float,
) -> None:
  data = valid_request_data()
  data["candidate_score"] = candidate_score

  with pytest.raises(ValidationError):
    GenerateExplanationRequest.model_validate(data)


@pytest.mark.parametrize("similarity", [-1.01, 1.01])
def test_request_rejects_invalid_experience_similarity(
    similarity: float,
) -> None:
  data = valid_request_data()
  experiences = data["matching_experiences"]
  assert isinstance(experiences, list)
  experience = experiences[0]
  assert isinstance(experience, dict)
  experience["similarity"] = similarity

  with pytest.raises(ValidationError):
    GenerateExplanationRequest.model_validate(data)
