import pytest
from pydantic import ValidationError

from app.models.explanation_models import GenerateExplanationRequest


def valid_request_data() -> dict[str, object]:
  """Return valid text-only input for request validation tests."""
  return {
    "candidate_id": "candidate-1",
    "opportunity_description": "Seeking an accountant.",
    "experiences": [
      {
        "experience_id": "experience-1",
        "job_title": None,
        "description": "Prepared financial reports.",
      }
    ],
  }


def test_request_contains_only_text_comparison_inputs() -> None:
  data = valid_request_data()

  request = GenerateExplanationRequest.model_validate(data)

  assert request.candidate_id == "candidate-1"
  assert request.opportunity_description == "Seeking an accountant."
  assert request.experiences[0].description == "Prepared financial reports."
  assert "rank" not in GenerateExplanationRequest.model_fields
  assert "candidate_score" not in GenerateExplanationRequest.model_fields


@pytest.mark.parametrize(
  "missing_field",
  [
    "candidate_id",
    "opportunity_description",
    "experiences",
  ],
)
def test_request_rejects_missing_required_inputs(
    missing_field: str,
) -> None:
  data = valid_request_data()
  del data[missing_field]

  with pytest.raises(ValidationError):
    GenerateExplanationRequest.model_validate(data)
