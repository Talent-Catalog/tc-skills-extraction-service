from __future__ import annotations

import json
from collections.abc import Generator
from typing import Any, Protocol

import httpx
import pytest
from botocore.credentials import ReadOnlyCredentials

from app.services.llm_client import (
  AwsCredentialsProvider,
  LlmAuthentication,
  LlmClient,
  LlmConfigurationError,
  LlmServiceUnavailableError,
  MalformedLlmResponseError,
)


class FakeAwsCredentials:
  """A minimal stand-in for botocore's lazily-refreshing credential objects."""

  def __init__(
      self,
      access_key: str,
      secret_key: str,
      token: str | None = None,
  ) -> None:
    self._access_key = access_key
    self._secret_key = secret_key
    self._token = token

  def get_frozen_credentials(self) -> Any:
    # botocore's ReadOnlyCredentials is a namedtuple from an unstubbed
    # package, so it isn't used as a type annotation here - only as the
    # real value SigV4Auth expects.
    return ReadOnlyCredentials(
      self._access_key,
      self._secret_key,
      self._token,
    )


class FakeAwsSession:
  """
  A minimal stand-in for botocore.session.Session, exposing only the two
  methods LlmClient relies on for AWS_SIGV4.
  """

  def __init__(
      self,
      credentials: FakeAwsCredentials | None,
      region: str | None,
  ) -> None:
    self._credentials = credentials
    self._region = region

  def get_credentials(self) -> FakeAwsCredentials | None:
    return self._credentials

  def get_config_variable(self, name: str) -> str | None:
    return self._region if name == "region" else None


class CreateLlmClient(Protocol):
  """Describes the configurable LLM client factory used by tests."""

  def __call__(
      self,
      transport: httpx.MockTransport,
      *,
      base_url: str = "http://llm.test/v1/",
      model_name: str = "test-model",
      authentication: LlmAuthentication = LlmAuthentication.NONE,
      api_key: str | None = None,
      aws_session: AwsCredentialsProvider | None = None,
  ) -> LlmClient:
    """Create an LLM client using the supplied test transport."""


@pytest.fixture
def create_client() -> Generator[
    CreateLlmClient,
    None,
    None,
]:
  """Create an LLM client backed by an in-memory HTTP transport."""
  clients: list[httpx.Client] = []

  def factory(
      transport: httpx.MockTransport,
      *,
      base_url: str = "http://llm.test/v1/",
      model_name: str = "test-model",
      authentication: LlmAuthentication = LlmAuthentication.NONE,
      api_key: str | None = None,
      aws_session: AwsCredentialsProvider | None = None,
  ) -> LlmClient:
    http_client = httpx.Client(transport=transport)
    clients.append(http_client)
    return LlmClient(
      base_url=base_url,
      model_name=model_name,
      timeout=12.0,
      http_client=http_client,
      authentication=authentication,
      api_key=api_key,
      aws_session=aws_session,
    )

  yield factory

  for client in clients:
    client.close()


def _success_response() -> httpx.Response:
  return httpx.Response(
    200,
    json={
      "choices": [
        {"message": {"content": "generated content"}}
      ]
    },
  )


def test_extracts_assistant_message_content(
    create_client: CreateLlmClient,
) -> None:
  def handler(request: httpx.Request) -> httpx.Response:
    assert request.url == "https://provider.test/openai/v1/chat/completions"
    assert "Authorization" not in request.headers
    payload = json.loads(request.content)
    assert payload == {
      "model": "configured-model",
      "messages": [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "evidence"},
      ],
      "temperature": 0.1,
    }
    assert "chat_template_kwargs" not in payload
    return _success_response()

  result = create_client(
    httpx.MockTransport(handler),
    base_url="https://provider.test/openai/v1/",
    model_name="configured-model",
  ).generate("system", "evidence")

  assert result == "generated content"


@pytest.mark.parametrize("api_key", [None, "", "   ", "unused-key"])
def test_none_authentication_sends_no_authorization_header(
    create_client: CreateLlmClient,
    api_key: str | None,
) -> None:
  """
  NONE must send no authentication headers, even if an API key happens to
  be configured alongside it.
  """
  def handler(request: httpx.Request) -> httpx.Response:
    assert "Authorization" not in request.headers
    return _success_response()

  result = create_client(
    httpx.MockTransport(handler),
    authentication=LlmAuthentication.NONE,
    api_key=api_key,
  ).generate("system", "evidence")

  assert result == "generated content"


def test_bearer_authentication_sends_bearer_authorization(
    create_client: CreateLlmClient,
) -> None:
  def handler(request: httpx.Request) -> httpx.Response:
    assert request.headers["Authorization"] == "Bearer secret-api-key"
    return _success_response()

  result = create_client(
    httpx.MockTransport(handler),
    authentication=LlmAuthentication.BEARER,
    api_key="secret-api-key",
  ).generate("system", "evidence")

  assert result == "generated content"


@pytest.mark.parametrize("api_key", [None, "", "   "])
def test_bearer_without_api_key_raises_configuration_error(
    create_client: CreateLlmClient,
    api_key: str | None,
) -> None:
  with pytest.raises(
      LlmConfigurationError,
      match="API key is required",
  ):
    create_client(
      httpx.MockTransport(lambda request: _success_response()),
      authentication=LlmAuthentication.BEARER,
      api_key=api_key,
    )


def test_aws_sigv4_signs_the_request_and_handles_session_token(
    create_client: CreateLlmClient,
) -> None:
  """
  AWS_SIGV4 must sign the request (proving credentials/region were used)
  and must forward a temporary session token as X-Amz-Security-Token.
  """
  def handler(request: httpx.Request) -> httpx.Response:
    authorization = request.headers["Authorization"]
    assert authorization.startswith("AWS4-HMAC-SHA256 ")
    assert "Credential=AKIA_TEST/" in authorization
    assert "/eu-west-2/bedrock/aws4_request" in authorization
    assert "x-amz-security-token" in authorization
    assert request.headers["X-Amz-Security-Token"] == "session-token-abc"
    assert "X-Amz-Date" in request.headers
    return _success_response()

  aws_session = FakeAwsSession(
    credentials=FakeAwsCredentials(
      access_key="AKIA_TEST",
      secret_key="secret",
      token="session-token-abc",
    ),
    region="eu-west-2",
  )

  result = create_client(
    httpx.MockTransport(handler),
    authentication=LlmAuthentication.AWS_SIGV4,
    aws_session=aws_session,
  ).generate("system", "evidence")

  assert result == "generated content"


def test_aws_sigv4_signature_depends_on_the_request_body(
    create_client: CreateLlmClient,
) -> None:
  """
  The signature must cover the body actually sent, not a fixed or
  independently re-serialized body - otherwise a body altered after
  signing (e.g. by httpx re-encoding JSON) would go undetected.
  """
  captured_authorizations: list[str] = []

  def handler(request: httpx.Request) -> httpx.Response:
    captured_authorizations.append(request.headers["Authorization"])
    return _success_response()

  aws_session = FakeAwsSession(
    credentials=FakeAwsCredentials(
      access_key="AKIA_TEST",
      secret_key="secret",
    ),
    region="eu-west-2",
  )

  client = create_client(
    httpx.MockTransport(handler),
    authentication=LlmAuthentication.AWS_SIGV4,
    aws_session=aws_session,
  )

  client.generate("system", "first evidence")
  client.generate("system", "second evidence")

  assert len(captured_authorizations) == 2
  assert captured_authorizations[0] != captured_authorizations[1]


def test_aws_sigv4_uses_the_configured_region_not_a_hardcoded_one(
    create_client: CreateLlmClient,
) -> None:
  def handler(request: httpx.Request) -> httpx.Response:
    assert "/us-east-1/bedrock/aws4_request" in request.headers["Authorization"]
    return _success_response()

  aws_session = FakeAwsSession(
    credentials=FakeAwsCredentials(
      access_key="AKIA_TEST",
      secret_key="secret",
    ),
    region="us-east-1",
  )

  create_client(
    httpx.MockTransport(handler),
    authentication=LlmAuthentication.AWS_SIGV4,
    aws_session=aws_session,
  ).generate("system", "evidence")


def test_aws_sigv4_without_credentials_raises_configuration_error(
    create_client: CreateLlmClient,
) -> None:
  aws_session = FakeAwsSession(credentials=None, region="eu-west-2")

  with pytest.raises(
      LlmConfigurationError,
      match="No AWS credentials were found",
  ):
    create_client(
      httpx.MockTransport(lambda request: _success_response()),
      authentication=LlmAuthentication.AWS_SIGV4,
      aws_session=aws_session,
    )


def test_aws_sigv4_without_region_raises_configuration_error(
    create_client: CreateLlmClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.delenv("AWS_REGION", raising=False)

  aws_session = FakeAwsSession(
    credentials=FakeAwsCredentials(access_key="AKIA_TEST", secret_key="secret"),
    region=None,
  )

  with pytest.raises(
      LlmConfigurationError,
      match="No AWS region was found",
  ):
    create_client(
      httpx.MockTransport(lambda request: _success_response()),
      authentication=LlmAuthentication.AWS_SIGV4,
      aws_session=aws_session,
    )


def test_http_failure_raises_service_unavailable_without_exposing_key(
    create_client: CreateLlmClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
  def handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
      503,
      json={"detail": "Provider unavailable"},
    )

  with pytest.raises(
      LlmServiceUnavailableError,
      match="LLM service is unavailable",
  ) as exception_info:
    create_client(
      httpx.MockTransport(handler),
      authentication=LlmAuthentication.BEARER,
      api_key="secret-api-key",
    ).generate("system", "evidence")

  assert "secret-api-key" not in str(exception_info.value)
  assert "secret-api-key" not in caplog.text


def test_invalid_json_raises_malformed_response(
    create_client: CreateLlmClient,
) -> None:
  def handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=b"not JSON")

  with pytest.raises(
      MalformedLlmResponseError,
      match="malformed chat-completion response",
  ):
    create_client(
      httpx.MockTransport(handler)
    ).generate("system", "evidence")


def test_malformed_chat_completion_raises_clear_error(
    create_client: CreateLlmClient,
) -> None:
  def handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"choices": []})

  with pytest.raises(
      MalformedLlmResponseError,
      match="malformed chat-completion response",
  ):
    create_client(
      httpx.MockTransport(handler)
    ).generate("system", "evidence")


def test_empty_assistant_content_is_malformed(
    create_client: CreateLlmClient,
) -> None:
  def handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
      200,
      json={
        "choices": [
          {"message": {"content": "   "}}
        ]
      },
    )

  with pytest.raises(
      MalformedLlmResponseError,
      match="empty or non-text",
  ):
    create_client(
      httpx.MockTransport(handler)
    ).generate("system", "evidence")
