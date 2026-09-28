from __future__ import annotations

import json
import os
from enum import StrEnum
from typing import Any, Protocol

import httpx
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.session import Session as BotocoreSession


class AwsCredentialsProvider(Protocol):
  """
  The minimal AWS session interface LlmClient needs for AWS_SIGV4: credential
  and region discovery.

  botocore.session.Session satisfies this structurally; declaring it here
  (rather than typing directly against that unstubbed botocore class) lets
  tests supply a lightweight fake session instead of a real one.
  """

  def get_credentials(self) -> Any: ...

  def get_config_variable(self, name: str) -> str | None: ...


class LlmAuthentication(StrEnum):
  """
  Identifies how LlmClient authenticates with the configured OpenAI-compatible
  endpoint.

  NONE sends no authentication headers, for endpoints such as a locally
  hosted vLLM server that require none. BEARER sends a static bearer token,
  such as a short-term Bedrock API key used for local development.
  AWS_SIGV4 signs each request with AWS Signature Version 4 using credentials
  from the standard AWS credential provider chain, such as an ECS task role.
  """

  NONE = "NONE"
  BEARER = "BEARER"
  AWS_SIGV4 = "AWS_SIGV4"


class LlmServiceUnavailableError(RuntimeError):
  """Raised when the configured LLM service cannot complete a request."""


class MalformedLlmResponseError(RuntimeError):
  """Raised when an LLM response is not a valid chat completion."""


class LlmConfigurationError(RuntimeError):
  """
  Raised when LlmClient is configured with an invalid or incomplete
  authentication setup.
  """


class LlmClient:
  """
  Communicates with any OpenAI-compatible LLM endpoint.

  This provider-independent boundary contains only chat transport concerns,
  allowing the inference server or model to change without changing domain
  prompt logic. Authentication is likewise a transport concern handled here:
  it determines how a request is signed or labelled, not which endpoint or
  model is being called.
  """

  AWS_SIGV4_SERVICE_NAME = "bedrock"

  def __init__(
      self,
      base_url: str,
      model_name: str,
      timeout: float,
      http_client: httpx.Client,
      authentication: LlmAuthentication = LlmAuthentication.NONE,
      api_key: str | None = None,
      aws_session: AwsCredentialsProvider | None = None,
  ) -> None:
    self._base_url = base_url.rstrip("/")
    self._model_name = model_name
    self._timeout = timeout
    self._http_client = http_client
    self._authentication = authentication
    self._api_key = api_key.strip() if api_key and api_key.strip() else None

    self._aws_credentials = None
    self._aws_region = None

    if self._authentication == LlmAuthentication.BEARER:
      if self._api_key is None:
        raise LlmConfigurationError(
          "An API key is required when LlmAuthentication.BEARER is "
          "configured"
        )

    elif self._authentication == LlmAuthentication.AWS_SIGV4:
      session = aws_session if aws_session is not None else BotocoreSession()

      credentials = session.get_credentials()
      if credentials is None:
        raise LlmConfigurationError(
          "No AWS credentials were found for LlmAuthentication.AWS_SIGV4. "
          "Configure the standard AWS credential provider chain, for "
          "example an ECS task role."
        )

      # botocore's own "region" config variable only resolves AWS_REGION via
      # the AWS_DEFAULT_REGION environment variable (and shared config
      # files); AWS_REGION is checked explicitly first since ECS/Fargate
      # tasks commonly set that instead.
      region = (
          os.environ.get("AWS_REGION")
          or session.get_config_variable("region")
      )
      if not region:
        raise LlmConfigurationError(
          "No AWS region was found for LlmAuthentication.AWS_SIGV4. Set "
          "AWS_REGION or AWS_DEFAULT_REGION."
        )

      self._aws_credentials = credentials
      self._aws_region = region

  def generate(
      self,
      system_prompt: str,
      user_prompt: str,
  ) -> str:
    """Generate assistant message content from the supplied prompts."""
    url = f"{self._base_url}/chat/completions"

    # The body is serialized once and reused unchanged for both signing
    # (AWS_SIGV4) and sending, so the signature always covers exactly the
    # bytes that are transmitted.
    body = json.dumps({
      "model": self._model_name,
      "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
      ],
      "temperature": 0.1,
    }).encode("utf-8")

    try:
      response = self._http_client.post(
        url,
        headers=self._build_headers(url=url, body=body),
        content=body,
        timeout=self._timeout,
      )
      response.raise_for_status()
    except httpx.HTTPError as exception:
      raise LlmServiceUnavailableError(
        "The LLM service is unavailable"
      ) from exception

    try:
      payload: Any = response.json()
      content = payload["choices"][0]["message"]["content"]
    except (
        ValueError,
        UnicodeError,
        KeyError,
        IndexError,
        TypeError,
    ) as exception:
      raise MalformedLlmResponseError(
        "The LLM service returned a malformed chat-completion response"
      ) from exception

    if not isinstance(content, str) or not content.strip():
      raise MalformedLlmResponseError(
        "The LLM service returned empty or non-text assistant content"
      )

    return content

  def _build_headers(
      self,
      url: str,
      body: bytes,
  ) -> dict[str, str]:
    """Return the request headers for the configured authentication mode."""
    headers = {"Content-Type": "application/json"}

    if self._authentication == LlmAuthentication.BEARER:
      headers["Authorization"] = f"Bearer {self._api_key}"
      return headers

    if self._authentication == LlmAuthentication.AWS_SIGV4:
      return self._sign_with_aws_sigv4(url=url, body=body, headers=headers)

    return headers

  def _sign_with_aws_sigv4(
      self,
      url: str,
      body: bytes,
      headers: dict[str, str],
  ) -> dict[str, str]:
    """
    Sign the request with AWS Signature Version 4.

    Credentials are re-frozen on every call (rather than once at
    construction) because ECS task role credentials are temporary and
    botocore transparently refreshes them behind this call as they near
    expiry.
    """
    frozen_credentials = self._aws_credentials.get_frozen_credentials()

    request = AWSRequest(
      method="POST",
      url=url,
      data=body,
      headers=headers,
    )

    SigV4Auth(
      frozen_credentials,
      self.AWS_SIGV4_SERVICE_NAME,
      self._aws_region,
    ).add_auth(request)

    return dict(request.headers.items())
