from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from docling.document_converter import DocumentConverter

from app.services.cv_extraction.pdf_to_doctags import (
  PdfConversionError,
  PdfToDoctagsConverter,
)

FAKE_PDF_BYTES = b"%PDF-1.4 fake resume content"


class FakeDoclingDocument:
  """Return a configured Doctags string, like Docling's real document object."""

  def __init__(self, doctags: str) -> None:
    self._doctags = doctags

  def export_to_doctags(self) -> str:
    return self._doctags


class FakeConversionResult:
  def __init__(self, doctags: str) -> None:
    self.document = FakeDoclingDocument(doctags)


class FakeDocumentConverter:
  """
  Return a configured Doctags string, or raise, without loading Docling's
  real layout/OCR models. Captures the PDF bytes it was given (read while
  the caller's temp file still exists) so tests can assert on them.

  Deliberately does NOT subclass DocumentConverter: its real convert() takes
  several extra parameters (headers, raises_on_error, max_num_pages, ...)
  that a fake would either have to replicate or violate Liskov by dropping.
  Callers pass this to PdfToDoctagsConverter via cast(DocumentConverter, ...)
  instead - the same approach test_explanation_service.py's FakeLlmClient
  uses for its own inert httpx.Client dependency.
  """

  def __init__(
      self,
      doctags: str | None = None,
      error: Exception | None = None,
  ) -> None:
    self._doctags = doctags
    self._error = error
    self.received_bytes: bytes | None = None

  def convert(self, source: Path) -> FakeConversionResult:
    self.received_bytes = source.read_bytes()
    if self._error is not None:
      raise self._error
    return FakeConversionResult(self._doctags or "")


def _as_document_converter(fake: FakeDocumentConverter) -> DocumentConverter:
  """
  Narrow a FakeDocumentConverter to DocumentConverter for
  PdfToDoctagsConverter's constructor param. A direct cast() is flagged by
  PyCharm since the two classes don't share a hierarchy; routing through
  `object` first - as the inspection itself suggests - confirms this is
  deliberate rather than subclassing DocumentConverter (see
  FakeDocumentConverter's docstring for why not).
  """
  return cast(DocumentConverter, cast(object, fake))


def test_convert_returns_doctags_markup() -> None:
  fake_converter = FakeDocumentConverter(doctags="<doctag>...</doctag>")
  converter = PdfToDoctagsConverter(converter=_as_document_converter(fake_converter))

  result = converter.convert(FAKE_PDF_BYTES)

  assert result == "<doctag>...</doctag>"


def test_convert_writes_pdf_bytes_to_the_file_docling_receives() -> None:
  fake_converter = FakeDocumentConverter(doctags="ignored")
  converter = PdfToDoctagsConverter(converter=_as_document_converter(fake_converter))

  converter.convert(FAKE_PDF_BYTES)

  assert fake_converter.received_bytes == FAKE_PDF_BYTES


def test_convert_wraps_any_docling_error_as_pdf_conversion_error() -> None:
  docling_error = RuntimeError("Docling could not parse this file")
  fake_converter = FakeDocumentConverter(error=docling_error)
  converter = PdfToDoctagsConverter(converter=_as_document_converter(fake_converter))

  with pytest.raises(PdfConversionError) as exception_info:
    converter.convert(b"not a real pdf")

  assert "valid PDF" in str(exception_info.value)
  assert exception_info.value.__cause__ is docling_error
