from __future__ import annotations

import tempfile
from pathlib import Path

from docling.document_converter import DocumentConverter


class PdfConversionError(RuntimeError):
  """Raised when an uploaded file cannot be converted to Doctags markup."""


class PdfToDoctagsConverter:
  """
  Converts PDF bytes to Docling's Doctags markup.

  Docling's DocumentConverter loads its layout/OCR models once per instance,
  so - like EmbeddingService's SentenceTransformer model - a single instance
  is created at application startup and reused across requests rather than
  constructed per request.

  Note: Docling's own exception hierarchy for "not a valid PDF" / "unsupported
  or corrupt file" was not enumerated file-by-file before writing this - the
  broad except below is a deliberate boundary (any conversion failure becomes
  a PdfConversionError) rather than a claim that every specific Docling
  failure mode has been accounted for individually.
  """

  def __init__(self, converter: DocumentConverter | None = None) -> None:
    # Accepts an injected converter (like EmbeddingService's model_provider and
    # LlmClient's http_client) so tests can substitute a fake instead of
    # loading Docling's real layout/OCR models.
    self._converter = converter or DocumentConverter()

  def convert(self, pdf_bytes: bytes) -> str:
    """Convert PDF bytes to a Doctags string."""
    try:
      with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_file:
        temp_file.write(pdf_bytes)
        temp_file.flush()
        result = self._converter.convert(Path(temp_file.name))
        return result.document.export_to_doctags()
    except Exception as exception:
      raise PdfConversionError(
        "The uploaded file could not be converted - is it a valid PDF?"
      ) from exception
