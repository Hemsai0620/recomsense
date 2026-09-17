"""Exception hierarchy used across RecomSense.

Every failure that the API is expected to translate into a meaningful HTTP
status code derives from :class:`RecomSenseError` and carries an
``http_status`` plus a machine readable ``code``.
"""

from __future__ import annotations


class RecomSenseError(Exception):
    """Base class for all errors raised by RecomSense."""

    http_status: int = 500
    code: str = "internal_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class DataValidationError(RecomSenseError):
    """Raised when the interaction dataset is missing or malformed."""

    http_status = 422
    code = "invalid_dataset"


class ArtifactNotFoundError(RecomSenseError):
    """Raised when the trained model bundle has not been produced yet."""

    http_status = 503
    code = "model_unavailable"


class InvalidTopNError(RecomSenseError):
    """Raised when the requested number of recommendations is out of range."""

    http_status = 422
    code = "invalid_top_n"


class UnknownUserError(RecomSenseError):
    """Raised when an unknown user is requested and cold start is disabled."""

    http_status = 404
    code = "unknown_user"
