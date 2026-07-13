"""Document processing lifecycle states."""

from enum import Enum


class DocumentStatus(str, Enum):
    """Represents the current processing state of an uploaded document."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
