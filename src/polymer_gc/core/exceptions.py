class AttachmentPointError(ValueError):
    """Raised when a requested attachment point is undefined for a monomer."""


class InvalidValenceError(ValueError):
    """Raised when generated graph violates atomic valence rules."""
