import pytest
from polymer_gc.core.exceptions import AttachmentPointError, InvalidValenceError


def test_attachment_point_error():
    with pytest.raises(AttachmentPointError, match="Test message"):
        raise AttachmentPointError("Test message")
    assert issubclass(AttachmentPointError, ValueError)


def test_invalid_valence_error():
    with pytest.raises(InvalidValenceError, match="Another test"):
        raise InvalidValenceError("Another test")
    assert issubclass(InvalidValenceError, ValueError)
