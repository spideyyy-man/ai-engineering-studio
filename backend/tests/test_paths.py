"""Unit tests for filename and path safety helpers."""
import pytest

from app.core.errors import ValidationError
from app.core.paths import codegen_path, dataset_path, safe_filename

BS = "\\"


@pytest.mark.parametrize("raw, expected", [
    ("data.csv", "data.csv"),
    ("../../evil.csv", "evil.csv"),
    (BS.join(["..", "..", "evil.csv"]), "evil.csv"),
    (BS.join(["C:", "Users", "me", "data.csv"]), "data.csv"),
    (".hidden.csv", "hidden.csv"),
    ("a b$%.csv", "a b_.csv"),
    ("", "file"),
    ("..", "file"),
])
def test_safe_filename(raw, expected):
    assert safe_filename(raw) == expected


def test_dataset_path_refuses_traversal():
    with pytest.raises(ValidationError):
        dataset_path("p1", "../../escape.csv")


def test_codegen_path_refuses_traversal():
    with pytest.raises(ValidationError):
        codegen_path("p1", "../..", "x.py")
    with pytest.raises(ValidationError):
        codegen_path("p1", "data_engineering", "../../x.py")
