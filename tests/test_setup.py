"""Smoke test: the pinned environment is what the project expects."""
import sys


def test_python_is_3_12() -> None:
    assert sys.version_info[:2] == (3, 12)


def test_core_dependencies_import() -> None:
    import pydantic
    import yaml

    assert pydantic.VERSION
    assert yaml.__version__


def test_src_package_importable() -> None:
    import src

    assert src.__name__ == "src"
