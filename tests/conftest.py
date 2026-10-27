import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import create_app  # noqa: E402
from config import TestConfig  # noqa: E402


@pytest.fixture()
def app(tmp_path):
    """App whose experiment folders live in a temp dir, so tests never touch real outputs/uploads."""

    class Cfg(TestConfig):
        EXPERIMENTS_DIR = tmp_path / "experiments"

    return create_app(Cfg)


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def real_client():
    """App using the real experiments folder (saved notebook outputs). Read-only usage."""
    return create_app(TestConfig).test_client()
