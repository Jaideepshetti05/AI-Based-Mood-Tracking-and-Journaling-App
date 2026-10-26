import os
import sys
from pathlib import Path

os.environ["MOOD_DATABASE_URL"] = "sqlite:///./test_mood_journal.db"
os.environ["MOOD_ENABLE_EMOTION_MODEL"] = "false"
sys.path.insert(0, str(Path(__file__).parents[1]))

import pytest
from fastapi.testclient import TestClient

from app.database import Base, engine
from app.main import app


@pytest.fixture(autouse=True)
def reset_db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client
