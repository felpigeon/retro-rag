import pytest
from src.app import app


@pytest.fixture
def client():
    app.config['TESTING'] = True

    client = app.test_client()
    with app.test_request_context():
        yield client

