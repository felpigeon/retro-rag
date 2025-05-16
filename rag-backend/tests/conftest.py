"""
Fixtures and configuration for pytest tests.
This file contains shared fixtures that can be used across test modules.
"""

import os
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Any

from flask import Flask
from flask.testing import FlaskClient

from src.app import app
from src.models.requests import IngestRequest, QuestionRequest


@pytest.fixture
def flask_app():
    """Create a Flask application for testing."""
    app.config.update({
        "TESTING": True,
        "DEBUG": True,
    })
    yield app


@pytest.fixture
def client(flask_app):
    """Create a test client for the Flask application."""
    with flask_app.test_client() as client:
        yield client

