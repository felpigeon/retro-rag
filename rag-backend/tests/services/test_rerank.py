"""
Tests for reranking functionality.
"""

import pytest
from unittest.mock import MagicMock
from aioresponses import aioresponses

from src.services.rerank import rerank


class TestRerank:
    """Tests for rerank service."""

    @pytest.fixture
    def mock_docs(self):
        """Create sample documents for testing."""
        doc1 = MagicMock()
        doc1.payload = {"text": "This is the first document about testing."}

        doc2 = MagicMock()
        doc2.payload = {"text": "Second document contains reranking information."}

        doc3 = MagicMock()
        doc3.payload = {"text": "Third document is not relevant."}

        return [doc1, doc2, doc3]

    @pytest.fixture
    def rerank_url(self):
        """Return the rerank service URL."""
        return 'http://gpu-service:5001/rerank'

    @pytest.mark.asyncio
    async def test_rerank_success(self, mock_docs, rerank_url):
        """Test successful reranking of documents."""
        # Setup
        query = "reranking test"
        expected_response = {
            'ranked_documents': [
                {"text": "Second document contains reranking information.", "score": 0.92},
                {"text": "This is the first document about testing.", "score": 0.75},
                {"text": "Third document is not relevant.", "score": 0.32}
            ]
        }
        expected_payload = {
            'query': 'reranking test',
            'documents': mock_docs
        }

        # Use aioresponses to mock the HTTP response
        with aioresponses() as m:
            m.post(rerank_url, status=200, payload=expected_response)

            # Execute
            result = await rerank(query, mock_docs)

            # Assert
            # Check results were returned correctly
            assert len(result) == 3
            assert result[0]["text"] == "Second document contains reranking information."
            assert result[0]["score"] == 0.92
            assert result[1]["text"] == "This is the first document about testing."
            assert result[1]["score"] == 0.75
            assert result[2]["text"] == "Third document is not relevant."
            assert result[2]["score"] == 0.32

    @pytest.mark.asyncio
    async def test_rerank_error(self, mock_docs, rerank_url):
        """Test error handling during reranking."""
        # Setup
        query = "reranking test"
        error_message = "Internal server error"

        # Use aioresponses to mock the HTTP error response
        with aioresponses() as m:
            m.post(rerank_url, status=500, body=error_message)

            # Execute & Assert
            with pytest.raises(Exception) as excinfo:
                await rerank(query, mock_docs)

            # Check that the correct error was raised
            assert "GPU service reranking failed" in str(excinfo.value)
            assert error_message in str(excinfo.value)

