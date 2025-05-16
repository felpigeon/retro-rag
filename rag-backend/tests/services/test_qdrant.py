"""
Tests for Qdrant vector database operations.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock, call
import numpy as np

from src.services.qdrant import create_articles_collection, upsert_articles, get_qdrant_client, COLLECTION_NAME
from src.models.document import Document
from qdrant_client.models import Distance, VectorParams, Modifier, SparseIndexParams, PointStruct


class TestQdrantService:
    """Tests for Qdrant service functions."""

    @pytest.fixture
    def mock_qdrant_client(self):
        """Mock the Qdrant client."""
        client_instance = AsyncMock()
        client_instance.collection_exists = AsyncMock()
        client_instance.create_collection = AsyncMock()
        client_instance.upsert = AsyncMock()

        with patch("src.services.qdrant.AsyncQdrantClient", return_value=client_instance):
            with patch("src.services.qdrant.get_qdrant_client", return_value=client_instance):
                yield client_instance

    @pytest.fixture
    def sample_sparse_vector(self):
        """Create a sample sparse vector for testing."""
        sparse_vec = MagicMock()
        sparse_vec.indices = [1, 4, 10, 20, 50]
        sparse_vec.values = [0.5, 0.8, 0.6, 0.9, 0.3]
        return sparse_vec

    @pytest.fixture
    def sample_dense_vector(self):
        """Create a sample dense vector for testing."""
        return [0.1, 0.2, 0.3, 0.4, 0.5] * 153  # 768-dimensional vector

    @pytest.fixture
    def sample_document(self, sample_sparse_vector, sample_dense_vector):
        """Create a sample Document for testing."""
        doc = MagicMock(spec=Document)
        doc.doc_id = "test123"
        doc.text = "This is a test document about Super Mario Bros on Nintendo Switch."
        doc.entities = {
            "Game": ["Super Mario Bros"],
            "Console": ["Nintendo Switch"],
            "Publisher": ["Nintendo"]
        }
        doc.sparse_vec = sample_sparse_vector
        doc.dense_vec = sample_dense_vector
        return doc

    @pytest.mark.asyncio
    async def test_create_articles_collection_when_not_exists(self, mock_qdrant_client):
        """Test creating a collection when it doesn't exist."""
        # Setup
        mock_qdrant_client.collection_exists.return_value = False

        # Execute
        await create_articles_collection()

        # Assert
        mock_qdrant_client.collection_exists.assert_called_once_with(collection_name="articles")
        mock_qdrant_client.create_collection.assert_called_once()

        # Check collection parameters
        call_args = mock_qdrant_client.create_collection.call_args[1]
        assert call_args["collection_name"] == COLLECTION_NAME
        assert "embedding" in call_args["vectors_config"]
        assert call_args["vectors_config"]["embedding"].size == 768
        assert call_args["vectors_config"]["embedding"].distance == Distance.COSINE
        assert "text" in call_args["sparse_vectors_config"]

    @pytest.mark.asyncio
    async def test_create_articles_collection_when_exists(self, mock_qdrant_client):
        """Test creating a collection when it already exists."""
        # Setup
        mock_qdrant_client.collection_exists.return_value = True

        # Execute
        await create_articles_collection()

        # Assert
        mock_qdrant_client.collection_exists.assert_called_once_with(collection_name="articles")
        mock_qdrant_client.create_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_upsert_articles_single_document(self, mock_qdrant_client, sample_document):
        """Test upserting a single article document."""
        # Setup
        with patch("src.services.qdrant.create_articles_collection") as mock_create:
            # Execute
            await upsert_articles(sample_document)

            # Assert
            mock_create.assert_called_once()
            mock_qdrant_client.upsert.assert_called_once()

            # Check upsert parameters
            call_args = mock_qdrant_client.upsert.call_args[1]
            assert call_args["collection_name"] == COLLECTION_NAME
            assert len(call_args["points"]) == 1

            point = call_args["points"][0]
            assert point.id == sample_document.doc_id
            assert point.payload["text"] == sample_document.text
            assert point.payload["Game"] == ["Super Mario Bros"]
            assert point.payload["Console"] == ["Nintendo Switch"]
            assert point.payload["Publisher"] == ["Nintendo"]

            assert "embedding" in point.vector
            assert point.vector["embedding"] == sample_document.dense_vec

            assert "text" in point.vector
            assert point.vector["text"].indices == sample_document.sparse_vec.indices
            assert point.vector["text"].values == sample_document.sparse_vec.values

            assert call_args["wait"] is True

    @pytest.mark.asyncio
    async def test_upsert_articles_multiple_documents(self, mock_qdrant_client, sample_document):
        """Test upserting multiple article documents."""
        # Setup
        doc1 = sample_document

        doc2 = MagicMock(spec=Document)
        doc2.doc_id = "test456"
        doc2.text = "Another test document."
        doc2.entities = {"Character": ["Luigi"]}
        doc2.sparse_vec = MagicMock()
        doc2.sparse_vec.indices = [2, 5, 15]
        doc2.sparse_vec.values = [0.3, 0.7, 0.5]
        doc2.dense_vec = [0.2, 0.3, 0.4] * 256

        with patch("src.services.qdrant.create_articles_collection") as mock_create:
            # Execute
            await upsert_articles([doc1, doc2])

            # Assert
            mock_create.assert_called_once()
            mock_qdrant_client.upsert.assert_called_once()

            # Check upsert parameters
            call_args = mock_qdrant_client.upsert.call_args[1]
            assert call_args["collection_name"] == COLLECTION_NAME
            assert len(call_args["points"]) == 2

            # Check first document
            assert call_args["points"][0].id == doc1.doc_id

            # Check second document
            assert call_args["points"][1].id == doc2.doc_id
            assert call_args["points"][1].payload["Character"] == ["Luigi"]

    @pytest.mark.asyncio
    async def test_upsert_articles_empty_list(self, mock_qdrant_client):
        """Test upserting an empty list of documents."""
        # Setup
        with patch("src.services.qdrant.create_articles_collection") as mock_create:
            # Execute
            await upsert_articles([])

            # Assert
            mock_create.assert_called_once()
            mock_qdrant_client.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_qdrant_client(self):
        """Test getting a Qdrant client."""
        with patch("src.services.qdrant.AsyncQdrantClient") as mock_client_class:
            # Setup
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Execute
            client = await get_qdrant_client()

            # Assert
            assert client == mock_client
            mock_client_class.assert_called_once()
