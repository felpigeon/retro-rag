import pytest
import os
import uuid
from unittest.mock import patch, MagicMock, AsyncMock, call

import duckdb

from src.services.ingest import ingest_documents, insert_entities
from src.models.requests import IngestRequest
from src.models.document import Document, SparseVector


class TestIngestDocuments:
    """Tests for the ingest_documents function."""

    @pytest.fixture
    def mock_uuid(self):
        """Create a mock UUID."""
        test_uuids = ["test-uuid-123", "test-uuid-456", "test-uuid-789"]
        mock_uuids = [MagicMock() for _ in test_uuids]
        for mock_uuid, test_uuid in zip(mock_uuids, test_uuids):
            mock_uuid.__str__.return_value = test_uuid
        return mock_uuids, test_uuids

    @pytest.mark.asyncio
    @patch("src.services.ingest.insert_entities")
    @patch("src.services.ingest.get_dense_embeddings")
    @patch("src.services.ingest.get_sparse_embeddings")
    @patch("src.services.ingest.upsert_articles")
    @patch("src.services.ingest.entity_extractor")
    @patch("uuid.uuid4")
    async def test_ingest_documents_with_preextracted_entities(
        self, mock_uuid4, mock_extractor, mock_upsert,
        mock_get_sparse, mock_get_dense, mock_insert_entities,
        mock_uuid
    ):
        """Test ingest_documents with pre-extracted entities."""
        # Setup mocks
        mock_uuid_objs, test_uuids = mock_uuid
        mock_uuid4.side_effect = mock_uuid_objs

        mock_get_dense.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

        sparse_return_values = [
            MagicMock(indices=[1, 2, 3], values=[0.1, 0.2, 0.3]),
            MagicMock(indices=[4, 5, 6], values=[0.4, 0.5, 0.6])
        ]
        mock_get_sparse.return_value = sparse_return_values

        # Test data
        documents_request = [
            IngestRequest(
                text="This is a test document about Final Fantasy Adventure.",
                entities={
                    "game": ["Final Fantasy Adventure"],
                    "console": ["Game Boy"]
                }
            ),
            IngestRequest(
                text="This is another test document about Zelda.",
                entities={
                    "game": ["Zelda"],
                    "console": ["NES"]
                }
            )
        ]

        # Execute function
        result = await ingest_documents(documents_request)

        # Verify results
        assert isinstance(result, list)
        assert len(result) == 2

        # Check function calls
        mock_extractor.extract_entities.assert_not_called()  # Should not be called since entities are provided
        mock_insert_entities.assert_called_once()

        # Verify the entities being inserted
        called_entities = mock_insert_entities.call_args[0][0]
        assert len(called_entities) == 2
        assert called_entities[0] == documents_request[0].entities
        assert called_entities[1] == documents_request[1].entities

        mock_get_dense.assert_called_once_with([doc.text for doc in documents_request])
        mock_get_sparse.assert_called_once_with([doc.text for doc in documents_request])

        # Check that Document objects were created correctly
        expected_documents = [
            Document(
                doc_id=test_uuids[0],
                text=documents_request[0].text,
                dense_vec=mock_get_dense.return_value[0],
                sparse_vec=SparseVector(
                    indices=sparse_return_values[0].indices,
                    values=sparse_return_values[0].values
                ),
                entities=documents_request[0].entities
            ),
            Document(
                doc_id=test_uuids[1],
                text=documents_request[1].text,
                dense_vec=mock_get_dense.return_value[1],
                sparse_vec=SparseVector(
                    indices=sparse_return_values[1].indices,
                    values=sparse_return_values[1].values
                ),
                entities=documents_request[1].entities
            )
        ]

        # Check that upsert_articles was called with the expected documents
        mock_upsert.assert_called_once()
        called_docs = mock_upsert.call_args[0][0]
        assert len(called_docs) == 2
        for i, doc in enumerate(called_docs):
            assert doc.text == expected_documents[i].text
            assert doc.dense_vec == expected_documents[i].dense_vec
            assert doc.sparse_vec.indices == expected_documents[i].sparse_vec.indices
            assert doc.sparse_vec.values == expected_documents[i].sparse_vec.values
            assert doc.entities == expected_documents[i].entities

    @pytest.mark.asyncio
    @patch("src.services.ingest.insert_entities")
    @patch("src.services.ingest.get_dense_embeddings")
    @patch("src.services.ingest.get_sparse_embeddings")
    @patch("src.services.ingest.upsert_articles")
    @patch("src.services.ingest.entity_extractor")
    @patch("uuid.uuid4")
    async def test_ingest_documents_with_extraction(
        self, mock_uuid4, mock_extractor, mock_upsert,
        mock_get_sparse, mock_get_dense, mock_insert_entities,
        mock_uuid
    ):
        """Test ingest_documents with automatic entity extraction."""
        # Setup mocks
        mock_uuid_objs, test_uuids = mock_uuid
        mock_uuid4.side_effect = mock_uuid_objs

        mock_get_dense.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

        sparse_return_values = [
            MagicMock(indices=[1, 2, 3], values=[0.1, 0.2, 0.3]),
            MagicMock(indices=[4, 5, 6], values=[0.4, 0.5, 0.6])
        ]
        mock_get_sparse.return_value = sparse_return_values

        # Mock entity extraction
        extracted_entities = [
            {"game": ["Final Fantasy"], "console": ["SNES"]},
            {"game": ["Zelda"], "console": ["NES"]}
        ]
        mock_extractor.extract_entities = AsyncMock(side_effect=extracted_entities)
        mock_extractor.entity_types = ["game", "console", "publisher"]

        # Test data
        documents_request = [
            IngestRequest(
                text="This is a test document about Final Fantasy Adventure.",
            ),
            IngestRequest(
                text="This is another test document about Zelda.",
            )
        ]

        # Execute function
        result = await ingest_documents(documents_request)

        # Verify results
        assert isinstance(result, list)
        assert len(result) == 2

        # Check function calls
        assert mock_extractor.extract_entities.call_count == 2
        mock_extractor.extract_entities.assert_has_calls([
            call(documents_request[0].text),
            call(documents_request[1].text)
        ])

        mock_insert_entities.assert_called_once_with(extracted_entities)
        mock_get_dense.assert_called_once_with([doc.text for doc in documents_request])
        mock_get_sparse.assert_called_once_with([doc.text for doc in documents_request])

        # Check that upsert_articles was called with the expected documents
        mock_upsert.assert_called_once()
        called_docs = mock_upsert.call_args[0][0]
        for i, doc in enumerate(called_docs):
            assert doc.text == documents_request[i].text
            assert doc.dense_vec == mock_get_dense.return_value[i]
            assert doc.entities == extracted_entities[i]
