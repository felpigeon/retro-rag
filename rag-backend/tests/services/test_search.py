"""
Tests for search functionality.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock, call
from qdrant_client import models

from src.services.search import create_entity_filter, search, process_search_results
from src.services.search_strategies import (
    BM25SearchStrategy, DenseSearchStrategy, HybridSearchStrategy, SearchStrategy
)


class TestSearchFilters:
    """Tests for search filter creation functionality."""

    def test_create_entity_filter_with_entities(self):
        """Test creating a filter with entities."""
        # Setup
        entities = {
            "game": ["Super Mario Bros", "Zelda"],
            "console": ["Nintendo Switch"]
        }

        # Execute
        result = create_entity_filter(entities)

        # Assert
        assert isinstance(result, models.Filter)
        assert len(result.should) == 2

        # Check game filter
        game_condition = result.should[0]
        assert game_condition.key == "game[]"
        assert isinstance(game_condition.match, models.MatchAny)
        assert game_condition.match.any == ["Super Mario Bros", "Zelda"]

        # Check console filter
        console_condition = result.should[1]
        assert console_condition.key == "console[]"
        assert isinstance(console_condition.match, models.MatchAny)
        assert console_condition.match.any == ["Nintendo Switch"]

    def test_create_entity_filter_none(self):
        """Test creating a filter with None input."""
        # Execute
        result = create_entity_filter(None)

        # Assert
        assert result is None


class TestSearchStrategies:
    """Tests for individual search strategies."""

    @pytest.fixture
    def mock_sparse_embeddings(self):
        """Mock sparse embeddings."""
        sparse_vec = MagicMock()
        sparse_vec.indices = [1, 4, 10, 20]
        sparse_vec.values = [0.5, 0.8, 0.6, 0.9]
        return sparse_vec

    @pytest.fixture
    def mock_dense_embeddings(self):
        """Mock dense embeddings."""
        return [0.1, 0.2, 0.3, 0.4, 0.5] * 153  # 768-dimensional vector

    @pytest.fixture
    def mock_search_results(self):
        """Mock search results from Qdrant."""
        points = []
        for i in range(5):
            point = MagicMock()
            point.id = f"doc{i+1}"
            point.payload = {"text": f"Document content {i+1}"}
            points.append(point)

        result = MagicMock()
        result.points = points
        return result

    @pytest.mark.asyncio
    async def test_bm25_strategy(self, mock_sparse_embeddings, mock_search_results):
        """Test BM25 search strategy."""
        # Setup
        query = "game console comparison"
        entities = {"game": ["Super Mario Bros"]}
        filter = create_entity_filter(entities)
        strategy = BM25SearchStrategy()

        with patch("src.services.search_strategies.get_sparse_embeddings", return_value=mock_sparse_embeddings) as mock_get_embeddings:
            with patch("src.services.search_strategies.get_qdrant_client") as mock_get_qdrant:
                mock_qdrant_client = AsyncMock()
                mock_qdrant_client.query_points = AsyncMock(return_value=mock_search_results)
                mock_get_qdrant.return_value = mock_qdrant_client

                # Execute
                results = await strategy.execute_search(query, k=5, filter=filter)

                # Assert
                mock_get_embeddings.assert_called_once_with(query)
                mock_qdrant_client.query_points.assert_called_once()

                # Check if the correct params were passed to query_points
                args = mock_qdrant_client.query_points.call_args[1]
                assert args["collection_name"] == "dev_articles"
                assert args["using"] == "text"
                assert isinstance(args["query"], models.SparseVector)
                assert args["query"].indices == mock_sparse_embeddings.indices
                assert args["query"].values == mock_sparse_embeddings.values
                assert args["limit"] == 5
                assert args["query_filter"] == filter

                # Check results
                assert len(results) == 5
                assert results[0].id == "doc1"
                assert results[0].payload["text"] == "Document content 1"

    @pytest.mark.asyncio
    async def test_dense_strategy(self, mock_dense_embeddings, mock_search_results):
        """Test dense vector search strategy."""
        # Setup
        query = "game console comparison"
        entities = {"console": ["Nintendo Switch"]}
        filter = create_entity_filter(entities)
        strategy = DenseSearchStrategy()

        with patch("src.services.search_strategies.get_dense_embeddings", return_value=mock_dense_embeddings) as mock_get_embeddings:
            with patch("src.services.search_strategies.get_qdrant_client") as mock_get_qdrant:
                mock_qdrant_client = AsyncMock()
                mock_qdrant_client.query_points = AsyncMock(return_value=mock_search_results)
                mock_get_qdrant.return_value = mock_qdrant_client

                # Execute
                results = await strategy.execute_search(query, k=5, filter=filter)

                # Assert
                mock_get_embeddings.assert_called_once_with(query)
                mock_qdrant_client.query_points.assert_called_once()

                # Check if the correct params were passed to query_points
                args = mock_qdrant_client.query_points.call_args[1]
                assert args["collection_name"] == "dev_articles"
                assert args["using"] == "embedding"
                assert args["query"] == mock_dense_embeddings
                assert args["limit"] == 5
                assert args["query_filter"] == filter

                # Check results
                assert len(results) == 5
                assert results[0].id == "doc1"
                assert results[0].payload["text"] == "Document content 1"

    @pytest.mark.asyncio
    async def test_hybrid_strategy(self, mock_sparse_embeddings, mock_dense_embeddings, mock_search_results):
        """Test hybrid search strategy."""
        # Setup
        query = "game console comparison"
        entities = {"publisher": ["Nintendo"]}
        filter = create_entity_filter(entities)
        strategy = HybridSearchStrategy()

        with patch("src.services.search_strategies.get_sparse_embeddings", return_value=[mock_sparse_embeddings]) as mock_get_sparse:
            with patch("src.services.search_strategies.get_dense_embeddings", return_value=[mock_dense_embeddings]) as mock_get_dense:
                with patch("src.services.search_strategies.get_qdrant_client") as mock_get_qdrant:
                    mock_qdrant_client = AsyncMock()
                    mock_qdrant_client.query_points = AsyncMock(return_value=mock_search_results)
                    mock_get_qdrant.return_value = mock_qdrant_client

                    # Execute
                    results = await strategy.execute_search(query, k=5, filter=filter)

                    # Assert
                    mock_get_sparse.assert_called_once_with(query)
                    mock_get_dense.assert_called_once_with(query)
                    mock_qdrant_client.query_points.assert_called_once()

                    # Check if the correct params were passed to query_points
                    args = mock_qdrant_client.query_points.call_args[1]
                    assert args["collection_name"] == "dev_articles"
                    assert isinstance(args["query"], models.FusionQuery)
                    assert args["query"].fusion == models.Fusion.RRF

                    # Check prefetch strategies
                    prefetch = args["prefetch"]
                    assert len(prefetch) == 2
                    assert prefetch[0]["using"] == "text"
                    assert prefetch[1]["using"] == "embedding"

                    # Check results
                    assert len(results) == 5
                    assert results[0].id == "doc1"
                    assert results[0].payload["text"] == "Document content 1"

class TestMainSearch:
    """Tests for the main search function."""

    @pytest.fixture
    def mock_search_results(self):
        """Mock search results from individual search strategies."""
        return [
            MagicMock(id="doc1", payload={"text": "Document 1 content"}),
            MagicMock(id="doc2", payload={"text": "Document 2 content"}),
            MagicMock(id="doc3", payload={"text": "Document 3 content"}),
            MagicMock(id="doc4", payload={"text": "Document 4 content"}),
            MagicMock(id="doc5", payload={"text": "Document 5 content"}),
        ]

    @pytest.mark.asyncio
    async def test_process_search_results(self, mock_search_results):
        """Test processing of search results."""
        query = "test query"

        # Test without reranking
        results = await process_search_results(mock_search_results, query, 5, False)

        assert len(results) == 5
        assert results[0]["id"] == "doc1"
        assert results[0]["text"] == "Document 1 content"

        # Test with reranking
        reranked_docs = [
            {"text": "Document 3 content", "score": 0.95, "id": "doc3"},
            {"text": "Document 1 content", "score": 0.85, "id": "doc1"},
            {"text": "Document 5 content", "score": 0.75, "id": "doc5"},
        ]

        with patch("src.services.search.rerank", AsyncMock(return_value=reranked_docs)) as mock_rerank:
            results = await process_search_results(mock_search_results, query, 3, True)

            mock_rerank.assert_called_once()
            assert len(results) == 3
            assert results[0]["id"] == "doc3"
            assert results[1]["id"] == "doc1"
            assert results[2]["id"] == "doc5"

    @pytest.mark.asyncio
    async def test_search_with_bm25_strategy(self, mock_search_results):
        """Test search function with BM25 strategy."""
        # Setup
        query = "game console comparison"

        with patch("src.services.search.entity_extractor.extract_entities", AsyncMock(return_value=None)):
            with patch.object(BM25SearchStrategy, "execute_search", AsyncMock(return_value=mock_search_results)):
                with patch("src.services.search.process_search_results", AsyncMock(return_value=[
                    {"id": "doc1", "text": "Document 1 content"},
                    {"id": "doc2", "text": "Document 2 content"},
                    {"id": "doc3", "text": "Document 3 content"},
                ])) as mock_process:

                    # Execute
                    results = await search(query, method="bm25", k=3, filter_by_entity=False, do_rerank=False)

                    # Assert
                    assert len(results) == 3
                    assert results[0]["id"] == "doc1"
                    assert results[0]["text"] == "Document 1 content"
                    mock_process.assert_called_once_with(mock_search_results, query, 3, False)

    @pytest.mark.asyncio
    async def test_search_with_entity_filter(self, mock_search_results):
        """Test search function with entity filtering enabled."""
        # Setup
        query = "game console comparison"
        entities = {"game": ["Super Mario Bros"]}

        with patch("src.services.search.entity_extractor.extract_entities", AsyncMock(return_value=entities)) as mock_extract:
            with patch("src.services.search.entity_extractor.match_entities", return_value=entities) as mock_match:
                with patch.object(HybridSearchStrategy, "execute_search", AsyncMock(return_value=mock_search_results)):
                    with patch("src.services.search.process_search_results", AsyncMock(return_value=[
                        {"id": "doc1", "text": "Document 1 content"},
                        {"id": "doc2", "text": "Document 2 content"},
                    ])) as mock_process:

                        # Execute
                        results = await search(query, method="hybrid", k=5, filter_by_entity=True, do_rerank=False)

                        # Assert
                        mock_extract.assert_called_once_with(query)
                        mock_match.assert_called_once_with(entities)
                        assert len(results) == 2
                        assert results[0]["id"] == "doc1"
                        assert results[1]["id"] == "doc2"

    @pytest.mark.asyncio
    async def test_search_with_reranking(self, mock_search_results):
        """Test search function with reranking enabled."""
        # Setup
        query = "game console comparison"

        # Mock reranked results
        reranked_results = [
            {"id": "doc3", "text": "Document 3 content"},
            {"id": "doc1", "text": "Document 1 content"},
        ]

        with patch("src.services.search.entity_extractor.extract_entities", AsyncMock(return_value=None)):
            with patch.object(DenseSearchStrategy, "execute_search", AsyncMock(return_value=mock_search_results)):
                with patch("src.services.search.process_search_results", AsyncMock(return_value=reranked_results)) as mock_process:

                    # Execute
                    results = await search(query, method="dense", k=2, filter_by_entity=False, do_rerank=True)

                    # Assert
                    # Should request 3x more docs for reranking
                    mock_process.assert_called_once_with(mock_search_results, query, 2, True)
                    assert len(results) == 2
                    assert results[0]["id"] == "doc3"
                    assert results[1]["id"] == "doc1"

    @pytest.mark.asyncio
    async def test_search_invalid_method(self):
        """Test search function with invalid method."""
        # Setup
        query = "game console comparison"

        # Execute & Assert
        with pytest.raises(ValueError) as excinfo:
            await search(query, method="invalid_method", k=5)

        assert "Invalid search method" in str(excinfo.value)
