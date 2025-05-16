from abc import ABC, abstractmethod
from qdrant_client import models

from src.services.embeddings import get_dense_embeddings, get_sparse_embeddings
from src.services.qdrant import get_qdrant_client
from src.utils.logger import get_logger

logger = get_logger(__name__)

class SearchStrategy(ABC):
    """Base strategy class for different search methods.

    This abstract class defines the interface for all search strategies.
    """

    @abstractmethod
    async def execute_search(self, query, k=5, filter=None):
        """Execute search using the specific strategy.

        Args:
            query: The search query text or embedding.
            k: The number of results to retrieve.
            filter: Optional filter to apply to the search.

        Returns:
            A list of matching document points.
        """
        pass

    async def handle_insufficient_results(self, query, k, current_results, filter):
        """Handle case where filter returns insufficient results.

        Args:
            query: The search query text or embedding.
            k: The desired number of results.
            current_results: The results obtained with filter.
            filter: The filter that was applied.

        Returns:
            A potentially extended list of document points.
        """
        if filter is not None and len(current_results) < k:
            logger.info("Insufficient results with filter, retrying without filter")
            additional_results = await self.execute_search(query, k, filter=None)
            current_results.extend(additional_results)
            return current_results[:k]
        return current_results


class BM25SearchStrategy(SearchStrategy):
    """Strategy for BM25 (sparse vector) search."""

    async def execute_search(self, query, k=5, filter=None):
        """Execute a BM25 search using sparse vectors.

        Args:
            query: The search query text.
            k: The number of results to retrieve.
            filter: Optional filter to apply to the search.

        Returns:
            A list of matching document points.
        """
        logger.info(f"Performing BM25 search with query: {query}, k={k}, filter={filter}")
        vec = get_sparse_embeddings(query)

        qdrant_client = await get_qdrant_client()
        docs = await qdrant_client.query_points(
            collection_name="dev_articles",
            using="text",
            query=models.SparseVector(
                indices=vec.indices,
                values=vec.values,
            ),
            query_filter=filter,
            limit=k,
        )

        docs = list(docs.points)
        logger.info(f"BM25 search returned {len(docs)} documents")

        return await self.handle_insufficient_results(query, k, docs, filter)


class DenseSearchStrategy(SearchStrategy):
    """Strategy for dense vector search."""

    async def execute_search(self, query, k=5, filter=None):
        """Execute a dense vector search.

        Args:
            query: The search query text.
            k: The number of results to retrieve.
            filter: Optional filter to apply to the search.

        Returns:
            A list of matching document points.
        """
        logger.info(f"Performing dense search with query: {query}, k={k}, filter={filter}")
        embedding = get_dense_embeddings(query)

        qdrant_client = await get_qdrant_client()
        docs = await qdrant_client.query_points(
            collection_name="dev_articles",
            using="embedding",
            query=embedding,
            query_filter=filter,
            limit=k,
        )

        docs = list(docs.points)
        logger.info(f"Dense search returned {len(docs)} documents")

        return await self.handle_insufficient_results(query, k, docs, filter)


class HybridSearchStrategy(SearchStrategy):
    """Strategy for hybrid search combining sparse and dense vectors."""

    async def execute_search(self, query, k=5, filter=None):
        """Execute a hybrid search combining BM25 and dense vector approaches.

        Args:
            query: The search query text.
            k: The number of results to retrieve.
            filter: Optional filter to apply to the search.

        Returns:
            A list of matching document points.
        """
        logger.info(f"Performing hybrid search with query: {query}, k={k}, filter={filter}")
        sparse = get_sparse_embeddings(query)[0]
        dense = get_dense_embeddings(query)[0]

        qdrant_client = await get_qdrant_client()
        docs = await qdrant_client.query_points(
            collection_name="dev_articles",
            query=models.FusionQuery(
                fusion=models.Fusion.RRF
            ),
            prefetch=[
                {
                    "query": models.SparseVector(
                        indices=sparse.indices,
                        values=sparse.values,
                    ),
                    "using": "text",
                    "limit": k,
                    "filter": filter,
                },
                {
                    "query": dense,
                    "using": "embedding",
                    "limit": k,
                    "filter": filter,
                }
            ],
            limit=k,
        )

        docs = list(docs.points)
        logger.info(f"Hybrid search returned {len(docs)} documents")

        return await self.handle_insufficient_results(query, k, docs, filter)
