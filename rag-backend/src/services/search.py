from qdrant_client import models

from src.services.entity import entity_extractor
from src.services.rerank import rerank
from src.utils.logger import get_logger
from src.services.search_strategies import BM25SearchStrategy, DenseSearchStrategy, HybridSearchStrategy

logger = get_logger(__name__)


def create_entity_filter(filter):
    """Creates a Qdrant filter object based on extracted entities.

    Args:
        filter: A dictionary where keys are entity types and values are lists of entity names.

    Returns:
        models.Filter: A Qdrant filter object or None if no filter is provided.
    """
    if filter is None:
        return None

    return models.Filter(
        should=[
            models.FieldCondition(
                key=f"{entity_type}[]",
                match=models.MatchAny(any=entity_names)
            )
            for entity_type, entity_names in filter.items()
        ]
    )


async def process_search_results(docs, query, k, do_rerank):
    """Process and format search results.

    Args:
        docs: List of document points from Qdrant.
        query: The original search query.
        k: Number of results to return.
        do_rerank: Whether to apply reranking.

    Returns:
        list: A list of formatted documents.
    """
    formatted_docs = [
        {"id": doc.id, "text": doc.payload["text"]}
        for doc in docs
    ]

    if do_rerank and len(formatted_docs):
        logger.info("Applying reranking")
        formatted_docs = (await rerank(query, formatted_docs))[:k]

    logger.info(f"Search completed, returning {len(formatted_docs)} documents")
    return formatted_docs


async def search(query, method, k=5, filter_by_entity=False, do_rerank=False):
    """Executes a search using the specified method and optional filters.

    Args:
        query: The search query text.
        method: The search method ('bm25', 'dense', or 'hybrid').
        k: The number of top results to return.
        filter_by_entity: Whether to filter results by extracted entities.
        do_rerank: Whether to rerank the results.

    Returns:
        list: A list of documents matching the query.

    Raises:
        ValueError: If the specified search method is invalid.
    """
    logger.info(f"Starting search with method={method}, k={k}, filter_by_entity={filter_by_entity}, do_rerank={do_rerank}")

    # Prepare filter if needed
    filter = None
    if filter_by_entity:
        entities = await entity_extractor.extract_entities(query)
        if entities:
            logger.info(f"Extracted entities for filtering: {entities}")
            entities = entity_extractor.match_entities(entities)
            logger.info(f"Matched entities: {entities}")
            filter = create_entity_filter(entities)

    # Calculate effective k for reranking
    k_eff = k*3 if do_rerank else k

    # Select strategy based on method
    strategies = {
        "bm25": BM25SearchStrategy(),
        "dense": DenseSearchStrategy(),
        "hybrid": HybridSearchStrategy()
    }

    if method not in strategies:
        logger.error(f"Invalid search method: {method}")
        raise ValueError("Invalid search method. Choose from 'bm25', 'dense', or 'hybrid'.")

    # Execute search with selected strategy
    docs = await strategies[method].execute_search(query, k=k_eff, filter=filter)

    # Process and return results
    return await process_search_results(docs, query, k, do_rerank)
