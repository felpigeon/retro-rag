import aiohttp
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def rerank(query, docs):
    """
    Rerank a list of documents based on their relevance to a query using a GPU service.

    Args:
        query (str): The query string to rank the documents against.
        docs (list): A list of documents, where each document is expected to have a `payload`
            dictionary containing a "text" key.

    Returns:
        list: A list of reranked documents in the order of their relevance.

    Raises:
        Exception: If the GPU service fails to rerank the documents.
    """
    logger.info(f"Reranking {len(docs)} documents using GPU service")

    async with aiohttp.ClientSession() as session:
        async with session.post(
            'http://gpu-service:5001/rerank',
            json={
                'query': query,
                'documents': docs
            }
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"GPU service reranking failed: {error_text}")
                raise Exception(f"GPU service reranking failed: {error_text}")

            result = await response.json()
            reranked_docs = result['ranked_documents']
            logger.debug(f"Reranking complete, returned {len(reranked_docs)} documents")

            return reranked_docs
