import os
from google import genai
from fastembed import SparseTextEmbedding
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_dense_embeddings(documents):
    """
    Generate dense embeddings for the given documents using Google's embedding model.

    Args:
        documents (str or list of str): A single document or a list of documents to embed.

    Returns:
        list or list of lists: Dense embeddings for the input documents. If a single document is provided,
        returns a single embedding. Otherwise, returns a list of embeddings.
    """
    logger.info("Generating dense embeddings")
    client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))

    if isinstance(documents, str):
        documents = [documents]

    embeddings = []
    batch_size = 100

    logger.debug(f"Processing {len(documents)} documents in batches of {batch_size}")
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        logger.debug(f"Processing batch {i//batch_size + 1} with {len(batch)} documents")

        result = client.models.embed_content(
            model="text-embedding-004",
            contents=batch,
        )

        embeddings += [e.values for e in result.embeddings]

    logger.debug(f"Returning {len(embeddings)} embeddings")
    logger.info("Dense embeddings generation complete")

    return embeddings


def get_sparse_embeddings(documents):
    """
    Generate sparse embeddings for the given documents using a BM25 model.

    Args:
        documents (str or list of str): A single document or a list of documents to embed.

    Returns:
        list or list of lists: Sparse embeddings for the input documents. If a single document is provided,
        returns a single embedding. Otherwise, returns a list of embeddings.
    """
    logger.info("Generating sparse embeddings with BM25")
    bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25")

    if isinstance(documents, str):
        documents = [documents]

    logger.debug(f"Processing {len(documents)} documents")
    embeddings = list(bm25_model.embed(documents))

    logger.debug(f"Returning {len(embeddings)} sparse embeddings")
    logger.info("Sparse embeddings generation complete")

    return embeddings
