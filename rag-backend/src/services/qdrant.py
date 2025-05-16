import os
from qdrant_client import AsyncQdrantClient, models
from src.utils.logger import get_logger
from src.models.document import Document
from typing import List, Union

logger = get_logger(__name__)

COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "articles")


async def get_qdrant_client():
    """Get or create an AsyncQdrantClient with the current event loop.

    Returns:
        AsyncQdrantClient: An instance of the Qdrant client.
    """
    host = os.environ.get("QDRANT_HOST", "http://localhost:6333")
    logger.debug(f"Creating Qdrant client with host: {host}")
    return AsyncQdrantClient(host)


async def create_articles_collection():
    """Create articles collection if it doesn't exist.

    Creates a collection in Qdrant for storing articles with both dense and sparse vectors.
    """
    qdrant_client = await get_qdrant_client()

    if await qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
        logger.debug(f"{COLLECTION_NAME} collection already exists")
        return

    logger.info(f"Creating {COLLECTION_NAME} collection")
    await qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "embedding": models.VectorParams(
                size=768,
                distance=models.Distance.COSINE
            ),
        },
        sparse_vectors_config={
            "text": models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False),
                modifier=models.Modifier.IDF,
            )
        },
    )
    logger.info(f"{COLLECTION_NAME} collection created successfully")


async def upsert_articles(documents: Union[Document, List[Document]]):
    """Insert or update one or multiple articles in the Qdrant database.

    Args:
        documents: A single Document or a list of Document objects to upsert.
    """
    await create_articles_collection()

    if not isinstance(documents, list):
        documents = [documents]

    if not documents:
        logger.warning("No documents provided for upserting")
        return

    logger.info(f"Upserting {len(documents)} articles")
    qdrant_client = await get_qdrant_client()

    points = []
    for doc in documents:
        points.append(
            models.PointStruct(
                id=doc.doc_id,
                payload={
                    "text": doc.text,
                    **{f"{entity_type}": entity_names for entity_type, entity_names in doc.entities.items()}
                },
                vector={
                    "embedding": doc.dense_vec,
                    "text": models.SparseVector(
                        indices=doc.sparse_vec.indices,
                        values=doc.sparse_vec.values,
                    )
                }
            )
        )

    await qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True,
    )
    logger.info(f"Successfully upserted {len(documents)} articles")