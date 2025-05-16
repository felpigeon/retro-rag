import os
from typing import List, Dict, Any
from uuid import uuid4
import duckdb
from src.utils.logger import get_logger

from src.models.requests import IngestRequest
from src.models.document import Document, SparseVector
from src.services.entity import entity_extractor
from src.services.embeddings import get_dense_embeddings, get_sparse_embeddings
from src.services.qdrant import upsert_articles


logger = get_logger(__name__)

COLLECTION_NAME = "articles"


async def ingest_documents(documentsRequest: List[IngestRequest]) -> Dict[str, Any]:
    """
    Process a batch of documents.

    Args:
        documents: List of documents to be ingested

    Returns:
        Dictionary containing summary and individual document results
    """
    logger.info(f"Starting batch ingestion of {len(documentsRequest)} documents")

    doc_ids = [
        str(uuid4()) for _ in range(len(documentsRequest))
    ]

    dense_embeddings = get_dense_embeddings(
        [doc.text for doc in documentsRequest]
    )
    sparse_embeddings = get_sparse_embeddings(
        [doc.text for doc in documentsRequest]
    )
    logger.debug(f"Generated dense and sparse embeddings for batch")

    entities = [
        doc.entities if doc.entities else await entity_extractor.extract_entities(doc.text)
        for doc in documentsRequest
    ]
    logger.debug(f"Extracted entities for batch")

    documents = [
        Document(
            doc_id=doc_ids[i],
            text=documentsRequest[i].text,
            dense_vec=dense_embeddings[i],
            sparse_vec=SparseVector(
                indices = sparse_embeddings[i].indices,
                values = sparse_embeddings[i].values
            ),
            entities=entities[i]
        )
        for i in range(len(documentsRequest))
    ]

    insert_entities(entities)
    logger.debug(f"Inserted entities into DuckDB")
    await upsert_articles(documents)
    logger.debug(f"Upserted documents to Qdrant")

    return doc_ids


def insert_entities(entities: List[dict]):
    """
    Insert extracted entities into a DuckDB database.

    Args:
        entities (List[dict]): A list of dictionaries where each dictionary contains
            entity types as keys and lists of entity values as values.
            Example: [{"Game": ["Mario Bross"], "Publisher": ["Acme Inc."]}, ...]

    Returns:
        None

    Note:
        The function uses environment variable DUCKDB_PATH to determine the
        database location, defaulting to "entities.db" if not specified.
    """
    entities_batch = {
        et: []
        for et in entity_extractor.entity_types
    }
    for entity in entities:
        for entity_type, entity_values in entity.items():
            if entity_type not in entities_batch: continue
            entities_batch[entity_type].extend([
                e.lower() for e in entity_values
            ])

    db_path = os.environ.get("DUCKDB_PATH", "entities.db")
    logger.info(f"Using database at: {db_path}")
    db_parent_path = os.path.dirname(db_path)
    if db_parent_path:
        os.makedirs(db_parent_path, exist_ok=True)
        logger.info(f"Created directory: {db_parent_path}")

    with duckdb.connect(db_path) as con:
        for entity_type in entities_batch:
            table_name = entity_type.lower() + "s"
            con.sql(f"CREATE TABLE IF NOT EXISTS {table_name} (name VARCHAR UNIQUE)")

            if entity_type not in entity_extractor.entity_types: continue
            if not len(entities_batch[entity_type]): continue

            values = [(v,) for v in entities_batch[entity_type]]
            logger.info(f"Inserting {len(values)} {entity_type} entities")

            # Create temporary table
            temp_table = f"temp_{table_name}_{uuid4().hex[:8]}"
            con.sql(f"CREATE TEMPORARY TABLE {temp_table} (name VARCHAR)")

            # Insert all values into temporary table
            con.executemany(
                f"INSERT INTO {temp_table} VALUES (?)",
                values
            )

            # Insert distinct values from temp table that don't exist in the main table
            con.sql(f"""
                INSERT INTO {table_name} (name)
                SELECT DISTINCT t.name
                FROM {temp_table} t
                WHERE NOT EXISTS (
                    SELECT 1 FROM {table_name} m WHERE m.name = t.name
                )
            """)

            # Drop temporary table
            con.sql(f"DROP TABLE {temp_table}")

            con.commit()
            logger.info(f"Committed {entity_type} entities")
