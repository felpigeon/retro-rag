import os
import duckdb
from src.utils.logger import get_logger
from src.baml_client.async_client import b
from typing import Dict, List, Optional, Any

logger = get_logger(__name__)

class EntityExtractor:
    """A class to handle entity extraction and matching operations."""

    def __init__(self, db_path: str = None):
        """
        Initialize the EntityExtractor.

        Args:
            db_path: Path to the DuckDB database file
        """
        self.db_path = db_path or os.environ.get("DUCKDB_PATH", 'entities.db')
        self.entity_types = ['Game', 'Console', 'Publisher']
        self.similarity_threshold = 0.1

    async def extract_entities(self, question: str) -> Optional[Dict[str, List[str]]]:
        """
        Extract entities from a given question.

        Args:
            question: The input question from which entities are to be extracted.

        Returns:
            A dictionary of formatted entities if found, otherwise None.
        """
        if not question or not question.strip():
            logger.warning("Empty question provided for entity extraction")
            return None

        logger.info(f"Extracting entities from question: {question}")

        try:
            raw_entities = await b.ExtractEntities(question)

            if not raw_entities:
                logger.warning("No entities found in the question")
                return None

            logger.info(f"Found {len(raw_entities)} raw entities")
            return self.format_entities(raw_entities)

        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}", exc_info=True)
            return None

    def format_entities(self, raw_entities: List[Any]) -> Dict[str, List[str]]:
        """
        Format raw entities into a structured dictionary.

        Args:
            raw_entities: A list of raw entities to format.
            match: Whether to match entities against the database.

        Returns:
            A dictionary of formatted entities grouped by type.
        """
        logger.info(f"Formatting {len(raw_entities)} entities")

        entities = {}

        for entity_type in self.entity_types:
            values = [
                e.name.lower() for e in raw_entities
                if e.type == entity_type
            ]

            if values:
                entities[entity_type] = values
                logger.info(f"Found {len(values)} {entity_type} entities")

        return entities if entities else None

    def match_entities(self, entities: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Match a list of entity terms against the database to find the closest matching entities.

        Args:
            entities: A dictionary mapping entity types to lists of entity terms.

        Returns:
            A dictionary of matched entity terms grouped by entity type, or None if no matches are found.
        """
        matches = {}
        for entity_type, entity_list in entities.items():
            matched_entities = []
            for term in entity_list:
                matched_entity = self.match_entity(term, entity_type)
                if matched_entity:
                    matched_entities.append(matched_entity)

            if matched_entities:
                matches[entity_type] = matched_entities
                logger.info(f"Matched {len(matches)} entities of type '{entity_type}'")

        return matches if matches else None

    def match_entity(self, term: str, entity_type: str) -> Optional[str]:
        """
        Match a given term to an entity in the database based on its type.

        Args:
            term: The term to match.
            entity_type: The type of the entity (e.g., 'Game', 'Console', 'Publisher').

        Returns:
            The matched entity name if found, otherwise None.
        """
        if not term or not entity_type:
            return None

        logger.info(f"Matching entity: term='{term}', type='{entity_type}'")

        try:
            with duckdb.connect(self.db_path) as conn:
                # Prepare statement to avoid SQL injection
                table_name = f"{entity_type.lower()}s"
                query = f"""
                    SELECT
                        name,
                        levenshtein(LOWER(name), LOWER(?)) / GREATEST(LENGTH(name), LENGTH(?)) AS distance
                    FROM {table_name}
                    WHERE distance < ?
                    ORDER BY distance ASC
                    LIMIT 1
                """
                entity = conn.execute(query, [term, term, self.similarity_threshold]).fetchall()

            if entity and entity[0]:
                entity_name = entity[0][0]
                logger.info(f"Matched entity: {entity_name}")
                return entity_name
            else:
                logger.warning(f"No match found for term '{term}' in {entity_type}")
                return None

        except Exception as e:
            logger.error(f"Error matching entity '{term}': {str(e)}", exc_info=True)
            return None

entity_extractor = EntityExtractor()
