from baml_client import b
import time
import duckdb
import os


def extract_entities(question, sleep=0):
    raw_entities = b.ExtractEntities(question)
    if not len(raw_entities):
        return

    entities = {}

    games = [e.name for e in raw_entities if e.type == 'Game']
    if games:
        entities['game'] = games

    consoles = [e.name for e in raw_entities if e.type == 'Console']
    if consoles:
        entities['console'] = consoles

    publishers = [e.name for e in raw_entities if e.type == 'Publisher']
    if publishers:
        entities['publisher'] = publishers

    if sleep:
        time.sleep(sleep)

    return entities


def match_entity(term, entity_type):
    with duckdb.connect(os.environ.get("DUCKDB_PATH", 'entities.db')) as conn:
        entity = conn.sql(f"""
            SELECT
                name,
                levenshtein(name, $${term.lower()}$$) / GREATEST(LENGTH(name), LENGTH($${term.lower()}$$)) AS distance,
            FROM {entity_type.lower()}s
            WHERE distance < 0.1
            ORDER BY distance ASC
            LIMIT 1
        """).fetchall()

    entity = entity[0][0] if len(entity) else None

    return entity