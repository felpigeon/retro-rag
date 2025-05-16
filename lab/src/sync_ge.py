import pandas as pd
from baml_client import b
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import duckdb
import numpy as np


TRIPLES_PATH = "../../data/triples.db"


model = SentenceTransformer("all-MiniLM-L6-v2")


def most_similar_triple(term):
    term = term.lower()

    with duckdb.connect(TRIPLES_PATH) as con:
        match = con.sql(f"""
        SELECT
            head,
            relation,
            tail,
            1 - levenshtein($${term}$$, flatten) / GREATEST(LENGTH($${term}$$), LENGTH(flatten)) as similarity
        FROM triples
        ORDER BY similarity DESC
        LIMIT 1
        """).fetchall()

    return match[0][:-1]


def get_neighbours(triple, side):
    with duckdb.connect(TRIPLES_PATH) as con:
        match = con.sql(f"""
        SELECT
            head,
            relation,
            tail,
        FROM triples
        WHERE {side} = $${triple}$$
        """).fetchall()

    return match



def find_neighbours(t):
    seen_triples = set()
    
    for side in ["head", "tail"]:
        for end in [t[0], t[-1]]:
            new_neighbours = get_neighbours(end, side)
            for n in new_neighbours:
                triple_tuple = tuple(n[:3])
                seen_triples.add(triple_tuple)

    neighbours = [
        [t] for t in seen_triples
    ]

    return neighbours


def triple_to_str(triple):
    return " ".join(triple)


def expand_beam(beam, query_embedding, gamma=4, n_neighbours=100):
    t, s = beam

    path_str = " ".join(triple_to_str(t_) for t_ in t)

    neighbours = find_neighbours(t[-1])
    neighbours = [
        n for n in neighbours
        if n[0] not in t
    ]

    neighbours_str = [
        path_str + " ".join(triple_to_str(n[0]))
        for n in neighbours
    ]

    n_emb = model.encode(neighbours_str)
    n_scores = model.similarity(query_embedding, n_emb).numpy()[0]

    for n_, s_ in zip(neighbours, n_scores):
        n_.append(s + s_)

    neighbours = sorted(neighbours, key=lambda x: x[-1], reverse=True)
    for i in range(len(neighbours)):
        neighbours[i][-1] = neighbours[i][-1] * np.exp(-min(i, gamma)/ gamma) 

    neighbours = sorted(neighbours, key=lambda x: x[-1], reverse=True)
    return neighbours[:n_neighbours]


def retrieve_docs_from_triples(triples):
    with duckdb.connect(TRIPLES_PATH) as con:
        uuids = con.sql(f"""
            SELECT DISTINCT uuid
            FROM triples
            WHERE flatten IN ({", ".join([f"$${t}$$" for t in triples])})
        """).fetchall()

    uuids = [u[0] for u in uuids]

    client = QdrantClient("http://localhost:6333")
    return client.retrieve(
        collection_name="dev_articles",
        ids=uuids
    )


def beam_search(beams, query_embedding, beam_size=10, length=2, gamma=2):
    beams_next = []
    for b in beams:
        new_beams = expand_beam(b, query_embedding, gamma=gamma)
        for new_beam in new_beams:
            new_beam[0] = b[0] + [new_beam[0]]
        beams_next += new_beams

    beams_next = sorted(beams_next, key=lambda x: x[-1], reverse=True)
    beams_next = beams_next[:beam_size]

    if length == 0:
        return beams_next

    return beam_search(beams_next, query_embedding, beam_size=beam_size, length=length-1)


def graph_expansion(question, documents):
    q_emb = model.encode(question)

    # Extract triples from retrieved documents
    documents_str = [f"Document {i}:\n{d.payload["text"]}" for i, d in enumerate(documents)]
    documents_str = "\n\n".join(documents_str)
    triples = b.ExtractTriples(question, documents_str)
    if not len(triples):
        return []

    triples_flatten = [" ".join(triple) for triple in triples]

    # Match triples to triples in the BD
    ts = [
        most_similar_triple(t)
        for t in triples_flatten
    ]

    # Create beams (score triples)
    t_emb = model.encode([" ".join(t) for t in ts])
    t_scores = model.similarity(q_emb, t_emb).numpy()[0]
    beams = [
        [[t], s]
        for t, s in zip(ts, t_scores)
    ]

    # Expand graph
    r = beam_search(beams, q_emb, length=1)

    # Format triples from graph expansion
    final_triples = sum([
        b[0]
        for b in r
    ], [])

    final_triples_flat = list(set([
        " ".join(t)
        for t in final_triples
    ]))

    # Retrieve corresponding documents
    docs_uuid = [d.id for d in documents]
    retrieved_docs =  retrieve_docs_from_triples(final_triples_flat)
    retrieved_docs = [d for d in retrieved_docs if d.id in docs_uuid]

    return retrieved_docs