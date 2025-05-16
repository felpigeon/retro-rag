from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding
from sentence_transformers import CrossEncoder
import time

from sync_ge import graph_expansion


client = QdrantClient("http://localhost:6333")
bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25")
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')


def create_entity_filter(filter):
    if filter is None: return

    filter = models.Filter(
        should=[
            models.FieldCondition(
                key=f"{entity_type}[]",
                match=models.MatchAny(any=entity_names)
            )
            for entity_type, entity_names in filter.items()
        ]
    )

    return filter


def search_bm25(query, k=5, filter=None):
    vec = list(bm25_model.embed(query))[0]

    docs = client.query_points(
        collection_name="dev_articles",
        using="text",
        query=models.SparseVector(
            indices = vec.indices,
            values = vec.values,
        ),
        query_filter=create_entity_filter(filter),
        limit=k,
    )

    docs = list(docs.points)

    if filter is not None and len(docs) < k:
        docs += search_bm25(query, k, filter=None)
        docs = docs[:k]

    return docs


def search_dense(embedding, k=5, filter=None):
    docs = client.query_points(
        collection_name="dev_articles",
        using="embedding",
        query=embedding,
        query_filter=create_entity_filter(filter),
        limit=k,
    )

    docs = list(docs.points)

    if filter is not None and len(docs) < k:
        docs += search_dense(embedding, k, filter=None)
        docs = docs[:k]

    return docs


def search_hybrid(query, embedding, k=5, filter=None):
    vec = list(bm25_model.embed(query))[0]

    docs = client.query_points(
        collection_name="dev_articles",
        query=models.FusionQuery(
            fusion=models.Fusion.RRF
        ),
        prefetch=[
            {
                "query": models.SparseVector(
                    indices = vec.indices,
                    values = vec.values,
                ),
                "using": "text",
                "limit": k,
                "filter": create_entity_filter(filter),
            },
            {
                "query": embedding,
                "using": "embedding",
                "limit": k,
                "filter": create_entity_filter(filter),
            }

        ],
        limit=k,
    )

    docs = list(docs.points)

    if filter is not None and len(docs) < k:
        docs += search_hybrid(query, embedding, k, filter=None)
        docs = docs[:k]

    return docs


def rerank(query, docs):
    scores = reranker.predict([
        (query, doc.payload["text"])
        for doc in docs
    ])

    sorted_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in sorted_docs]


def search(doc, method, k=5, filter_by_entity=False, do_rerank=False, do_graph_expansion=False, sleep=0):
    filter = doc.entities if filter_by_entity else None

    k_eff = k*3 if do_rerank else k

    if method == "bm25":
        docs = search_bm25(doc.question, k=k_eff, filter=filter)
    elif method == "dense":
        docs = search_dense(doc.question_embedding, k=k_eff, filter=filter)
    elif method == "hybrid":
        docs = search_hybrid(doc.question, doc.question_embedding, k=k_eff, filter=filter)
    else:
        raise ValueError("Invalid search method. Choose from 'bm25', 'dense', or 'hybrid'.")

    if do_graph_expansion:
        docs += graph_expansion(
            doc.question,
            docs,
        )

    if do_rerank and len(docs):
        docs = rerank(doc.question, docs)[:k]

    if sleep:
        time.sleep(sleep)

    return docs