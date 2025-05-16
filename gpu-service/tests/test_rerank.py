
def test_rerank(client):
    """
    Test the '/rerank' endpoint.

    Verifies that the endpoint correctly processes a query and a list of documents,
    returning the query and a ranked list of documents based on relevance.
    """
    params = {
        "query": "What is the capital of France?",
        "documents": [
            {"text": "Paris is the capital of France."},
            {"text": "Berlin is the capital of Germany."},
            {"text": "Madrid is the capital of Spain."}
        ]
    }
    response = client.post('/rerank', json=params)
    assert response.status_code == 200
    assert response.json["query"] == "What is the capital of France?"
    assert len(response.json["ranked_documents"]) == 3
    assert response.json["ranked_documents"][0]["text"] == "Paris is the capital of France."
