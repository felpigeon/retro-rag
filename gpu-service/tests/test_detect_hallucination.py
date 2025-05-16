
def test_detect_hallucination_true(client):
    """
    Test the '/detect_hallucination' endpoint for hallucination detection.

    Checks if the endpoint correctly identifies hallucinations in the response
    when the response contains conflicting or incorrect information.
    """
    params = {
        "query": "What is the capital of France?",
        "context": "Paris is the capital of France. It is known for its art, culture, and history.",
        "response": "The capital of France is Paris. The capital of France is Berlin."
    }
    response = client.post('/detect_hallucination', json=params)
    assert response.status_code == 200
    assert response.json['hallucination_detected'] is True


def test_detect_hallucination_false(client):
    """
    Test the '/detect_hallucination' endpoint for no hallucination detection.

    Verifies that the endpoint does not flag hallucinations when the response
    aligns with the provided context and query.
    """
    params = {
        "query": "What is the capital of France?",
        "context": "Paris is the capital of France. It is known for its art, culture, and history.",
        "response": "The capital of France is Paris."
    }
    response = client.post('/detect_hallucination', json=params)
    assert response.status_code == 200
    assert response.json['hallucination_detected'] is False
