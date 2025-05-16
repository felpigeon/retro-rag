
def test_summarize(client):
    """
    Test the '/summarize' endpoint.

    Ensures that the endpoint generates a summary of the provided document
    based on the specified query and length.
    """
    params = {
        "query": "What are the key points?",
        "document": "Flask is a lightweight WSGI web application framework in Python. It is designed with simplicity and flexibility in mind. Flask is easy to learn and use, making it a popular choice for developers.",
        "length": 10
    }
    response = client.post('/summarize', json=params)
    assert response.status_code == 200
    assert response.json["summary"] == "It is designed with simplicity and flexibility in mind."
