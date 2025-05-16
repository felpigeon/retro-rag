import pytest
from src.services.embeddings import (
    get_sparse_embeddings,
    get_dense_embeddings,
)
from fastembed import SparseEmbedding


class MockResponse:
    def __init__(self, embeddings):
        self.embeddings = embeddings

class MockEmbedding:
    def __init__(self, values):
        self.values = values


@pytest.fixture
def mock_genai_client(mocker):
    mock_client = mocker.MagicMock()
    mock_models = mocker.MagicMock()
    mock_embed = mocker.MagicMock(return_value=MockResponse([MockEmbedding([0.1, 0.2, 0.3])]))

    # Patch the client
    mocker.patch("src.services.embeddings.genai.Client", return_value=mock_client)

    # Set up the chain
    mock_client.models = mock_models
    mock_models.embed_content = mock_embed

    return {
        "client": mock_client,
        "models": mock_models,
        "embed": mock_embed
    }


def test_sparse_embed_text():
    document = "Document test"

    embeddings = get_sparse_embeddings(document)

    assert isinstance(embeddings, list)
    assert isinstance(embeddings[0], SparseEmbedding)
    assert len(embeddings) == 1


def test_sparse_embed_texts():
    documents = [
        "Document test",
        "Document test 2"
    ]

    embeddings = get_sparse_embeddings(documents)

    assert isinstance(embeddings, list)
    assert isinstance(embeddings[0], SparseEmbedding)

def test_dense_embed_text(mock_genai_client):
    document = "Document test"
    embeddings = get_dense_embeddings(document)

    mock_genai_client["embed"].assert_called_once()
    assert isinstance(embeddings, list)
    assert isinstance(embeddings[0], list)
    assert len(embeddings) == 1


def test_dense_embed_texts(mock_genai_client):
    documents = [
        "Document test",
        "Document test 2"
    ]*100
    embeddings = get_dense_embeddings(documents)

    assert isinstance(embeddings, list)
    assert isinstance(embeddings[0], list)
    assert len(embeddings) == 2
    assert mock_genai_client["embed"].call_count == 2