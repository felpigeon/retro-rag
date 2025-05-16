import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from src.services.qa_pipeline import qa_pipeline


@pytest.fixture
def mock_query_expansion():
    mock = AsyncMock()
    mock.return_value = MagicMock(question="expanded question", language="en")
    return mock


@pytest.fixture
def mock_search():
    mock = AsyncMock()
    mock.return_value = [
        {"id": "1", "text": "This is document 1", "metadata": {"title": "Doc 1"}},
        {"id": "2", "text": "This is document 2", "metadata": {"title": "Doc 2"}},
    ]
    return mock


@pytest.fixture
def mock_summarize():
    mock = AsyncMock()
    mock.return_value = "Summarized text"
    return mock


@pytest.fixture
def mock_ask_question():
    mock = AsyncMock()
    mock.return_value = MagicMock(answer="This is the answer")
    return mock


@pytest.mark.asyncio
@patch("src.services.qa_pipeline.b.QueryExpansion")
@patch("src.services.qa_pipeline.search")
@patch("src.services.qa_pipeline.b.AskQuestion")
async def test_qa_pipeline_happy_path(mock_ask, mock_search, mock_query_exp):
    # Setup
    mock_query_exp.return_value = MagicMock(question="expanded question", language="en")
    mock_search.return_value = [
        {"id": "1", "text": "Short document", "metadata": {"title": "Doc 1"}},
        {"id": "2", "text": "Another short doc", "metadata": {"title": "Doc 2"}},
    ]
    mock_ask.return_value = MagicMock(answer="This is the answer")

    # Execute
    question = "What is the meaning of life?"
    result = await qa_pipeline(question)

    # Assert
    mock_query_exp.assert_called_once_with(question)
    mock_search.assert_called_once_with(
        "expanded question", method="hybrid", k=5, filter_by_entity=False, do_rerank=False
    )
    mock_ask.assert_called_once_with(question, ["Short document", "Another short doc"], "en")

    assert result["answer"] == "This is the answer"
    assert len(result["docs"]) == 2


@pytest.mark.asyncio
@patch("src.services.qa_pipeline.b.QueryExpansion")
@patch("src.services.qa_pipeline.search")
@patch("src.services.qa_pipeline.summarize")
@patch("src.services.qa_pipeline.b.AskQuestion")
async def test_qa_pipeline_with_summarization(mock_ask, mock_summarize, mock_search, mock_query_exp):
    # Setup
    mock_query_exp.return_value = MagicMock(question="expanded question", language="en")
    # Create a document that exceeds the chunk size
    long_text = "x" * 1200  # 1200 chars, exceeding default 1000 chunk size
    short_text = "Short document"
    mock_search.return_value = [
        {"id": "1", "text": long_text, "metadata": {"title": "Doc 1"}},
        {"id": "2", "text": short_text, "metadata": {"title": "Doc 2"}},
    ]
    mock_summarize.return_value = "Summarized text"
    mock_ask.return_value = MagicMock(answer="This is the answer")

    # Execute
    result = await qa_pipeline("What is the meaning of life?")

    # Assert
    assert mock_summarize.call_count == 1  # Only called for the long document
    mock_ask.assert_called_once_with(
        "What is the meaning of life?", ["Summarized text", short_text], "en"
    )
    assert result["answer"] == "This is the answer"


@pytest.mark.asyncio
@patch("src.services.qa_pipeline.b.QueryExpansion")
@patch("src.services.qa_pipeline.search")
@patch("src.services.qa_pipeline.b.AskQuestion")
async def test_qa_pipeline_with_custom_params(mock_ask, mock_search, mock_query_exp):
    # Setup
    mock_query_exp.return_value = MagicMock(question="expanded question", language="fr")
    mock_search.return_value = [{"id": "1", "text": "Document", "metadata": {"title": "Doc 1"}}]
    mock_ask.return_value = MagicMock(answer="Custom answer")

    # Execute with custom parameters
    result = await qa_pipeline(
        "Custom question?",
        method="bm25",
        k=10,
        filter_by_entity=True,
        do_rerank=True,
        chunk_size=2000,
    )

    # Assert
    mock_search.assert_called_once_with(
        "expanded question", method="bm25", k=10, filter_by_entity=True, do_rerank=True
    )
    assert result["answer"] == "Custom answer"


@pytest.mark.asyncio
@patch("src.services.qa_pipeline.b.QueryExpansion")
@patch("src.services.qa_pipeline.search")
@patch("src.services.qa_pipeline.b.AskQuestion")
async def test_qa_pipeline_empty_results(mock_ask, mock_search, mock_query_exp):
    # Setup
    mock_query_exp.return_value = MagicMock(question="expanded question", language="en")
    mock_search.return_value = []  # No search results
    mock_ask.return_value = MagicMock(answer="I don't have enough information to answer.")

    # Execute
    result = await qa_pipeline("What is the meaning of life?")

    # Assert
    mock_ask.assert_called_once_with("What is the meaning of life?", [], "en")
    assert result["answer"] == "I don't have enough information to answer."
    assert result["docs"] == []


@pytest.mark.asyncio
@patch("src.services.qa_pipeline.b.QueryExpansion")
@patch("src.services.qa_pipeline.search")
async def test_qa_pipeline_search_exception(mock_search, mock_query_exp):
    # Setup
    mock_query_exp.return_value = MagicMock(question="expanded question", language="en")
    mock_search.side_effect = Exception("Search failed")

    # Execute and Assert
    with pytest.raises(Exception, match="Search failed"):
        await qa_pipeline("What is the meaning of life?")
