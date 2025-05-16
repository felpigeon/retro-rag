import pytest
from aioresponses import aioresponses
import json
from src.services.chunking import summarize

@pytest.mark.asyncio
async def test_summarize_success_with_aioresponses():
    """Test successful summarization using aioresponses."""

    query = "What is the main point?"
    document = "This is a lengthy document that should be summarized."
    expected_summary = "This is a summary."

    with aioresponses() as m:
        m.post('http://gpu-service:5001/summarize',
               payload={"summary": expected_summary},
               status=200)

        result = await summarize(query, document)

        assert result == expected_summary