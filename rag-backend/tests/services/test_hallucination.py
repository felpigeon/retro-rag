import pytest
import aiohttp
from aioresponses import aioresponses

from src.services.hallucination import detect_hallucination


@pytest.fixture
def mock_aioresponse():
    with aioresponses() as m:
        yield m


@pytest.mark.asyncio
async def test_detect_hallucination_success(mock_aioresponse):
    """Test that detect_hallucination successfully processes valid input and returns expected results."""
    # Test data
    query = "Who is the Chief Justice of Canada?"
    context = "The Chief Justice of Canada is Richard Wagner since 2017."
    response = "Richard Wagner is the Chief Justice of Canada."

    # Expected response from the GPU service
    expected_result = {
        "hallucination_score": 0.05,
        "hallucinated_sections": [],
        "explanation": "The response is fully supported by the context."
    }

    # Mock the HTTP endpoint
    mock_aioresponse.post(
        'http://gpu-service:5001/detect_hallucination',
        status=200,
        payload=expected_result
    )

    # Call the function being tested
    result = await detect_hallucination(query, context, response)

    # Assert the function returned the expected result
    assert result == expected_result
    assert result["hallucination_score"] == 0.05
    assert len(result["hallucinated_sections"]) == 0


@pytest.mark.asyncio
async def test_detect_hallucination_with_hallucination(mock_aioresponse):
    """Test that detect_hallucination correctly identifies hallucinated content."""
    # Test data
    query = "Who is the Chief Justice of Canada?"
    context = "The Chief Justice of Canada is Richard Wagner since 2017."
    response = "Richard Wagner has been the Chief Justice of Canada since 2015 and previously served as a Quebec judge."

    # Mock response containing hallucination details
    expected_result = {
        "hallucination_score": 0.75,
        "hallucinated_sections": ["since 2015", "previously served as a Quebec judge"],
        "explanation": "The response contains information not found in the context."
    }

    # Mock the HTTP endpoint
    mock_aioresponse.post(
        'http://gpu-service:5001/detect_hallucination',
        status=200,
        payload=expected_result
    )

    # Call the function being tested
    result = await detect_hallucination(query, context, response)

    # Assert the function returns the hallucination details
    assert result["hallucination_score"] == 0.75
    assert len(result["hallucinated_sections"]) == 2
    assert "since 2015" in result["hallucinated_sections"]


@pytest.mark.asyncio
async def test_detect_hallucination_error_response(mock_aioresponse):
    """Test that detect_hallucination raises an exception when the GPU service returns an error."""
    # Test data
    query = "Who is the Chief Justice of Canada?"
    context = "The Chief Justice of Canada is Richard Wagner since 2017."
    response = "Richard Wagner is the Chief Justice of Canada."

    # Mock error response
    mock_aioresponse.post(
        'http://gpu-service:5001/detect_hallucination',
        status=500,
        body="Internal Server Error"
    )

    # Assert the function raises an exception with the correct error message
    with pytest.raises(Exception) as excinfo:
        await detect_hallucination(query, context, response)

    assert "GPU service hallucination detection failed: Internal Server Error" in str(excinfo.value)


@pytest.mark.asyncio
async def test_detect_hallucination_connection_error(mock_aioresponse):
    """Test that detect_hallucination handles connection errors to the GPU service."""
    # Test data
    query = "Who is the Chief Justice of Canada?"
    context = "The Chief Justice of Canada is Richard Wagner since 2017."
    response = "Richard Wagner is the Chief Justice of Canada."

    # Mock a connection error
    mock_aioresponse.post(
        'http://gpu-service:5001/detect_hallucination',
        exception=aiohttp.ClientConnectorError(
            connection_key=None,
            os_error=ConnectionRefusedError(111, "Connection refused")
        )
    )

    # Assert the function propagates the connection error
    with pytest.raises(aiohttp.ClientConnectorError):
        await detect_hallucination(query, context, response)
