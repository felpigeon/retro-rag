import aiohttp
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def detect_hallucination(query, context, response):
    """
    Detect hallucinations in an LLM response by comparing it with the provided context.

    This function communicates with a specialized GPU service that analyzes whether
    the response contains information not supported by the context or query.

    Args:
        query (str): The original user query that prompted the response
        context (str): The reference information that should support the response
        response (str): The LLM-generated response to check for hallucinations

    Returns:
        dict: Results of the hallucination detection, typically containing:
            - hallucination_score: Float indicating likelihood of hallucination
            - hallucinated_sections: List of text segments that may be hallucinated

    Raises:
        Exception: If the GPU service returns a non-200 status code
    """
    logger.info("Starting hallucination detection")
    logger.debug(f"Query length: {len(query)}, context length: {len(context)}, response length: {len(response)}")

    async with aiohttp.ClientSession() as session:
        logger.debug("Sending request to GPU service for hallucination detection")
        async with session.post(
            'http://gpu-service:5001/detect_hallucination',
            json={
                "query": query,
                "context": context,
                "response": response,
            }
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"GPU service hallucination detection failed: {error_text}")
                raise Exception(f"GPU service hallucination detection failed: {error_text}")

            result = await response.json()
            logger.debug(f"Hallucination detection complete: score={result.get('hallucination_score', 'N/A')}")
            logger.info("Hallucination detection completed successfully")

            return result
