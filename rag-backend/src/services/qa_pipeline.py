from src.baml_client.async_client import b
from src.utils.logger import get_logger
from src.services.search import search
from src.services.chunking import summarize

logger = get_logger(__name__)


async def qa_pipeline(
    question,
    method="hybrid",
    k=5,
    filter_by_entity=False,
    do_rerank=False,
    chunk_size=1000
):
    """
    Executes a question-answering pipeline using a combination of search, summarization, and question-answering.

    Args:
        question (str): The input question to be answered.
        method (str, optional): The search method to use (e.g., "hybrid", "bm25"). Defaults to "hybrid".
        k (int, optional): The number of top documents to retrieve. Defaults to 5.
        filter_by_entity (bool, optional): Whether to filter results by entity. Defaults to False.
        do_rerank (bool, optional): Whether to rerank the search results. Defaults to False.
        chunk_size (int, optional): The maximum size of text chunks for summarization. Defaults to 1000.

    Returns:
        str: The final answer to the input question.

    Raises:
        Exception: If any error occurs during the pipeline execution.
    """
    logger.info(f"Starting QA pipeline with question: '{question}', method={method}")

    query = await b.QueryExpansion(question)
    logger.debug(f"Expanded query: {query}")

    logger.info(f"Searching with method={method}, k={k}")
    docs = await search(
        query.question,
        method=method,
        k=k,
        filter_by_entity=filter_by_entity,
        do_rerank=do_rerank,
    )
    logger.debug(f"Search returned {len(docs)} documents")

    for doc in docs:
        if len(doc["text"]) > chunk_size:
            logger.debug(f"Document exceeds chunk size ({len(doc['text'])} > {chunk_size}), summarizing")
            doc["text"] = await summarize(query.question, doc["text"])
            logger.debug(f"Document summarized to {len(doc['text'])} characters")

    docs_str = [doc['text'] for doc in docs]

    logger.info("Generating answer from LLM")
    answer = await b.AskQuestion(question, docs_str, query.language)
    answer = answer.answer
    logger.debug("LLM returned answer")

    logger.info("QA pipeline completed successfully")
    return {
        "answer": answer,
        "docs": docs,
    }
