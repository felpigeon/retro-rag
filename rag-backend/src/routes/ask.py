from flask import Blueprint, request, jsonify

from src.services.qa_pipeline import qa_pipeline
from src.services.hallucination import detect_hallucination
from src.models.requests import QuestionRequest
from src.utils.logger import get_logger


ask_bp = Blueprint('ask', __name__)
logger = get_logger(__name__)


@ask_bp.route('/ask', methods=['GET'])
async def ask():
    """
    Ask a question and get an answer from the QA pipeline.
    ---
    tags:
      - Ask
    parameters:
      - in: query
        name: question
        schema:
          type: string
        required: true
        description: The question to be answered.
      - in: query
        name: method
        schema:
          type: string
        required: false
        description: The method to use for the search (bm25/dense/hybrid).
      - in: query
        name: k
        schema:
          type: integer
        required: false
        description: The number of top results to consider.
      - in: query
        name: filter_by_entity
        schema:
          type: boolean
        required: false
        description: Whether to filter articles by entity when searching.
      - in: query
        name: do_rerank
        schema:
          type: boolean
        required: false
        description: Whether to rerank the articles (will search k*3 articles and use the top k).
    responses:
      200:
        description: A successful response containing the question and its answer.
        schema:
          type: object
          properties:
            question:
              type: string
              description: The question asked.
            answer:
              type: string
              description: The answer to the question.
      400:
        description: Bad request. Either the question is missing or an error occurred.
        schema:
          type: object
          properties:
            error:
              type: string
              description: The error message.
    """
    try:
        logger.info("Received question request")
        question = request.args.get('question')
        if not question:
            logger.warning("Request missing required 'question' parameter")
            return jsonify({'error': 'Question is required'}), 400

        question_data = QuestionRequest(**request.args)
        logger.debug(f"Processing question: '{question_data.question}' with parameters: "
                     f"method={question_data.method}, k={question_data.k}, "
                     f"filter_by_entity={question_data.filter_by_entity}, do_rerank={question_data.do_rerank}")

        result = await qa_pipeline(
            question_data.question,
            method=question_data.method,
            k=question_data.k,
            filter_by_entity=question_data.filter_by_entity,
            do_rerank=question_data.do_rerank,
        )
        logger.debug("QA pipeline completed successfully")

        hallucination = await detect_hallucination(
            question_data.question,
            "\n".join(doc["text"] for doc in result['docs']),
            result['answer'],
        )
        result.update(hallucination=hallucination)
        logger.debug(f"Hallucination detection result: {hallucination}")

        logger.info(f"Successfully processed question: '{question_data.question[:30]}...'")
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error processing question request: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 400
