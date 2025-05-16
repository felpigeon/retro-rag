from flask import Blueprint, request, jsonify
import numpy as np
from sentence_splitter import split_text_into_sentences
from src.services.crossencoder import crossencoder
from src.utils.logger import get_logger


logger = get_logger("summarize_service")
summarize_bp = Blueprint('summarize', __name__)


@summarize_bp.route('/summarize', methods=['POST'])
async def summarize():
    """
    Summarize a document based on a query.
    ---
    tags:
      - Summarize
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            query:
              type: string
              description: The query string.
            document:
              type: string
              description: The document to summarize.
            length:
              type: integer
              description: The desired summary length.
    responses:
      200:
        description: Generated summary.
        schema:
          type: object
          properties:
            summary:
              type: string
      400:
        description: Error message.
    """
    try:
        logger.info("Received summarize request")
        data = request.get_json()

        query = data["query"]
        if not isinstance(query, str):
            logger.warning("Invalid query format: query should be a string")
            return jsonify({'error': 'query should be a string'}), 400
        if len(query) == 0:
            logger.warning("Empty query received")
            return jsonify({'error': 'query should not be empty'}), 400

        document = data["document"]
        if not isinstance(document, str):
            logger.warning("Invalid document format: document should be a string")
            return jsonify({'error': 'document should be a string'}), 400
        if len(document) == 0:
            logger.warning("Empty document received")
            return jsonify({'error': 'document should not be empty'}), 400

        doc_length = data["length"]
        if not isinstance(doc_length, int):
            logger.warning("Invalid length format: length should be an integer")
            return jsonify({'error': 'length should be an integer'}), 400
        if doc_length <= 0:
            logger.warning("Invalid length value: length should be greater than 0")
            return jsonify({'error': 'length should be greater than 0'}), 400

        logger.debug(f"Summarizing document of length {len(document)} with target length {doc_length}")

        sentences = split_text_into_sentences(
            text=document,
            language='en',
        )

        sentences = [s for s in sentences if len(s) > 0]
        logger.debug(f"Split document into {len(sentences)} sentences")

        sentences_length = [len(s) for s in sentences]
        sentences = np.array(sentences)

        logger.debug(f"Calculating relevance scores for sentences based on query: '{query[:50]}...'")
        scores = crossencoder.predict([(query, s) for s in sentences])

        index = np.argsort(scores).tolist()[::-1]

        total_length = 0
        for i in range(len(index)):
            total_length += sentences_length[index[i]]
            if total_length > doc_length:
                break

        index = index[:i+1]
        index = np.sort(index)

        sentences = sentences[index]
        summary = "\n".join(sentences)

        logger.info(f"Summarization completed successfully. Summary length: {len(summary)}")
        return jsonify({'summary': summary})
    except KeyError as ke:
        error_msg = f"Missing required parameter: {str(ke)}"
        logger.error(error_msg)
        return jsonify({'error': error_msg}), 400
    except Exception as e:
        logger.exception(f"Error in summarize service: {str(e)}")
        return jsonify({'error': str(e)}), 400
