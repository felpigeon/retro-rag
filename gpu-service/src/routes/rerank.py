from flask import Blueprint, request, jsonify
from sentence_transformers import CrossEncoder
from src.services.crossencoder import crossencoder
from src.utils.logger import get_logger


logger = get_logger("rerank_service")
rerank_bp = Blueprint('rerank', __name__)


@rerank_bp.route('/rerank', methods=['POST'])
async def rerank():
    """
    Re-rank documents based on their relevance to a query. Any payload can be added with the documents.
    ---
    tags:
      - Rerank
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
            documents:
              type: array
              items:
                type: object
                properties:
                  text:
                    type: string
                    description: The document text.
    responses:
      200:
        description: Ranked documents.
        schema:
          type: object
          properties:
            query:
              type: string
            ranked_documents:
              type: array
              items:
                type: object
      400:
        description: Error message.
    """
    try:
        logger.info("Received rerank request")
        data = request.get_json()

        query = data["query"]
        if not isinstance(query, str):
            logger.warning("Invalid query format: query should be a string")
            return jsonify({'error': 'query should be a string'}), 400

        documents = data["documents"]
        if not documents:
            logger.info("Empty documents list received, returning empty result")
            return jsonify({
                'query': query,
                'ranked_documents': [],
            })
        if not isinstance(documents, list):
            logger.warning("Invalid documents format: documents should be a list")
            return jsonify({'error': 'documents should be a list'}), 400
        if not all(isinstance(doc, dict) and 'text' in doc for doc in documents):
            logger.warning("Invalid document format: each document should be a dict with a text field")
            return jsonify({'error': 'each document should be a dict with a text field'}), 400
        if not all(isinstance(doc["text"], str) for doc in documents):
            logger.warning("Invalid document text format: each document text should be a string")
            return jsonify({'error': 'each document text should be a string'}), 400

        logger.debug(f"Reranking {len(documents)} documents for query: '{query[:50]}...'")

        texts = [
            doc["text"] for doc in documents
        ]

        scores = crossencoder.predict([
            (query, text)
            for text in texts
        ])

        sorted_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        sorted_docs = [d[0] for d in sorted_docs]

        logger.info(f"Rerank completed successfully for {len(documents)} documents")
        return jsonify({
            'query': query,
            'ranked_documents': sorted_docs,
        })
    except KeyError as ke:
        error_msg = f"Missing required parameter: {str(ke)}"
        logger.error(error_msg)
        return jsonify({'error': error_msg}), 400
    except Exception as e:
        logger.exception(f"Error in rerank service: {str(e)}")
        return jsonify({'error': str(e)}), 400
