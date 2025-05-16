from flask import Blueprint, request, jsonify
from src.services.hallucination import hdm
from src.utils.logger import get_logger


logger = get_logger("hallucination_detection")
detect_hallucination_bp = Blueprint('detect_hallucination', __name__)


@detect_hallucination_bp.route('/detect_hallucination', methods=['POST'])
async def detect_hallucination():
    """
    Detect hallucinations in a response based on a query and context.
    ---
    tags:
      - Hallucination Detection
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
            context:
              type: string
              description: The context text.
            response:
              type: string
              description: The response to analyze.
    responses:
      200:
        description: Hallucination detection results.
        schema:
          type: object
          properties:
            hallucination_detected:
              type: boolean
            hallucination_severity:
              type: number
              format: float
            ck_results:
              type: array
              items:
                type: object
                properties:
                  hallucination_probability:
                    type: number
                    format: float
                  non_hallucination_probability:
                    type: number
                    format: float
                  prediction:
                    type: integer
                  text:
                    type: string
      400:
        description: Error message.
    """
    try:
        logger.info("Received hallucination detection request")
        data = request.get_json()

        query = data["query"]
        context = data["context"]
        response = data["response"]

        logger.debug(f"Processing query: '{query[:50]}...' with context length: {len(context)}")

        results = hdm.apply(query, context, response)

        keys = ["hallucination_detected", "hallucination_severity", "ck_results"]
        results = {
            k: results[k]
            for k in keys
        }

        hallucination_detected = results["hallucination_detected"]
        logger.info(f"Hallucination detection completed. Result: {hallucination_detected}")

        return jsonify(results)
    except KeyError as ke:
        error_msg = f"Missing required parameter: {str(ke)}"
        logger.error(error_msg)
        return jsonify({'error': error_msg}), 400
    except Exception as e:
        logger.exception(f"Error in hallucination detection: {str(e)}")
        return jsonify({'error': str(e)}), 400

