from flask import Blueprint, request, jsonify

from src.services.ingest import ingest_documents
from src.models.requests import IngestRequest, BatchIngestRequest
from src.utils.logger import get_logger


ingest_bp = Blueprint('ingest', __name__)
logger = get_logger(__name__)


@ingest_bp.route('/ingest', methods=['POST'])
async def ingest():
    """
    Ingest a document with associated entities and triples.
    ---
    tags:
      - Ingest
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            text:
              type: string
              description: The text of the document to ingest.
            entities:
              type: array
              items:
                type: string
              description: A list of entities associated with the document.
            triples:
              type: array
              items:
                type: object
                properties:
                  subject:
                    type: string
                  predicate:
                    type: string
                  object:
                    type: string
              description: A list of triples associated with the document.
    responses:
      200:
        description: Successfully ingested the document.
        schema:
          type: object
          properties:
            message:
              type: string
              example: Successfully ingested text
            id:
              type: string
              description: The ID of the ingested document.
      400:
        description: Bad request. Either no data was provided or an error occurred during processing.
        schema:
          type: object
          properties:
            error:
              type: string
              example: No data provided
    """
    try:
        logger.info("Received document ingest request")
        data = request.get_json()
        if not data:
            logger.warning("Request missing required JSON data")
            return jsonify({'error': 'No data provided'}), 400

        document = IngestRequest(**data)
        logger.debug(f"Processing ingest request with document length: {len(document.text)} chars, "
                     f"{len(document.entities)} entities")

        docs_ids = await ingest_documents([document])

        logger.info(f"Successfully ingested document with ID: {docs_ids[0]}")
        return jsonify({"id": docs_ids[0]})
    except Exception as e:
        logger.error(f"Error ingesting document: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 400


@ingest_bp.route('/ingest_batch', methods=['POST'])
async def ingest_batch():
    """
    Ingest a batch of documents with associated entities and triples.
    ---
    tags:
      - Ingest
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            documents:
              type: array
              items:
                type: object
                properties:
                  text:
                    type: string
                    description: The text of the document to ingest.
                  entities:
                    type: object
                    additionalProperties:
                      type: array
                      items:
                        type: string
                    description: Dictionary of entity types to their values.
                  triples:
                    type: array
                    items:
                      type: array
                      items:
                        type: string
                    description: A list of triples associated with the document.
              description: List of documents to ingest
    responses:
      200:
        description: Results of the batch ingestion operation.
        schema:
          type: object
          properties:
            ids:
              type: array
              items:
                type: string
              description: List of document IDs that were successfully ingested
      400:
        description: Bad request. Either no data was provided or the data format is invalid.
        schema:
          type: object
          properties:
            error:
              type: string
              example: No data provided
    """
    try:
        logger.info("Received batch ingest request")
        data = request.get_json()
        if not data:
            logger.warning("Request missing required JSON data")
            return jsonify({'error': 'No data provided'}), 400

        # Validate request data
        batch_request = BatchIngestRequest(**data)
        document_count = len(batch_request.documents)
        logger.info(f"Processing batch ingest request with {document_count} documents")

        # Process the batch
        docs_ids = await ingest_documents(batch_request.documents)

        logger.info(f"Successfully ingested document with ID: {docs_ids[0]}")
        return jsonify({"ids": docs_ids})

    except Exception as e:
        logger.error(f"Error processing batch ingest: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 400
