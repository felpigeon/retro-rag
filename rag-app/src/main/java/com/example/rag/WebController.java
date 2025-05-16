package com.example.rag;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ExceptionHandler;
import com.example.rag.exception.QuestionRequiredException;
import com.example.rag.model.BatchIngestRequest;
import com.example.rag.model.BatchIngestResponse;
import com.example.rag.model.IngestRequest;
import com.example.rag.model.QuestionRequest;
import com.example.rag.model.RagResponse;
import com.example.rag.service.RagService;
import org.springframework.http.ResponseEntity;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.MissingServletRequestParameterException;
import org.springframework.web.bind.annotation.ResponseBody;
import java.util.Map;
import java.util.Collections;
import java.util.HashMap;


/**
 * Main controller for the RAG application.
 * Handles HTTP requests for question answering and text ingestion.
 * Provides both API endpoints and web interface.
 */
@Controller
@RequiredArgsConstructor
@Slf4j
public class WebController {

	private final RagService ragService;

	/**
	 * Serves the main application page.
	 *
	 * @return The name of the index view template
	 */
	@GetMapping("/")
	public String index() {
		log.debug("Serving index page");
		return "index.html";
	}

	/**
	 * Processes a question and returns the RAG system's response.
	 *
	 * @param question The question to be answered
	 * @param depth Search depth parameter for knowledge exploration
	 * @param method The RAG method to use (e.g., hybrid, dense, sparse)
	 * @param k Number of documents to retrieve
	 * @param filterByEntity Whether to filter results by entity
	 * @param doRerank Whether to rerank the retrieved documents
	 * @return A JSON response containing the answer and supporting documents
	 * @throws QuestionRequiredException If the question parameter is empty or null
	 */
	@GetMapping("/ask")
	@ResponseBody
	public ResponseEntity<RagResponse> ask(
		@RequestParam(required = true) String question,
		@RequestParam(required = false, defaultValue = "0") int depth,
		@RequestParam(required = false, defaultValue = "hybrid") String method,
		@RequestParam(required = false, defaultValue = "5") int k,
		@RequestParam(required = false, defaultValue = "false") boolean filterByEntity,
		@RequestParam(required = false, defaultValue = "false") boolean doRerank
	) {
		log.info("Received question request: '{}' with depth={}, method={}, k={}",
			question, depth, method, k);

		if (question == null || question.trim().isEmpty()) {
			log.warn("Empty question received");
			throw new QuestionRequiredException();
		}

		QuestionRequest request = new QuestionRequest(
			question, method, k, filterByEntity, doRerank
		);

		log.debug("Forwarding question request to RAG service");
		RagResponse response = ragService.askQuestion(request);
		log.info("Question processed successfully");
		return ResponseEntity.ok(response);
	}

	/**
	 * Ingests text into the RAG system's knowledge base.
	 *
	 * @param request Contains the text to be ingested and optional metadata
	 * @return A response indicating success or failure of the ingestion process
	 */
	@PostMapping("/ingest")
	@ResponseBody
	public ResponseEntity<String> ingest(@RequestBody IngestRequest request) {
		log.info("Received ingest");

		if (request.getText() == null || request.getText().trim().isEmpty()) {
			log.warn("Empty text received for ingestion");
			return ResponseEntity.badRequest().body("Text parameter is required");
		}

		log.debug("Forwarding ingest request to RAG service");
		String result = ragService.ingestText(request);
		log.info("Text ingestion completed successfully");
		return ResponseEntity.ok(result);
	}


	/**
	 * Ingests a batch of text documents into the RAG system's knowledge base.
	 *
	 * @param request Contains a list of documents to be ingested with optional metadata
	 * @return A response containing the IDs of the ingested documents
	 */
	@PostMapping("/ingest_batch")
	@ResponseBody
	public ResponseEntity<BatchIngestResponse> ingestBatch(@RequestBody BatchIngestRequest request) {
		log.info("Received batch ingest request with {} documents",
				 request.getDocuments() != null ? request.getDocuments().size() : 0);

		if (request.getDocuments() == null || request.getDocuments().isEmpty()) {
			log.warn("Empty document list received for batch ingestion");
			return ResponseEntity.badRequest().body(new BatchIngestResponse());
		}

		// Validate each document in the batch and ensure entities is never null
		for (IngestRequest doc : request.getDocuments()) {
			if (doc.getText() == null || doc.getText().trim().isEmpty()) {
				log.warn("Empty text found in batch ingestion request");
				return ResponseEntity.badRequest().body(new BatchIngestResponse());
				}

			if (doc.getEntities() == null) {
				doc.setEntities(Collections.emptyMap());
			}
		}

		log.debug("Forwarding batch ingest request to RAG service");
		BatchIngestResponse result = ragService.ingestBatch(request);
		log.info("Batch ingestion completed successfully for {} documents",
				 result.getIds() != null ? result.getIds().size() : 0);
		return ResponseEntity.ok(result);
	}

	/**
	 * Exception handler for missing required request parameters.
	 *
	 * @param ex The exception that was thrown
	 * @return A formatted error response with BAD_REQUEST status
	 */
	@ExceptionHandler(MissingServletRequestParameterException.class)
	public ResponseEntity<Map<String, String>> handleMissingParams(MissingServletRequestParameterException ex) {
		log.warn("Missing required parameter: {}", ex.getMessage());
		Map<String, String> errorResponse = new HashMap<>();
		errorResponse.put("error", "An error occurred");
		errorResponse.put("message", ex.getMessage());
		return new ResponseEntity<>(errorResponse, HttpStatus.BAD_REQUEST);
	}

	/**
	 * Exception handler for missing question parameter.
	 *
	 * @param ex The exception that was thrown
	 * @return A formatted error response with BAD_REQUEST status
	 */
	@ExceptionHandler(QuestionRequiredException.class)
	public ResponseEntity<Map<String, String>> handleQuestionRequiredException(QuestionRequiredException ex) {
		log.warn("Question required exception: {}", ex.getMessage());
		Map<String, String> errorResponse = new HashMap<>();
		errorResponse.put("error", "Question is required");
		errorResponse.put("message", ex.getMessage());
		return new ResponseEntity<>(errorResponse, HttpStatus.BAD_REQUEST);
	}

	/**
	 * General exception handler for all unhandled exceptions.
	 *
	 * @param ex The exception that was thrown
	 * @return A formatted error response with INTERNAL_SERVER_ERROR status
	 */
	@ExceptionHandler(Exception.class)
	public ResponseEntity<Map<String, String>> handleGenericException(Exception ex) {
		log.error("Unhandled exception occurred", ex);
		Map<String, String> errorResponse = new HashMap<>();
		errorResponse.put("error", "An error occurred");
		errorResponse.put("message", ex.getMessage());
		return new ResponseEntity<>(errorResponse, HttpStatus.INTERNAL_SERVER_ERROR);
	}
}
