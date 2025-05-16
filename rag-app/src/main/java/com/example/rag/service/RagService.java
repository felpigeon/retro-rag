package com.example.rag.service;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.http.ResponseEntity;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.web.util.UriComponentsBuilder;
import com.example.rag.model.IngestRequest;
import com.example.rag.model.QuestionRequest;
import com.example.rag.model.RagResponse;
import com.example.rag.model.BatchIngestRequest;
import com.example.rag.model.BatchIngestResponse;

/**
 * Service responsible for communicating with the Python RAG backend.
 * Handles question processing and text ingestion by forwarding requests
 * to the appropriate Python service endpoints.
 */
@Service
@Slf4j
public class RagService {

    private final RestTemplate restTemplate;
    private final String pythonServiceUrl;

    /**
     * Constructs a new RagService with the specified Python service URL.
     *
     * @param pythonServiceUrl The URL of the Python RAG backend service
     */
    public RagService(@Value("${python.service.url:http://localhost:5000}") String pythonServiceUrl) {
        this.restTemplate = new RestTemplate();
        this.pythonServiceUrl = pythonServiceUrl;
        log.info("RagService initialized with Python service URL: {}", pythonServiceUrl);
    }

    /**
     * Processes a question using the RAG system.
     *
     * @param request The question request containing all parameters needed for processing
     * @return A response containing the answer and supporting documents
     */
    public RagResponse askQuestion(QuestionRequest request) {
        log.info("Processing question: '{}'", request.getQuestion());
        String url = pythonServiceUrl + "/ask";

        UriComponentsBuilder builder = UriComponentsBuilder.fromHttpUrl(url)
            .queryParam("question", request.getQuestion())
            .queryParam("method", request.getMethod())
            .queryParam("k", request.getK())
            .queryParam("filter_by_entity", request.isFilterByEntity())
            .queryParam("do_rerank", request.isDoRerank());

        log.debug("Sending request to: {}", builder.toUriString());
        RagResponse response = restTemplate.getForObject(builder.toUriString(), RagResponse.class);
        log.info("Received response from Python service");
        return response;
    }

    /**
     * Ingests text into the RAG system's knowledge base.
     *
     * @param request The ingest request containing the text and optional metadata
     * @return A response message indicating success or failure
     */
    public String ingestText(IngestRequest request) {
        log.info("Processing text ingest request");
        String url = pythonServiceUrl + "/ingest";
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);

        HttpEntity<IngestRequest> entity = new HttpEntity<>(request, headers);
        log.debug("Sending ingest request to: {}", url);
        ResponseEntity<String> response = restTemplate.postForEntity(url, entity, String.class);

        log.info("Text ingestion completed with status: {}", response.getStatusCode());
        return response.getBody();
    }

    /**
     * Ingests a batch of texts into the RAG system's knowledge base.
     *
     * @param request The batch ingest request containing multiple texts and optional metadata
     * @return A response containing the IDs of successfully ingested documents
     */
    public BatchIngestResponse ingestBatch(BatchIngestRequest request) {
        log.info("Processing batch ingest request with {} documents", request.getDocuments().size());
        String url = pythonServiceUrl + "/ingest_batch";
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);

        HttpEntity<BatchIngestRequest> entity = new HttpEntity<>(request, headers);
        log.debug("Sending batch ingest request to: {}", url);
        ResponseEntity<BatchIngestResponse> response = restTemplate.postForEntity(
            url, entity, BatchIngestResponse.class);

        log.info("Batch ingestion completed with status: {}", response.getStatusCode());
        return response.getBody();
    }
}