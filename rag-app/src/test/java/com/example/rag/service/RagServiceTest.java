package com.example.rag.service;

import com.example.rag.model.HallucinationDetails;
import com.example.rag.model.IngestRequest;
import com.example.rag.model.QuestionRequest;
import com.example.rag.model.RagResponse;
import com.example.rag.model.BatchIngestRequest;
import com.example.rag.model.BatchIngestResponse;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.test.util.ReflectionTestUtils;
import org.springframework.web.client.RestTemplate;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
public class RagServiceTest {

    @Mock
    private RestTemplate restTemplate;

    private RagService ragService;

    private final String pythonServiceUrl = "http://localhost:5000";

    @BeforeEach
    public void setUp() {
        ragService = new RagService(pythonServiceUrl);
        // Use reflection to replace the automatically created RestTemplate with our mock
        ReflectionTestUtils.setField(ragService, "restTemplate", restTemplate);
    }

    @Test
    public void testAskQuestion() {
        // Prepare test data
        QuestionRequest request = new QuestionRequest(
            "What is RAG?", "hybrid", 5, false, false
        );

        // Prepare mock response
        RagResponse mockResponse = new RagResponse();
        mockResponse.setQuestion("What is RAG?");
        mockResponse.setAnswer("RAG stands for Retrieval-Augmented Generation.");
        mockResponse.setDocs(new ArrayList<>());
        HallucinationDetails hDetails = new HallucinationDetails();
        hDetails.setHallucination_detected(false);
        hDetails.setHallucination_severity(0.0);
        mockResponse.setHallucination(hDetails);

        // Configure mock
        when(restTemplate.getForObject(contains("/ask"), eq(RagResponse.class)))
            .thenReturn(mockResponse);

        // Execute
        RagResponse result = ragService.askQuestion(request);

        // Verify
        assertNotNull(result);
        assertEquals("What is RAG?", result.getQuestion());
        assertEquals("RAG stands for Retrieval-Augmented Generation.", result.getAnswer());
        assertFalse(result.getHallucination().isHallucination_detected());
    }

    @Test
    public void testIngestText() {
        // Prepare test data
        IngestRequest request = new IngestRequest();
        request.setText("Sample text for ingestion");

        // Prepare mock response
        ResponseEntity<String> mockResponse = new ResponseEntity<>("Text ingested successfully", HttpStatus.OK);

        // Configure mock
        when(restTemplate.postForEntity(eq(pythonServiceUrl + "/ingest"), any(HttpEntity.class), eq(String.class)))
            .thenReturn(mockResponse);

        // Execute
        String result = ragService.ingestText(request);

        // Verify
        assertEquals("Text ingested successfully", result);
    }

    @Test
    public void testIngestBatch() {
        // Prepare test data
        BatchIngestRequest request = new BatchIngestRequest();
        List<IngestRequest> documents = new ArrayList<>();

        IngestRequest doc1 = new IngestRequest();
        doc1.setText("Document 1 text");
        documents.add(doc1);

        IngestRequest doc2 = new IngestRequest();
        doc2.setText("Document 2 text");
        documents.add(doc2);

        request.setDocuments(documents);

        // Prepare mock response
        BatchIngestResponse mockResponse = new BatchIngestResponse();
        mockResponse.setIds(Arrays.asList("doc-001", "doc-002"));
        ResponseEntity<BatchIngestResponse> responseEntity =
            new ResponseEntity<>(mockResponse, HttpStatus.OK);

        // Configure mock
        when(restTemplate.postForEntity(
            eq(pythonServiceUrl + "/ingest_batch"),
            any(HttpEntity.class),
            eq(BatchIngestResponse.class)))
            .thenReturn(responseEntity);

        // Execute
        BatchIngestResponse result = ragService.ingestBatch(request);

        // Verify
        assertNotNull(result);
        assertEquals(2, result.getIds().size());
        assertEquals("doc-001", result.getIds().get(0));
        assertEquals("doc-002", result.getIds().get(1));
    }
}
