package com.example.rag;

import com.example.rag.model.BatchIngestRequest;
import com.example.rag.model.BatchIngestResponse;
import com.example.rag.model.HallucinationDetails;
import com.example.rag.model.IngestRequest;
import com.example.rag.model.QuestionRequest;
import com.example.rag.model.RagResponse;
import com.example.rag.service.RagService;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;
import static org.hamcrest.Matchers.hasSize;

import java.util.Arrays;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.hamcrest.Matchers.containsString;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@WebMvcTest(WebController.class)
public class WebControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @MockBean
    private RagService ragService;

    @Autowired
    private ObjectMapper objectMapper;

    @Test
    public void testIndexPage() throws Exception {
        mockMvc.perform(get("/"))
               .andExpect(status().isOk())
               .andExpect(view().name("index.html"));
    }

    @Test
    public void testAskQuestion_Success() throws Exception {
        // Prepare mock response
        RagResponse mockResponse = new RagResponse();
        mockResponse.setQuestion("What is RAG?");
        mockResponse.setAnswer("RAG stands for Retrieval-Augmented Generation.");
        mockResponse.setDocs(new ArrayList<>());
        HallucinationDetails hDetails = new HallucinationDetails();
        hDetails.setHallucination_detected(false);
        hDetails.setHallucination_severity(0.0);
        mockResponse.setHallucination(hDetails);

        // Configure mock service
        when(ragService.askQuestion(any(QuestionRequest.class))).thenReturn(mockResponse);

        // Execute and verify
        mockMvc.perform(get("/ask")
                .param("question", "What is RAG?")
                .accept(MediaType.APPLICATION_JSON))
               .andExpect(status().isOk())
               .andExpect(jsonPath("$.question").value("What is RAG?"))
               .andExpect(jsonPath("$.answer").value("RAG stands for Retrieval-Augmented Generation."))
               .andExpect(jsonPath("$.hallucination.hallucination_detected").value(false));
    }

    @Test
    public void testAskQuestion_EmptyQuestion() throws Exception {
        // Execute and verify
        mockMvc.perform(get("/ask")
                .param("question", "")
                .accept(MediaType.APPLICATION_JSON))
               .andExpect(status().isBadRequest())
               .andExpect(jsonPath("$.error").value("Question is required"));
    }

    @Test
    public void testAskQuestion_MissingQuestion() throws Exception {
        // Execute and verify
        mockMvc.perform(get("/ask")
                .accept(MediaType.APPLICATION_JSON))
                .andExpect(status().is(400))  // Explicitly check for 400 status code
                .andExpect(content().contentType(MediaType.APPLICATION_JSON))
                .andExpect(jsonPath("$.error").value("An error occurred"))
                .andExpect(jsonPath("$.message").value("Required request parameter 'question' for method parameter type String is not present"));
    }


    @Test
    public void testIngestBatch_Success() throws Exception {
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

        // Configure mock response
        BatchIngestResponse mockResponse = new BatchIngestResponse();
        mockResponse.setIds(Arrays.asList("doc-001", "doc-002"));
        when(ragService.ingestBatch(any(BatchIngestRequest.class))).thenReturn(mockResponse);

        // Execute and verify
        mockMvc.perform(post("/ingest_batch")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(request)))
               .andExpect(status().isOk())
               .andExpect(jsonPath("$.ids", hasSize(2)))
               .andExpect(jsonPath("$.ids[0]").value("doc-001"))
               .andExpect(jsonPath("$.ids[1]").value("doc-002"));
    }

    @Test
    public void testIngestBatch_EmptyBatch() throws Exception {
        // Prepare test data with empty document list
        BatchIngestRequest request = new BatchIngestRequest();
        request.setDocuments(Collections.emptyList());

        // Execute and verify
        mockMvc.perform(post("/ingest_batch")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(request)))
               .andExpect(status().isBadRequest());
    }

    @Test
    public void testIngestBatch_NullBatch() throws Exception {
        // Prepare test data with null document list
        BatchIngestRequest request = new BatchIngestRequest();
        request.setDocuments(null);

        // Execute and verify
        mockMvc.perform(post("/ingest_batch")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(request)))
               .andExpect(status().isBadRequest());
    }

    @Test
    public void testIngestBatch_EmptyTextInDocument() throws Exception {
        // Prepare test data with an empty text in one document
        BatchIngestRequest request = new BatchIngestRequest();
        List<IngestRequest> documents = new ArrayList<>();

        IngestRequest doc1 = new IngestRequest();
        doc1.setText("Document 1 text");
        documents.add(doc1);

        IngestRequest doc2 = new IngestRequest();
        doc2.setText(""); // Empty text
        documents.add(doc2);

        request.setDocuments(documents);

        // Execute and verify
        mockMvc.perform(post("/ingest_batch")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(request)))
               .andExpect(status().isBadRequest());
    }

    @Test
    public void testIngestText_Success() throws Exception {
        // Prepare test data
        IngestRequest request = new IngestRequest();
        request.setText("Sample text for ingestion");

        // Configure mock service
        when(ragService.ingestText(any(IngestRequest.class))).thenReturn("Text ingested successfully");

        // Execute and verify
        mockMvc.perform(post("/ingest")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(request)))
               .andExpect(status().isOk())
               .andExpect(content().string("Text ingested successfully"));
    }

    @Test
    public void testIngestText_EmptyText() throws Exception {
        // Prepare test data
        IngestRequest request = new IngestRequest();
        request.setText("");

        // Execute and verify
        mockMvc.perform(post("/ingest")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(request)))
               .andExpect(status().isBadRequest())
               .andExpect(content().string(containsString("Text parameter is required")));
    }

    @Test
    public void testAskQuestion_WithAllParameters() throws Exception {
        // Prepare mock response
        RagResponse mockResponse = new RagResponse();
        mockResponse.setQuestion("Complex question");
        mockResponse.setAnswer("Detailed answer");
        mockResponse.setDocs(Collections.emptyList());
        HallucinationDetails hDetails = new HallucinationDetails();
        hDetails.setHallucination_detected(false);
        hDetails.setHallucination_severity(0.0);
        mockResponse.setHallucination(hDetails);

        // Configure mock service
        when(ragService.askQuestion(any(QuestionRequest.class))).thenReturn(mockResponse);

        // Execute and verify
        mockMvc.perform(get("/ask")
                .param("question", "Complex question")
                .param("depth", "2")
                .param("method", "dense")
                .param("k", "10")
                .param("filterByEntity", "true")
                .param("doRerank", "true")
                .accept(MediaType.APPLICATION_JSON))
               .andExpect(status().isOk())
               .andExpect(jsonPath("$.question").value("Complex question"))
               .andExpect(jsonPath("$.answer").value("Detailed answer"));
    }
}
