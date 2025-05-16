package com.example.rag.model;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

public class ModelTest {

    @Test
    public void testQuestionRequest() {
        // Test default constructor
        QuestionRequest defaultRequest = new QuestionRequest();
        assertEquals("hybrid", defaultRequest.getMethod());
        assertEquals(5, defaultRequest.getK());
        assertFalse(defaultRequest.isFilterByEntity());
        assertFalse(defaultRequest.isDoRerank());

        // Test constructor with question
        String question = "What is RAG?";
        QuestionRequest questionRequest = new QuestionRequest(question);
        assertEquals(question, questionRequest.getQuestion());
        assertEquals("hybrid", questionRequest.getMethod());

        // Test full constructor
        QuestionRequest fullRequest = new QuestionRequest(
                "Full question", "dense", 10, true, true);
        assertEquals("Full question", fullRequest.getQuestion());
        assertEquals("dense", fullRequest.getMethod());
        assertEquals(10, fullRequest.getK());
        assertTrue(fullRequest.isFilterByEntity());
        assertTrue(fullRequest.isDoRerank());
    }

    @Test
    public void testIngestRequest() {
        IngestRequest request = new IngestRequest();
        request.setText("Test text");

        Map<String, List<String>> entities = new HashMap<>();
        List<String> persons = new ArrayList<>();
        persons.add("John Doe");
        entities.put("PERSON", persons);
        request.setEntities(entities);

        assertEquals("Test text", request.getText());
        assertEquals(1, request.getEntities().size());
        assertEquals("John Doe", request.getEntities().get("PERSON").get(0));
    }

    @Test
    public void testRagResponse() {
        RagResponse response = new RagResponse();
        response.setQuestion("What is RAG?");
        response.setAnswer("RAG is Retrieval-Augmented Generation");

        List<Map<String, Object>> docs = new ArrayList<>();
        Map<String, Object> doc = new HashMap<>();
        doc.put("text", "RAG is a technique...");
        doc.put("score", 0.95);
        docs.add(doc);
        response.setDocs(docs);

        HallucinationDetails hallucination = new HallucinationDetails();
        hallucination.setHallucination_detected(false);
        hallucination.setHallucination_severity(0.1);
        response.setHallucination(hallucination);

        assertEquals("What is RAG?", response.getQuestion());
        assertEquals("RAG is Retrieval-Augmented Generation", response.getAnswer());
        assertEquals(1, response.getDocs().size());
        assertEquals(0.95, response.getDocs().get(0).get("score"));
        assertFalse(response.getHallucination().isHallucination_detected());
        assertEquals(0.1, response.getHallucination().getHallucination_severity());
    }

    @Test
    public void testBatchIngestRequest() {
        BatchIngestRequest request = new BatchIngestRequest();
        List<IngestRequest> documents = new ArrayList<>();

        IngestRequest doc1 = new IngestRequest();
        doc1.setText("Document 1");
        documents.add(doc1);

        IngestRequest doc2 = new IngestRequest();
        doc2.setText("Document 2");
        documents.add(doc2);

        request.setDocuments(documents);

        assertEquals(2, request.getDocuments().size());
        assertEquals("Document 1", request.getDocuments().get(0).getText());
        assertEquals("Document 2", request.getDocuments().get(1).getText());
    }

    @Test
    public void testBatchIngestResponse() {
        BatchIngestResponse response = new BatchIngestResponse();
        List<String> ids = new ArrayList<>();
        ids.add("doc-001");
        ids.add("doc-002");

        response.setIds(ids);

        assertEquals(2, response.getIds().size());
        assertEquals("doc-001", response.getIds().get(0));
        assertEquals("doc-002", response.getIds().get(1));
    }
}
