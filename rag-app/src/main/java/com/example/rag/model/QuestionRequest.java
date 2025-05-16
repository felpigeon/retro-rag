package com.example.rag.model;

import lombok.Data;

/**
 * Represents a question request to the RAG system.
 * Contains all parameters needed to process a question.
 */
@Data
public class QuestionRequest {
    /**
     * The question text to be answered
     */
    private String question;

    /**
     * The RAG method to use (e.g., hybrid, dense, sparse)
     */
    private String method;

    /**
     * Number of documents to retrieve
     */
    private int k;

    /**
     * Whether to filter results by entity
     */
    private boolean filterByEntity;

    /**
     * Whether to rerank the retrieved documents
     */
    private boolean doRerank;

    // Default constructor
    public QuestionRequest() {
        this.method = "hybrid";
        this.k = 5;
        this.filterByEntity = false;
        this.doRerank = false;
    }

    // Constructor with required question
    public QuestionRequest(String question) {
        this();
        this.question = question;
    }

    // Full constructor
    public QuestionRequest(
        String question,
        String method, int k,
        boolean filterByEntity,
        boolean doRerank
    ) {
        this.question = question;
        this.method = method;
        this.k = k;
        this.filterByEntity = filterByEntity;
        this.doRerank = doRerank;
    }
}
