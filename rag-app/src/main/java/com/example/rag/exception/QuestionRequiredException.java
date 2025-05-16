package com.example.rag.exception;

/**
 * Exception thrown when a required question parameter is missing or empty.
 * This indicates that the client provided an invalid request.
 */
public class QuestionRequiredException extends RuntimeException {

    /**
     * Constructs a new QuestionRequiredException with the default message.
     */
    public QuestionRequiredException() {
        super("Question parameter is required and cannot be empty");
    }

    /**
     * Constructs a new QuestionRequiredException with the specified message.
     *
     * @param message The detail message
     */
    public QuestionRequiredException(String message) {
        super(message);
    }
}