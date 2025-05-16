package com.example.rag.exception;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.method.annotation.MethodArgumentTypeMismatchException;
import org.springframework.web.bind.MissingServletRequestParameterException;

/**
 * Global exception handler for the RAG application.
 * This class provides centralized exception handling across all controllers in the application.
 * It translates exceptions to appropriate HTTP responses with error details.
 */
@ControllerAdvice
public class GlobalExceptionHandler {

    /**
     * Handles exceptions thrown when a question is required but not provided.
     *
     * @param ex The QuestionRequiredException that was thrown
     * @return ResponseEntity containing error details with HTTP 400 (Bad Request) status
     */
    @ExceptionHandler(QuestionRequiredException.class)
    public ResponseEntity<ErrorResponse> handleQuestionRequiredException(QuestionRequiredException ex) {
        ErrorResponse error = new ErrorResponse(HttpStatus.BAD_REQUEST.value(), ex.getMessage());
        return new ResponseEntity<>(error, HttpStatus.BAD_REQUEST);
    }

    /**
     * Handles exceptions thrown when a required request parameter is missing.
     *
     * @param ex The MissingServletRequestParameterException that was thrown
     * @return ResponseEntity containing error details with HTTP 400 (Bad Request) status
     */
    @ExceptionHandler(MissingServletRequestParameterException.class)
    public ResponseEntity<ErrorResponse> handleMissingServletRequestParameterException(MissingServletRequestParameterException ex) {
        ErrorResponse error = new ErrorResponse(HttpStatus.BAD_REQUEST.value(), "Required parameter '" + ex.getParameterName() + "' is missing");
        return new ResponseEntity<>(error, HttpStatus.BAD_REQUEST);
    }

    /**
     * Handles exceptions thrown when a method argument has the wrong type.
     *
     * @param ex The MethodArgumentTypeMismatchException that was thrown
     * @return ResponseEntity containing error details with HTTP 400 (Bad Request) status
     */
    @ExceptionHandler(MethodArgumentTypeMismatchException.class)
    public ResponseEntity<ErrorResponse> handleMethodArgumentTypeMismatchException(MethodArgumentTypeMismatchException ex) {
        ErrorResponse error = new ErrorResponse(HttpStatus.BAD_REQUEST.value(), "Invalid parameter value for '" + ex.getName() + "'");
        return new ResponseEntity<>(error, HttpStatus.BAD_REQUEST);
    }

    /**
     * Fallback handler for all other unhandled exceptions.
     * This provides a generic error response for unexpected errors.
     *
     * @param ex The Exception that was thrown
     * @return ResponseEntity containing error details with HTTP 500 (Internal Server Error) status
     */
    @ExceptionHandler(Exception.class)
    public ResponseEntity<ErrorResponse> handleGlobalException(Exception ex) {
        ErrorResponse error = new ErrorResponse(HttpStatus.INTERNAL_SERVER_ERROR.value(), "An unexpected error occurred");
        return new ResponseEntity<>(error, HttpStatus.INTERNAL_SERVER_ERROR);
    }
}