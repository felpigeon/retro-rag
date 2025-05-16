package com.example.rag.exception;

/**
 * Error response model used for returning structured error information to clients.
 * This class encapsulates HTTP status codes and error messages in a standardized format
 * that can be serialized as JSON in API responses.
 */
public class ErrorResponse {
    /**
     * The HTTP status code associated with this error.
     */
    private int status;

    /**
     * A descriptive message explaining the error.
     */
    private String message;

    /**
     * Constructs a new ErrorResponse with the specified status code and message.
     *
     * @param status  The HTTP status code representing the error
     * @param message A descriptive message explaining the error
     */
    public ErrorResponse(int status, String message) {
        this.status = status;
        this.message = message;
    }

    /**
     * Gets the HTTP status code for this error.
     *
     * @return The HTTP status code
     */
    public int getStatus() {
        return status;
    }

    /**
     * Sets the HTTP status code for this error.
     *
     * @param status The HTTP status code to set
     */
    public void setStatus(int status) {
        this.status = status;
    }

    /**
     * Gets the descriptive error message.
     *
     * @return The error message
     */
    public String getMessage() {
        return message;
    }

    /**
     * Sets the descriptive error message.
     *
     * @param message The error message to set
     */
    public void setMessage(String message) {
        this.message = message;
    }
}