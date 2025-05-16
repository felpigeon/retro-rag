package com.example.rag.model;

import lombok.Data;
import java.util.List;

/**
 * Contains details about potential hallucinations detected in the generated answer.
 * Hallucinations are parts of the answer that aren't supported by the provided context.
 */
@Data
public class HallucinationDetails {
    /**
     * Indicates whether a hallucination was detected
     */
    private boolean hallucination_detected;

    /**
     * The severity of the detected hallucination (0.0 to 1.0)
     */
    private double hallucination_severity;

    /**
     * Detailed results from the consistency checking process
     */
    private List<Object> ck_results;
}
