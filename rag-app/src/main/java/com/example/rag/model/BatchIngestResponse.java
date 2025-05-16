package com.example.rag.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import java.util.List;
import java.util.ArrayList;

/**
 * Response model for batch document ingestion.
 * Contains the IDs of the documents successfully ingested into the RAG system.
 */
@Data
@AllArgsConstructor
@NoArgsConstructor
public class BatchIngestResponse {
    private List<String> ids = new ArrayList<>();
}
