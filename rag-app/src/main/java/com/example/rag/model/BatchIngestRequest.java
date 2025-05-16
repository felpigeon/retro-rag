package com.example.rag.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import java.util.List;

/**
 * Request model for batch document ingestion.
 * Contains a list of documents to be ingested into the RAG system.
 */
@Data
@AllArgsConstructor
@NoArgsConstructor
public class BatchIngestRequest {
    private List<IngestRequest> documents;
}
