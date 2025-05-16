package com.example.rag.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import java.util.Map;
import java.util.List;
import java.util.HashMap;

/**
 * Request model for document ingestion.
 * Contains the text to be ingested and optional entities.
 */
@Data
@AllArgsConstructor
@NoArgsConstructor
public class IngestRequest {
    private String text;
    private Map<String, List<String>> entities = new HashMap<>();
}