from pydantic import BaseModel, Field
from typing import List, Dict, Literal


class QuestionRequest(BaseModel):
    """Request model for question answering operations that specifies the question and retrieval parameters."""
    question: str = Field(..., min_length=1, description="The question to be answered")
    method: Literal["bm25", "dense", "hybrid"] = Field(
        default="hybrid",
        description="Search method to use"
    )
    k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of documents to retrieve (1-100)"
    )
    filter_by_entity: bool = Field(default=False, description="Whether to filter results by entity")
    do_rerank: bool = Field(default=False, description="Whether to rerank the results")


class IngestRequest(BaseModel):
    """Request model for ingesting a single document with optional entity metadata."""
    text: str = Field(..., min_length=1, description="The document to be ingested")
    entities: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Dictionary of entity types to their values (e.g., {'Game': ['Mario'], 'Console': ['Switch']})"
    )

class BatchIngestRequest(BaseModel):
    """Request for batch ingestion of documents."""
    documents: List[IngestRequest] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of documents to be ingested"
    )