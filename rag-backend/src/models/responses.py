from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Literal, Any, Union


class IngestResult(BaseModel):
    """Result of a single document ingestion."""
    id: str = Field(..., description="The ID of the ingested document")
    status: str = Field(default="success", description="Status of the ingestion operation")
    error: Optional[str] = Field(default=None, description="Error message, if any")


class BatchIngestResult(BaseModel):
    """Result of a batch ingestion operation."""
    summary: Dict[str, int] = Field(
        ...,
        description="Summary statistics of the batch operation"
    )
    results: List[Dict[str, Any]] = Field(
        ...,
        description="List of individual document ingestion results"
    )