from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Literal, Any, Union


class SparseVector(BaseModel):
    """Representation of a sparse vector with indices and values."""
    indices: List[int] = Field(..., description="Indices of non-zero elements")
    values: List[float] = Field(..., description="Values of non-zero elements")


class Document(BaseModel):
    """Represents a document with its text, vector representations, and entity information."""
    doc_id: str = Field(..., description="Unique identifier for the document")
    text: str = Field(..., description="The text content of the document")
    sparse_vec: SparseVector = Field(..., description="Sparse vector representation of the document")
    dense_vec: List[float] = Field(..., description="Dense vector representation of the document")
    entities: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Dictionary of entity types to their values"
    )
