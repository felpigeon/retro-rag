"""
Tests for data models validation.
"""

import pytest
from pydantic import ValidationError

from src.models.requests import QuestionRequest, IngestRequest, BatchIngestRequest
from src.models.responses import IngestResult, BatchIngestResult
from src.models.document import Document, SparseVector


class TestQuestionRequest:
    """Tests for QuestionRequest model."""

    def test_question_request_required_fields(self):
        """Test that question is required."""
        with pytest.raises(ValidationError):
            QuestionRequest()

        with pytest.raises(ValidationError):
            QuestionRequest(question=None)

    def test_question_request_defaults(self):
        """Test default values are set correctly."""
        request = QuestionRequest(question="What is the law?")
        assert request.question == "What is the law?"
        assert request.method == "hybrid"
        assert request.k == 5
        assert request.filter_by_entity is False
        assert request.do_rerank is False

    def test_question_request_validation(self):
        """Test field validation rules."""
        # Test invalid method
        with pytest.raises(ValidationError):
            QuestionRequest(question="test", method="invalid_method")

        # Test valid methods
        for method in ["bm25", "dense", "hybrid"]:
            request = QuestionRequest(question="test", method=method)
            assert request.method == method

        # Test k range validation
        with pytest.raises(ValidationError):
            QuestionRequest(question="test", k=0)

        with pytest.raises(ValidationError):
            QuestionRequest(question="test", k=101)  # Max k is 100

        # Test valid k values
        for k in [1, 50, 100]:
            request = QuestionRequest(question="test", k=k)
            assert request.k == k

    def test_question_request_full_initialization(self):
        """Test full initialization with all parameters."""
        request = QuestionRequest(
            question="What is the law?",
            method="bm25",
            k=20,
            filter_by_entity=True,
            do_rerank=False
        )

        assert request.question == "What is the law?"
        assert request.method == "bm25"
        assert request.k == 20
        assert request.filter_by_entity is True
        assert request.do_rerank is False


class TestIngestRequest:
    """Tests for IngestRequest model."""

    def test_ingest_request_required_fields(self):
        """Test that text is required."""
        with pytest.raises(ValidationError):
            IngestRequest()

        with pytest.raises(ValidationError):
            IngestRequest(text=None)

        # Minimal valid request
        request = IngestRequest(text="This is a test document")
        assert request.text == "This is a test document"
        assert request.entities == {}

    def test_ingest_request_entities_validation(self):
        """Test entities field validation."""
        # Valid entities
        valid_entities = {
            "Person": ["John Doe", "Jane Smith"],
            "Organization": ["Acme Corp"]
        }
        request = IngestRequest(text="test", entities=valid_entities)
        assert request.entities == valid_entities

        # Invalid entities (not a dict)
        with pytest.raises(ValidationError):
            IngestRequest(text="test", entities=["John", "Jane"])

        # Invalid entities (values not a list)
        with pytest.raises(ValidationError):
            IngestRequest(text="test", entities={"Person": "John"})

    def test_ingest_request_full_initialization(self):
        """Test full initialization with all parameters."""
        text = "This is a test document about John and Acme Corp."
        entities = {
            "Person": ["John"],
            "Organization": ["Acme Corp"]
        }

        request = IngestRequest(
            text=text,
            entities=entities
        )

        assert request.text == text
        assert request.entities == entities


class TestBatchIngestRequest:
    """Tests for BatchIngestRequest model."""

    def test_batch_ingest_request_required_fields(self):
        """Test that documents list is required and validated."""
        with pytest.raises(ValidationError):
            BatchIngestRequest()

        with pytest.raises(ValidationError):
            BatchIngestRequest(documents=[])  # Empty list is invalid

    def test_batch_ingest_request_validation(self):
        """Test batch request validation."""
        documents = [
            IngestRequest(text="Document 1"),
            IngestRequest(text="Document 2", entities={"Person": ["John"]})
        ]

        request = BatchIngestRequest(documents=documents)
        assert len(request.documents) == 2
        assert request.documents[0].text == "Document 1"
        assert request.documents[1].entities == {"Person": ["John"]}


class TestResponseModels:
    """Tests for response models."""

    def test_ingest_result(self):
        """Test IngestResult model."""
        # Success case
        result = IngestResult(id="doc123")
        assert result.id == "doc123"
        assert result.status == "success"
        assert result.error is None

        # Error case
        result = IngestResult(id="doc456", status="error", error="Failed to process")
        assert result.id == "doc456"
        assert result.status == "error"
        assert result.error == "Failed to process"

    def test_batch_ingest_result(self):
        """Test BatchIngestResult model."""
        summary = {"total": 10, "success": 8, "error": 2}
        results = [
            {"id": "doc1", "status": "success"},
            {"id": "doc2", "status": "error", "error": "Invalid format"}
        ]

        batch_result = BatchIngestResult(summary=summary, results=results)
        assert batch_result.summary == summary
        assert len(batch_result.results) == 2
        assert batch_result.results[0]["id"] == "doc1"
        assert batch_result.results[1]["error"] == "Invalid format"


class TestDocumentModel:
    """Tests for Document model."""

    def test_document_required_fields(self):
        """Test required fields for Document model."""
        with pytest.raises(ValidationError):
            Document()

        with pytest.raises(ValidationError):
            Document(doc_id="doc1", text="content")  # Missing vectors

    def test_document_full_initialization(self):
        """Test full initialization of Document model."""
        doc = Document(
            doc_id="doc123",
            text="This is a test document",
            sparse_vec=SparseVector(indices=[1, 5, 10], values=[0.5, 0.3, 0.8]),
            dense_vec=[0.1, 0.2, 0.3, 0.4],
            entities={"Person": ["John"]}
        )

        assert doc.doc_id == "doc123"
        assert doc.text == "This is a test document"
        assert doc.sparse_vec.indices == [1, 5, 10]
        assert doc.sparse_vec.values == [0.5, 0.3, 0.8]
        assert doc.dense_vec == [0.1, 0.2, 0.3, 0.4]
        assert doc.entities == {"Person": ["John"]}
