"""
Tests for entity extraction, matching, and formatting functionality.
"""

import pytest
from unittest.mock import patch, MagicMock, call, AsyncMock
import os

from src.services.entity import EntityExtractor


class TestEntityExtraction:
    """Tests for entity extraction functionality."""

    @pytest.fixture
    def entity_extractor(self):
        """Create an instance of EntityExtractor for testing."""
        return EntityExtractor(db_path="test_entities.db")

    @pytest.fixture
    def mock_baml_client(self):
        """Mock the BAML client."""
        with patch("src.services.entity.b") as mock:
            mock.ExtractEntities = AsyncMock()
            yield mock

    @pytest.fixture
    def mock_formatted_entities(self):
        """Return mock formatted entities."""
        return {
            "Game": ["super mario bros"],
            "Console": ["nintendo switch"]
        }

    @pytest.mark.asyncio
    async def test_extract_entities(self, entity_extractor, mock_baml_client, mock_formatted_entities):
        """Test extracting entities from a question."""
        # Setup
        question = "What is Super Mario Bros on Nintendo Switch?"

        # Create mock entities
        mario_entity = MagicMock()
        mario_entity.name = "Super Mario Bros"
        mario_entity.type = "Game"

        switch_entity = MagicMock()
        switch_entity.name = "Nintendo Switch"
        switch_entity.type = "Console"

        expected_entities = [mario_entity, switch_entity]
        mock_baml_client.ExtractEntities.return_value = expected_entities

        # Execute with mocked format_entities
        with patch.object(entity_extractor, 'format_entities', return_value=mock_formatted_entities) as mock_format:
            result = await entity_extractor.extract_entities(question)

        # Assert
        mock_baml_client.ExtractEntities.assert_awaited_once_with(question)
        mock_format.assert_called_once_with(expected_entities)
        assert result == mock_formatted_entities

    @pytest.mark.asyncio
    async def test_extract_entities_empty(self, entity_extractor, mock_baml_client):
        """Test extracting entities when none are found."""
        # Setup
        question = "What is the meaning of life?"
        mock_baml_client.ExtractEntities.return_value = []

        # Execute
        result = await entity_extractor.extract_entities(question)

        # Assert
        mock_baml_client.ExtractEntities.assert_awaited_once_with(question)
        assert result is None

    @pytest.mark.asyncio
    async def test_extract_entities_empty_question(self, entity_extractor, mock_baml_client):
        """Test extracting entities with an empty question."""
        # Setup
        question = "   "

        # Execute
        result = await entity_extractor.extract_entities(question)

        # Assert
        mock_baml_client.ExtractEntities.assert_not_called()
        assert result is None


class TestEntityMatching:
    """Tests for entity matching functionality."""

    @pytest.fixture
    def entity_extractor(self):
        """Create an instance of EntityExtractor for testing."""
        return EntityExtractor(db_path="test_entities.db")

    @pytest.fixture
    def mock_duckdb_connect(self):
        """Mock DuckDB connection."""
        with patch("src.services.entity.duckdb.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_connect.return_value.__enter__.return_value = mock_conn
            yield mock_connect, mock_conn

    def test_match_entity_found(self, entity_extractor, mock_duckdb_connect):
        """Test matching an entity that exists in the database."""
        # Setup
        mock_connect, mock_conn = mock_duckdb_connect
        term = "Mario"
        entity_type = "Game"
        mock_conn.execute.return_value.fetchall.return_value = [("Super Mario Bros.", 0.05)]

        # Execute
        result = entity_extractor.match_entity(term, entity_type)

        # Assert
        mock_connect.assert_called_once_with("test_entities.db")
        assert result == "Super Mario Bros."

    def test_match_entity_not_found(self, entity_extractor, mock_duckdb_connect):
        """Test matching an entity that does not exist in the database."""
        # Setup
        mock_connect, mock_conn = mock_duckdb_connect
        term = "NonexistentGame"
        entity_type = "Game"
        mock_conn.execute.return_value.fetchall.return_value = []

        # Execute
        result = entity_extractor.match_entity(term, entity_type)

        # Assert
        mock_connect.assert_called_once()
        assert result is None

    def test_match_entity_empty_term(self, entity_extractor):
        """Test matching with an empty term."""
        # Execute
        result = entity_extractor.match_entity("", "Game")

        # Assert
        assert result is None

    def test_match_entities(self, entity_extractor):
        """Test matching multiple entities."""
        # Setup
        entities = {
            "Game": ["mario", "zelda"],
            "Console": ["switch"]
        }

        with patch.object(entity_extractor, 'match_entity') as mock_match:
            def match_side_effect(term, entity_type):
                if term == "mario" and entity_type == "Game":
                    return "Super Mario Bros."
                elif term == "zelda" and entity_type == "Game":
                    return "The Legend of Zelda"
                elif term == "switch" and entity_type == "Console":
                    return "Nintendo Switch"
                return None

            mock_match.side_effect = match_side_effect

            # Execute
            result = entity_extractor.match_entities(entities)

            # Assert
            assert mock_match.call_count == 3
            assert result["Game"] == ["Super Mario Bros.", "The Legend of Zelda"]
            assert result["Console"] == ["Nintendo Switch"]


class TestEntityFormatting:
    """Tests for entity formatting functionality."""

    @pytest.fixture
    def entity_extractor(self):
        """Create an instance of EntityExtractor for testing."""
        return EntityExtractor(db_path="test_entities.db")

    @pytest.fixture
    def raw_entities(self):
        """Return sample raw entities for testing."""
        # Create mock entities that match the structure expected by format_entities
        game_entity = MagicMock()
        game_entity.name = "Mario"
        game_entity.type = "Game"

        console_entity = MagicMock()
        console_entity.name = "Switch"
        console_entity.type = "Console"

        publisher_entity = MagicMock()
        publisher_entity.name = "Nintendo"
        publisher_entity.type = "Publisher"

        unknown_entity = MagicMock()
        unknown_entity.name = "Unknown"
        unknown_entity.type = "Unknown"

        return [game_entity, console_entity, publisher_entity, unknown_entity]

    def test_format_entities(self, entity_extractor, raw_entities):
        """Test formatting entities."""
        # Execute
        result = entity_extractor.format_entities(raw_entities)

        # Assert
        assert "Game" in result
        assert "Console" in result
        assert "Publisher" in result
        assert "mario" in result["Game"]
        assert "switch" in result["Console"]
        assert "nintendo" in result["Publisher"]
        assert len(result) == 3  # Should not include unknown entity type

    def test_format_entities_empty_result(self, entity_extractor):
        """Test formatting with no matching entities."""
        # Setup
        unknown_entity = MagicMock()
        unknown_entity.name = "Unknown"
        unknown_entity.type = "Unknown"

        # Execute
        result = entity_extractor.format_entities([unknown_entity])

        # Assert
        assert result is None
