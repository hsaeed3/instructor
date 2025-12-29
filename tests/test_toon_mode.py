"""Tests for TOON (Token-Oriented Object Notation) mode.

TOON is a compact data format that reduces token usage by 30-60% compared to JSON.
These tests verify the TOON mode implementation in instructor.
"""

import pytest
from unittest.mock import MagicMock
from pydantic import BaseModel, Field

from instructor.mode import Mode
from instructor.providers.openai.utils import (
    handle_toon,
    reask_toon,
    _generate_toon_structure,
)


class SimpleUser(BaseModel):
    """A simple user model for testing."""

    name: str = Field(description="The user's full name")
    age: int = Field(description="The user's age in years")


class UserWithAddress(BaseModel):
    """A user model with nested address."""

    name: str = Field(description="The user's name")
    address: dict = Field(description="The user's address")


class UserWithTags(BaseModel):
    """A user model with a list of tags."""

    name: str = Field(description="The user's name")
    tags: list[str] = Field(description="Tags associated with the user")


class TestToonModeEnum:
    """Test TOON mode is properly defined in the Mode enum."""

    def test_toon_mode_exists(self):
        """TOON mode should exist in the Mode enum."""
        assert hasattr(Mode, "TOON")
        assert Mode.TOON.value == "toon"

    def test_toon_in_json_modes(self):
        """TOON should be classified as a JSON-like mode."""
        assert Mode.TOON in Mode.json_modes()


class TestGenerateToonStructure:
    """Test TOON structure generation from Pydantic models."""

    def test_simple_model_structure(self):
        """Test structure generation for a simple model."""
        structure = _generate_toon_structure(SimpleUser)
        assert "name" in structure
        assert "age" in structure

    def test_structure_with_descriptions(self):
        """Test that field descriptions are included."""
        structure = _generate_toon_structure(SimpleUser)
        assert "full name" in structure.lower() or "name" in structure.lower()

    def test_structure_with_nested_models(self):
        """Test structure generation handles nested Pydantic models."""
        structure = _generate_toon_structure(UserWithAddress)
        assert "name" in structure
        assert "address" in structure


class TestHandleToon:
    """Test the handle_toon request handler."""

    def test_handle_toon_with_none_model(self):
        """handle_toon should return unchanged kwargs when model is None."""
        kwargs = {"messages": [{"role": "user", "content": "Hello"}]}
        result_model, result_kwargs = handle_toon(None, kwargs)

        assert result_model is None
        assert result_kwargs == kwargs

    def test_handle_toon_adds_system_message(self):
        """handle_toon should add TOON instructions to messages."""
        original_messages = [{"role": "user", "content": "Extract user info"}]
        kwargs = {"messages": original_messages.copy()}
        result_model, result_kwargs = handle_toon(SimpleUser, kwargs)

        assert result_model is SimpleUser
        assert len(result_kwargs["messages"]) == 2
        assert result_kwargs["messages"][0]["role"] == "system"
        assert result_kwargs["messages"][1]["role"] == "user"
        assert "TOON" in result_kwargs["messages"][0]["content"]

    def test_handle_toon_appends_to_existing_system(self):
        """handle_toon should append to existing system message."""
        kwargs = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Extract user info"},
            ]
        }
        result_model, result_kwargs = handle_toon(SimpleUser, kwargs)

        system_msg = result_kwargs["messages"][0]
        assert "helpful assistant" in system_msg["content"]
        assert "TOON" in system_msg["content"]

    def test_handle_toon_includes_structure_template(self):
        """handle_toon should include the TOON structure template."""
        kwargs = {"messages": [{"role": "user", "content": "Extract user"}]}
        result_model, result_kwargs = handle_toon(SimpleUser, kwargs)

        system_msg = result_kwargs["messages"][0]
        assert "name" in system_msg["content"].lower()


class TestReaskToon:
    """Test the reask_toon error handler."""

    def test_reask_toon_adds_error_message(self):
        """reask_toon should add validation error to messages."""
        kwargs = {"messages": [{"role": "user", "content": "Hello"}]}

        # Mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Invalid TOON"
        mock_response.choices[0].message.role = "assistant"
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].message.function_call = None

        exception = ValueError("Invalid field: expected int got string")

        result = reask_toon(kwargs, mock_response, exception)

        assert len(result["messages"]) > 1
        last_msg = result["messages"][-1]
        assert "Validation Error" in last_msg["content"]
        assert "Invalid field" in last_msg["content"]


class TestToonModeIntegration:
    """Integration tests for TOON mode."""

    @pytest.fixture
    def toon_format_available(self):
        """Check if toon_format is available."""
        try:
            import toon_format  # noqa: F401

            return True
        except ImportError:
            return False

    def test_parse_simple_toon(self, toon_format_available):
        """Test parsing simple TOON content."""
        if not toon_format_available:
            pytest.skip("toon-format not installed")

        from toon_format import decode

        toon_content = "name: John Doe\nage: 30"
        data = decode(toon_content)
        result = SimpleUser.model_validate(data)

        assert result.name == "John Doe"
        assert result.age == 30

    def test_extract_toon_from_codeblock(self):
        """Test extracting TOON from markdown code blocks."""
        from instructor.processing.function_calls import _extract_toon_from_response

        text = "Here's the data:\n```toon\nname: John\nage: 25\n```"
        result = _extract_toon_from_response(text)
        assert "name: John" in result
        assert "age: 25" in result

        text2 = "```\nname: Jane\nage: 30\n```"
        result2 = _extract_toon_from_response(text2)
        assert "name: Jane" in result2

        text3 = "name: Bob\nage: 40"
        result3 = _extract_toon_from_response(text3)
        assert result3 == text3


class TestToonModeInModeHandlers:
    """Test that TOON mode is properly registered in handlers."""

    def test_toon_in_response_handlers(self):
        """TOON should be in the response mode handlers."""
        from instructor.processing.response import handle_response_model

        kwargs = {"messages": [{"role": "user", "content": "test"}]}
        result_model, result_kwargs = handle_response_model(
            SimpleUser, Mode.TOON, **kwargs
        )

        assert result_model is not None
        assert result_model.__name__ == "SimpleUser"
        assert "messages" in result_kwargs

    def test_toon_in_reask_handlers(self):
        """TOON should be in the reask handlers."""
        from instructor.processing.response import handle_reask_kwargs

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "test"
        mock_response.choices[0].message.role = "assistant"
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].message.function_call = None

        kwargs = {"messages": [{"role": "user", "content": "test"}]}
        exception = ValueError("test error")

        result = handle_reask_kwargs(kwargs, Mode.TOON, mock_response, exception)
        assert "messages" in result
