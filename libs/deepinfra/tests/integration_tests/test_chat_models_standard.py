"""Standard LangChain interface tests"""

from typing import Optional, Type
import logging

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_tests.integration_tests import ChatModelIntegrationTests
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from langchain_deepinfra import ChatDeepInfra

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize the rate limiter in global scope
rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.5,
)


class TestDeepInfraStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatDeepInfra

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "rate_limiter": rate_limiter,
        }

    @property
    def has_tool_calling(self) -> bool:
        """Disable tool calling tests until properly implemented"""
        return False

    @property
    def has_structured_output(self) -> bool:
        """Disable structured output tests until properly implemented"""
        return False

    @property
    def tool_choice_value(self) -> Optional[str]:
        """Value to use for tool choice when used in tests."""
        return None

    @pytest.mark.xfail(reason="Not yet supported.")
    def test_usage_metadata_streaming(self, model: BaseChatModel) -> None:
        super().test_usage_metadata_streaming(model)
