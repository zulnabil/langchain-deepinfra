"""Standard LangChain interface tests"""

from typing import Tuple, Type

from langchain_core.language_models import BaseChatModel
from langchain_tests.unit_tests import (  # type: ignore[import-not-found]
    ChatModelUnitTests,  # type: ignore[import-not-found]
)

from langchain_deepinfra import ChatDeepInfra


class TestDeepInfraStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatDeepInfra

    @property
    def chat_model_params(self) -> dict:
        return {"model": "meta-llama/Meta-Llama-3.1-8B-Instruct"}

    @property
    def init_from_env_params(self) -> Tuple[dict, dict, dict]:
        return (
            {
                "DEEPINFRA_API_TOKEN": "api_key",
                "DEEPINFRA_API_BASE": "https://base.com",
            },
            {},
            {
                "deepinfra_api_token": "api_key",
                "deepinfra_api_base": "https://base.com",
            },
        )
