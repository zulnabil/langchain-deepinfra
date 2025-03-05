"""Standard LangChain interface tests"""

from typing import Tuple, Type

from langchain_core.embeddings import Embeddings
from langchain_tests.unit_tests import EmbeddingsUnitTests

from langchain_deepinfra import DeepInfraEmbeddings


class TestDeepInfraStandard(EmbeddingsUnitTests):
    @property
    def embeddings_class(self) -> Type[Embeddings]:
        return DeepInfraEmbeddings

    @property
    def embeddings_params(self) -> dict:
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
