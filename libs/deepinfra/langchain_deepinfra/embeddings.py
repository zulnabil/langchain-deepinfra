"""Wrapper around DeepInfra's Embeddings API."""

import logging
import warnings
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import openai
from langchain_core.embeddings import Embeddings
from langchain_core.utils import from_env, get_pydantic_field_names, secret_from_env
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    model_validator,
)
from typing_extensions import Self

logger = logging.getLogger(__name__)


class DeepInfraEmbeddings(BaseModel, Embeddings):
    """DeepInfra embedding model integration.

    Setup:
        Install ``langchain_deepinfra`` and set environment variable
        ``DEEPINFRA_API_TOKEN``.

        .. code-block:: bash

            pip install -U langchain_deepinfra
            export DEEPINFRA_API_TOKEN="your-api-key"

    Key init args — completion params:
        model: str
            Name of DeepInfra model to use.

    Key init args — client params:
      api_key: Optional[SecretStr]

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        .. code-block:: python

            from __module_name__ import DeepInfraEmbeddings

            embed = DeepInfraEmbeddings(
                model="BAAI/bge-base-en-v1.5",
                # api_key="...",
                # other params...
            )

    Embed single text:
        .. code-block:: python

            input_text = "The meaning of life is 42"
            vector = embed.embed_query(input_text)
            print(vector[:3])

        .. code-block:: python

            [-0.024603435769677162, -0.007543657906353474, 0.0039630369283258915]

    Embed multiple texts:
        .. code-block:: python

             input_texts = ["Document 1...", "Document 2..."]
            vectors = embed.embed_documents(input_texts)
            print(len(vectors))
            # The first 3 coordinates for the first vector
            print(vectors[0][:3])

        .. code-block:: python

            2
            [-0.024603435769677162, -0.007543657906353474, 0.0039630369283258915]

    Async:
        .. code-block:: python

            vector = await embed.aembed_query(input_text)
           print(vector[:3])

            # multiple:
            # await embed.aembed_documents(input_texts)

        .. code-block:: python

            [-0.009100092574954033, 0.005071679595857859, -0.0029193938244134188]
    """

    client: Any = Field(default=None, exclude=True)  #: :meta private:
    async_client: Any = Field(default=None, exclude=True)  #: :meta private:
    model: str = "BAAI/bge-base-en-v1.5"
    """Embeddings model name to use.
    Instead, use 'BAAI/bge-base-en-v1.5' for example.
    """
    dimensions: Optional[int] = None
    """The number of dimensions the resulting output embeddings should have.

    Not yet supported.
    """
    deepinfra_api_token: Optional[SecretStr] = Field(
        alias="api_key",
        default_factory=secret_from_env("DEEPINFRA_API_TOKEN", default=None),
    )
    """DeepInfra API key.

    Automatically read from env variable `DEEPINFRA_API_TOKEN` if not provided.
    """
    deepinfra_api_base: str = Field(
        default_factory=from_env(
            "DEEPINFRA_API_BASE", default="https://api.deepinfra.com/v1/openai"
        ),
        alias="base_url",
    )
    """Endpoint URL to use."""
    embedding_ctx_length: int = 4096
    """The maximum number of tokens to embed at once.

    Not yet supported.
    """
    allowed_special: Union[Literal["all"], Set[str]] = set()
    """Not yet supported."""
    disallowed_special: Union[Literal["all"], Set[str], Sequence[str]] = "all"
    """Not yet supported."""
    chunk_size: int = 1000
    """Maximum number of texts to embed in each batch.

    Not yet supported.
    """
    max_retries: int = 2
    """Maximum number of retries to make when generating."""
    request_timeout: Optional[Union[float, Tuple[float, float], Any]] = Field(
        default=None, alias="timeout"
    )
    """Timeout for requests to DeepInfra embedding API. Can be float, httpx.Timeout or
        None."""
    show_progress_bar: bool = False
    """Whether to show a progress bar when embedding.

    Not yet supported.
    """
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""
    skip_empty: bool = False
    """Whether to skip empty strings when embedding or raise an error.
    Defaults to not skipping.

    Not yet supported."""
    default_headers: Union[Mapping[str, str], None] = None
    default_query: Union[Mapping[str, object], None] = None
    # Configure a custom httpx client. See the
    # [httpx documentation](https://www.python-httpx.org/api/#client) for more details.
    http_client: Union[Any, None] = None
    """Optional httpx.Client. Only used for sync invocations. Must specify
        http_async_client as well if you'd like a custom client for async invocations.
    """
    http_async_client: Union[Any, None] = None
    """Optional httpx.AsyncClient. Only used for async invocations. Must specify
        http_client as well if you'd like a custom client for sync invocations."""
    encoding_format: str = "float"

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        protected_namespaces=(),
    )

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: Dict[str, Any]) -> Any:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name in extra:
                raise ValueError(f"Found {field_name} supplied twice.")
            if field_name not in all_required_field_names:
                warnings.warn(
                    f"""WARNING! {field_name} is not default parameter.
                    {field_name} was transferred to model_kwargs.
                    Please confirm that {field_name} is what you intended."""
                )
                extra[field_name] = values.pop(field_name)

        invalid_model_kwargs = all_required_field_names.intersection(extra.keys())
        if invalid_model_kwargs:
            raise ValueError(
                f"Parameters {invalid_model_kwargs} should be specified explicitly. "
                f"Instead they were passed in as part of `model_kwargs` parameter."
            )

        values["model_kwargs"] = extra
        return values

    @model_validator(mode="after")
    def post_init(self) -> Self:
        """Logic that will post Pydantic initialization."""
        client_params: dict = {
            "api_key": (
                self.deepinfra_api_token.get_secret_value()
                if self.deepinfra_api_token
                else None
            ),
            "base_url": self.deepinfra_api_base,
            "timeout": self.request_timeout,
            "max_retries": self.max_retries,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }
        if not (self.client or None):
            sync_specific: dict = (
                {"http_client": self.http_client} if self.http_client else {}
            )
            self.client = openai.OpenAI(**client_params, **sync_specific).embeddings
        if not (self.async_client or None):
            async_specific: dict = (
                {"http_client": self.http_async_client}
                if self.http_async_client
                else {}
            )
            self.async_client = openai.AsyncOpenAI(
                **client_params, **async_specific
            ).embeddings
        return self

    @property
    def _invocation_params(self) -> Dict[str, Any]:
        """Get the parameters used to invoke the model."""
        params: Dict = {
            "model": self.model,
            "encoding_format": self.encoding_format,
            **self.model_kwargs
        }
        if self.dimensions is not None:
            params["dimensions"] = self.dimensions
        return params

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document texts using passage model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        embeddings = []
        params = self._invocation_params

        for text in texts:
            response = self.client.create(input=text, **params)

            if not isinstance(response, dict):
                response = response.model_dump()
            embeddings.extend([i["embedding"] for i in response["data"]])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed query text using query model.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        params = self._invocation_params
        params["model"] = params["model"]

        response = self.client.create(input=text, **params)

        if not isinstance(response, dict):
            response = response.model_dump()
        return response["data"][0]["embedding"]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document texts using passage model asynchronously.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        embeddings = []
        params = self._invocation_params
        params["model"] = params["model"]

        for text in texts:
            response = await self.async_client.create(input=text, **params)

            if not isinstance(response, dict):
                response = response.model_dump()
                embeddings.extend([i["embedding"] for i in response["data"]])
        return embeddings

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous Embed query text using query model.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        params = self._invocation_params
        params["model"] = params["model"]

        response = await self.async_client.create(input=text, **params)

        if not isinstance(response, dict):
            response = response.model_dump()
        return response["data"][0]["embedding"]
