"""Test DeepInfra API wrapper.
In order to run this test, you need to have an DeepInfra api key.
You can get it by registering for free at https://deepinfra.com/dash
A test key can be found at https://deepinfra.com/dash/api_keys
You'll then need to set DEEPINFRA_API_TOKEN environment variable to your api key.
"""

import pytest as pytest  # type: ignore[import-not-found]

from langchain_deepinfra import DeepInfra


def test_deepinfra_call() -> None:
    """Test simple call to deepinfra."""
    llm = DeepInfra(  # type: ignore[call-arg]
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        temperature=0.2,
        max_tokens=250,
    )
    output = llm.invoke("Say foo:")

    assert llm._llm_type == "deepinfra"
    assert isinstance(output, str)
    assert len(output) > 0


async def test_deepinfra_acall() -> None:
    """Test simple call to deepinfra."""
    llm = DeepInfra(  # type: ignore[call-arg]
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        temperature=0.2,
        max_tokens=250,
    )
    output = await llm.agenerate(["Say foo:"], stop=["bar"])

    assert llm._llm_type == "deepinfra"
    output_text = output.generations[0][0].text
    assert isinstance(output_text, str)
    assert output_text.count("bar") <= 1
