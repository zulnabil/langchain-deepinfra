"""Test embedding model integration."""

import os

import pytest  # type: ignore[import-not-found]

from langchain_deepinfra import DeepInfraEmbeddings

os.environ["DEEPINFRA_API_TOKEN"] = "foo"


def test_initialization() -> None:
    """Test embedding model initialization."""
    DeepInfraEmbeddings()


def test_deepinfra_invalid_model_kwargs() -> None:
    with pytest.raises(ValueError):
        DeepInfraEmbeddings(model_kwargs={"model": "foo"})


def test_deepinfra_incorrect_field() -> None:
    with pytest.warns(match="not default parameter"):
        llm = DeepInfraEmbeddings(foo="bar")  # type: ignore[call-arg]
    assert llm.model_kwargs == {"foo": "bar"}
