"""Test DeepInfra embeddings."""

from langchain_deepinfra import DeepInfraEmbeddings


def test_langchain_deepinfra_embed_documents() -> None:
    """Test DeepInfra embeddings."""
    documents = ["foo bar", "bar foo"]
    embedding = DeepInfraEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 2
    assert len(output[0]) > 0


def test_langchain_deepinfra_embed_query() -> None:
    """Test DeepInfra embeddings."""
    query = "foo bar"
    embedding = DeepInfraEmbeddings()
    output = embedding.embed_query(query)
    assert len(output) > 0


async def test_langchain_deepinfra_aembed_documents() -> None:
    """Test DeepInfra embeddings asynchronous."""
    documents = ["foo bar", "bar foo"]
    embedding = DeepInfraEmbeddings()
    output = await embedding.aembed_documents(documents)
    assert len(output) == 2
    assert len(output[0]) > 0


async def test_langchain_deepinfra_aembed_query() -> None:
    """Test DeepInfra embeddings asynchronous."""
    query = "foo bar"
    embedding = DeepInfraEmbeddings()
    output = await embedding.aembed_query(query)
    assert len(output) > 0
