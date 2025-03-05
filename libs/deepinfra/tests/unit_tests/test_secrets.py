from langchain_deepinfra import ChatDeepInfra, DeepInfraEmbeddings


def test_chat_deepinfra_secrets() -> None:
    o = ChatDeepInfra(deepinfra_api_token="foo")  # type: ignore[call-arg]
    s = str(o)
    assert "foo" not in s


def test_deepinfra_embeddings_secrets() -> None:
    o = DeepInfraEmbeddings(deepinfra_api_token="foo")  # type: ignore[call-arg]
    s = str(o)
    assert "foo" not in s
