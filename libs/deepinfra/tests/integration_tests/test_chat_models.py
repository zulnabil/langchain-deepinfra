import pytest  # type: ignore[import-not-found]
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from langchain_deepinfra import ChatDeepInfra


def test_chat_deepinfra_model() -> None:
    """Test ChatDeepInfra wrapper handles model_name."""
    chat = ChatDeepInfra(model="foo")
    assert chat.model_name == "foo"
    chat = ChatDeepInfra(model_name="bar")  # type: ignore[call-arg]
    assert chat.model_name == "bar"


def test_chat_deepinfra_system_message() -> None:
    """Test ChatDeepInfra wrapper with system message."""
    chat = ChatDeepInfra(max_tokens=10)
    system_message = SystemMessage(content="You are to chat with the user.")
    human_message = HumanMessage(content="Hello")
    response = chat([system_message, human_message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_chat_deepinfra_llm_output_contains_model_name() -> None:
    """Test llm_output contains model_name."""
    chat = ChatDeepInfra(max_tokens=10)
    message = HumanMessage(content="Hello")
    llm_result = chat.generate([[message]])
    assert llm_result.llm_output is not None
    assert llm_result.llm_output["model_name"] == chat.model_name


def test_chat_deepinfra_streaming_llm_output_contains_model_name() -> None:
    """Test llm_output contains model_name."""
    chat = ChatDeepInfra(max_tokens=10, streaming=True)
    message = HumanMessage(content="Hello")
    llm_result = chat.generate([[message]])
    assert llm_result.llm_output is not None
    assert llm_result.llm_output["model_name"] == chat.model_name


def test_chat_deepinfra_invalid_streaming_params() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    with pytest.raises(ValueError):
        ChatDeepInfra(
            max_tokens=10,
            streaming=True,
            temperature=0,
            n=5,
        )


def test_chat_deepinfra_extra_kwargs() -> None:
    """Test extra kwargs to chat deepinfra."""
    # Check that foo is saved in extra_kwargs.
    llm = ChatDeepInfra(foo=3, max_tokens=10)  # type: ignore[call-arg]
    assert llm.max_tokens == 10
    assert llm.model_kwargs == {"foo": 3}

    # Test that if extra_kwargs are provided, they are added to it.
    llm = ChatDeepInfra(foo=3, model_kwargs={"bar": 2})  # type: ignore[call-arg]
    assert llm.model_kwargs == {"foo": 3, "bar": 2}

    # Test that if provided twice it errors
    with pytest.raises(ValueError):
        ChatDeepInfra(foo=3, model_kwargs={"foo": 2})  # type: ignore[call-arg]


def test_stream() -> None:
    """Test streaming tokens from DeepInfra."""
    llm = ChatDeepInfra()

    for token in llm.stream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


async def test_astream() -> None:
    """Test streaming tokens from DeepInfra."""
    llm = ChatDeepInfra()

    async for token in llm.astream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


async def test_abatch() -> None:
    """Test streaming tokens from ChatDeepInfra."""
    llm = ChatDeepInfra()

    result = await llm.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


async def test_abatch_tags() -> None:
    """Test batch tokens from ChatDeepInfra."""
    llm = ChatDeepInfra()

    result = await llm.abatch(
        ["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token.content, str)


def test_batch() -> None:
    """Test batch tokens from ChatDeepInfra."""
    llm = ChatDeepInfra()

    result = llm.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


async def test_ainvoke() -> None:
    """Test invoke tokens from ChatDeepInfra."""
    llm = ChatDeepInfra()

    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result.content, str)


def test_invoke() -> None:
    """Test invoke tokens from ChatDeepInfra."""
    llm = ChatDeepInfra()

    result = llm.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
    assert isinstance(result.content, str)
