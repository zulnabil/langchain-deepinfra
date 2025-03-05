from typing import cast

from pydantic import SecretStr
from pytest import CaptureFixture, MonkeyPatch  # type: ignore[import-not-found]

from langchain_deepinfra import DeepInfra


def test_deepinfra_api_key_is_secret_string() -> None:
    """Test that the API key is stored as a SecretStr."""
    llm = DeepInfra(
        deepinfra_api_token="secret-api-key",  # type: ignore[call-arg]
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        temperature=0.2,
        max_tokens=250,
    )
    assert isinstance(llm.deepinfra_api_token, SecretStr)


def test_deepinfra_api_key_masked_when_passed_from_env(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture
) -> None:
    """Test that the API key is masked when passed from an environment variable."""
    monkeypatch.setenv("DEEPINFRA_API_TOKEN", "secret-api-key")
    llm = DeepInfra(  # type: ignore[call-arg]
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        temperature=0.2,
        max_tokens=250,
    )
    print(llm.deepinfra_api_token, end="")
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_deepinfra_api_key_masked_when_passed_via_constructor(
    capsys: CaptureFixture,
) -> None:
    """Test that the API key is masked when passed via the constructor."""
    llm = DeepInfra(
        deepinfra_api_token="secret-api-key",  # type: ignore[call-arg]
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        temperature=0.2,
        max_tokens=250,
    )
    print(llm.deepinfra_api_token, end="")
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_deepinfra_uses_actual_secret_value_from_secretstr() -> None:
    """Test that the actual secret value is correctly retrieved."""
    llm = DeepInfra(
        deepinfra_api_token="secret-api-key",  # type: ignore[call-arg]
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        temperature=0.2,
        max_tokens=250,
    )
    assert cast(SecretStr, llm.deepinfra_api_token).get_secret_value() == "secret-api-key"


def test_deepinfra_uses_actual_secret_value_from_secretstr_api_key() -> None:
    """Test that the actual secret value is correctly retrieved."""
    llm = DeepInfra(
        deepinfra_api_token="secret-api-key",  # type: ignore[arg-type]
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        temperature=0.2,
        max_tokens=250,
    )
    assert cast(SecretStr, llm.deepinfra_api_token).get_secret_value() == "secret-api-key"


def test_deepinfra_model_params() -> None:
    # Test standard tracing params
    llm = DeepInfra(
        deepinfra_api_token="secret-api-key",  # type: ignore[arg-type]
        model="foo",
    )
    ls_params = llm._get_ls_params()
    assert ls_params == {
        "ls_provider": "deepinfra",
        "ls_model_type": "llm",
        "ls_model_name": "foo",
        "ls_max_tokens": 200,
    }

    llm = DeepInfra(
        deepinfra_api_token="secret-api-key",  # type: ignore[arg-type]
        model="foo",
        temperature=0.2,
        max_tokens=250,
    )
    ls_params = llm._get_ls_params()
    assert ls_params == {
        "ls_provider": "deepinfra",
        "ls_model_type": "llm",
        "ls_model_name": "foo",
        "ls_max_tokens": 250,
        "ls_temperature": 0.2,
    }
