from langchain_deepinfra import __all__

EXPECTED_ALL = ["ChatDeepInfra", "DeepInfraEmbeddings", "DeepInfra"]


def test_all_imports() -> None:
    assert sorted(EXPECTED_ALL) == sorted(__all__)
