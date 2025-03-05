# 🦜️🔗 LangChain DeepInfra

This repository contains 1 package with DeepInfra integrations with LangChain:

- [langchain-deepinfra](https://pypi.org/project/langchain-deepinfra/)

## Setup for Testing

```bash
cd libs/deepinfra
poetry install --with lint,typing,test,test_integration,
```

## Running the Unit Tests

```bash
cd libs/deepinfra
make tests
```

## Running the Integration Tests

```bash
cd libs/deepinfra
export DEEPINFRA_API_TOKEN=<your-api-key>
make integration_tests
```
