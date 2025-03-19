from dataclasses import dataclass
from featuristic.features.feature import FeatureDefinition, PromptFeatureDefinition, PromptFeatureConfiguration
from featuristic.classification import Distribution


def test_feature_definition():
    name = "number of sentences"
    feature = FeatureDefinition(name=name,
                                preprocess_callback=lambda x: len(x.split('.')), distribution=Distribution.GAUSSIAN)

    assert feature.name == name
    assert feature.preprocess_callback(
        "This is a sentence. This is another sentence") == 2


def test_prompt_feature_definition():
    prompt = "Whether or not the notion of cats is mentioned"
    name = "mention of cats"

    config = PromptFeatureConfiguration(
        system_prompt="You are a helpful assistant.",
        preprocess_callback=None,
        api_key="test-key",
        api_base="https://test-endpoint.com",
        api_version="2023-10-01",
        model="gpt-4o"
    )

    feature = PromptFeatureDefinition(
        name=name,
        prompt=prompt,
        llm_return_type=bool,
        feature_post_callback=None,
        distribution=Distribution.BERNOULLI,
        config=config
    )

    assert feature.name == name
    assert feature.prompt == prompt
    assert feature.llm_return_type == bool
    assert feature.feature_post_callback is None


def test_prompt_feature_configuration():
    config = PromptFeatureConfiguration(
        system_prompt="You are a helpful assistant.",
        preprocess_callback=None,
        api_key="test-key",
        api_base="https://test-endpoint.com",
        api_version="2023-10-01",
        model="gpt-4o"
    )

    assert config.system_prompt == "You are a helpful assistant."
    assert config.preprocess_callback is None


def test_prompt_feature_definition_group_with_preprocess_callback():
    config = PromptFeatureConfiguration(
        system_prompt="You are a helpful assistant.",
        preprocess_callback=lambda x: x.text.strip("!"),
        api_key="test-key",
        api_base="https://test-endpoint.com",
        api_version="2023-10-01",
        model="gpt-4o"
    )

    @dataclass
    class Data:
        text: str

    assert config.system_prompt == "You are a helpful assistant."
    assert config.preprocess_callback is not None
    assert config.preprocess_callback(
        Data("This is a test!")) == "This is a test"
