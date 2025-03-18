from dataclasses import dataclass
from featuristic.features.feature import FeatureDefinition, PromptFeatureDefinition, PromptFeatureConfiguration
from featuristic.classification import Distribution
import pytest


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
        aoai_api_key="test-key",
        aoai_api_endpoint="https://test-endpoint.com",
        gpt4o_deployment="test-deployment"
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
        aoai_api_key="test-key",
        aoai_api_endpoint="https://test-endpoint.com",
        gpt4o_deployment="test-deployment"
    )

    assert config.system_prompt == "You are a helpful assistant."
    assert config.preprocess_callback is None


def test_prompt_feature_definition_group_with_preprocess_callback():
    config = PromptFeatureConfiguration(
        system_prompt="You are a helpful assistant.",
        preprocess_callback=lambda x: x.text.strip("!"),
        aoai_api_key="test-key",
        aoai_api_endpoint="https://test-endpoint.com",
        gpt4o_deployment="test-deployment"
    )

    @dataclass
    class Data:
        text: str

    assert config.system_prompt == "You are a helpful assistant."
    assert config.preprocess_callback is not None
    assert config.preprocess_callback(
        Data("This is a test!")) == "This is a test"


def test_prompt_feature_configuration_post_init():
    """Test the post-init validation of PromptFeatureConfiguration."""

    # Test that a valid configuration is accepted
    config = PromptFeatureConfiguration(
        aoai_api_key="test-key",
        aoai_api_endpoint="https://test-endpoint.com",
        gpt4o_deployment="test-deployment"
    )
    assert config.aoai_api_key == "test-key"
    assert config.aoai_api_endpoint == "https://test-endpoint.com"
    assert config.gpt4o_deployment == "test-deployment"

    # Test that missing API key raises ValueError
    with pytest.raises(ValueError, match="AOAI_API_KEY is not set"):
        PromptFeatureConfiguration(
            aoai_api_key="",
            aoai_api_endpoint="https://test-endpoint.com",
            gpt4o_deployment="test-deployment"
        )

    # Test that missing API endpoint raises ValueError
    with pytest.raises(ValueError, match="AOAI_API_ENDPOINT is not set"):
        PromptFeatureConfiguration(
            aoai_api_key="test-key",
            aoai_api_endpoint="",
            gpt4o_deployment="test-deployment"
        )

    # Test that missing deployment raises ValueError
    with pytest.raises(ValueError, match="GPT4O_DEPLOYMENT is not set"):
        PromptFeatureConfiguration(
            aoai_api_key="test-key",
            aoai_api_endpoint="https://test-endpoint.com",
            gpt4o_deployment=""
        )
