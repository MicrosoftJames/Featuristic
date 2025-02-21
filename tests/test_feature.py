from dataclasses import dataclass
from featuristic.feature import Feature, FeatureDefinition, PromptFeature, PromptFeatureDefinition, PromptFeatureDefinitionGroup


def test_feature_definition():
    name = "number of sentences"
    feature = FeatureDefinition(name=name,
                                preprocess_callback=lambda x: len(x.split('.')))

    assert feature.name == name
    assert feature.preprocess_callback(
        "This is a sentence. This is another sentence") == 2


def test_prompt_feature_definition():

    prompt = "Whether or not the notion of cats is mentioned"
    name = "mention of cats"

    feature = PromptFeatureDefinition(
        name=name,
        prompt=prompt,
        llm_return_type=bool,
        feature_post_callback=None
    )

    assert feature.name == name
    assert feature.prompt == prompt
    assert feature.llm_return_type == bool
    assert feature.feature_post_callback is None


def test_prompt_feature_definition_group():
    mention_of_cats = PromptFeatureDefinition(
        name="mention of cats",
        prompt="Whether or not the notion of cats is mentioned",
        llm_return_type=bool
    )

    mention_of_dogs = PromptFeatureDefinition(
        name="mention of dogs",
        prompt="Whether or not the notion of dogs is mentioned",
        llm_return_type=bool
    )

    group = PromptFeatureDefinitionGroup(
        features=[mention_of_cats, mention_of_dogs],
        system_prompt="You are a helpful assistant.",
        preprocess_callback=None
    )

    assert len(group.features) == 2
    assert group.features == [mention_of_cats, mention_of_dogs]
    assert group.system_prompt == "You are a helpful assistant."
    assert group.preprocess_callback is None
    assert group.features[0].name == "mention of cats"
    assert group.features[1].name == "mention of dogs"
    assert group.features[0].prompt == "Whether or not the notion of cats is mentioned"
    assert group.features[1].prompt == "Whether or not the notion of dogs is mentioned"
    assert group.features[0].llm_return_type == bool
    assert group.features[1].llm_return_type == bool


def test_prompt_feature_definition_group_with_preprocess_callback():
    mention_of_cats = PromptFeatureDefinition(
        name="mention of cats",
        prompt="Whether or not the notion of cats is mentioned",
        llm_return_type=bool
    )

    mention_of_dogs = PromptFeatureDefinition(
        name="mention of dogs",
        prompt="Whether or not the notion of dogs is mentioned",
        llm_return_type=bool
    )

    group = PromptFeatureDefinitionGroup(
        features=[mention_of_cats, mention_of_dogs],
        system_prompt="You are a helpful assistant.",
        preprocess_callback=lambda x: x.text.strip("!")
    )

    @dataclass
    class Data:
        text: str

    assert len(group.features) == 2
    assert group.features == [mention_of_cats, mention_of_dogs]
    assert group.system_prompt == "You are a helpful assistant."
    assert group.preprocess_callback is not None
    assert group.preprocess_callback(
        Data("This is a test!")) == "This is a test"


def test_feature():
    name = "number of sentences"
    value = 5
    feature = Feature(name=name, value=value)
    assert feature.name == name
    assert feature.value == value


def test_prompt_feature():
    name = "number of sentences"
    value = 5
    llm_response = 5
    prompt = "The number of sentences in the text"
    prompt_feature = PromptFeature(
        name=name,
        value=value,
        llm_response=llm_response,
        prompt=prompt,
        llm_return_type=bool
    )
    assert prompt_feature.name == name
    assert prompt_feature.value == value
    assert prompt_feature.llm_response == llm_response
    assert prompt_feature.prompt == prompt
