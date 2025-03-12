import asyncio
from typing import List
from unittest.mock import patch

from pydantic import BaseModel
import pytest
from featuristic.features import extract
from featuristic.features.feature import Feature, FeatureDefinition, PromptFeatureDefinition, PromptFeatureConfiguration
from featuristic.classification import Distribution


def test_dynamic_pydantic_model():
    feature = PromptFeatureDefinition(name='simple feature', prompt='simple prompt',
                                      feature_post_callback=None, llm_return_type=int, distribution=Distribution.MULTINOMIAL, config=None)

    dynamic_pydantic_model = extract._get_dynamic_pydantic_model(
        [feature])

    assert dynamic_pydantic_model.model_json_schema(
    )['properties'].keys() == {'simple feature'}
    assert dynamic_pydantic_model.model_json_schema(
    )['properties']['simple feature']['description'] == 'simple prompt'
    assert dynamic_pydantic_model.model_json_schema(
    )['properties']['simple feature']['type'] == 'integer'
    assert dynamic_pydantic_model.__name__ == 'FeaturesSchema'


def test_preprocess_data():
    feature = FeatureDefinition(
        name='simple feature', preprocess_callback=lambda x: x*2, distribution=Distribution.MULTINOMIAL)

    data = [1, 2, 3]
    preprocessed_data = extract._preprocess_data(data, feature)
    assert preprocessed_data == [2, 4, 6]


def test_extract_feature():
    feature = FeatureDefinition(
        name='simple feature', preprocess_callback=lambda x: x*2, distribution=Distribution.GAUSSIAN)

    data = [1, 2, 3]
    extracted_feature = extract._extract_feature(data, feature)
    assert len(extracted_feature.values) == 3
    assert isinstance(extracted_feature, Feature)
    assert extracted_feature.name == 'simple feature'
    assert extracted_feature.values == [2, 4, 6]


@pytest.mark.asyncio
@patch('featuristic.features.extract._extract_features_with_llm')
async def test_extract_prompt_features(mock_extract_features_with_llm):

    class Response(BaseModel):
        animal_list: List[str]

    mock_extract_features_with_llm.return_value = Response(animal_list=[
                                                           "cat", "dog"])

    config = PromptFeatureConfiguration(
        aoai_api_endpoint="https://example.com",
        aoai_api_key="example",
        preprocess_callback=None)
    feature = PromptFeatureDefinition(
        name='animal_list', prompt='extract a list of animals',
        llm_return_type=List[str], feature_post_callback=lambda x, _: len(x),
        distribution=Distribution.MULTINOMIAL, config=config)

    data = ["The cat and dog are friends."]
    extracted_features = await extract._extract_prompt_features(data, [feature], config)
    assert len(extracted_features) == 1  # one feature
    assert isinstance(extracted_features, List)
    assert extracted_features[0].name == 'animal_list'
    assert extracted_features[0].values == [2]


@patch('featuristic.features.extract._extract_features_with_llm')
def test_extract_features(mock_extract_features_with_llm):

    data = ["The cat and dog are friends.", "The cow is in the field."]

    class Response(BaseModel):
        animal_list: List[str]
        contains_cow: bool

    def _side_effect(*args, **kwargs):
        string = args[0]
        if string == data[0]:
            return Response(animal_list=["cat", "dog"], contains_cow=False)

        if string == data[1]:
            return Response(animal_list=["cow"], contains_cow=True)

    mock_extract_features_with_llm.side_effect = _side_effect

    config = PromptFeatureConfiguration(
        aoai_api_endpoint="https://example.com",
        aoai_api_key="example",
        preprocess_callback=None)
    animal_list = PromptFeatureDefinition(
        name='animal_list',
        prompt='extract a list of animals',
        llm_return_type=List[str],
        feature_post_callback=lambda x, _: len(x),
        distribution=Distribution.MULTINOMIAL,
        config=config)

    contains_cow = PromptFeatureDefinition(
        name='contains_cow',
        prompt='whether the text contains cow',
        llm_return_type=bool,
        feature_post_callback=None,
        distribution=Distribution.BERNOULLI,
        config=config)

    char_count = FeatureDefinition(
        name='char_count', preprocess_callback=lambda x: len(x), distribution=Distribution.GAUSSIAN)

    features = asyncio.run(extract.extract_features(data,
                                                    feature_definitions=[animal_list, contains_cow, char_count]))

    assert isinstance(features, List)
    assert len(features) == 3
    assert len(features[0].values) == 2
    assert len(features[1].values) == 2
    assert len(features[2].values) == 2

    expected_features = [
        Feature(name='animal_list', values=[2, 1]),
        Feature(name='contains_cow', values=[False, True]),
        Feature(name='char_count', values=[28, 24])
    ]

    assert len(features) == len(expected_features)

    for i in range(len(features)):
        assert features[i].values == expected_features[i].values


def test_error_if_no_feature_definitions():
    with pytest.raises(ValueError):
        asyncio.run(extract.extract_features([1, 2, 3], []))


def test_get_unique_prompt_feature_configs():
    config1 = PromptFeatureConfiguration(
        aoai_api_key="example1", aoai_api_endpoint="https://example1.com")
    config2 = PromptFeatureConfiguration(
        aoai_api_key="example2",
        aoai_api_endpoint="https://example2.com")

    feature_definitions = [
        PromptFeatureDefinition(name='feature1', prompt='prompt1',
                                llm_return_type=str, distribution=Distribution.MULTINOMIAL, config=config1),
        PromptFeatureDefinition(name='feature2', prompt='prompt2',
                                llm_return_type=str, distribution=Distribution.MULTINOMIAL, config=config2),
        PromptFeatureDefinition(name='feature3', prompt='prompt3',
                                llm_return_type=str, distribution=Distribution.MULTINOMIAL, config=config1),
    ]

    unique_configs = extract._get_unique_prompt_feature_configs(
        feature_definitions)

    assert len(unique_configs) == 2
    assert config1 in unique_configs
    assert config2 in unique_configs


def test_get_prompt_feature_definitions_with_config():
    config1 = PromptFeatureConfiguration(
        aoai_api_key="example1", aoai_api_endpoint="https://example1.com")
    config2 = PromptFeatureConfiguration(
        aoai_api_key="example2",
        aoai_api_endpoint="https://example2.com")

    feature_definitions = [
        PromptFeatureDefinition(name='feature1', prompt='prompt1',
                                llm_return_type=str, distribution=Distribution.MULTINOMIAL, config=config1),
        PromptFeatureDefinition(name='feature2', prompt='prompt2',
                                llm_return_type=str, distribution=Distribution.MULTINOMIAL, config=config2),
        PromptFeatureDefinition(name='feature3', prompt='prompt3',
                                llm_return_type=str, distribution=Distribution.MULTINOMIAL, config=config1),
    ]

    results = extract._get_prompt_feature_definitions_with_config(
        feature_definitions, config1)

    assert len(results) == 2
    assert results[0] == feature_definitions[0]
    assert results[1] == feature_definitions[2]

    results = extract._get_prompt_feature_definitions_with_config(
        feature_definitions, config2)

    assert len(results) == 1
    assert results[0] == feature_definitions[1]
