import asyncio
from typing import List
from unittest.mock import patch

from pydantic import BaseModel
import pytest
from featuristic.features.feature_extractor import FeatureExtractor
from featuristic.features.feature import Feature, FeatureDefinition, PromptFeature, PromptFeatureDefinition, PromptFeatureDefinitionGroup
from featuristic.classification import Distribution


def test_dynamic_pydantic_model():
    feature = PromptFeatureDefinition(name='simple feature', prompt='simple prompt',
                                      feature_post_callback=None, llm_return_type=int, distribution=Distribution.MULTINOMIAL)

    group = PromptFeatureDefinitionGroup(
        features=[feature], preprocess_callback=None)

    f = FeatureExtractor(aoai_api_endpoint="test", aoai_api_key="test")
    f.add_feature_definition(group)

    assert f._feature_definitions == [group]

    dynamic_pydantic_model = f._get_dynamic_pydantic_model(group)

    assert dynamic_pydantic_model.model_json_schema(
    )['properties'].keys() == {'simple feature'}
    assert dynamic_pydantic_model.model_json_schema(
    )['properties']['simple feature']['description'] == 'simple prompt'
    assert dynamic_pydantic_model.model_json_schema(
    )['properties']['simple feature']['type'] == 'integer'
    assert dynamic_pydantic_model.__name__ == 'FeaturesSchema'

    assert isinstance(f._feature_definitions[0], PromptFeatureDefinitionGroup)
    assert f._feature_definitions[0].features[0].name == 'simple feature'


def test_featuristic_init():
    f = FeatureExtractor(aoai_api_endpoint="test", aoai_api_key="test")
    assert f._feature_definitions == []

    feature = PromptFeatureDefinition(name='simple feature', prompt='simple prompt',
                                      feature_post_callback=None, llm_return_type=int, distribution=Distribution.MULTINOMIAL)
    group = PromptFeatureDefinitionGroup(
        features=[feature], preprocess_callback=None)
    f.add_feature_definition(group)
    assert f._feature_definitions == [group]

    feature = FeatureDefinition(
        name='simple feature', preprocess_callback=lambda x: x, distribution=Distribution.MULTINOMIAL)
    f.add_feature_definition(feature)

    assert len(f._feature_definitions) == 2
    assert f._feature_definitions == [group, feature]


def test_preprocess_data():
    f = FeatureExtractor(aoai_api_endpoint="test", aoai_api_key="test")
    feature = FeatureDefinition(
        name='simple feature', preprocess_callback=lambda x: x*2, distribution=Distribution.MULTINOMIAL)

    f.add_feature_definition(feature)

    data = [1, 2, 3]
    preprocessed_data = f._preprocess_data(data, feature)
    assert preprocessed_data == [2, 4, 6]


def test_extract_features():
    f = FeatureExtractor(aoai_api_endpoint="test", aoai_api_key="test")
    feature = FeatureDefinition(
        name='simple feature', preprocess_callback=lambda x: x*2, distribution=Distribution.GAUSSIAN)
    f.add_feature_definition(feature)

    data = [1, 2, 3]
    extracted_features = f._extract_features(data, feature)
    assert len(extracted_features) == 3
    assert all(isinstance(f, Feature)
               for f in extracted_features)
    assert extracted_features[0].name == 'simple feature'
    assert extracted_features[0].value == 2
    assert extracted_features[1].value == 4
    assert extracted_features[2].value == 6


@pytest.mark.asyncio
@patch('featuristic.features.feature_extractor.extract_features')
async def test_extract_prompt_features(mock_extract_features):

    class Response(BaseModel):
        animal_list: List[str]

    mock_extract_features.return_value = Response(animal_list=["cat", "dog"])

    f = FeatureExtractor(aoai_api_endpoint="test", aoai_api_key="test")
    feature = PromptFeatureDefinition(
        name='animal_list', prompt='extract a list of animals', llm_return_type=List[str], feature_post_callback=lambda x, _: len(x), distribution=Distribution.MULTINOMIAL)
    group = PromptFeatureDefinitionGroup(
        features=[feature], preprocess_callback=None)

    data = ["The cat and dog are friends."]
    extracted_features = await f._extract_prompt_features(data, group)
    assert len(extracted_features) == 1
    assert isinstance(extracted_features[0], List)
    assert extracted_features[0][0].name == 'animal_list'
    assert extracted_features[0][0].value == 2


@patch('featuristic.features.feature_extractor.extract_features')
def test_extract(mock_extract_features):

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

    mock_extract_features.side_effect = _side_effect

    animal_list = PromptFeatureDefinition(
        name='animal_list', prompt='extract a list of animals', llm_return_type=List[str], feature_post_callback=lambda x, _: len(x), distribution=Distribution.MULTINOMIAL)

    contains_cow = PromptFeatureDefinition(
        name='contains_cow', prompt='whether the text contains cow', llm_return_type=bool, feature_post_callback=None, distribution=Distribution.BERNOULLI)

    group = PromptFeatureDefinitionGroup(
        features=[animal_list, contains_cow], preprocess_callback=None)

    char_count = FeatureDefinition(
        name='char_count', preprocess_callback=lambda x: len(x), distribution=Distribution.GAUSSIAN)

    f = FeatureExtractor(aoai_api_endpoint="test", aoai_api_key="test")
    f.add_feature_definition(group)
    f.add_feature_definition(char_count)

    features = asyncio.run(f.extract(data))

    assert len(features) == 2
    assert len(features[0]) == 3
    assert len(features[1]) == 3

    expected_features = [
        [PromptFeature(name='animal_list', value=2, llm_response=['cat', 'dog']),
         PromptFeature(name='contains_cow', value=False, llm_response=False),
         Feature(name='char_count', value=28)],
        [PromptFeature(name='animal_list', value=1, llm_response=['cow']),
         PromptFeature(name='contains_cow', value=True, llm_response=True),
         Feature(name='char_count', value=24)]
    ]

    for i in range(len(features)):
        for j in range(len(features[i])):
            assert features[i][j].name == expected_features[i][j].name
            assert features[i][j].value == expected_features[i][j].value

            if isinstance(features[i][j], PromptFeature):
                assert features[i][j].llm_response == expected_features[i][j].llm_response


def test_error_if_no_feature_definitions():
    f = FeatureExtractor(aoai_api_endpoint="test", aoai_api_key="test")
    with pytest.raises(ValueError):
        asyncio.run(f.extract([1, 2, 3]))
