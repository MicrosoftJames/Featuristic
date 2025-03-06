import asyncio
from typing import List
from unittest.mock import patch

from pydantic import BaseModel
import pytest
from featuristic.feature_extractor import FeatureExtractor
from featuristic.feature import Feature, FeatureDefinition, PromptFeatureDefinition, PromptFeatureDefinitionGroup


def test_dynamic_pydantic_model():
    feature = PromptFeatureDefinition(name='simple feature', prompt='simple prompt',
                                      feature_post_callback=None, llm_return_type=int)

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
                                      feature_post_callback=None, llm_return_type=int)
    group = PromptFeatureDefinitionGroup(
        features=[feature], preprocess_callback=None)
    f.add_feature_definition(group)
    assert f._feature_definitions == [group]

    feature = FeatureDefinition(
        name='simple feature', preprocess_callback=lambda x: x)
    f.add_feature_definition(feature)

    assert len(f._feature_definitions) == 2
    assert f._feature_definitions == [group, feature]


def test_preprocess_data():
    f = FeatureExtractor(aoai_api_endpoint="test", aoai_api_key="test")
    feature = FeatureDefinition(
        name='simple feature', preprocess_callback=lambda x: x*2)
    f.add_feature_definition(feature)

    data = [1, 2, 3]
    preprocessed_data = f._preprocess_data(data, feature)
    assert preprocessed_data == [2, 4, 6]


def test_extract_features():
    f = FeatureExtractor(aoai_api_endpoint="test", aoai_api_key="test")
    feature = FeatureDefinition(
        name='simple feature', preprocess_callback=lambda x: x*2)
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
@patch('featuristic.feature_extractor.extract_features')
async def test_extract_prompt_features(mock_ainvoke):

    class Response(BaseModel):
        animal_list: List[str]

    mock_ainvoke.return_value = Response(animal_list=["cat", "dog"])
    f = FeatureExtractor(aoai_api_endpoint="test", aoai_api_key="test")
    feature = PromptFeatureDefinition(
        name='animal_list', prompt='extract a list of animals', llm_return_type=List[str], feature_post_callback=lambda x, _: len(x))
    group = PromptFeatureDefinitionGroup(
        features=[feature], preprocess_callback=None)

    data = ["The cat and dog are friends."]
    extracted_features = await f._extract_prompt_features(data, group)
    assert len(extracted_features) == 1
    assert isinstance(extracted_features[0], List)
    assert extracted_features[0][0].name == 'animal_list'
    assert extracted_features[0][0].value == 2


def test_error_if_no_feature_definitions():
    f = FeatureExtractor(aoai_api_endpoint="test", aoai_api_key="test")
    with pytest.raises(ValueError):
        asyncio.run(f.extract([1, 2, 3]))
