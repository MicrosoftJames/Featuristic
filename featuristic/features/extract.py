import asyncio
import pandas as pd
from typing import List, Optional, Union


from pydantic import BaseModel, Field, create_model

from featuristic.features.feature import FeatureDefinition,  PromptFeatureDefinition, PromptFeatureConfiguration
from featuristic.features.llm import _extract_features_with_llm


def _get_dynamic_pydantic_model(prompt_feature_definitions: List[PromptFeatureDefinition]):
    d = {}
    for feature in prompt_feature_definitions:
        d[feature.name] = (
            feature.llm_return_type, Field(description=feature.prompt))

    FeaturesSchema = create_model('FeaturesSchema', **d)
    return FeaturesSchema


async def _extract_features_batch(texts: List[str], schema: BaseModel, config: PromptFeatureConfiguration):
    batch_size = 50
    iterations = (len(texts) // batch_size) + \
        min(len(texts) % batch_size, 1)

    results = []
    for i in range(iterations):
        tasks = []
        for text in texts[i * batch_size: (i + 1) * batch_size]:
            task = asyncio.create_task(
                _extract_features_with_llm(text, schema, config.system_prompt,
                                           config.api_key, config.api_base, config.api_version, config.model))
            tasks.append(task)
        results.extend(await asyncio.gather(*tasks))

    return results


def _extract_feature(data: List, feature_definition: FeatureDefinition) -> pd.DataFrame:
    preprocessed_data_points = [
        feature_definition.preprocess_callback(d) for d in data]

    return pd.DataFrame({
        feature_definition.name: preprocessed_data_points
    })


async def _extract_prompt_features(data: List, prompt_feature_definitions: PromptFeatureDefinition,
                                   config: PromptFeatureConfiguration) -> pd.DataFrame:
    preprocessed_data_points = [
        config.preprocess_callback(d) for d in data] if config.preprocess_callback else data

    schema = _get_dynamic_pydantic_model(prompt_feature_definitions)
    llm_responses: List[BaseModel] = await _extract_features_batch(
        preprocessed_data_points, schema, config)

    features = pd.DataFrame()
    for definition in prompt_feature_definitions:
        values = []
        for response, data_point in zip(llm_responses, data):
            v = getattr(response, definition.name)
            v = definition.feature_post_callback(
                v, data_point) if definition.feature_post_callback else v
            values.append(v)

        features[definition.name] = values

    return features


def _get_unique_prompt_feature_configs(feature_definitions: List[Union[FeatureDefinition, PromptFeatureDefinition]]):
    unique_prompt_feature_configs = set()
    for fd in feature_definitions:
        if isinstance(fd, PromptFeatureDefinition) and fd.config:
            unique_prompt_feature_configs.add(fd.config)
    return unique_prompt_feature_configs


def _get_prompt_feature_definitions_with_config(feature_definitions: List[Union[FeatureDefinition, PromptFeatureDefinition]],
                                                config: PromptFeatureConfiguration) -> List[PromptFeatureDefinition]:
    prompt_feature_definitions = []
    for fd in feature_definitions:
        if isinstance(fd, PromptFeatureDefinition) and fd.config == config:
            prompt_feature_definitions.append(fd)
    return prompt_feature_definitions


async def extract_features(data: List, feature_definitions: List[Union[FeatureDefinition, PromptFeatureDefinition]], ids: Optional[List] = None) -> pd.DataFrame:
    """Extract features from the data using the provided feature definitions.

    Args:
        data (List): The input data to extract features from.
        feature_definitions (List[Union[FeatureDefinition, PromptFeatureDefinition]]): A list of feature definitions.
            Each feature definition can be either a FeatureDefinition or a PromptFeatureDefinition.
            FeatureDefinition is a class that defines a python-based feature extraction method.
            PromptFeatureDefinition is a class that defines a feature extraction method using a language model.
        ids (Optional[List]): An optional list of ids to use as the index for the DataFrame. The length of this list must match the length of the data.

    Returns:
        pd.DataFrame: A DataFrame containing the extracted features. The columns of the DataFrame are named according to the feature definitions provided.

    """
    if len(feature_definitions) == 0:
        raise ValueError(
            "No feature definitions have been added to the Featuristic object.")

    if ids is not None and len(data) != len(ids):
        raise ValueError(
            "The length of data and ids must be the same.")

    unique_prompt_feature_configs = _get_unique_prompt_feature_configs(
        feature_definitions)

    features = pd.DataFrame()
    for config in unique_prompt_feature_configs:
        prompt_feature_definitions_with_config = _get_prompt_feature_definitions_with_config(
            feature_definitions, config)

        prompt_features = await _extract_prompt_features(
            data, prompt_feature_definitions_with_config, config)

        features = pd.concat([features, prompt_features], axis=1)

    # Extract non-prompt features
    for feature_definition in feature_definitions:
        if isinstance(feature_definition, FeatureDefinition):
            feature = _extract_feature(
                data, feature_definition)
            features = pd.concat([features, feature], axis=1)

    # Add ids to the DataFrame as the index if provided
    if ids is not None:
        features.index = ids

    # Sort features based on the order of feature definitions provided
    feature_names = [
        feature_definition.name for feature_definition in feature_definitions]

    # Ensure the order of features in the DataFrame matches the order of feature definitions
    features = features[feature_names]

    return features
