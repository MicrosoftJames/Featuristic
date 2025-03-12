import asyncio
from typing import List, Union


from pydantic import BaseModel, Field, create_model

from featuristic.features.feature import Feature, FeatureDefinition, Feature, PromptFeatureDefinition, PromptFeatureConfiguration
from featuristic.features.llm import extract_features_with_llm


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
                extract_features_with_llm(text, schema, config.system_prompt,
                                          config.aoai_api_key, config.aoai_api_endpoint, config.gpt4o_deployment))
            tasks.append(task)
        results.extend(await asyncio.gather(*tasks))

    return results


def _preprocess_data(data, feature_definition: Union[FeatureDefinition, PromptFeatureDefinition]):
    return [feature_definition.preprocess_callback(d) for d in data]


def _extract_feature(data: List, feature_definition: FeatureDefinition) -> Feature:
    preprocessed_data_points = _preprocess_data(
        data, feature_definition)

    return Feature(
        name=feature_definition.name,
        values=preprocessed_data_points)


async def _extract_prompt_features(data: List, prompt_feature_definitions: PromptFeatureDefinition,
                                   config: PromptFeatureConfiguration) -> List[Feature]:
    preprocessed_data_points = _preprocess_data(
        data, prompt_feature_definitions) if config.preprocess_callback else data

    schema = _get_dynamic_pydantic_model(prompt_feature_definitions)
    llm_responses: List[BaseModel] = await _extract_features_batch(
        preprocessed_data_points, schema, config)

    features = []
    for definition in prompt_feature_definitions:
        values = []
        for response, data_point in zip(llm_responses, data):
            v = getattr(response, definition.name)
            v = definition.feature_post_callback(
                v, data_point) if definition.feature_post_callback else v
            values.append(v)
        features.append(
            Feature(definition.name, values))

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


async def extract_features(data: List, feature_definitions: List[Union[FeatureDefinition, PromptFeatureDefinition]]) -> List[Feature]:
    if len(feature_definitions) == 0:
        raise ValueError(
            "No feature definitions have been added to the Featuristic object.")

    unique_prompt_feature_configs = _get_unique_prompt_feature_configs(
        feature_definitions)

    features = []
    for config in unique_prompt_feature_configs:
        prompt_feature_definitions_with_config = _get_prompt_feature_definitions_with_config(
            feature_definitions, config)

        prompt_features = await _extract_prompt_features(
            data, prompt_feature_definitions_with_config, config)

        features.extend(prompt_features)

    # Extract non-prompt features
    for feature_definition in feature_definitions:
        if isinstance(feature_definition, FeatureDefinition):
            feature = _extract_feature(
                data, feature_definition)
            features.append(feature)

    # Sort features based on the order of feature definitions provided
    features.sort(
        key=lambda x: [f.name for f in feature_definitions].index(x.name))

    return features
