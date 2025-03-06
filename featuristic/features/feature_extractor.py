import asyncio
from typing import List, Optional, Union


from pydantic import BaseModel, Field, create_model

from featuristic.features.feature import BaseFeatureDefinition, Feature, FeatureDefinition, PromptFeature, PromptFeatureDefinitionGroup
from featuristic.features.llm import extract_features


class FeatureExtractor:
    def __init__(self, aoai_api_key: Optional[str] = None, aoai_api_endpoint: Optional[str] = None, gpt4o_deployment: str = "gpt-4o"):
        self._aoai_api_key = aoai_api_key
        self._aoai_api_endpoint = aoai_api_endpoint
        self._gpt4o_deployment = gpt4o_deployment
        self._feature_definitions: List[Union[PromptFeatureDefinitionGroup, FeatureDefinition]] = [
        ]

    def add_feature_definition(self, features: Union[PromptFeatureDefinitionGroup, FeatureDefinition]):
        assert isinstance(
            features, (PromptFeatureDefinitionGroup, FeatureDefinition))

        if not all([self._aoai_api_endpoint, self._aoai_api_key, self._gpt4o_deployment]):
            raise ValueError(
                "Azure OpenAI API key, endpoint, and deployment must be provided for prompt-based features.")
        self._feature_definitions.append(features)

    def _get_dynamic_pydantic_model(self, feature_definition: PromptFeatureDefinitionGroup):
        d = {}
        for feature in feature_definition.features:
            d[feature.name] = (
                feature.llm_return_type, Field(description=feature.prompt))

        FeaturesSchema = create_model('FeaturesSchema', **d)
        return FeaturesSchema

    async def _extract_features_batch(self, texts: List[str], schema: BaseModel, system_prompt: str):
        batch_size = 50
        iterations = (len(texts) // batch_size) + \
            min(len(texts) % batch_size, 1)

        results = []
        for i in range(iterations):
            tasks = []
            for text in texts[i * batch_size: (i + 1) * batch_size]:
                task = asyncio.create_task(
                    extract_features(text, schema, system_prompt,
                                     self._aoai_api_key, self._aoai_api_endpoint, self._gpt4o_deployment))
                tasks.append(task)
            results.extend(await asyncio.gather(*tasks))

        return results

    @staticmethod
    def _preprocess_data(data, feature_definition: BaseFeatureDefinition):
        return [feature_definition.preprocess_callback(d) for d in data]

    def _extract_features(self, data: List, feature_definition: FeatureDefinition):
        preprocessed_data_points = self._preprocess_data(
            data, feature_definition)

        return [Feature(feature_definition.name, d) for d in preprocessed_data_points]

    async def _extract_prompt_features(self, data: List, feature_definition: PromptFeatureDefinitionGroup):
        preprocessed_data_points = self._preprocess_data(
            data, feature_definition) if feature_definition.preprocess_callback else data

        schema = self._get_dynamic_pydantic_model(feature_definition)
        llm_responses: List[BaseModel] = await self._extract_features_batch(
            preprocessed_data_points, schema, feature_definition.system_prompt)

        prompt_features = []
        for response, data_point in zip(llm_responses, data):
            data_point_responses = []
            for feature in feature_definition.features:
                v = getattr(response, feature.name)
                v = feature.feature_post_callback(
                    v, data_point) if feature.feature_post_callback else v

                data_point_responses.append(
                    PromptFeature(name=feature.name, value=v, llm_response=getattr(response, feature.name), prompt=feature.prompt, llm_return_type=feature.llm_return_type))

            prompt_features.append(data_point_responses)

        return prompt_features

    async def extract(self, data: List):
        if len(self._feature_definitions) == 0:
            raise ValueError(
                "No feature definitions have been added to the Featuristic object.")

        features = [[] for _ in range(len(data))]
        for feature_definition in self._feature_definitions:

            if isinstance(feature_definition, PromptFeatureDefinitionGroup):
                feature_group_responses = await self._extract_prompt_features(
                    data, feature_definition)
            else:
                feature_group_responses = self._extract_features(
                    data, feature_definition)

            for i, feature_responses in enumerate(feature_group_responses):
                features[i].append(feature_responses)

        return features
