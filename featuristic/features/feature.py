from dataclasses import dataclass
from typing import Callable, Optional
from featuristic.classification import Distribution
from featuristic.features import API_BASE, API_KEY, API_VERSION, MODEL

SYSTEM_MESSAGE = """You are helpful AI assistant that extracts machine learning features from text.
You will be given a text input and your job is to extract features according to the JSON schema provided."""


@dataclass(frozen=True)
class PromptFeatureConfiguration:
    """A configuration class for prompt-based features. All PromptFeatureDefinition instances 
    that share this configuration will be extracted using the same call to the LLM. The model must 
    support structured outputs, such as gpt-4o. The api variables follow litellm.

    Args:
        api_key (str): The API key for the service.
        api_base (str): The API base URL for the service.
        api_version (str): The API version to use.
        model (str): The deployment name for the model.
        preprocess_callback (Optional[Callable]): A function to preprocess the data before extracting the feature.
            The function should take a single argument, which is the data point, and return the preprocessed data point.
        system_prompt (Optional[str]): The system prompt to be used for the LLM. Defaults to SYSTEM_MESSAGE.
    """
    api_key: str = API_KEY
    api_base: str = API_BASE
    api_version: str = API_VERSION
    model: str = MODEL
    preprocess_callback: Optional[Callable] = None
    system_prompt: Optional[str] = SYSTEM_MESSAGE

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)


@dataclass
class FeatureDefinition:
    """A python-based feature.
    Args:
        name (str): The name of the feature.
        preprocess_callback (Optional[Callable]): A function to preprocess the data before extracting the feature.
            The function should take a single argument, which is the data point, and return the preprocessed data point.
        distribution (Distribution): The distribution of the feature.
    """
    preprocess_callback: Optional[Callable]
    name: str
    distribution: Distribution


@dataclass
class PromptFeatureDefinition:
    """A custom prompt-based feature.

        Args:
            name (str): The name of the feature.
            prompt (str): The prompt to be used to extract the feature.
            distribution (Distribution): The distribution of the feature.
            config (PromptFeatureConfiguration): The configuration for the prompt feature. All PromptFeatureDefinition instances
                that share this configuration will be extracted using the same call to the LLM.
            llm_return_type (type, optional): The return type of the feature. Defaults to str.
            feature_post_callback (Optional[Callable], optional): A post-processing function to apply to the feature. Defaults to None.
                The post-processing function will be called with the feature value and the original data object.

                Example for feature_post_callback:
                ```python
                >>> @dataclass
                ... class MyData:
                ...     text: str
                >>> def feature_post_callback(x, my_data: MyData):
                ...     return x.upper()
                >>> feature_definition = PromptFeatureDefinition(
                ...     name="sentiment",
                ...     prompt="What is the sentiment of this text?",
                ...     llm_return_type=str,
                ...     feature_post_callback=feature_post_callback
                ... )
                ```
    """
    name: str
    prompt: str
    distribution: Distribution
    config: PromptFeatureConfiguration
    llm_return_type: type = str
    feature_post_callback: Optional[Callable] = None
