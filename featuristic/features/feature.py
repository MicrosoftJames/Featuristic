from dataclasses import dataclass
from typing import Callable, Optional, List, Union
from featuristic.classification import Distribution

SYSTEM_MESSAGE = """You are helpful AI assistant that extracts machine learning features from text.
You will be given a text input and your job is to extract features according to the JSON schema provided."""


@dataclass(frozen=True)
class PromptFeatureConfiguration:
    aoai_api_key: str = None
    aoai_api_endpoint: str = None
    gpt4o_deployment: str = "gpt-4o"
    preprocess_callback: Optional[Callable] = None
    system_prompt: Optional[str] = SYSTEM_MESSAGE

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)


@dataclass
class FeatureDefinition:
    preprocess_callback: Optional[Callable]
    name: str
    distribution: Distribution


@dataclass
class PromptFeatureDefinition:
    """
    A custom prompt-based feature.

    Args:
        name (str): The name of the feature.
        prompt (str): The prompt to be used to extract the feature.
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

        >>> feature = Feature(
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


@dataclass
class Feature:
    name: str
    values: List[Union[int, float]]
    distribution: Distribution
