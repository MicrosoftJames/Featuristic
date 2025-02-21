from dataclasses import dataclass
from typing import Callable, Optional, List

SYSTEM_MESSAGE = """You are helpful AI assistant that extracts machine learning features from text.
You will be given a text input and your job is to extract features according to the JSON schema provided."""


class PromptFeatureDefinition():
    def __init__(self, name, prompt: str, llm_return_type: type = str, feature_post_callback: Optional[Callable] = None):
        """
        A custom prompt-based feature.
<<<<<<< HEAD

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
=======
>>>>>>> 8c82d3f (Add initial project setup with .gitignore, requirements, and core features)
        """
        self.name = name
        self.feature_post_callback = feature_post_callback
        self.prompt = prompt
        self.llm_return_type = llm_return_type


class PromptFeature(PromptFeatureDefinition):
    def __init__(self, name, prompt, value, llm_response, feature_post_callback: Optional[Callable] = None, llm_return_type=str):
        super().__init__(name, prompt,
                         feature_post_callback, llm_return_type)
        self.value = value
        self.llm_response = llm_response

    def __repr__(self):
        return f"PromptFeature(name='{self.name}', value={repr(self.value)})"


class Feature():
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __repr__(self):
        return f"Feature(name='{self.name}', value={repr(self.value)})"


@dataclass
class BaseFeatureDefinition():
    preprocess_callback: Optional[Callable]


@dataclass
class PromptFeatureDefinitionGroup(BaseFeatureDefinition):
    features: List[PromptFeatureDefinition]
    system_prompt: Optional[str] = SYSTEM_MESSAGE


@dataclass
class FeatureDefinition(BaseFeatureDefinition):
    name: str
