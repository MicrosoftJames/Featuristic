# Featuristic

Featuristic is a Python library designed to extract features from text data using both traditional and prompt-based methods. It leverages the power of large language models (LLMs) to extract complex features from text.

## Features

- **Traditional Feature Extraction**: Define features using simple preprocessing callbacks.
- **Prompt-based Feature Extraction**: Define features using prompts and extract them using LLMs.
- **Batch Processing**: Efficiently process large datasets in batches.

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

### Define Features

You can define both traditional and prompt-based features.

#### Traditional Feature

```python
from featuristic.feature import FeatureDefinition

num_sentences = FeatureDefinition(
    name="number of sentences", preprocess_callback=lambda x: len(x.text.split('.')))
```

#### Prompt-based Feature

```python
from featuristic.feature import PromptFeatureDefinition, PromptFeatureDefinitionGroup

mention_of_war = PromptFeatureDefinition(
    name="mention of war", prompt="Whether or not the notion of war is mentioned", llm_return_type=bool)

group = PromptFeatureDefinitionGroup(
    features=[mention_of_war], preprocess_callback=lambda x: x.text)  # x.text is used to access the text data to be processed
```

### Initialize Featuristic

```python
from featuristic.featuristic import Featuristic

featuristic = Featuristic(aoai_api_key="your_api_key", aoai_api_endpoint="your_api_endpoint")
```

### Add Feature Definitions

```python
featuristic.add_feature_definition(num_sentences)
featuristic.add_feature_definition(group)
```

### Extract Features

```python
import asyncio

@dataclass
class MyData:
    text: str


data = [MyData("The United States and Russia are at war.")]
results = asyncio.run(featuristic.extract(data))
print(results)
```

The output will be:

```
[[Feature(name='number of sentences', value=2), [PromptFeature(name='mention of war', value=True)]]]
```

## Running Tests

To run the tests, use:

```bash
python3 -m pytest tests/
```
