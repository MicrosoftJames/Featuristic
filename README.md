<img src="img/logo.png" alt="Featuristic"  width="200px" height="auto" style="display: block; margin-left: auto; margin-right: auto">

# Featuristic

Featuristic is a Python library that combines LLM-based feature extraction with bayesian classification. It allows you to define features using both traditional methods (i.e. pure python) and prompt-based methods using LLMs, and then train a bayesian classifier on the extracted features.

## Key Features

- **Prompt-based Feature Extraction**: Define features using prompts and extract them using LLMs.
- **Batch Processing**: Efficiently process large datasets in batched calls to the LLM.
- **Bayesian Classification**: Train a bayesian classifier on the extracted features, allowing for mixed distribution types.

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

### Define Features

You can define both pure python features and prompt-based features.

#### Pure Python Feature

```python
from featuristic import FeatureDefinition, Distribution

num_sentences = FeatureDefinition(
    name="number of sentences", 
    preprocess_callback=lambda x: len(x.text.split('.')),
    distribution=Distribution.GAUSSIAN)
```

#### Prompt-based Feature

```python
from featuristic import PromptFeatureDefinition, PromptFeatureConfiguration, Distribution

# Define the prompt feature configuration
# This configuration is used to call the LLM
# and extract features.
# All PromptFeatureDefinitions that use this configuration
# will be extracted using the same LLM call using 'structured outputs'.
config = PromptFeatureConfiguration(
    api_key="your_api_key",
    api_base="your_api_endpoint",
    model="gpt-4o", # any model that supports structured outputs
    api_version="api_version",
    preprocess_callback=lambda x: x.text,
)

mention_of_war = PromptFeatureDefinition(
    name="mention of war", 
    prompt="Whether or not the notion of war is mentioned",
    distribution=Distribution.BERNOULLI,
    llm_return_type=bool,
    config=config
)

```

### Initialize Featuristic

```python
from featuristic import FeaturisticClassifier

feature_definitions = [
    num_sentences,
    mention_of_war]

featuristic = FeaturisticClassifier(distributions=[d.distribution for d in feature_definitions])
```

### Extract Features

```python
from featuristic import extract_features
from dataclasses import dataclass

@dataclass
class MyData:
    text: str

data = [MyData("The United States and Russia are not at war."), # related to war
        MyData("Free healthcare should be a human right. For everyone."),]
feature_df = await extract_features(data, feature_definitions)

print(feature_df)
```
will output:

|| number of sentences | mention of war |
|---|---------------------|----------------|
|0| 2                   | True           |
|1| 3                   | False          |

### Train Classifier
```python
from featuristic import FeaturisticClassifier

# Define the classifier
classifier = FeaturisticClassifier(
    distributions=[d.distribution for d in feature_definitions],
)

# Train the classifier
classifier.fit(
    feature_df,
    Y=[1, 0],  # Labels for the data
)

# Predict using the classifier
test_data = [
    MyData("The battle of Stalingrad was a turning point in WW2.") # related to war, expected label is 1
]

test_feature_df = await extract_features(test_data, feature_definitions)

predictions = classifier.predict(
    test_feature_df,
)
print(predictions)
```
will output:

```
[1]
```

## Running Tests
To run the tests, use:

```bash
python3 -m pytest tests/
```
