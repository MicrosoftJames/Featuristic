
from typing import List
from unittest import mock
from litellm import ContentPolicyViolationError
import pytest

from pydantic import BaseModel

from featuristic.features.llm import _extract_features_with_llm


@pytest.mark.asyncio
@mock.patch('featuristic.features.llm.acompletion')
async def test_extract_features_with_llm(mock_acompletion):
    class MockResponse:
        class MockChoice:
            class MockMessage:
                content = '{"animal_list": ["cat", "dog"]}'
            message = MockMessage()
        choices = [MockChoice()]

    mock_acompletion.return_value = MockResponse()

    data = "The cat and dog are friends."
    system_prompt = "Extract the list of animals from the text"
    api_key = "example"
    api_base = "https://example.com"
    api_version = "2023-10-01"
    model = "gpt-4o"

    class FeaturesSchema(BaseModel):
        animal_list: List[str]

    schema = FeaturesSchema
    result = await _extract_features_with_llm(data, schema, system_prompt,
                                              api_key, api_base, api_version, model)
    assert result.animal_list == ["cat", "dog"]
    assert mock_acompletion.called
    assert mock_acompletion.call_count == 1
    assert mock_acompletion.call_args[1]['model'] == model


@pytest.mark.asyncio
@mock.patch('featuristic.features.llm.acompletion')
async def test_extract_features_with_llm_content_violation_exception(mock_acompletion):

    data = "Cats aren't pigs like dogs."
    system_prompt = "Extract the list of animals from the text"
    api_key = "example"
    api_base = "https://example.com"
    api_version = "2023-10-01"
    model = "gpt-4o"

    class FeaturesSchema(BaseModel):
        animal_list: List[str]

    mock_acompletion.side_effect = ContentPolicyViolationError(
        "Content policy violation: test", model, "test provider")

    with pytest.raises(ContentPolicyViolationError, match="Content policy violation: test"):
        await _extract_features_with_llm(data, FeaturesSchema, system_prompt,
                                         api_key, api_base, api_version, model)
