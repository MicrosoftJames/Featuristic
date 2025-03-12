
from typing import List
from unittest import mock
import pytest

from pydantic import BaseModel

from featuristic.features.llm import _extract_features_with_llm


@pytest.mark.asyncio
@mock.patch('featuristic.features.llm.AzureChatOpenAI.with_structured_output')
async def test_extract_features_with_llm(mock_with_structured_output):
    mock_with_structured_output.return_value = mock.Mock()

    class Response(BaseModel):
        animal_list: List[str]
        additional_kwargs: dict = {"parsed": True}

    async def mock_ainvoke(*args, **kwargs):
        return Response(animal_list=["cat", "dog"])

    mock_with_structured_output.return_value.ainvoke.side_effect = mock_ainvoke

    data = "The cat and dog are friends."
    system_prompt = "Extract the list of animals from the text"
    aoai_api_key = "test"
    aoai_api_endpoint = "test"

    class FeaturesSchema(BaseModel):
        animal_list: List[str]

    schema = FeaturesSchema
    result = await _extract_features_with_llm(data, schema, system_prompt, aoai_api_key, aoai_api_endpoint)
    assert result.animal_list == ["cat", "dog"]
