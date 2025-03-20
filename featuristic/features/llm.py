from litellm import acompletion, supports_response_schema
from pydantic import BaseModel


async def _extract_features_with_llm(text, schema: BaseModel, system_prompt: str, api_key: str, api_base: str, api_version: str, model: str, use_cache: bool = False):
    """Extract features using the Azure OpenAI LLM."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
    ]

    if not supports_response_schema(model=model):
        raise ValueError(
            "The provided model does not support json_schema response format.")

    resp = await acompletion(model=model,
                             messages=messages,
                             response_format=schema,
                             api_key=api_key,
                             base_url=api_base,
                             api_version=api_version,
                             temperature=0,
                             caching=use_cache)

    return schema.model_validate_json(resp.choices[0].message.content)
