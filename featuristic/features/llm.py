from litellm import acompletion, supports_response_schema
from pydantic import BaseModel


async def _extract_features_with_llm(text, schema: BaseModel, system_prompt: str, aoai_api_key: str, aoai_api_endpoint: str, gpt4o_deployment: str = "gpt-4o"):
    """Extract features using the Azure OpenAI LLM."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
    ]

    if not supports_response_schema(model=gpt4o_deployment):
        raise ValueError(
            "The provided model does not support json_schema response format.")

    resp = await acompletion(model=gpt4o_deployment,
                             messages=messages,
                             response_format=schema,
                             api_key=aoai_api_key,
                             base_url=aoai_api_endpoint,
                             api_version="2024-08-01-preview",
                             temperature=0)

    return schema.model_validate_json(resp.choices[0].message.content)
