from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel


async def _extract_features_with_llm(text, schema: BaseModel, system_prompt: str, aoai_api_key: str, aoai_api_endpoint: str, gpt4o_deployment: str = "gpt-4o"):
    """Extract features using the Azure OpenAI LLM."""
    llm = AzureChatOpenAI(
        azure_deployment=gpt4o_deployment, api_key=aoai_api_key, azure_endpoint=aoai_api_endpoint, api_version="2024-08-01-preview", temperature=0)

    llm = llm.with_structured_output(schema)

    return await llm.ainvoke(
        [SystemMessage(system_prompt), HumanMessage(text)])
