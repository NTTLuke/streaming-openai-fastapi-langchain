from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
from langchain.llms import AzureOpenAI


def azure_openai_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        deployment="embeddings", chunk_size=1, embedding_ctx_length=1000
    )


def azure_chat_openai_llm() -> AzureChatOpenAI:
    return AzureChatOpenAI(deployment_name="chat", temperature=0, max_tokens=500)


def azure_openai_llm() -> AzureOpenAI:
    return AzureOpenAI(deployment_name="chat", temperature=0, max_tokens=500)
