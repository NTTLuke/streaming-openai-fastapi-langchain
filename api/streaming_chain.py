from dotenv import load_dotenv
from pydantic import BaseModel
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from fastapi.responses import StreamingResponse
from fastapi import APIRouter
import asyncio
from typing import AsyncIterable, Tuple

load_dotenv()


router_chain = APIRouter()


class Message(BaseModel):
    content: str


def init_llm_and_chain() -> Tuple[LLMChain, AsyncIteratorCallbackHandler]:
    # ref https://python.langchain.com/docs/modules/callbacks/
    callback_handler = AsyncIteratorCallbackHandler()

    llm = AzureChatOpenAI(
        deployment_name="chat",
        temperature=0,
        streaming=True,
        verbose=True,
        callbacks=[callback_handler],
    )

    prompt = PromptTemplate(
        input_variables=["topic"],
        template="Create a rock song lyric starting from this topic: {topic}. Use max 3 verses.",
    )

    # TODO: replace with the chain you want to use
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain, callback_handler


# start the app
chain, callback_handler = init_llm_and_chain()


async def ask_chat_llm(content: str) -> AsyncIterable[str]:
    """Ask the LLM to generate a text based on the input content using streaming."""

    task = asyncio.create_task(chain.arun(content))
    try:
        async for token in callback_handler.aiter():
            yield token
    except Exception as e:
        print(f"Bro, something went wrong ...fix it or run far away! {e}")
    finally:
        callback_handler.done.set()
    await task


@router_chain.post("/stream_chat/")
async def stream_chat(message: Message):
    response = ask_chat_llm(message.content)
    return StreamingResponse(response, media_type="text/event-stream")
