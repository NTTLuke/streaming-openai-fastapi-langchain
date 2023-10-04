# TODO: ADD ENV FILE WITH THESE VARIABLES
# OPENAI_API_TYPE=azure
# OPENAI_API_KEY=
# OPENAI_API_BASE=
# OPENAI_API_VERSION=2023-03-15-preview


from dotenv import load_dotenv
from pydantic import BaseModel

from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.schema import HumanMessage
from langchain.chat_models import AzureChatOpenAI

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi import FastAPI

import asyncio
from typing import AsyncIterable

load_dotenv()


class Message(BaseModel):
    content: str


app = FastAPI(
    title="Streaming API with LangChain and FastAPI",
    description="FastAPI with LangChain and streaming endpoint",
    version="0.1.0",
    docs_url="/",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def ask_chat_llm(content: str) -> AsyncIterable[str]:
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate

    """
    Invoking the AzureChatOpenAI model with streaming enabled and running it as async
    the callback_handler will be used to get the response from the model
    see https://python.langchain.com/docs/modules/callbacks/async_callbacks
    """
    callback_handler = AsyncIteratorCallbackHandler()

    llm = AzureChatOpenAI(
        deployment_name="chat",
        temperature=0,
        streaming=True,
        verbose=True,
        callbacks=[callback_handler],
    )

    # prompt example
    prompt = PromptTemplate(
        input_variables=["topic"],
        template="Create a rock song lyrics starting from this topic. Use max 3 verses  : {topic}?",
    )

    # TODO: replace with the type of chain you need
    chain = LLMChain(llm=llm, prompt=prompt, callbacks=[callback_handler])

    task = asyncio.create_task(
        # llm.agenerate(messages=[[HumanMessage(content=content)]])
        chain.arun(content)
    )

    try:
        async for token in callback_handler.aiter():
            yield token
    except Exception as e:
        # LOL
        print(f"Bro, something went wrong ...fix it or run far away! {e}")
    finally:
        # see on_llm_end method in the callback_handler: it seems the task is closed by design
        # BUT just to be sure we close the task every time
        callback_handler.done.set()

    await task


@app.post("/stream_chat/")
async def stream_chat(message: Message):
    """The endpoint for streaming"""

    # prepare the response as a stream
    response = ask_chat_llm(message.content)

    # return the response as a stream.
    # Remember: media_type MUST be set to text/event-stream
    return StreamingResponse(response, media_type="text/event-stream")
