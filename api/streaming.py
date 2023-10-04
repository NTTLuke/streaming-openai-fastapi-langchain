from dotenv import load_dotenv
from pydantic import BaseModel
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.schema import HumanMessage
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi import FastAPI
import asyncio
from typing import AsyncIterable, Tuple

load_dotenv()


class Message(BaseModel):
    content: str


def init_fastapi() -> FastAPI:
    """Initialize FastAPI"""
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
    return app


def init_llm_and_chain() -> Tuple[LLMChain, AsyncIteratorCallbackHandler]:
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
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain, callback_handler


# start the app
app = init_fastapi()
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


@app.post("/stream_chat/")
async def stream_chat(message: Message):
    response = ask_chat_llm(message.content)
    return StreamingResponse(response, media_type="text/event-stream")
