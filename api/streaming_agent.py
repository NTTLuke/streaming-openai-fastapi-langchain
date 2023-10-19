import asyncio
from typing import Any

import uvicorn
from fastapi import FastAPI, APIRouter, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import AzureChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.schema import LLMResult

from tools.image_generator_tool import ImageGeneratorTool
from tools.sentimenta_tool import SentimentTool


router_agent = APIRouter()

# initialize the agent (we need to do this for the callbacks)
llm = AzureChatOpenAI(
    deployment_name="chat",
    temperature=0,
    streaming=True,
    verbose=True,
    callbacks=[],
)

memory = ConversationBufferWindowMemory(
    memory_key="chat_history", k=5, return_messages=True, output_key="output"
)


agent_tools = [ImageGeneratorTool()]
agent_tools.append(SentimentTool())

agent = initialize_agent(
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    tools=agent_tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method="generate",
    memory=memory,
    return_intermediate_steps=False,
)


class AsyncCallbackHandler(AsyncIteratorCallbackHandler):
    """Callback handler for async iterator.
    Methods are used for analyzing the response from the LLM.
    If the Final Answer is reached, we put the tokens in a queue.
    Otherwise, we just reset the content.
    """

    content: str = ""
    final_answer: bool = False

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.content += token
        # if we passed the final answer, we put tokens in queue
        if self.final_answer:
            if '"action_input": "' in self.content:
                if token not in ['"', "}"]:
                    self.queue.put_nowait(token)
        elif "Final Answer" in self.content:
            self.final_answer = True
            self.content = ""

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        if self.final_answer:
            self.content = ""
            self.final_answer = False
            self.done.set()
        else:
            self.content = ""


async def run_call(query: str, stream_it: AsyncCallbackHandler):
    # assign callback handler
    agent.agent.llm_chain.llm.callbacks = [stream_it]
    # now query
    await agent.acall(inputs={"input": query})


# request input format
class Message(BaseModel):
    content: str


async def create_gen(query: str, stream_it: AsyncCallbackHandler):
    task = asyncio.create_task(run_call(query, stream_it))
    try:
        async for token in stream_it.aiter():
            yield token
    except Exception as e:
        # LOL
        print(f"Bro, something went wrong ...fix it or run far away! {e}")
    finally:
        stream_it.done.set()
    await task


@router_agent.post("/stream_chat/")
async def chat(
    message: Message = Body(
        ..., example={"content": "Tell me more about low-fi music style."}
    )
):
    stream_it = AsyncCallbackHandler()
    gen = create_gen(message.content, stream_it)
    return StreamingResponse(gen, media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run("app:app", host="localhost", port=8000, reload=True)
