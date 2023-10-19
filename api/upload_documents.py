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


router_upload = APIRouter()
