from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from api.streaming_agent import router_agent
from api.streaming_chain import router_chain

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(router_agent, prefix="/chat")
app.include_router(router_chain, prefix="/chat")
