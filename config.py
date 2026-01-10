import os
from dotenv import load_dotenv
from langchain_fireworks import ChatFireworks

load_dotenv()

model = ChatFireworks(
    model="accounts/fireworks/models/minimax-m2p1",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
