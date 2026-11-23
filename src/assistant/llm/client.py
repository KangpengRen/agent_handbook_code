"""大模型客户端"""
from dotenv import load_dotenv
import os
import requests

load_dotenv(override=True)


class DeepSeekClient:

    def __init__(self):
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.api_url = os.getenv("DEEPSEEK_API_URL")
        self.model = os.getenv("DEEPSEEK_MODEL")


    def call(self, messages: list, tools: dict) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "stream": False,
            "temperature": 0.7,
            "max_tokens": 1024
        }

        reply = requests.post(url=self.api_url, headers=headers, json=payload, timeout=30)
        if reply.status_code != 200:
            raise RuntimeError(f"API {reply.status_code}: {reply.text}")
        return reply.json()