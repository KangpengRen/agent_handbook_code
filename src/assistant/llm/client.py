"""大模型客户端"""
from dotenv import load_dotenv
import os
import requests
import time

from assistant.core.agent_assistant_exception import LLMRequestException, LLMRetryException

load_dotenv(override=True)


class DeepSeekClient:

    def __init__(self):
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.api_url = os.getenv("DEEPSEEK_API_URL")
        self.reason_model = os.getenv("DEEPSEEK_REASON_MODEL")
        self.model = os.getenv("DEEPSEEK_MODEL")

    def call(self, messages: list, tools: dict, reasoning_effort: str = None) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "messages": messages,
            "tools": tools,
            "stream": False,
            "temperature": 0.7,
            "max_tokens": 4096
        }


        # 【非对称思考预算】Planner 用推理模型，Executor 用快速模型
        if reasoning_effort == "high":
            payload["model"] = self.reason_model
        else:
            payload["model"] = self.model

        for attempt in range(3):
            try:
                reply = requests.post(url=self.api_url, headers=headers, json=payload, timeout=30)

                # 成功直接返回
                if reply.status_code == 200:
                    return reply.json()

                # 如果是 429 限流 或 5xx 服务端错误，进行指数退避重试
                if reply.status_code == 429 or reply.status_code >= 00:
                    raise LLMRetryException(f"可重试的 API 错误 {reply.status_code}: {reply.text}")

                # 其他客户端错误（如 401 鉴权失败、400 参数错误），重试也没用，直接抛出
                raise LLMRequestException(f"不可重试的 API 错误 {reply.status_code}: {reply.text}")
            except LLMRetryException:
                if attempt <= 2:
                    time.sleep(2 ** attempt)

        raise LLMRequestException("LLM API请求重试耗尽")