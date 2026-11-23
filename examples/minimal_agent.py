"""最小的可运行agent"""
from pathlib import Path
from typing import Optional, Any
import requests
import json
import os
import sys
from dotenv import load_dotenv

load_dotenv(override=True)


class ToolException(Exception):
    """工具执行异常"""
    pass


# 1. 工具定义
def list_files(inp: Optional[dict] = None) -> str:
    """读取指定目录下的文件，默认为当前目录"""
    inp = inp or {}
    return "\n".join(sorted(p.name for p in Path(inp.get("dir", ".")).iterdir()))


def read_file(inp: dict) -> str:
    """读取指定路径的文件，以防超长阶段2000字符"""
    if not (p := inp.get("path")):
        raise ToolException(f"read_file 工具调用参数异常，缺失参数：path")
    if not Path(p).exists():
        raise ToolException(f"文件{inp.get('path')}不存在")
    return Path(p).read_text(encoding="utf-8")[:2000]


def write_file(inp: dict) -> str:
    """向指定路径的文件写入指定内容，覆盖写入"""
    if not (p := inp.get("path")):
        raise ToolException(f"write_file 工具调用参数异常，缺失参数：path")
    if not (c := inp.get("content")):
        raise ToolException(f"write_file 工具调用参数异常，缺失参数：content")
    count = Path(p).write_text(data=c, encoding="utf-8")
    return f"向文件写入{count}字符"


TOOLS: list[dict[str, Any]] = [
    {
        "name": "list_files",
        "description": "读取指定目录下的文件，默认为当前目录",
        "parameters": {
            "type": "object",
            "properties": {"path": {"description": "列出文件的指定目录，为空即为读取当前目录", "type": "string"}},
            "required": []
        },
        "run": list_files
    },
    {
        "name": "read_file",
        "description": "读取指定路径的文件，以防超长阶段2000字符",
        "parameters": {
            "type": "object",
            "properties": {"path": {"description": "读取文件的指定目录", "type": "string"}},
            "required": ["path"]
        },
        "run": read_file
    },
    {
        "name": "write_file",
        "description": "向指定路径的文件写入指定内容，覆盖写入",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"description": "写入文件的指定目录", "type": "string"},
                "content": {"description": "写入文件的内容", "type": "string"}
            },
            "required": ["path", "content"]
        },
        "run": write_file
    }
]
TOOLS_BY_NAME: dict[str, Any] = {t["name"]: t for t in TOOLS}

tools: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["parameters"]
        }
    }
    for t in TOOLS
]

# 2. 定义大模型请求
API_KEY = os.getenv("DEEPSEEK_API_KEY")
API_URL = os.getenv("DEEPSEEK_API_URL")
MODEL = os.getenv("DEEPSEEK_MODEL")


def call_llm(messages: list) -> dict[str, Any]:
    """向大模型发送请求"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "deepseek-chat",
        "messages": messages,
        "tools": tools,
        "stream": False,
        "temperature": 0.7,
        "max_tokens": 1024
    }

    reply = requests.post(url=API_URL, headers=headers, json=payload, timeout=30)
    if reply.status_code != 200:
        raise RuntimeError(f"API {reply.status_code}: {reply.text}")
    return reply.json()


SYSTEM_PROMPT = """
你是示例助手，一个企业内部任务执行Agent。
工作方式：先思考需要哪些信息，再调用工具获取，信息足够后完成任务，直接给出最终答复。
约束：只操作当前工作目录；文件不存在或读取失败时如实报告，绝不编造内容。
"""


def run_agent(task: str) -> str:
    """运行agent循环"""
    # 初始化messages列表
    messages: list = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task}
    ]
    print("=" * 10, " 进入Agent循环 ", "=" * 10)

    while True:
        # 发送消息给大模型
        reply = call_llm(messages)

        choice = reply["choices"][0]
        message = choice["message"]
        # 将大模型回复消息加入到消息列表中
        messages.append(message)

        # 大模型回复
        text = message.get("content", "").strip()
        if text:
            print("-" * 50)
            print(f"[思考] LLM：{text}")
            print("-" * 50, "\n")

        # 判断是否到达结束条件
        if choice["finish_reason"] != "tool_calls":
            print("=" * 10, " 任务完成！ ", "=" * 10)
            return text

        # 执行工具调用
        for tool_call in message["tool_calls"]:
            name = tool_call["function"]["name"]
            args = json.loads(tool_call["function"]["arguments"])

            try:
                print("-" * 50)
                print(f"[行动] 调用工具：{name}({args})")
                print("-" * 50, "\n")
                if not (tool := TOOLS_BY_NAME.get(name, None)):
                    raise ToolException(f"工具[{name}]未找到")
                tool_result = tool["run"](args)
            except Exception as e:
                tool_result = f"工具调用失败：{e}"

            print("-" * 50)
            print(f"[观察] 调用结果：{tool_result}")
            print("-" * 50, "\n")
            # 将工具调用结果加入到消息列表中
            messages.append({"role": "tool", "tool_call_id": tool_call["id"], "content": tool_result})


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "请列出当前目录下的所有文件名，然后写一份200字以内的 summary.md 总结一下"
    print(f"\n ==== 开始执行任务 ====\n")
    print(f"\n ==== 最终答复 ====\n{run_agent(task)}")
