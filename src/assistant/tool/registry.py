"""
工具注册
"""
from typing import Any

from ..core.types import Tool


def build_tool_registry(tools: list[Tool]) -> dict[str, Tool]:
    """转化为工具名-工具的映射关系"""
    return {t.name: t for t in tools}


def tools_to_wire(tools: list[Tool]) -> list[dict[str, Any]]:
    """将工具转化为适配大模型的格式"""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters
            }
        }
        for t in tools
    ]
