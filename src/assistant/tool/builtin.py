"""
具体工具实现
"""
from typing import Any
from pathlib import Path

from ..core.types import Tool
from ..core.agent_assistant_exception import ToolException


class ListFilesTool(Tool):
    name = "list_files"
    description = "读取指定目录下的文件，默认为当前目录"
    parameters = {
        "type": "object",
        "properties": {"path": {"description": "列出文件的指定目录，为空即为读取当前目录", "type": "string"}},
        "required": []
    }

    def execute(self, inp: dict[str, Any]) -> str:
        """读取指定目录下的文件，默认为当前目录"""
        inp = inp or {}
        target_dir = inp.get("path") or "."
        return "\n".join(sorted(p.name for p in Path(target_dir).iterdir()))


class ReadFileTool(Tool):
    name = "read_file"
    description = "读取指定路径的文件，以防超长阶段2000字符"
    parameters = {
        "type": "object",
        "properties": {"path": {"description": "读取文件的指定目录", "type": "string"}},
        "required": ["path"]
    }

    def execute(self, inp: dict[str, Any]) -> str:
        """读取指定路径的文件，以防超长阶段2000字符"""
        if not (p := inp.get("path")):
            raise ToolException(f"read_file 工具调用参数异常，缺失参数：path")
        if not Path(p).exists():
            raise ToolException(f"文件{inp.get('path')}不存在")
        return Path(p).read_text(encoding="utf-8")[:2000]


class WriteFileTool(Tool):
    name = "write_file"
    description = "向指定路径的文件写入指定内容，覆盖写入"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"description": "写入文件的指定目录", "type": "string"},
            "content": {"description": "写入文件的内容", "type": "string"}
        },
        "required": ["path", "content"]
    }

    def execute(self, inp: dict[str, Any]) -> str:
        """向指定路径的文件写入指定内容，覆盖写入"""
        if not (p := inp.get("path")):
            raise ToolException(f"write_file 工具调用参数异常，缺失参数：path")
        if not (c := inp.get("content")):
            raise ToolException(f"write_file 工具调用参数异常，缺失参数：content")
        count = Path(p).write_text(data=c, encoding="utf-8")
        return f"向文件写入{count}字符"
