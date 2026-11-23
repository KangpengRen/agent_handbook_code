"""
Agent循环
"""
from typing import Any
import json

from .agent_assistant_exception import ToolException, AgentLoopException
from .types import AgentLoopOptions, AgentEvent
from ..llm.client import DeepSeekClient
from ..tool.registry import build_tool_registry, tools_to_wire


class AgentLoop:
    """Agent循环类"""

    def __init__(self, opts: AgentLoopOptions):
        self.opts = opts
        self.llm = DeepSeekClient()
        self.tool_by_name = build_tool_registry(opts.tools)
        self.tools = tools_to_wire(opts.tools)

    def _emit(self, event: str, turn: int, **payload: Any):
        if self.opts.on_event:
            self.opts.on_event(AgentEvent(type=event, turn=turn, payload=payload))

    def run_agent(self, task: str) -> str:
        """Agent循环"""
        messages = [
            {"role": "system", "content": self.opts.system_prompt},
            {"role": "user", "content": task}
        ]

        for turn in range(1, self.opts.max_turns + 1):
            self._emit("llm_start", turn)

            reply = self.llm.call(messages=messages, tools=self.tools)
            choice = reply["choices"][0]
            message = choice["message"]
            # 将大模型回复消息加入到消息列表中
            messages.append(message)

            # 大模型回复
            text = message.get("content", "").strip()
            if text:
                self._emit(event="llm_text", turn=turn, text=text)

            # 判断是否到达结束条件
            if choice["finish_reason"] != "tool_calls":
                self._emit(event="done", turn=turn, text=text)
                return text

            # 执行工具调用
            for tool_call in message["tool_calls"]:
                name = tool_call["function"]["name"]
                args = json.loads(tool_call["function"]["arguments"])

                try:
                    self._emit(event="tool_call", turn=turn, name=name, args=args)
                    if not (tool := self.tool_by_name.get(name, None)):
                        raise ToolException(f"工具[{name}]未找到")
                    tool_result = tool.execute(args)
                    is_error = False
                except Exception as e:
                    tool_result = f"工具调用失败：{e}"
                    is_error = True

                self._emit(event="tool_result", turn=turn, tool_result=tool_result, is_error=is_error)
                # 将工具调用结果加入到消息列表中
                messages.append({"role": "tool", "tool_call_id": tool_call["id"], "content": tool_result})

        self._emit(event="aborted", turn=self.opts.max_turns, reason="循环次数消耗完毕")
        raise AgentLoopException(f"达到 {self.opts.max_turns} 轮上限，未完成任务")
