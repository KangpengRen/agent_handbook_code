"""
Agent循环
"""
from typing import Any
import json

from .agent_assistant_exception import ToolException, AgentLoopException
from .budget import AgentState, StopCondition
from .types import AgentLoopOptions, AgentEvent
from ..llm.client import DeepSeekClient
from ..tool.registry import build_tool_registry, tools_to_wire


class AgentLoop:
    """Agent循环类"""

    def __init__(self, opts: AgentLoopOptions, stop_conditions: list[StopCondition], verifier=None):
        self.opts = opts
        self.stop_conditions = stop_conditions
        self.llm = DeepSeekClient()
        self.tool_by_name = build_tool_registry(opts.tools)
        self.tools = tools_to_wire(opts.tools)
        self.verifier = verifier

    def _emit(self, event: str, turn: int, **payload: Any):
        if self.opts.on_event:
            self.opts.on_event(AgentEvent(type=event, turn=turn, payload=payload))

    def run_agent(self, task: str) -> str:
        """Agent循环"""
        messages = [
            {"role": "system", "content": self.opts.system_prompt},
            {"role": "user", "content": task}
        ]

        state = AgentState()  # 初始化Agent状态

        while True:
            state.turn += 1

            for cond in self.stop_conditions:
                d = cond.check(state=state)
                if d.should_stop:
                    self._emit(event="aborted", turn=state.turn, reason=d.reason)
                    return self._finalize(messages=messages, reason=d.reason, state=state)


            self._emit("llm_start", state.turn)

            try:
                reply = self.llm.call(messages=messages, tools=self.tools)
            except Exception as e:
                self._emit(event="aborted", turn=state.turn, reason=f"大模型服务调用异常：{e}")
                return f"❌ 系统提示：大模型服务连接失败，Agent已安全停止。原因：{e}"
            state.record_usage(reply.get("usage", {}))

            choice = reply["choices"][0]
            message = choice["message"]
            # 调用完大模型，记录token使用信息
            # 将大模型回复消息加入到消息列表中
            messages.append(message)

            # 大模型回复
            text = message.get("content", "").strip()
            if text:
                self._emit(event="done", turn=state.turn, text=text)

                if choice.get("finish_reason") == "length":
                    self._emit("aborted", state.turn, reason="max_tokens阶段，正在请求精简重试")

                    messages.pop()
                    messages.append({
                        "role": "user",
                        "content": "你刚才的输出触发了最大长度限制被硬截断了。"
                                   "请精简你的回答，或者分多次调用工具。如果你是准备调用工具，请确保参数不要过长。"
                    })
                    continue

                self._emit(event="llm_text", turn=state.turn, text=text)

            # 判断是否到达结束条件
            if choice["finish_reason"] != "tool_calls":
                # 通过方法严格校验是否完成任务
                if self.verifier and (err := self.verifier(messages)):
                    self._emit(event="verification_failed", turn=state.turn, error=err)
                    messages.append({
                        "role": "user",
                        "content": f"任务未通过外部校验：{err}，请根据要求重新修改"
                    })
                    continue

                self._emit(event="done", turn=state.turn, text=text)
                return text

            # 执行工具调用
            for tool_call in message["tool_calls"]:
                name = tool_call["function"]["name"]
                args = json.loads(tool_call["function"]["arguments"])

                try:
                    self._emit(event="tool_call", turn=state.turn, name=name, args=args)
                    if not (tool := self.tool_by_name.get(name, None)):
                        raise ToolException(f"工具[{name}]未找到")
                    tool_result = tool.execute(args)
                    is_error = False
                except Exception as e:
                    tool_result = f"工具调用失败：{e}"
                    is_error = True

                self._emit(event="tool_result", turn=state.turn, tool_result=tool_result, is_error=is_error)
                # 将工具调用结果加入到消息列表中
                messages.append({"role": "tool", "tool_call_id": tool_call["id"], "content": tool_result})

    def _finalize(self, messages: list[dict[str, Any]], reason: str, state: AgentState) -> str:
        """收尾轮：禁用工具的最后一次调用，把残局翻译成可交接的总结"""
        messages.append({
            "role": "user",
            "content": f"运行预算已耗尽（{reason}）。请立即停止行动，"
                       f"总结：已完成事项、未完成事项、下一步建议"
        })
        # 禁用工具，强制总结
        reply = self.llm.call(messages, tools=[])
        state.record_usage(reply.get("usage", {}))
        choice = reply["choices"][0]
        message = choice["message"]
        return f"任务因 {reason} 终止\n" + message.get("content", "")