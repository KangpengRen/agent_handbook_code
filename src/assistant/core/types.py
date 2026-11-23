"""
定义通用接口
"""
from typing import Protocol, Any, Optional, Callable, Literal
from dataclasses import dataclass, field


class Tool(Protocol):
    """工具格式规范化"""
    name: str
    description: str
    parameters: dict[str, Any]

    def execute(self, inp: dict[str, Any]) -> str: ...


@dataclass
class AgentEvent:
    type: Literal["llm_start", "llm_text", "tool_call", "tool_result", "done", "aborted"]
    turn: int
    payload: dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentLoopOptions:
    max_turns: int
    system_prompt: str
    tools: list[Tool]
    on_event: Optional[Callable[[AgentEvent], None]] = None