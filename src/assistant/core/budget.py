"""
预算熔断
"""
import time
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class AgentState:
    turn: int = 0  # 轮次
    input_tokens: int = 0  # 输入的token
    output_tokens: int = 0  # 输出的token
    start_at: float = field(default_factory=time.monotonic)

    def record_usage(self, usage: dict) -> None:
        # 兼容 DeepSeek/OpenAI 和 Claude 的字段名
        self.input_tokens += usage.get("prompt_tokens", usage.get("input_tokens", 0))
        self.output_tokens += usage.get("completion_tokens", usage.get("output_tokens", 0))


@dataclass
class StopDecision:
    should_stop: bool
    reason: str


class StopCondition(Protocol):
    def check(self, state: AgentState) -> StopDecision: ...


@dataclass
class MaxTurn(StopCondition):
    limit: int

    def check(self, state: AgentState) -> StopDecision:
        return StopDecision(should_stop=state.turn >= self.limit,
                            reason=f"max_turns: {state.turn}/{self.limit}")


@dataclass
class TokenBudget(StopCondition):
    limit: int

    def check(self, state: AgentState) -> StopDecision:
        used = state.input_tokens + state.output_tokens
        return StopDecision(should_stop=used >= self.limit,
                            reason=f"token_budget {used}/{self.limit}")


@dataclass
class CostBudget(StopCondition):
    limit_used: float
    price_in: float = 0.14
    price_out: float = 0.28

    def check(self, state: AgentState) -> StopDecision:
        cost = (self.price_in * state.input_tokens + self.price_out * state.output_tokens) / 1e6
        return StopDecision(should_stop=cost >= self.limit_used,
                            reason=f"limit_used ${cost:.4f}/${self.limit_used:.4f}")


@dataclass
class WallBudget(StopCondition):
    limit_seconds: float

    def check(self, state: AgentState) -> StopDecision:
        time_cost = time.monotonic() - state.start_at
        return StopDecision(should_stop=time_cost >= self.limit_seconds,
                            reason=f"wall_budget {time_cost:.2f}s/{self.limit_seconds:.2f}s")
