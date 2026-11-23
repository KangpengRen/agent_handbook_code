"""
推理和规划
"""
import json
from dataclasses import dataclass
from typing import Literal

from src.assistant.core.agent_assistant_exception import AgentPlanException
from src.assistant.llm.client import DeepSeekClient


@dataclass
class Step:
    id: int
    goal: str  # 该步完成目标（交给执行器的任务描述）
    done_when: str  # 完成依据：可被 Verifier 检查
    status: Literal["pending", "done", "failed"] = "pending"
    result: str = ""  # 执行摘要：只保留结论，供后续步骤引用


PLAN_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_plan",
        "description": "提交任务执行计划。每步必须有可验证的完成判据；一般 3-8 步。",
        "parameters": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "goal": {"type": "string", "description": "要执行的具体动作"},
                            "done_when": {"type": "string", "description": "判断该步完成的客观标准"}
                        },
                        "required": ["goal", "done_when"]
                    }
                }
            },
            "required": ["steps"]
        }
    }
}

class PlanThenExecute:
    """一次显示规划 + 逐步执行 + 增量重规划"""

    def __init__(self, llm: DeepSeekClient, make_executor, max_replans: int = 2):
        self.llm, self.make_executor = llm, make_executor
        self.max_replans = max_replans

    def plan(self, task: str, context: str = "") -> list[Step]:
        reply = self.llm.call(messages=[{"role": "user", "content": f"任务：{task}\n{context}\n请规划执行步骤"}],
                              tools=[PLAN_TOOL],
                              reasoning_effort="high")

        message = reply["choices"][0]["message"]

        if not (tool_calls := message.get("tool_calls", "")):
            raise AgentPlanException("Planner 未返回结构化计划")
        steps_data = json.loads(tool_calls[0]["function"]["arguments"]).get("steps", [])
        return [Step(i, s["goal"], s["done_when"]) for i, s in enumerate(steps_data, 1)]

    def run(self, task: str) -> str:
        print(f"\n[Planner] 正在拆解任务: {task}")
        steps = self.plan(task)
        for s in steps:
            print(f"  -> 步骤 {s.id}: {s.goal} (完成判据: {s.done_when})")

        replans = 0
        while pending := [s for s in steps if s.status == "pending"]:
            step = pending[0]
            done_ctx = "\n".join(f"步骤{s.id}[{s.goal}]：{s.result}" for s in steps if s.status == "done")

            print(f"\n[Executor] 开始执行步骤 {step.id}: {step.goal}")

            # 把当前的 step 传给 make_executor，方便动态挂载 verifier
            executor = self.make_executor(step)
            try:
                # 执行器只看到"已完成结论 + 当前步骤"
                step.result = executor.run_agent(
                    f"已完成进展：\n{done_ctx or '（无）'}\n\n"
                    f"当前步骤：{step.goal}\n完成判据：{step.done_when}"
                )
                step.status = "done"
                print(f"✅ 步骤 {step.id} 完成")
            except Exception as e:
                step.status = "failed"
                step.result = str(e)
                print(f"❌ 步骤 {step.id} 失败: {e}")

            # 触发增量重规划
            if any(s.status == "failed" for s in steps):
                if replans >= self.max_replans:
                    print(f"🚨 达到最大重规划次数 ({self.max_replans})，终止任务。")
                    return self._abort_summary(steps)

                replans += 1
                print(f"\n[Replanner] 触发增量重规划 (第 {replans} 次)...")
                steps = self._replan(task, steps)
                for s in steps:
                    if s.status != "done":
                        print(f"  -> 新步骤 {s.id}: {s.goal}")

        return "\n\n".join(f"## 步骤 {s.id}: {s.goal}\n{s.result}" for s in steps if s.status == "done")

    def _replan(self, task: str, steps: list[Step]) -> list[Step]:
        """增量重规划：done步骤保留为资产，只重排剩余工作"""
        done = [s for s in steps if s.status == "done"]
        report = "\n".join(f"[{s.status}] {s.goal}: {s.result[:200]}" for s in steps)
        fresh = self.plan(task, context=f"既有进展与失败记录：\n{report}\n"
                                        f"只规划剩余步骤，不要重复已完成步骤。")

        return done + [Step(len(done) + i, s.goal, s.done_when) for i, s in enumerate(fresh, 1)]

    def _abort_summary(self, steps: list[Step]) -> str:
        return ("[重规划次数耗尽，任务中止]\n" +
                "\n".join(f"[{s.status}] {s.goal}: {s.result[:200]}" for s in steps))