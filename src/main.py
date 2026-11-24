from pathlib import Path
import json

from src.assistant.core.planning import PlanThenExecute
from src.assistant.core.budget import MaxTurn, TokenBudget, CostBudget, WallBudget
from src.assistant.core.agent_loop import AgentLoop
from src.assistant.core.types import AgentEvent, AgentLoopOptions
from src.assistant.tool.builtin import ListFilesTool, ReadFileTool, WriteFileTool
from src.assistant.llm.client import DeepSeekClient

EXECUTOR_PROMPT = """你是底层执行器，负责完成大计划中的某个特定小步骤。
工作方式：根据步骤目标和完成判据，调用工具完成任务。
"""

def custom_logger(e: AgentEvent):
    prefix = "      "
    if e.type == "llm_start":
        print(f"{prefix}--- 执行器 第 {e.turn} 轮思考开始 ---")

    elif e.type == "llm_text":
        # payload 是一个字典，从中取出 text
        print(f"{prefix} 执行器🤖 模型说: {e.payload.get('text')}")

    elif e.type == "tool_call":
        tool_name = e.payload.get("name")
        tool_input = e.payload.get("args")
        print(f"{prefix} 🛠️ 决定调用工具: {tool_name}")
        print(f"{prefix}    参数: {json.dumps(tool_input, ensure_ascii=False)}")

    elif e.type == "tool_result":
        output = str(e.payload.get("tool_result"))
        is_error = e.payload.get("is_error")
        status = "❌ 失败" if is_error else "✅ 成功"
        # 截断输出，防止刷屏
        print(f"{prefix} 👀 工具执行结果 ({status}): {output[:150]}...")

    elif e.type == "done":
        print(f"{prefix}  ✅ 第 {e.turn} 轮结束，任务完成。")

    # 新增：处理外部校验失败的事件
    elif e.type == "verification_failed":
        print(f"{prefix} ⚠️ 外部校验未通过: {e.payload.get('error')}，正在要求模型修正...")

    elif e.type == "aborted":
        reason = e.payload.get("reason", "未知原因")
        print(f"{prefix} 🚨 任务被强制中止！(原因: {reason})，当前轮数: {e.turn}")

def check_summary_file():
    """硬性校验器：专门用来约束 summary.md 的字数"""
    p = Path("summary.md")
    if not p.exists():
        return "你声称完成了，但 summary.md 不存在！请立即重写。"
    content = p.read_text(encoding="utf-8")
    if len(content) > 500:
        return f"文件字数严重超标（当前 {len(content)} 字，限制 500 字）。请立即删减并覆盖原文件！"
    return None


def make_executor(step) -> AgentLoop:
    """ 接收 step 参数，动态决定是否挂载字数校验器"""
    opts = AgentLoopOptions(
        system_prompt=EXECUTOR_PROMPT,
        tools=[ListFilesTool(), ReadFileTool(), WriteFileTool()],
        on_event=custom_logger
    )

    # 动态判断：只有当前任务与 summary.md 相关的写入，才挂载 verifier
    verifier = None
    if "summary.md" in step.goal and ("写" in step.goal or "覆盖" in step.goal or "撰写" in step.goal):
        verifier = check_summary_file

    return AgentLoop(
        opts,
        stop_conditions=[
            MaxTurn(12),
            TokenBudget(50_000),
            WallBudget(120),
            CostBudget(5.0)
        ],
        verifier=verifier
    )


def main():
    client = DeepSeekClient()

    p2e_agent = PlanThenExecute(
        llm=client,
        make_executor=make_executor,
        max_replans=2
    )

    task = "请根据这个项目的结构、内容，最后写一份 500 字以内的 summary.md"

    print(f"\n ==== 启动 Plan-then-Execute ====")
    final_report = p2e_agent.run(task)

    print("\n\n==== 最终交付报告 ====")
    print(final_report)


if __name__ == "__main__":
    main()