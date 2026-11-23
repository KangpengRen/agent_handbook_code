"""
异常类型
"""

class ToolException(Exception):
    """工具执行异常"""
    pass

class AgentLoopException(Exception):
    """Agent循环执行异常"""
    pass