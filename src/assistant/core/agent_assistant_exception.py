"""
异常类型
"""

class ToolException(Exception):
    """工具执行异常"""
    pass

class AgentLoopException(Exception):
    """Agent循环执行异常"""
    pass

class LLMRetryException(Exception):
    """LLM请求失败【重试】异常"""
    pass

class LLMRequestException(Exception):
    """LLM请求失败异常"""
    pass