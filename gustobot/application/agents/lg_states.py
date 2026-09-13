from pydantic import BaseModel, Field
from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, List, Literal, Optional
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

# 给用户查询进行分类
# Router 类 是 pydantic 类！！！
class Router(BaseModel):
    """Classify user query (同时兼容旧字段/属性访问)."""

    # 类的声明，同时也是pydantic的校验规则
    logic: str = ""  # 路由理由
    type: Literal[
        "general-query",
        "additional-query",
        "kb-query",
        "graphrag-query",
        "image-query",
        "file-query",
        "text2sql-query",
    ] = "kb-query"
    question: str = ""
    decision: Optional[str] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None

    class Config:
        extra = "allow"

    def get(self, key: str, default: Any = None) -> Any:
        """字典风格访问兼容旧逻辑。"""
        return getattr(self, key, self.__dict__.get(key, default))


@dataclass(kw_only=True)
class RouteResult:
    """Route selection result for downstream nodes."""
    route: str
    confidence: float = 0.0
    next_node: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, '1' or '0'"
    )

@dataclass(kw_only=True)
class InputState:
    """Represents the input state for the agent.

    This class defines the structure of the input state, which includes
    the messages exchanged between the user and the agent. 
    """

    messages: Annotated[list[AnyMessage], add_messages]

    memory: Dict[str, Any] = field(default_factory=dict)
    """结构化会话记忆：窗口之外的历史被抽取成固定字段（覆盖式，不随轮数累加）。

    它不出现在 messages 里，避免被 add_messages 无限累加、也避免 SystemMessage
    夹在对话中间；由 build_llm_messages() 统一拼到 system 提示之后。"""

    """Messages track the primary execution state of the agent.

    Typically accumulates a pattern of Human/AI/Human/AI messages; if
    you were to combine this template with a tool-calling ReAct agent pattern,
    it may look like this:

    1. HumanMessage - user input
    2. AIMessage with .tool_calls - agent picking tool(s) to use to collect
         information
    3. ToolMessage(s) - the responses (or errors) from the executed tools
    
        (... repeat steps 2 and 3 as needed ...)
    4. AIMessage without .tool_calls - agent responding in unstructured
        format to the user.

    5. HumanMessage - user responds with the next conversational turn.

        (... repeat steps 2-5 as needed ... )
    

    Merges two lists of messages, updating existing messages by ID.

    By default, this ensures the state is "append-only", unless the
    new message has the same ID as an existing message.
    

    Returns:
        A new list of messages with the messages from `right` merged into `left`.
        If a message in `right` has the same ID as a message in `left`, the
        message from `right` will replace the message from `left`."""
    

@dataclass(kw_only=True)
class AgentState(InputState):
    """State of the retrieval graph / agent."""
    router: Router = field(default_factory=lambda: Router(type="general-query", logic=""))
    """The router's classification of the user's query."""
    steps: list[str] = field(default_factory=list)
    """Populated by the retriever. This is a list of documents that the agent can reference."""
    documents: list[str] = field(default_factory=list)
    """Documents retrieved from the knowledge base."""
    question: str = field(default_factory=str)
    answer: str = field(default_factory=str)
    hallucination: GradeHallucinations = field(default_factory=lambda: GradeHallucinations(binary_score="0"))
    sources: list = field(default_factory=list)
    """Sources from knowledge base queries."""
