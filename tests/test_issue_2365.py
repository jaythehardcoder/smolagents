from unittest.mock import MagicMock

from smolagents.agents import ToolCallingAgent
from smolagents.memory import ActionStep
from smolagents.models import ChatMessage, ChatMessageToolCall, ChatMessageToolCallFunction, MessageRole


def test_issue_2365(test_tool):
    agent = ToolCallingAgent(tools=[test_tool], model=MagicMock())
    chat_message = ChatMessage(
        role=MessageRole.ASSISTANT,
        content="",
        tool_calls=[
            ChatMessageToolCall(
                id="call_1",
                type="function",
                function=ChatMessageToolCallFunction(name="test_tool", arguments={"input": "first result"}),
            ),
            ChatMessageToolCall(
                id="call_2",
                type="function",
                function=ChatMessageToolCallFunction(name="test_tool", arguments={"input": "second result"}),
            ),
        ],
    )
    memory_step = ActionStep(step_number=1, timing=MagicMock())

    list(agent.process_tool_calls(chat_message, memory_step))

    assert memory_step.observations == ["Processed: first result", "Processed: second result"]
