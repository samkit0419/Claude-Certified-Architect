"""Formats claude_agent_sdk messages into human-readable terminal lines."""

from __future__ import annotations

import json

from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)


def format_message(message) -> str | None:
    if isinstance(message, SystemMessage):
        return _format_system(message)
    if isinstance(message, AssistantMessage):
        return _format_assistant(message)
    if isinstance(message, UserMessage):
        return _format_user(message)
    if isinstance(message, ResultMessage):
        return _format_result(message)
    return None


def _format_system(message: SystemMessage) -> str | None:
    if message.subtype != "init":
        return None
    data = message.data
    tools = data.get("tools", [])
    return (
        f"[Session] id={data.get('session_id')} model={data.get('model')} "
        f"cwd={data.get('cwd')}\n"
        f"  tools available: {', '.join(tools)}"
    )


def _format_assistant(message: AssistantMessage) -> str | None:
    lines = []
    for block in message.content:
        if isinstance(block, ThinkingBlock):
            lines.append(f"[Thinking] {block.thinking[:200]}...")
        elif isinstance(block, TextBlock):
            lines.append(f"[Claude] {block.text}")
        elif isinstance(block, ToolUseBlock):
            lines.append(f"[Tool: {block.name}] Input: {json.dumps(block.input)}")
    return "\n".join(lines) if lines else None


def _format_user(message: UserMessage) -> str | None:
    content = message.content
    if isinstance(content, str):
        return None
    lines = []
    for block in content:
        if isinstance(block, ToolResultBlock):
            label = "[Tool Error]" if block.is_error else "[Tool Result]"
            lines.append(f"{label} {block.content}")
    return "\n".join(lines) if lines else None


def _format_result(message: ResultMessage) -> str:
    cost = message.total_cost_usd if message.total_cost_usd is not None else 0.0
    result_line = message.result if message.subtype == "success" else f"({message.subtype})"
    return (
        f"[Result] {result_line}\n"
        f"  stop_reason={message.stop_reason!r} cost=${cost:.4f} "
        f"duration={message.duration_ms}ms"
    )
