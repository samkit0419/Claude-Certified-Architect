"""Demonstrates stop_reason in the Claude Agent SDK using a weather tool.

Run with:
    python main.py "What's the weather in Paris?"

Walks through:
  * the "tool_use" stop_reason on the AssistantMessage where Claude calls
    the get_weather tool,
  * the SDK appending the tool's result back into the conversation as a
    UserMessage,
  * the "end_turn" stop_reason on the final ResultMessage once Claude has
    enough information to answer.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from claude_agent_sdk import ClaudeAgentOptions, create_sdk_mcp_server, query, tool

from lib.sdk_parser import format_message

_FAKE_WEATHER = {
    "paris": "15°C, light rain",
    "san francisco": "18°C, foggy",
    "tokyo": "24°C, clear skies",
}


@tool("get_weather", "Get the current weather for a city", {"city": str})
async def get_weather(args: dict[str, Any]) -> dict[str, Any]:
    city = args["city"].strip().lower()
    report = _FAKE_WEATHER.get(city, "22°C, sunny (default demo reading)")
    return {"content": [{"type": "text", "text": f"Weather in {args['city']}: {report}"}]}


async def main() -> None:
    prompt = " ".join(sys.argv[1:]) or "What's the weather in Paris?"

    weather_server = create_sdk_mcp_server(
        name="weather",
        version="1.0.0",
        tools=[get_weather],
    )

    options = ClaudeAgentOptions(
        mcp_servers={"weather": weather_server},
        allowed_tools=["mcp__weather__get_weather"],
        max_turns=5,
    )

    print(f"Prompt: {prompt}\n")

    try:
        async for message in query(prompt=prompt, options=options):
            line = format_message(message)
            if line:
                print(line)
    except Exception as error:
        print(f"Session ended with an error: {error}")


if __name__ == "__main__":
    asyncio.run(main())
