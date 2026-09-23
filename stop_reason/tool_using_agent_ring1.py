# Ring 1: Single tool, single turn.

import json
import anthropic
import random
from dotenv import load_dotenv

# Load ANTHROPIC_API_KEY from a local .env file (not committed to git).
load_dotenv()

# Create a client. It reads ANTHROPIC_API_KEY from the environment.
client = anthropic.Anthropic()

tools = [
    {
        "name": "roll_dice",
        "description": "Roll a 20-sided die and return the result as an integer between 1 and 20.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
  ]
# Send the user's request along with the tool definition. Claude decides
# whether to call the tool based on the request and the tool description.
response = client.messages.create(
    model="claude-opus-5",
    max_tokens=1024,
    tools=tools,
    tool_choice={"type": "auto", "disable_parallel_tool_use": True},
    messages=[
        {
            "role": "user",
            "content": "I need you to simulate a dice roll for a D&D game. Please roll a 20-sided die and give me the result.",
        }
    ],
)





def roll_dice() -> int:
    """Return a random integer between 1 and 20, inclusive."""
    return random.randint(1, 20)