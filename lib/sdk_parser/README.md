# sdk_parser

Turns `claude_agent_sdk` message objects into human-readable terminal lines instead of raw
`repr()` output like `AssistantMessage(content=[ThinkingBlock(...)])`.

## Usage

```python
from lib.sdk_parser import format_message

async for message in query(prompt="...", options=options):
    line = format_message(message)
    if line:
        print(line)
```

`format_message` returns `None` for messages with nothing worth printing (e.g. non-`init`
`SystemMessage`s), so callers should skip falsy results.

Scripts importing `lib.sdk_parser` from a subfolder (e.g. `stop_reason/main.py`) need the repo
root on `sys.path`; see the `sys.path.insert(...)` shim near the top of that file for the
pattern.
