# hello_world

Minimal Claude Agent SDK example: an agentic loop that reviews `utils.py` for crash-causing
bugs and fixes them, auto-approving file edits.

## Run

```bash
cd hello_world
python main.py
```

Run from inside this folder — the prompt references `utils.py` by a bare relative path, which
Claude's `Read`/`Edit` tools resolve against the process's working directory.

## What it does

- `main.py` streams `claude_agent_sdk` messages from `query()`, printing Claude's reasoning
  text and each tool call by name, then a final `Done: <subtype>` line from the `ResultMessage`.
- `utils.py` is a small stub with two intentional bugs (`average` divides by zero on an empty
  list, `get_last_item` indexes one past the end) for the agent to find and fix.

This example prints messages with plain `hasattr` checks rather than the shared
[`lib/sdk_parser`](../lib/sdk_parser/README.md) formatter — see `stop_reason/main.py` for an
example using that shared formatter instead.
