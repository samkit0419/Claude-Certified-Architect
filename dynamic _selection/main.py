"""Dynamic subagent selection vs. a rigid pipeline.

A rigid coordinator treats itself as a fixed conveyor belt: every request
runs through the same subagents, in the same order, whether they're needed
or not. A dynamic coordinator analyzes the request first, skips subagents
whose capability isn't needed, and — when it genuinely needs several calls
to the same subagent type — partitions them into non-overlapping scopes
instead of letting them duplicate each other's work.

Same three (role-played, no real internet/files) subagents throughout:

  web_search_agent        — "searches the web" for a given, scoped query
  document_analysis_agent — analyzes only the document text it's handed
  synthesis_agent         — combines 2+ findings into one narrative

Two scenarios isolate the two principles so the difference is unambiguous:

  Scenario 1 (tests skipping): summarize two already-provided documents.
  No web search is needed at all — a rigid pipeline runs one anyway.

  Scenario 2 (tests partitioning): compare EU vs. US remote-work privacy
  regulation. No documents are involved — a rigid pipeline analyzes a
  nonexistent document anyway, and has no `scope` field to keep its two
  web searches from covering the same ground.

Run with:
    python main.py
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from anthropic import AsyncAnthropic, DefaultAioHttpClient
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("dynamic_selection_demo")

MODEL = "claude-haiku-5-5"
MAX_STEPS = 6

DOC_V1 = (
    "RemoteWorkPolicy_v1.pdf (Jan 2024): Employees may work remotely up to 2 "
    "days/week. Company equipment must stay within the home country. No "
    "explicit data residency clause."
)
DOC_V2 = (
    "RemoteWorkPolicy_v2.pdf (Sep 2024): Employees may work remotely up to 4 "
    "days/week. Equipment may travel internationally with manager approval. "
    "New clause: customer data must stay within the EU or US depending on "
    "the employee's contracting entity."
)

SCENARIO_1 = (
    "Here are two uploaded documents:\n\n"
    f"--- RemoteWorkPolicy_v1.pdf ---\n{DOC_V1}\n\n"
    f"--- RemoteWorkPolicy_v2.pdf ---\n{DOC_V2}\n\n"
    "Please summarize what changed between them."
)

SCENARIO_2 = (
    "I need a comparison of how the EU and the US currently regulate "
    "remote-work data privacy."
)


# ---------------------------------------------------------------------------
# Subagents — identical for both coordinators.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Subagent:
    name: str
    system_prompt: str

    async def run(self, client: AsyncAnthropic, task: str) -> str:
        response = await client.messages.create(
            model=MODEL,
            max_tokens=200,
            system=self.system_prompt,
            messages=[{"role": "user", "content": task}],
        )
        return next((b.text for b in response.content if b.type == "text"), "(no response)")


SUBAGENTS = {
    s.name: s
    for s in [
        Subagent(
            "web_search_agent",
            "You are a web search agent (simulated — no real internet access, but broad "
            "general knowledge). Given a query and a scope, respond as if you searched the "
            "web and found sources ON THAT SCOPE ONLY. Stay strictly within the scope — do "
            "not wander into adjacent topics. 2-3 sentences.",
        ),
        Subagent(
            "document_analysis_agent",
            "You are a document analysis agent. You only know the document text you're "
            "given in the task — nothing else. If no document text was given, say so in one "
            "short sentence instead of inventing content. Otherwise analyze only what's "
            "asked, briefly.",
        ),
        Subagent(
            "synthesis_agent",
            "You are a synthesis agent. You'll be given two or more findings; combine them "
            "into one clear, well-organized narrative. Do not add facts you weren't given.",
        ),
    ]
}

RIGID_TOOLS = [
    {
        "name": "web_search_agent",
        "description": "Search the web for the given query.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "What to search for"}},
            "required": ["query"],
        },
    },
    {
        "name": "document_analysis_agent",
        "description": "Analyze the given document text for the given focus.",
        "input_schema": {
            "type": "object",
            "properties": {
                "focus": {"type": "string"},
                "document_text": {"type": "string", "description": "Leave empty if none applies"},
            },
            "required": ["focus"],
        },
    },
    {
        "name": "synthesis_agent",
        "description": "Combine the given findings into one narrative.",
        "input_schema": {
            "type": "object",
            "properties": {"findings": {"type": "string"}},
            "required": ["findings"],
        },
    },
]

DYNAMIC_TOOLS = [
    {
        "name": "web_search_agent",
        "description": "Search the web for the given query, within a specific non-overlapping scope.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for"},
                "scope": {
                    "type": "string",
                    "description": (
                        "The exact, narrow slice of the topic this call covers — must NOT "
                        "overlap with any other web_search_agent call you make for this "
                        "request (e.g. one specific region, source type, or subtopic)."
                    ),
                },
            },
            "required": ["query", "scope"],
        },
    },
    RIGID_TOOLS[1],  # document_analysis_agent — identical in both
    RIGID_TOOLS[2],  # synthesis_agent — identical in both
]

RIGID_SYSTEM_PROMPT = (
    "You are a research coordinator. You run a FIXED pipeline for every request, in this "
    "exact order, regardless of what the request actually asks for:\n"
    "  1. Call web_search_agent once for the request.\n"
    "  2. Call document_analysis_agent once (pass whatever document text was given, or leave "
    "it empty if none was).\n"
    "  3. Call synthesis_agent once, combining whatever you collected, to produce the final answer.\n"
    "Always run all three steps, even if a step doesn't seem necessary — this pipeline is "
    "fixed and does not change based on the request. If part of the request needs more than "
    "one web search, just call web_search_agent again with another query."
)

DYNAMIC_SYSTEM_PROMPT = (
    "You are a research coordinator with three subagents: web_search_agent, "
    "document_analysis_agent, synthesis_agent. Before delegating anything:\n"
    "1. Decide which subagents THIS SPECIFIC request actually needs. Skip any subagent whose "
    "capability isn't needed — e.g. if the request only needs already-provided documents "
    "analyzed, do not call web_search_agent at all; if no documents were given, do not call "
    "document_analysis_agent at all.\n"
    "2. If you need more than one web_search_agent call, partition them into non-overlapping "
    "scopes (distinct subtopics, regions, or source types) using the `scope` field — never "
    "issue two calls that would return the same information.\n"
    "3. Only call synthesis_agent if you end up with two or more findings that actually need "
    "combining into one narrative. If there's just one finding, use it directly.\n"
    "Then delegate accordingly and give one final reply."
)


async def _delegate(client: AsyncAnthropic, block, tag: str, calls: list[dict]) -> dict:
    name = block.name
    calls.append({"tag": tag, "name": name, "input": dict(block.input)})
    log.info(f"{tag} call -> {name}({block.input})")
    try:
        if name == "web_search_agent":
            task = f"Query: {block.input['query']}"
            if "scope" in block.input:
                task += f"\nScope (stay within this only): {block.input['scope']}"
        elif name == "document_analysis_agent":
            task = f"Focus: {block.input['focus']}\nDocument text:\n{block.input.get('document_text') or '(none provided)'}"
        else:
            task = f"Findings to combine:\n{block.input['findings']}"
        answer = await SUBAGENTS[name].run(client, task)
        log.info(f"{tag}   [{name}] {answer}")
        return {"type": "tool_result", "tool_use_id": block.id, "content": answer}
    except Exception as e:  # a failing subagent must not take down the coordinator
        log.warning(f"{tag}   [{name}] FAILED: {e}")
        return {"type": "tool_result", "tool_use_id": block.id, "content": f"Subagent failed: {e}", "is_error": True}


async def run_pipeline(client: AsyncAnthropic, tag: str, system_prompt: str, tools: list[dict], user_message: str, calls: list[dict]) -> str:
    log.info(f"\n{'=' * 70}\n{tag} request: {user_message[:120]}{'...' if len(user_message) > 120 else ''}\n{'=' * 70}")
    messages = [{"role": "user", "content": user_message}]
    for _ in range(MAX_STEPS):
        response = await client.messages.create(
            model=MODEL, max_tokens=1024, system=system_prompt, tools=tools, messages=messages,
        )
        log.info(f"{tag} stop_reason={response.stop_reason}")
        if response.stop_reason != "tool_use":
            return next((b.text for b in response.content if b.type == "text"), "(no text)")
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        results = await asyncio.gather(*(_delegate(client, b, tag, calls) for b in tool_uses))
        messages += [
            {"role": "assistant", "content": response.content},
            {"role": "user", "content": list(results)},
        ]
    return "(gave up after MAX_STEPS)"


def _summarize(label: str, calls: list[dict]) -> None:
    from collections import Counter
    counts = Counter(c["name"] for c in calls)
    log.info(f"\n{label} subagent calls: {dict(counts)} (total={len(calls)})")
    for c in calls:
        if c["name"] == "web_search_agent":
            log.info(f"{label}   web_search_agent query={c['input'].get('query')!r} scope={c['input'].get('scope', '(none — no partitioning available)')!r}")


async def main() -> None:
    async with AsyncAnthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        http_client=DefaultAioHttpClient(),
    ) as client:
        rigid_1_calls, dynamic_1_calls, rigid_2_calls, dynamic_2_calls = [], [], [], []

        rigid_1 = await run_pipeline(client, "[RIGID|doc-only]", RIGID_SYSTEM_PROMPT, RIGID_TOOLS, SCENARIO_1, rigid_1_calls)
        dynamic_1 = await run_pipeline(client, "[DYNAMIC|doc-only]", DYNAMIC_SYSTEM_PROMPT, DYNAMIC_TOOLS, SCENARIO_1, dynamic_1_calls)
        rigid_2 = await run_pipeline(client, "[RIGID|eu-us]", RIGID_SYSTEM_PROMPT, RIGID_TOOLS, SCENARIO_2, rigid_2_calls)
        dynamic_2 = await run_pipeline(client, "[DYNAMIC|eu-us]", DYNAMIC_SYSTEM_PROMPT, DYNAMIC_TOOLS, SCENARIO_2, dynamic_2_calls)

        print(f"\n{'#' * 70}\n# COMPARISON\n{'#' * 70}")

        print("\n--- Scenario 1: summarize two already-provided documents (no web search needed) ---")
        _summarize("[RIGID|doc-only]", rigid_1_calls)
        _summarize("[DYNAMIC|doc-only]", dynamic_1_calls)
        print(f"\n[RIGID|doc-only] final reply:\n{rigid_1}\n")
        print(f"[DYNAMIC|doc-only] final reply:\n{dynamic_1}\n")

        print("\n--- Scenario 2: compare EU vs. US regulation (no documents involved) ---")
        _summarize("[RIGID|eu-us]", rigid_2_calls)
        _summarize("[DYNAMIC|eu-us]", dynamic_2_calls)
        print(f"\n[RIGID|eu-us] final reply:\n{rigid_2}\n")
        print(f"[DYNAMIC|eu-us] final reply:\n{dynamic_2}\n")


if __name__ == "__main__":
    asyncio.run(main())
