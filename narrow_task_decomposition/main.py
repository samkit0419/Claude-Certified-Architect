"""Narrow vs. coverage-checked task decomposition.

Two coordinators get the SAME customer message and delegate to the SAME three
isolated specialists (billing / technical / account). The only difference is
architectural:

  NarrowHub   — identifies issues, delegates each one once, pastes the
                specialist answers together. This is the pattern from
                coordinating_agent_basic/main.py: decomposition happens once,
                up front, and is never re-checked.

  CoverageHub — after the first round of delegation, the loop MECHANICALLY
                forces one more call with tool_choice locked to
                `evaluate_coverage`, so the model cannot skip straight to a
                final reply. If that check finds a gap — in particular, two
                specialists CONTRADICTING each other, which neither spoke can
                see on its own since each is isolated — the gap is fed back
                in and one more delegation round runs before the final reply.

The customer message below is deliberately built so the two specialists give
answers that are individually correct but mutually contradictory. Narrow
decomposition has no step that would ever notice that; coverage-checking does.

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
log = logging.getLogger("decomposition_demo")

MODEL = "claude-haiku-4-5-20251001"
MAX_STEPS = 6

CUSTOMER_MESSAGE = (
    "I upgraded to the Premium plan last week and was charged $89, but my "
    "dashboard still shows a 'downgrade pending' banner and some premium "
    "features are greyed out. Am I actually on Premium, and was the $89 "
    "charge correct?"
)


# ---------------------------------------------------------------------------
# Spokes — isolated specialists, unchanged across both coordinators.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Spoke:
    name: str
    description: str
    system_prompt: str

    async def run(self, client: AsyncAnthropic, task: str) -> str:
        response = await client.messages.create(
            model=MODEL,
            max_tokens=300,
            system=self.system_prompt,
            messages=[{"role": "user", "content": task}],
        )
        return next((b.text for b in response.content if b.type == "text"), "(no response)")


SPOKES = {
    s.name: s
    for s in [
        Spoke(
            "billing_specialist",
            "invoices, charges, refunds, payment methods",
            "You are a billing specialist. You can see charge records but NOT the "
            "customer's dashboard or feature flags. Answer only the billing part of "
            "the task you're given, briefly and concretely.",
        ),
        Spoke(
            "technical_specialist",
            "bugs, UI glitches, feature-access issues",
            "You are a technical support specialist. You can see dashboard state and "
            "feature flags but NOT billing/charge records. Answer only the technical "
            "part of the task you're given, briefly and concretely.",
        ),
        Spoke(
            "account_specialist",
            "logins, account access, actual plan/subscription status of record",
            "You are an account specialist. You can see the account's actual "
            "plan/subscription status of record. Answer only the account part of the "
            "task you're given, briefly and concretely.",
        ),
    ]
}

DELEGATE_TOOL = {
    "name": "delegate_to_specialist",
    "description": (
        "Hand off one specific piece of the customer's request to a specialist "
        "agent and get their answer back. The specialist cannot see the customer's "
        "message or other specialists' answers — only the task text you give it."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "specialist": {"type": "string", "enum": list(SPOKES)},
            "task": {"type": "string", "description": "Self-contained task for that specialist"},
        },
        "required": ["specialist", "task"],
    },
}

EVALUATE_COVERAGE_TOOL = {
    "name": "evaluate_coverage",
    "description": (
        "Required self-check before you may give your final reply. State the single "
        "underlying question the customer actually needs resolved, and whether the "
        "specialist answers collected so far — taken together — actually resolve it, "
        "including whether they agree with each other."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "customer_core_question": {
                "type": "string",
                "description": "The core outcome the customer needs resolved, in your own words",
            },
            "sufficient": {
                "type": "boolean",
                "description": "True only if the specialist answers resolve that core question without contradiction",
            },
            "gap": {
                "type": "string",
                "description": "If not sufficient: the specific unresolved question or contradiction between specialists",
            },
        },
        "required": ["customer_core_question", "sufficient"],
    },
}


async def _complete(client: AsyncAnthropic, system_prompt: str, tools: list[dict], messages: list[dict], tool_choice: dict):
    return await client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=system_prompt,
        tools=tools,
        tool_choice=tool_choice,
        messages=messages,
    )


async def _delegate(client: AsyncAnthropic, block, tag: str) -> dict:
    specialist, task = block.input["specialist"], block.input["task"]
    log.info(f"{tag} delegates -> {specialist}: {task}")
    try:
        answer = await SPOKES[specialist].run(client, task)
        log.info(f"{tag}   [{specialist}] {answer}")
        return {"type": "tool_result", "tool_use_id": block.id, "content": answer}
    except Exception as e:  # a failing spoke must not take down the hub
        log.warning(f"{tag}   [{specialist}] FAILED: {e}")
        return {"type": "tool_result", "tool_use_id": block.id, "content": f"Specialist failed: {e}", "is_error": True}


# ---------------------------------------------------------------------------
# NarrowHub — decompose once, delegate once, paste answers together.
# ---------------------------------------------------------------------------

NARROW_SYSTEM_PROMPT = (
    "You are a customer support coordinator. You don't resolve issues yourself — "
    "you break the customer's request into pieces and delegate each piece to the "
    "right specialist:\n"
    + "\n".join(f"  - {s.name}: {s.description}" for s in SPOKES.values())
    + "\n\nWhen handling a request:\n"
    "1. Identify every distinct issue in the customer's message.\n"
    "2. Delegate each issue to the matching specialist (one call per issue).\n"
    "3. Once all specialists have responded, combine their answers into a single, "
    "coherent reply addressed directly to the customer.\n\n"
    "Use your tools step-by-step — delegate first, then respond."
)


async def run_narrow(client: AsyncAnthropic) -> str:
    tag = "[NARROW]"
    log.info(f"\n{'=' * 70}\n{tag} customer: {CUSTOMER_MESSAGE}\n{'=' * 70}")
    messages = [{"role": "user", "content": CUSTOMER_MESSAGE}]
    for _ in range(MAX_STEPS):
        response = await _complete(client, NARROW_SYSTEM_PROMPT, [DELEGATE_TOOL], messages, {"type": "auto"})
        log.info(f"{tag} stop_reason={response.stop_reason}")
        if response.stop_reason != "tool_use":
            return next((b.text for b in response.content if b.type == "text"), "(no text)")
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        results = await asyncio.gather(*(_delegate(client, b, tag) for b in tool_uses))
        messages += [
            {"role": "assistant", "content": response.content},
            {"role": "user", "content": list(results)},
        ]
    return "(gave up after MAX_STEPS)"


# ---------------------------------------------------------------------------
# CoverageHub — same spokes, but a coverage check is mechanically forced
# after the first delegation round, before any final reply is allowed.
# ---------------------------------------------------------------------------

COVERAGE_SYSTEM_PROMPT = (
    "You are a customer support coordinator. You don't resolve issues yourself — "
    "you break the customer's request into pieces and delegate each piece to the "
    "right specialist:\n"
    + "\n".join(f"  - {s.name}: {s.description}" for s in SPOKES.values())
    + "\n\nWhen handling a request:\n"
    "1. Identify every distinct issue in the customer's message and delegate each "
    "one to the matching specialist.\n"
    "2. You will be required to call evaluate_coverage before replying — use it "
    "honestly, especially checking whether specialists' answers CONTRADICT each "
    "other, not just whether each one individually answered its own piece.\n"
    "3. If evaluate_coverage finds a gap, delegate again — to whichever specialist "
    "can resolve it, giving them the conflicting context — before your final reply.\n"
    "4. Once there's no remaining gap, combine everything into a single, coherent, "
    "accurate reply to the customer."
)


async def run_coverage_checked(client: AsyncAnthropic) -> str:
    tag = "[COVERAGE-CHECK]"
    log.info(f"\n{'=' * 70}\n{tag} customer: {CUSTOMER_MESSAGE}\n{'=' * 70}")
    messages = [{"role": "user", "content": CUSTOMER_MESSAGE}]
    tools = [DELEGATE_TOOL, EVALUATE_COVERAGE_TOOL]
    forced_check_done = False

    for _ in range(MAX_STEPS):
        # Right after the first delegation round resolves, force the coverage
        # check — the model cannot opt out of this step the way plain prompt
        # instructions let it.
        if not forced_check_done and len(messages) > 1 and messages[-1]["role"] == "user":
            forced_check_done = True
            forced = await _complete(client, COVERAGE_SYSTEM_PROMPT, tools, messages, {"type": "tool", "name": "evaluate_coverage"})
            block = next(b for b in forced.content if b.type == "tool_use")
            core_q = block.input["customer_core_question"]
            sufficient = block.input["sufficient"]
            gap = block.input.get("gap", "")

            if sufficient:
                log.info(f"{tag} coverage check: core question = {core_q!r}; sufficient=True")
            else:
                log.warning(f"{tag} coverage check: core question = {core_q!r}; sufficient=False")
                log.warning(f"{tag} GAP CAUGHT (a narrow decomposition would have missed this): {gap}")

            messages += [
                {"role": "assistant", "content": forced.content},
                {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": "Coverage check recorded." if sufficient else f"Gap recorded: {gap}. Resolve it before replying.",
                    }],
                },
            ]
            continue  # let the next round act on the check (delegate or finish)

        response = await _complete(client, COVERAGE_SYSTEM_PROMPT, tools, messages, {"type": "auto"})
        log.info(f"{tag} stop_reason={response.stop_reason}")
        if response.stop_reason != "tool_use":
            return next((b.text for b in response.content if b.type == "text"), "(no text)")

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        messages.append({"role": "assistant", "content": response.content})
        results = []
        for b in tool_uses:
            if b.name == "delegate_to_specialist":
                results.append(await _delegate(client, b, tag))
            else:  # model volunteered another evaluate_coverage — honor it
                core_q, sufficient, gap = b.input["customer_core_question"], b.input["sufficient"], b.input.get("gap", "")
                log.info(f"{tag} coverage re-check: sufficient={sufficient} core_question={core_q!r}" + (f" gap={gap}" if gap else ""))
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": "Recorded."})
        messages.append({"role": "user", "content": results})
    return "(gave up after MAX_STEPS)"


async def main() -> None:
    async with AsyncAnthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        http_client=DefaultAioHttpClient(),
    ) as client:
        narrow_reply = await run_narrow(client)
        good_reply = await run_coverage_checked(client)

        print(f"\n{'#' * 70}\n# COMPARISON\n{'#' * 70}")
        print(f"\n[NARROW] final reply to customer:\n{narrow_reply}\n")
        print(f"\n[COVERAGE-CHECK] final reply to customer:\n{good_reply}\n")


if __name__ == "__main__":
    asyncio.run(main())
