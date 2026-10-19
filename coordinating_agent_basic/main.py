"""Hub-and-spoke multi-agent pattern.

                     ┌────────────────────┐
                     │  billing spoke     │
                     └─────────▲──────────┘
                               │
    customer ──► HUB ──────────┼───────────► technical spoke
                (coordinator)  │
                     ┌─────────▼──────────┐
                     │  account spoke     │
                     └────────────────────┘

Rules that make it hub-and-spoke (and not just "several agents"):
  1. ALL communication goes through the hub. Spokes never call each other.
  2. Spokes are isolated: each sees only the task the hub hands it — not the
     customer's full message, not the other spokes' answers.
  3. The hub owns routing (which spoke) and synthesis (one final reply).
  4. A spoke failing is the hub's problem to handle, not a crash.
"""

import os
import asyncio
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
from anthropic import AsyncAnthropic, DefaultAioHttpClient

load_dotenv(Path(__file__).parent.parent / ".env")

MODEL = "claude-haiku-5-5"


# ---------------------------------------------------------------------------
# Spoke: a focused agent with a narrow system prompt. It knows nothing about
# the hub, the other spokes, or the original customer message.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Spoke:
    name: str
    description: str  # shown to the hub so it can route
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
            "You are a billing specialist. You handle invoices, charges, refunds, "
            "and payment methods. Answer only the billing-related part of the task "
            "you're given, briefly and concretely.",
        ),
        Spoke(
            "technical_specialist",
            "bugs, errors, troubleshooting",
            "You are a technical support specialist. You handle bugs, error messages, "
            "and troubleshooting steps. Answer only the technical part of the task "
            "you're given, briefly and concretely.",
        ),
        Spoke(
            "account_specialist",
            "logins, passwords, account access, profile settings",
            "You are an account specialist. You handle logins, passwords, account "
            "access, and profile settings. Answer only the account-related part of "
            "the task you're given, briefly and concretely.",
        ),
    ]
}


# ---------------------------------------------------------------------------
# Hub: the only component that talks to the customer and to the spokes.
# ---------------------------------------------------------------------------

class Hub:
    def __init__(self, client: AsyncAnthropic, spokes: dict[str, Spoke]):
        self.client = client
        self.spokes = spokes

    # -- what the hub model sees ------------------------------------------

    @property
    def system_prompt(self) -> str:
        roster = "\n".join(f"  - {s.name}: {s.description}" for s in self.spokes.values())
        return (
            "You are a customer support coordinator. You don't resolve issues\n"
            "yourself — you break the customer's request into pieces and delegate each\n"
            f"piece to the right specialist:\n{roster}\n\n"
            "When handling a request:\n"
            "1. Identify every distinct issue in the customer's message, AND whether any\n"
            "   issue depends on another being resolved first (e.g. the customer can't\n"
            "   verify a charge because they're locked out of their account).\n"
            "2. Delegate independent issues in the same turn. For a dependent issue,\n"
            "   delegate the blocker first, wait for its result, then delegate the\n"
            "   dependent one — and since each specialist only sees the task text you\n"
            "   give it (not the customer's message or other specialists' answers),\n"
            "   include whatever context from earlier results it actually needs.\n"
            "3. If part of the request doesn't clearly match any specialist above, do not\n"
            "   force it onto the closest one. Leave it undelegated and say plainly in\n"
            "   your final reply that you can't help with that part (and, if obvious,\n"
            "   who can).\n"
            "4. Before replying, check whether a specialist's answer surfaced a new or\n"
            "   related issue (e.g. a billing hold causing a login failure). If so,\n"
            "   delegate that too — up to two extra rounds of this before you stop and\n"
            "   note it as a known gap in your reply instead of resolving it yourself.\n"
            "5. Once every issue is either answered or explicitly called out as\n"
            "   unsupported, combine the results into a single, coherent reply addressed\n"
            "   directly to the customer. If a specialist failed, say so honestly rather\n"
            "   than guessing its answer.\n\n"
            "Use your tools step-by-step — delegate first, then respond."
        )

    @property
    def tools(self) -> list[dict]:
        return [{
            "name": "delegate_to_specialist",
            "description": (
                "Hand off one specific piece of the customer's request to a specialist "
                "agent and get their answer back. Call once per specialist you need, "
                "each with a focused, self-contained task description (the specialist "
                "cannot see the customer's message or other specialists' answers)."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "specialist": {
                        "type": "string",
                        "enum": list(self.spokes),
                        "description": "Which specialist to delegate to",
                    },
                    "task": {
                        "type": "string",
                        "description": "The self-contained task/question for that specialist",
                    },
                },
                "required": ["specialist", "task"],
            },
        }]

    # -- routing: hub -> spoke -> hub -------------------------------------

    async def _delegate(self, block) -> dict:
        """Run one delegation; always returns a tool_result (never raises)."""
        specialist, task = block.input["specialist"], block.input["task"]
        print(f"  → Delegates to: {specialist} — {task}")
        try:
            answer = await self.spokes[specialist].run(self.client, task)
            print(f"    [{specialist}] {answer}")
            return {"type": "tool_result", "tool_use_id": block.id, "content": answer}
        except Exception as e:  # a failing spoke must not take down the hub
            print(f"    [{specialist}] FAILED: {e}")
            return {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": f"Specialist failed: {e}",
                "is_error": True,
            }

    async def handle(self, user_message: str) -> str:
        messages = [{"role": "user", "content": user_message}]
        step = 0
        while True:
            step += 1
            response = await self.client.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=self.system_prompt,
                tools=self.tools,
                messages=messages,
            )
            print(f"[Hub step {step}] stop_reason={response.stop_reason}")

            if response.stop_reason != "tool_use":
                return next((b.text for b in response.content if b.type == "text"), "(no text)")

            # Fan out: independent delegations run concurrently (spokes are isolated,
            # so there's nothing to coordinate between them). Fan in: results come
            # back to the hub in order, and only the hub sees them all together.
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            tool_results = await asyncio.gather(*(self._delegate(b) for b in tool_uses))

            messages += [
                {"role": "assistant", "content": response.content},
                {"role": "user", "content": list(tool_results)},
            ]


async def main() -> None:
    async with AsyncAnthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        http_client=DefaultAioHttpClient(),
    ) as client:
        user_message = (
            "I was double-charged on my last invoice, and now I can't log into my "
            "account to check it because it keeps saying my password is wrong."
        )
        print(f"User: {user_message}\n")
        reply = await Hub(client, SPOKES).handle(user_message)
        print(f"\nHub: {reply}")


if __name__ == "__main__":
    asyncio.run(main())
