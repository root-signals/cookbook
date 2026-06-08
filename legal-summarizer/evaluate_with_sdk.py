"""Example 1 — evaluate with the Scorable SDK (synchronous, in-process).

Run the agent, then score its output by calling the Faithfulness_to_Citations
evaluator directly. The score comes back here, in the same process, right after
the run — nothing else to set up.

    uv run --env-file .env python evaluate_with_sdk.py --url https://scorable.ai

Requires SCORABLE_API_KEY (and OPENAI_API_KEY for the agent) in .env.
"""

from __future__ import annotations

import argparse
import asyncio

from scorable import Scorable

from agent import run_agent


async def main(url: str) -> None:
    run = await run_agent(url)
    print(run.summary)

    # Faithfulness_to_Citations pulls the cited URLs out of the response and
    # fetches them server-side, so we pass only request + response.
    client = Scorable(run_async=True)
    result = await client.evaluators.Faithfulness_to_Citations(request=url, response=run.summary)

    print("\n" + "=" * 72)
    print(f"Faithfulness to Citations: {float(result.score):.2f} / 1.00")
    print("=" * 72)
    print(result.justification or "")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Company website URL, e.g. https://scorable.ai")
    args = parser.parse_args()
    asyncio.run(main(args.url))
