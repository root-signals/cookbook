"""Example 2 — evaluate via OpenTelemetry traces (asynchronous, server-side).

Instrument the agent so every run is exported to Scorable as an OTel trace.
You don't call an evaluator here: a one-time *evaluation filter* (see README)
matches the incoming traces and scores them server-side. The result shows up
in the Scorable dashboard, embedded in the trace.

    # one-time setup — wire the evaluator to incoming traces (see README):
    scorable otel-filter create \\
        --name "legal-summarizer-faithfulness" \\
        --evaluator-id <Faithfulness to Citations evaluator id>

    uv run --env-file .env python evaluate_with_otel.py --url https://scorable.ai

Requires SCORABLE_OTEL_ENDPOINT + SCORABLE_OTEL_API_KEY (and OPENAI_API_KEY) in .env.
"""

from __future__ import annotations

import argparse
import asyncio

from agent import agent, run_agent
from tracing import instrument_agent


async def main(url: str) -> None:
    provider = instrument_agent(agent)
    try:
        run = await run_agent(url)
        print(run.summary)
        print("\n" + "=" * 72)
        print("Trace sent to Scorable.")
        print("The evaluation filter scores it server-side — view the score in")
        print("Monitoring → Traces (or `scorable otel-trace list`).")
        print("=" * 72)
    finally:
        # Flush the BatchSpanProcessor before this short-lived script exits,
        # otherwise the trace may never leave the process.
        provider.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Company website URL, e.g. https://scorable.ai")
    args = parser.parse_args()
    asyncio.run(main(args.url))
