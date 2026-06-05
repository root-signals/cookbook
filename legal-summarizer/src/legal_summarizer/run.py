"""CLI: run the legal-summarizer agent and score it with Scorable.

uv run legal-summarizer --url https://scorable.ai --md-out report.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic_ai.messages import ModelMessage, ToolCallPart
from scorable import Scorable

from legal_summarizer.agent import LegalSummarizerDeps, agent
from legal_summarizer.render import render_markdown

logger = logging.getLogger(__name__)


@dataclass
class AgentRun:
    summary: str
    fetched_urls: list[str]


def _extract_fetched_urls(messages: list[ModelMessage]) -> list[str]:
    """Pull every URL the agent passed to a web_fetch tool call.

    Works whether WebFetch is satisfied by the model's native tool or by the
    local fallback — both surface as `ToolCallPart`s with the URL in args.
    """
    urls: list[str] = []
    seen: set[str] = set()
    for message in messages:
        for part in getattr(message, "parts", []):
            if not isinstance(part, ToolCallPart):
                continue
            args = part.args
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    continue
            if not isinstance(args, dict):
                continue
            url = args.get("url") or args.get("uri")
            if isinstance(url, str) and url not in seen:
                seen.add(url)
                urls.append(url)
    return urls


async def run_agent(company_url: str, *, chat_id: str) -> AgentRun:
    deps = LegalSummarizerDeps(company_url=company_url, chat_id=chat_id)
    result = await agent.run(company_url, deps=deps)
    return AgentRun(
        summary=result.output,
        fetched_urls=_extract_fetched_urls(result.all_messages()),
    )


@dataclass
class FaithfulnessResult:
    score: float
    justification: str


async def evaluate_faithfulness_to_citations(*, request: str, response: str) -> FaithfulnessResult:
    """Call Scorable's Faithfulness_to_Citations evaluator.

    The evaluator extracts cited URLs from the response itself and fetches
    them server-side — we don't pass contexts.
    """

    client = Scorable(run_async=True)
    result = await client.evaluators.Faithfulness_to_Citations(
        request=request,
        response=response,
    )
    return FaithfulnessResult(
        score=float(result.score),
        justification=result.justification or "",
    )


_URL_RE = re.compile(r"https?://[^\s)\]]+")


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    chat_id = args.chat_id or str(uuid.uuid4())
    run = await run_agent(args.url, chat_id=chat_id)

    output: dict[str, Any] = {
        "company_url": args.url,
        "summary": run.summary,
        "fetched_urls": run.fetched_urls,
    }

    if args.no_evaluate:
        return output

    if not os.environ.get("SCORABLE_API_KEY"):
        logger.info("SCORABLE_API_KEY not set — skipping Faithfulness evaluation.")
        return output

    if not _URL_RE.search(run.summary):
        logger.warning(
            "Agent output contains no URLs — Faithfulness_to_Citations needs cited URLs "
            "in the response. Skipping evaluation."
        )
        output["faithfulness_error"] = "no URLs cited in agent output"
        return output

    try:
        eval_result = await evaluate_faithfulness_to_citations(
            request=args.url,
            response=run.summary,
        )
    except Exception as e:
        logger.warning("Faithfulness_to_Citations failed: %s", e)
        output["faithfulness_error"] = str(e)
    else:
        output["faithfulness"] = {
            "score": eval_result.score,
            "justification": eval_result.justification,
        }

    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--url", required=True, help="Company website URL, e.g. https://scorable.ai")
    parser.add_argument("--chat-id", default=None, help="Conversation ID (default: random UUID)")
    parser.add_argument("--md-out", type=Path, help="Write markdown report to this path")
    parser.add_argument("--json", action="store_true", help="Print full result as JSON")
    parser.add_argument(
        "--no-evaluate", action="store_true", help="Skip Faithfulness_to_Citations evaluation"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    output = asyncio.run(_run(args))

    if args.md_out:
        args.md_out.write_text(render_markdown(output), encoding="utf-8")
        logger.info("Wrote markdown report to %s", args.md_out)

    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return

    print("=" * 72)
    print(f"Company: {output['company_url']}")
    print("=" * 72)
    print(output["summary"])

    score = output.get("faithfulness")
    if score:
        print()
        print("=" * 72)
        print(f"Faithfulness to Citations: {score['score']:.3f}")
        print("=" * 72)
        print(f"Justification: {score['justification']}")
    elif output.get("faithfulness_error"):
        print(f"\n[faithfulness evaluation failed: {output['faithfulness_error']}]", file=sys.stderr)


if __name__ == "__main__":
    main()
