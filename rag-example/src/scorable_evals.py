"""RAG evals using Scorable's built-in context-aware evaluators.

No custom judge prompts — Scorable provides Faithfulness and Truthfulness
as first-class evaluators designed for RAG pipelines.

Requires:
    pip install scorable
    export SCORABLE_API_KEY="sk-..."

Run with:
    pytest src/scorable_evals.py -v -s
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest
from openai import AsyncOpenAI
from scorable import Scorable
from src.rag import SYSTEM_PROMPT, Deps, agent as rag_agent, database_connect

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)

# ---------------------------------------------------------------------------
# Test dataset — questions with ground truth answers for Truthfulness
# ---------------------------------------------------------------------------

TEST_CASES = [
    {
        'question': 'How do I run a RAG evaluation with Scorable?',
        'expected': 'Use the Faithfulness and Truthfulness evaluators, passing the request, response, and retrieved context chunks.',
    },
    {
        'question': 'What is the difference between an evaluator and a judge in Scorable?',
        'expected': 'An evaluator is a single scoring check; a judge is a collection of evaluators grouped to evaluate a specific interaction.',
    },
    {
        'question': 'How do I run batch evaluations?',
        'expected': 'Use the batch evaluation cookbook: upload a dataset and run a judge against all rows in one call.',
    },
]

MIN_SCORE = 0.5

TAGS = ['unit-test']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_rag_components(result) -> tuple[str | None, list[str]]:
    """Return (search_query, context_chunks) from an agent RunResult.

    Scorable evaluators expect contexts as a list of strings, matching
    the shape of retrieved document chunks.
    """
    search_query: str | None = None
    contexts: list[str] = []

    for msg in result.all_messages():
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, ToolCallPart) and part.tool_name == 'retrieve':
                    args = part.args
                    if isinstance(args, dict):
                        search_query = args.get('search_query')
                    elif isinstance(args, str):
                        search_query = json.loads(args).get('search_query')

        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, ToolReturnPart) and part.tool_name == 'retrieve':
                    # The retrieve tool returns all chunks as a single string;
                    # split on the double-newline separator used in rag.py
                    contexts = [c.strip() for c in part.content.split('\n\n') if c.strip()]

    return search_query, contexts


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
async def rag_deps() -> AsyncIterator[object]:
    client = AsyncOpenAI()
    async with database_connect(False) as pool:
        yield Deps(openai=client, pool=pool)


@pytest.fixture(scope='module')
def scorable() -> Scorable:
    # Reads SCORABLE_API_KEY from the environment
    return Scorable()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
@pytest.mark.parametrize('case', TEST_CASES, ids=[c['question'][:40] for c in TEST_CASES])
async def test_faithfulness(case: dict, rag_deps, scorable: Scorable) -> None:
    """Answer must be grounded in the retrieved context."""
    rag_result = await rag_agent.run(case['question'], deps=rag_deps)
    _, contexts = extract_rag_components(rag_result)

    result = scorable.evaluators.Faithfulness(
        request=case['question'],
        response=rag_result.output,
        contexts=contexts,
        tags=TAGS,
        system_prompt=SYSTEM_PROMPT,
    )
    print(f'\n[faithfulness] {result.score:.2f} — {result.justification}')

    assert result.score >= MIN_SCORE, f'Faithfulness too low ({result.score:.2f}): {result.justification}'


@pytest.mark.anyio
@pytest.mark.parametrize('case', TEST_CASES, ids=[c['question'][:40] for c in TEST_CASES])
async def test_truthfulness(case: dict, rag_deps, scorable: Scorable) -> None:
    """Retrieved context must contain enough information to produce the correct answer."""
    rag_result = await rag_agent.run(case['question'], deps=rag_deps)
    _, contexts = extract_rag_components(rag_result)

    result = scorable.evaluators.Truthfulness(
        request=case['question'],
        contexts=contexts,
        expected_output=case['expected'],
        tags=TAGS,
        system_prompt=SYSTEM_PROMPT,
    )
    print(f'\n[truthfulness] {result.score:.2f} — {result.justification}')

    assert result.score >= MIN_SCORE, f'Truthfulness too low ({result.score:.2f}): {result.justification}'
