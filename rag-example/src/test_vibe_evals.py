"""Vibe-coded RAG evals — LLM-as-judge scorers written without an eval framework.

Requires a running pgvector instance and a populated search DB.
See rag.py for setup instructions.

Run with:
    pytest src/test_vibe_evals.py -v -s
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from src.rag import Deps, agent as rag_agent, database_connect

from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart

# ---------------------------------------------------------------------------
# Shared output schema
# ---------------------------------------------------------------------------


class EvalScore(BaseModel):
    score: float = Field(ge=0.0, le=1.0, description='Score between 0 (bad) and 1 (perfect)')
    justification: str = Field(description='One or two sentences explaining the score')


# ---------------------------------------------------------------------------
# Scorer agents
# ---------------------------------------------------------------------------

faithfulness_agent = Agent(
    'openai:gpt-5.2',
    output_type=EvalScore,
    system_prompt="""You are evaluating whether an AI answer is faithful to the context it was given.

Faithfulness means: every factual claim in the answer can be traced back to the retrieved context.
An answer that invents facts not present in the context scores low, even if those facts happen to be true.

Score 1.0 if the answer only uses information from the context.
Score 0.0 if the answer contains significant claims not supported by the context.
Score in between for partial faithfulness.""",
)

relevance_agent = Agent(
    'openai:gpt-5.2',
    output_type=EvalScore,
    system_prompt="""You are evaluating whether an AI answer is relevant to the question asked.

Relevance means: the answer directly addresses what the user asked.
An answer that is technically accurate but about the wrong topic scores low.

Score 1.0 if the answer directly and completely addresses the question.
Score 0.0 if the answer is off-topic or ignores the question.
Score in between for partial relevance.""",
)

tool_call_agent = Agent(
    'openai:gpt-5.2',
    output_type=EvalScore,
    system_prompt="""You are evaluating whether a RAG agent used its retrieval tool appropriately.

You will be given:
- The user's question
- The search query the agent used when calling the retrieve tool

A good search query should:
- Be semantically related to the question
- Be specific enough to retrieve useful results
- Not be copy-pasted verbatim from the question (reformulation adds value)

Score 1.0 for an excellent, targeted query.
Score 0.0 if no retrieval was attempted, or the query is clearly wrong.
Score in between otherwise.""",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_rag_components(result) -> tuple[str | None, str | None]:
    """Return (search_query, retrieved_context) from an agent RunResult."""
    search_query: str | None = None
    retrieved_context: str | None = None

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
                    retrieved_context = part.content

    return search_query, retrieved_context


async def score_faithfulness(question: str, context: str, answer: str) -> EvalScore:
    result = await faithfulness_agent.run(f'Question: {question}\n\nRetrieved context:\n{context}\n\nAnswer:\n{answer}')
    return result.output


async def score_relevance(question: str, answer: str) -> EvalScore:
    result = await relevance_agent.run(f'Question: {question}\n\nAnswer:\n{answer}')
    return result.output


async def score_tool_call(question: str, search_query: str | None) -> EvalScore:
    if search_query is None:
        return EvalScore(score=0.0, justification='The retrieve tool was never called.')
    result = await tool_call_agent.run(f'User question: {question}\n\nSearch query used: {search_query}')
    return result.output


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
async def rag_deps() -> AsyncIterator[object]:
    client = AsyncOpenAI()
    async with database_connect(False) as pool:
        yield Deps(openai=client, pool=pool)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

QUESTIONS = [
    'How do I run a RAG evaluation with Scorable?',
    'What is the difference between an evaluator and a judge in Scorable?',
    'How do I run batch evaluations?',
]

MIN_SCORE = 0.5  # threshold — below this the eval is considered a failure


@pytest.mark.anyio
@pytest.mark.parametrize('question', QUESTIONS)
async def test_faithfulness(question: str, rag_deps) -> None:
    rag_result = await rag_agent.run(question, deps=rag_deps)
    _search_query, context = extract_rag_components(rag_result)

    score = await score_faithfulness(question, context or '', rag_result.output)
    print(f'\n[faithfulness] {score.score:.2f} — {score.justification}')

    assert score.score >= MIN_SCORE, f'Faithfulness too low ({score.score:.2f}): {score.justification}'


@pytest.mark.anyio
@pytest.mark.parametrize('question', QUESTIONS)
async def test_relevance(question: str, rag_deps) -> None:
    rag_result = await rag_agent.run(question, deps=rag_deps)
    score = await score_relevance(question, rag_result.output)
    print(f'\n[relevance] {score.score:.2f} — {score.justification}')

    assert score.score >= MIN_SCORE, f'Relevance too low ({score.score:.2f}): {score.justification}'


@pytest.mark.anyio
@pytest.mark.parametrize('question', QUESTIONS)
async def test_tool_call_quality(question: str, rag_deps) -> None:
    rag_result = await rag_agent.run(question, deps=rag_deps)
    search_query, _ = extract_rag_components(rag_result)

    score = await score_tool_call(question, search_query)
    print(f'\n[tool call] {score.score:.2f} — {score.justification}')

    assert score.score >= MIN_SCORE, f'Tool call quality too low ({score.score:.2f}): {score.justification}'
