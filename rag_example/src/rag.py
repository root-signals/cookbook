"""RAG agent — vector search over the Scorable documentation.

Adjusted from https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/rag.py
"""

from __future__ import annotations as _annotations

import asyncio
import logging
import re
import sys
import unicodedata
from contextlib import asynccontextmanager
from dataclasses import dataclass

import asyncpg
import httpx
import pydantic_core
from openai import AsyncOpenAI
from typing_extensions import AsyncGenerator

from pydantic_ai import Agent, RunContext

logger = logging.getLogger(__name__)

LLMS_TXT = 'https://docs.scorable.ai/llms.txt'
BASE_URL = 'https://docs.scorable.ai'


@dataclass
class Deps:
    openai: AsyncOpenAI
    pool: asyncpg.Pool


SYSTEM_PROMPT = """
You are a helpful assistant that can answer questions about the Scorable documentation.

You are talking to users who understand the basics of LLM based agents and workflows but are not necessarily data scientists.

They are likely not familiar with the concept of evals and how to use them to improve the quality of the agent.
"""


agent = Agent('openai:gpt-5.2', deps_type=Deps, system_prompt=SYSTEM_PROMPT)


@agent.tool
async def retrieve(context: RunContext[Deps], search_query: str) -> str:
    """Retrieve documentation sections based on a search query.

    Args:
        context: The call context.
        search_query: The search query.
    """
    embedding = await context.deps.openai.embeddings.create(
        input=search_query,
        model='text-embedding-3-small',
    )
    assert len(embedding.data) == 1, f'Expected 1 embedding, got {len(embedding.data)}, doc query: {search_query!r}'
    vector = embedding.data[0].embedding
    embedding_json = pydantic_core.to_json(vector).decode()
    rows = await context.deps.pool.fetch(
        'SELECT url, title, content FROM doc_sections ORDER BY embedding <-> $1 LIMIT 8',
        embedding_json,
    )
    return '\n\n'.join(f'# {row["title"]}\nDocumentation URL: {row["url"]}\n\n{row["content"]}\n' for row in rows)


async def run_agent(question: str) -> None:
    """Run the agent and print the answer."""
    openai = AsyncOpenAI()
    logger.info('Asking %r', question)
    async with database_connect(False) as pool:
        deps = Deps(openai=openai, pool=pool)
        answer = await agent.run(question, deps=deps)
    print(answer.output)


#######################################################
# The rest of this file is dedicated to preparing the #
# search database, and some utilities.                #
#######################################################


@dataclass
class DocSection:
    url: str
    title: str
    content: str

    def embedding_content(self) -> str:
        return '\n\n'.join((f'url: {self.url}', f'title: {self.title}', self.content))


def parse_llms_txt(text: str) -> list[str]:
    """Extract page URLs from an llms.txt file.

    Parses markdown links of the form [Title](/path/to/page.md) and returns
    full URLs for each linked page.
    """
    return [f'{BASE_URL}{match.group(1)}' for match in re.finditer(r'\[.*?\]\((/[^)]+\.md)\)', text)]


def parse_markdown_sections(page_url: str, markdown: str) -> list[DocSection]:
    """Split a markdown page into sections, one per heading.

    Each section gets a URL with an anchor derived from the heading slug.
    Sections with no meaningful text content are skipped.
    """
    sections: list[DocSection] = []
    current_title: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        if current_title is None:
            return
        content = '\n'.join(current_lines).strip()
        if content:
            sections.append(
                DocSection(
                    url=f'{page_url}#{slugify(current_title, "-")}',
                    title=current_title,
                    content=content,
                )
            )

    for line in markdown.splitlines():
        if line.startswith('#'):
            flush()
            current_title = line.lstrip('#').strip()
            current_lines = []
        else:
            current_lines.append(line)

    flush()
    return sections


async def fetch_all_sections(client: httpx.AsyncClient) -> list[DocSection]:
    """Crawl llms.txt and return all sections from all linked pages."""
    response = await client.get(LLMS_TXT)
    response.raise_for_status()
    page_urls = parse_llms_txt(response.text)
    logger.info('Found %d pages in llms.txt', len(page_urls))

    sem = asyncio.Semaphore(10)

    async def fetch_page(url: str) -> list[DocSection]:
        async with sem:
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                return parse_markdown_sections(url, resp.text)
            except httpx.HTTPError as e:
                logger.warning('Failed to fetch %s: %s', url, e)
                return []

    results = await asyncio.gather(*[fetch_page(url) for url in page_urls])
    sections = [s for page in results for s in page]
    logger.info('Parsed %d sections total', len(sections))
    return sections


async def build_search_db() -> None:
    """Fetch Scorable docs via llms.txt and populate the pgvector search database."""
    async with httpx.AsyncClient() as http:
        sections = await fetch_all_sections(http)

    openai = AsyncOpenAI()

    async with database_connect(True) as pool:
        async with pool.acquire() as conn, conn.transaction():
            await conn.execute(DB_SCHEMA)

        sem = asyncio.Semaphore(10)

        async def embed_and_insert(section: DocSection) -> None:
            async with sem:
                embedding = await openai.embeddings.create(
                    input=section.embedding_content(),
                    model='text-embedding-3-small',
                )
                assert len(embedding.data) == 1
                vector = embedding.data[0].embedding
                embedding_json = pydantic_core.to_json(vector).decode()
                result = await pool.execute(
                    'INSERT INTO doc_sections (url, title, content, embedding)'
                    ' VALUES ($1, $2, $3, $4) ON CONFLICT (url) DO NOTHING',
                    section.url,
                    section.title,
                    section.content,
                    embedding_json,
                )
                if result == 'INSERT 0 1':
                    logger.info('Inserted %s', section.url)
                else:
                    logger.info('Skipping %s', section.url)

        await asyncio.gather(*[embed_and_insert(s) for s in sections])


# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
@asynccontextmanager
async def database_connect(
    create_db: bool = False,
) -> AsyncGenerator[asyncpg.Pool, None]:
    server_dsn, database = 'postgresql://api:api@localhost:5432', 'rag_example'
    if create_db:
        conn = await asyncpg.connect(server_dsn)
        try:
            db_exists = await conn.fetchval('SELECT 1 FROM pg_database WHERE datname = $1', database)
            if not db_exists:
                await conn.execute(f'CREATE DATABASE {database}')
        finally:
            await conn.close()

    pool = await asyncpg.create_pool(f'{server_dsn}/{database}')
    try:
        yield pool
    finally:
        await pool.close()


DB_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS doc_sections (
    id serial PRIMARY KEY,
    url text NOT NULL UNIQUE,
    title text NOT NULL,
    content text NOT NULL,
    -- text-embedding-3-small returns a vector of 1536 floats
    embedding vector(1536) NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_doc_sections_embedding ON doc_sections USING hnsw (embedding vector_l2_ops);
"""


def slugify(value: str, separator: str, unicode: bool = False) -> str:
    """Slugify a string, to make it URL friendly."""
    if not unicode:
        value = unicodedata.normalize('NFKD', value)
        value = value.encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    return re.sub(rf'[{separator}\s]+', separator, value)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    action = sys.argv[1] if len(sys.argv) > 1 else None
    if action == 'build':
        asyncio.run(build_search_db())
    elif action == 'search':
        q = sys.argv[2] if len(sys.argv) == 3 else 'How do I run a RAG evaluation?'
        asyncio.run(run_agent(q))
    else:
        print('python -m src.rag build|search', file=sys.stderr)
        sys.exit(1)
