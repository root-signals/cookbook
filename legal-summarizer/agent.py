"""Legal-document summarizer agent (shared by both evaluation examples).

A Pydantic AI agent that, given a company's website URL, fetches the
company's legal documents (privacy policy, terms of service, cookie policy,
DPA, sub-processors, acceptable-use, security/trust) and produces a single
markdown summary with inline `[N]` citations.

Discovery starts from the site's sitemap rather than the homepage — sitemaps
list every public URL by design, so the agent doesn't have to interpret nav
menus or guess footer conventions.

This module is intentionally evaluation-agnostic: it defines the agent and a
`run_agent` helper, but wires up no Scorable evaluation. The two example
scripts (`evaluate_with_sdk.py`, `evaluate_with_otel.py`) show the two ways to
score its output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebFetch
from pydantic_ai.messages import ModelMessage, ToolCallPart

AGENT_NAME = "legal-summarizer"

LEGAL_SUMMARIZER_SYSTEM_PROMPT = """
You are a legal-document summarizer for company websites.

Given a company's base URL (in `company_url`), produce a single markdown
document that summarizes the company's user-facing legal documents.

# Discovery — start from the sitemap

1. WebFetch `<company_url>/sitemap.xml`.
2. If that returns 404 or empty, try in order:
     - `<company_url>/sitemap_index.xml`
     - `<company_url>/robots.txt` (which usually lists the real sitemap URL)
3. If the sitemap is a sitemap-index (links to other sitemap files), pick
   the one most likely to contain top-level pages (e.g. `sitemap-pages.xml`,
   `pages-sitemap.xml`, or just the first/smallest one) and WebFetch it.
4. If no sitemap can be found at all, fall back to fetching the homepage at
   `company_url` and look for legal links in the footer / nav.

# Selecting legal URLs

From the sitemap (or homepage), pick URLs whose paths suggest one of these
document types. Match flexibly — e.g. `/privacy-policy`, `/en/legal/terms`,
`/legal/cookies` all count.

  - Privacy Policy           — paths containing `privacy`
  - Terms of Service         — paths containing `terms`, `tos`, `terms-of-service`, `terms-of-use`
  - Cookie Policy            — paths containing `cookie`
  - Data Processing Agreement — paths containing `dpa`, `data-processing`
  - Sub-processors           — paths containing `subprocessor`, `sub-processors`
  - Acceptable Use Policy    — paths containing `acceptable-use`, `aup`
  - Security / Trust         — paths containing `security`, `trust`

Ignore everything else in the sitemap. Do NOT fetch product pages, blog
posts, careers pages, etc.

# Fetching and summarizing

5. WebFetch each candidate URL you selected.
6. Produce a single markdown document with one `##` section per document
   type you actually found. Inside each section, give a concise bulleted
   summary of the key points (data collected, retention, user rights,
   prohibited uses, etc. — whatever is actually material).

# Citation format (mandatory — the evaluator depends on this)

7. Number each unique source URL starting at 1 in the order you first cite
   it. Every factual claim MUST end with one or more inline references in
   the form `[N]` (e.g. `… retained for 90 days [1].`). Multiple sources
   for the same claim use `[1][2]`.
8. End the document with a single `## References` section listing every
   numbered source, one per line, in this EXACT format:

       [1]: https://example.com/privacy-policy
       [2]: https://example.com/terms

   The `[N]:` prefix, the colon, and the literal URL are required. No
   extra text, no markdown link syntax, no titles. Use the exact URL you
   actually fetched.

# Rules

- Only state facts that appear in the fetched content. Do NOT infer,
  generalize, or fill in from prior knowledge of the company.
- If a document type isn't in the sitemap and isn't found via fallback,
  OMIT that section entirely. Don't write "no policy found".
- Keep it readable: aim for ~5-10 bullets per section, not exhaustive
  paragraph-by-paragraph quoting.
- The whole summary should be valid markdown.
""".strip()


class LegalSummarizerDeps(BaseModel):
    company_url: str = Field(description="Base URL of the company website, e.g. https://scorable.ai")


agent: Agent[LegalSummarizerDeps, str] = Agent(
    model="openai:gpt-5.4-mini",
    name=AGENT_NAME,
    deps_type=LegalSummarizerDeps,
    capabilities=[WebFetch(local=True)],
    system_prompt=LEGAL_SUMMARIZER_SYSTEM_PROMPT,
)


@dataclass
class AgentRun:
    summary: str
    fetched_urls: list[str]


def _extract_fetched_urls(messages: list[ModelMessage]) -> list[str]:
    """Pull every URL the agent passed to a web_fetch tool call."""
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


async def run_agent(company_url: str) -> AgentRun:
    result = await agent.run(company_url, deps=LegalSummarizerDeps(company_url=company_url))
    return AgentRun(summary=result.output, fetched_urls=_extract_fetched_urls(result.all_messages()))
