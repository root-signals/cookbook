# Legal Summarizer

A Pydantic AI agent that, given a company's website URL, fetches the company's
legal documents (privacy policy, terms of service, cookie policy, DPA,
sub-processors, acceptable-use, security/trust) and produces a single markdown
summary with inline `[source: <url>]` citations.

The summary is then scored by Scorable's **Faithfulness_to_Citations** evaluator,
which extracts the cited URLs from the response, fetches them server-side, and
checks whether every claim in the summary is actually supported by what those
URLs say.

## How it works

1. **Discovery from the sitemap.** The agent's first `WebFetch` is
   `<company_url>/sitemap.xml`. It falls back to `/sitemap_index.xml`, then
   `/robots.txt`, then the homepage if needed. Sitemaps list every public URL
   on a site, so there's no need to interpret nav menus or guess footer
   conventions.
2. **Filter for legal paths.** From the sitemap entries, the agent picks URLs
   whose paths match conventional legal-document keywords (`privacy`, `terms`,
   `tos`, `cookie`, `dpa`, `subprocessor`, `acceptable-use`, `aup`, `security`,
   `trust`).
3. **Fetch each one.** Using `WebFetch` again — one call per selected URL.
4. **Summarize with citations.** The agent produces one markdown doc with a
   `##` section per document type it actually found. Every factual claim is
   followed by `[source: <url>]`.
5. **Score.** `scorable.evaluators.Faithfulness_to_Citations(request, response)`.
   The evaluator pulls URLs out of the response and grounds the score in their
   real contents.

## Setup

```sh
cd cookbook/legal-summarizer
uv sync
```

Required environment variables:

- `OPENAI_API_KEY` — for the pydantic-ai agent (uses `openai:gpt-5.4-mini`).
- `SCORABLE_API_KEY` — for the Faithfulness_to_Citations evaluator.
  Get one at <https://scorable.ai/settings/api-keys>. If unset, the run still
  produces a summary; evaluation is skipped with a log line.

A `.env` file is gitignored — fill in the keys there and uv will load it
automatically with `--env-file`:

```sh
cp .env .env.local   # optional, only if you want a separate file
$EDITOR .env
```

## Run

```sh
uv run --env-file .env legal-summarizer --url https://scorable.ai --md-out scorable.md
```

Flags:

| Flag             | Meaning                                                |
| ---------------- | ------------------------------------------------------ |
| `--url`          | Required. Company base URL (e.g. `https://example.com`). |
| `--md-out PATH`  | Write a markdown report (summary + score + justification + fetched URLs). |
| `--json`         | Print the full result as JSON instead of formatted text. |
| `--no-evaluate`  | Skip the Faithfulness_to_Citations call.               |
| `-v` / `--verbose` | Debug logging.                                       |

## Notes

- The agent uses Pydantic AI's `WebFetch(local=True)` capability — native when
  the model supports it, with a local markdownify-based fallback otherwise. No
  web search is used: discovery is purely sitemap-driven.
- Faithfulness_to_Citations doesn't need `contexts` because it reads the
  cited URLs straight out of the response. That's the whole point of this
  evaluator vs. regular `Faithfulness`.
