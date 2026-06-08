# Legal Summarizer

A Pydantic AI agent that, given a company's website URL, fetches the company's
legal documents (privacy policy, terms of service, cookie policy, DPA,
sub-processors, acceptable-use, security/trust) and produces a single markdown
summary with inline `[N]` citations.

The interesting part isn't the agent — it's the **two ways to score it with
Scorable**, shown side by side:

| File | How it evaluates | When the score arrives |
| --- | --- | --- |
| `evaluate_with_sdk.py` | Calls the evaluator directly via the **SDK**, in-process | Synchronously, right after the run |
| `evaluate_with_otel.py` | Ships the run to Scorable as an **OpenTelemetry trace**; a filter scores it server-side | Asynchronously, in the dashboard |

Both import the same agent from `agent.py`. The agent is identical; only the
evaluation wiring differs.

## How the agent works

1. **Discovery from the sitemap.** First `WebFetch` is `<url>/sitemap.xml`,
   falling back to `/sitemap_index.xml`, `/robots.txt`, then the homepage.
2. **Filter for legal paths** (`privacy`, `terms`, `cookie`, `dpa`, …).
3. **Fetch each one** with `WebFetch`.
4. **Summarize with numbered citations** — one `##` section per document type,
   every claim ending in `[N]`, and a `## References` list mapping `[N]` → URL.

`Faithfulness_to_Citations` then pulls those URLs out of the response, fetches
them, and checks whether every claim is actually supported.

## Setup

```sh
cd cookbook/legal-summarizer
uv sync
```

Fill in `.env` (gitignored — `uv run --env-file .env` loads it automatically):

```sh
OPENAI_API_KEY=...            # the agent uses openai:gpt-5.4-mini

# For evaluate_with_sdk.py:
SCORABLE_API_KEY=...          # get one at https://scorable.ai/settings/api-keys

# For evaluate_with_otel.py:
SCORABLE_OTEL_ENDPOINT=https://api.scorable.ai/otel/v1/traces
SCORABLE_OTEL_API_KEY=...     # a Scorable API key
```

---

## Path 1 — evaluate with the SDK (in-process)

The simplest path: run the agent, then call the evaluator directly. The score
comes back in the same process, right after the run.

```sh
uv run --env-file .env python evaluate_with_sdk.py --url https://scorable.ai
```

Prints the summary, then:

```
========================================================================
Faithfulness to Citations: 0.93 / 1.00
========================================================================
110/116 assertions supported; errors concentrated in …
```

The whole integration is two lines:

```python
client = Scorable(run_async=True)
result = await client.evaluators.Faithfulness_to_Citations(request=url, response=summary)
```

Use this when you want the score inline — gating a response, a CI check, a
quick experiment.

---

## Path 2 — evaluate via OpenTelemetry traces (server-side)

Here you don't call an evaluator. You **instrument the agent**, so every run is
exported to Scorable as an OTel trace, and a one-time **evaluation filter**
scores matching traces server-side. The score appears in the dashboard,
embedded in the trace.

### One-time setup: create the filter (via the CLI)

Find the evaluator's id, then wire it to incoming traces:

```sh
# install the CLI once: curl -sSL https://scorable.ai/cli/install.sh | sh
scorable login

# look up the evaluator id
scorable evaluator list --name "Faithfulness to Citations"

# create a filter that runs it on every incoming trace
scorable otel-filter create \
    --name "legal-summarizer-faithfulness" \
    --evaluator-id <Faithfulness to Citations evaluator id>
```

(Scope it to this agent's traces with
`--filter-criteria '{"conditions":[{"column":"resource","type":"string","key":"service.name","operator":"=","value":"legal-summarizer"}]}'`
if you have other traces flowing in. List existing filters with
`scorable otel-filter list`.)

### Run

```sh
uv run --env-file .env python evaluate_with_otel.py --url https://scorable.ai
```

Prints the summary and confirms the trace was sent. The score is produced
asynchronously by the filter

```sh
scorable otel-trace list
```

The integration is one line — attach a tracer provider to the agent:

```python
agent.instrument = InstrumentationSettings(tracer_provider=provider)
```

Use this when evaluation should be decoupled from your app: score production
traffic continuously, without putting an evaluator call on the request path.

---

