---
name: scorable-otel-evaluation
description: Wire end-to-end OpenTelemetry-based tracing and evaluation for an LLM application — instrument the application with OTEL via OpenInference, ship traces to Scorable's OTLP endpoint, then use the Scorable CLI to query traces and create evaluation filters that auto-run an evaluator or judge against matching traces. Use when users want OTEL/OpenTelemetry tracing for their LLM app, want to monitor production LLM traffic, want auto-evaluation on a fraction of traffic, mention OTLP, OpenInference, Arize, pydantic-ai, openai-agents, LangChain, or "tracing my agent."
---

# Add End-to-End OTEL Tracing + Evaluation with Scorable

These instructions guide you through instrumenting an LLM application with OpenTelemetry, shipping traces to Scorable, and configuring server-side evaluation filters that automatically score matching traces. The result: every (or a sampled subset of) production LLM call gets observed, traced, and scored — and the score lands back on the same trace as a child span carrying the OpenTelemetry GenAI evaluation attributes.

## Execution Contract

You are responsible for completing the OTEL tracing + evaluation setup end-to-end in as few turns as possible.

- You MUST analyze the codebase to identify the LLM framework and entry points.
- You MUST install and use the Scorable CLI directly.
- You MUST add the OTEL instrumentation code yourself.
- You MUST run a test request and verify traces land in Scorable.
- You MUST create at least one evaluation filter via the CLI.
- You MUST verify the filter triggered an evaluation by inspecting the trace's spans.
- You MUST update project documentation for usage.
- You MUST NOT delegate technical steps to the user except where explicitly required (API key entry, secrets).
- You MUST continue until implementation is complete or a hard blocker is reached.

## Overview

Your role is to:

1. **Analyze the codebase** to identify the LLM framework in use (OpenAI SDK, openai-agents, pydantic-ai, LangChain, Anthropic, LlamaIndex, etc.) and the entry points where requests are made.
2. **Install and authenticate the Scorable CLI** so you can manage filters and inspect traces.
3. **Instrument the application** using the OpenInference instrumentation that matches the framework, plus the OTLP exporter pointing at Scorable.
4. **Send a test request** and verify traces appear in Scorable using `scorable otel-trace list`.
5. **Create an evaluation filter** that runs an evaluator or judge against matching traces.
6. **Verify the filter triggered** by inspecting the resulting eval span on the trace.
7. **Provide usage + next-step documentation**.

If the user wants a custom judge but doesn't have one yet, generate it using the **scorable-integration** skill (its Step 3 covers `scorable judge generate` with the right intent prompts), then come back here with the resulting `judge_id` for filter creation.

## Step 0: Explain the Process

Before performing any analysis or technical steps, brief the user clearly. Explain that you will:

- Analyze the codebase to identify the LLM framework and entry points
- Install the Scorable CLI and authenticate
- Add OTEL instrumentation to ship traces to Scorable's OTLP endpoint
- Send a test request and verify traces land in Scorable
- Create an evaluation filter that auto-scores matching traces
- Provide usage documentation

Mention that this setup is composable: any other observability backend that accepts OTLP HTTP/protobuf can be a second sink later — Scorable just receives the same OTLP traffic via OpenTelemetry, so they can keep their existing tracing if they have it.

---

## Step 1: Analyze the Application

Identify:

- **Framework** — OpenAI SDK directly, openai-agents, pydantic-ai, LangChain / LangGraph, LlamaIndex, Anthropic SDK, Mastra, Vercel AI SDK, etc. Each has a corresponding OpenInference instrumentation library.
- **Language** — Python or TypeScript/JavaScript (instrumentation libraries exist for both).
- **Entry points** — where the application kicks off an LLM call. The OTEL setup needs to run *before* those calls execute.
- **Existing observability** — if the project already has OpenTelemetry configured (e.g. exports to Honeycomb, Datadog, Tempo), you'll add a second exporter rather than replace the setup.

If the framework is unfamiliar, default to looking it up in the OpenInference catalog: https://github.com/Arize-ai/openinference. Most popular SDKs are covered there.

If multiple LLM-using subsystems exist, help the user prioritize. Recommend starting with the most critical agent first — additional services can each get their own `service.name` and their own filter later.

---

## Step 2: Install Scorable CLI & Authenticate

Install the Scorable CLI:

```bash
curl -sSL https://scorable.ai/cli/install.sh | sh
```

Or with npm:

```bash
npm install -g @root-signals/scorable-cli
```

Or run without installing via npx:

```bash
npx @root-signals/scorable-cli judge list
```

Then ask the user which authentication option they prefer:

### Option A: Permanent API Key (Recommended)

Direct them to: https://scorable.ai/api-key-setup to create an API key, then set it via the CLI:

```bash
scorable auth set-key
# paste the key when prompted

# or alternatively:
scorable auth set-key <your-api-key>
```

**Security:** Use environment variables or the project's secret management. The same key is used both by the CLI *and* by the application's OTEL exporter. Read existing `.env` files if available, otherwise ask the user where they want to store the secret. Do not paste the key into this session.

### Option B: Temporary API Key (Testing Only)

Get a free demo key (no registration required):

```bash
scorable auth demo-key
```

Warn the user appropriately that:
- Filters and evaluators created with it will be public and visible to everyone
- The key only works for a limited time
- For private setups, they should create a permanent key at https://scorable.ai/register

### Option C: Existing API Key

If they have an account: https://scorable.ai/settings/api-keys

```bash
scorable auth set-key <your-api-key>
# or
export SCORABLE_API_KEY="sk-your-api-key"
```

Verify the CLI works before continuing:

```bash
scorable judge list
```

If this errors, do not move on — the OTEL exporter uses the same key.

---

## Step 3: Instrument the Application with OTEL

Choose the OpenInference instrumentation library that matches the framework you identified in Step 1.

### Python frameworks

See [references/python-instrumentation.md](references/python-instrumentation.md) for installation snippets and the canonical setup for OpenAI SDK, openai-agents, pydantic-ai, LangChain / LangGraph, Anthropic SDK, and LlamaIndex.

### TypeScript / JavaScript frameworks

See [references/typescript-instrumentation.md](references/typescript-instrumentation.md) for OpenAI Node SDK, LangChain.js, and Vercel AI SDK setup.

### Other languages

OpenTelemetry SDKs exist for Go, Java, Ruby, .NET, etc., but OpenInference auto-instrumentation coverage is currently strongest in Python and TypeScript. For other languages, point the OTLP HTTP/protobuf exporter at `https://api.scorable.ai/otel/v1/traces` with header `Authorization: Api-Key <your-api-key>` and instrument LLM calls manually with custom spans following the GenAI semantic conventions: https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/ . Note that you must check this as this keeps constantly updating. For example Scorable expects for example gen_ai.output.messages and gen_ai.input.messages.

### The OTLP exporter (universal Python snippet)

After picking the right framework instrumentor, add the exporter that ships spans to Scorable:

```python
import os
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

exporter = OTLPSpanExporter(
    endpoint="https://api.scorable.ai/otel/v1/traces",
    headers={"Authorization": f"Api-Key {os.environ['SCORABLE_API_KEY']}"},
)

resource = Resource.create({"service.name": "my-agent"})
provider = TracerProvider(resource=resource)
provider.add_span_processor(BatchSpanProcessor(exporter))
trace.set_tracer_provider(provider)
```

Critical pieces:

- **`service.name`** is the most important resource attribute. Use a stable, descriptive name (e.g. `customer-support-agent`, `code-review-bot`, `sales-bot-prod`). It's the strongest filter target in Scorable. Per-environment naming (`my-agent-prod` vs `my-agent-staging`) is fine and recommended.
- **`Api-Key <key>` header** — the same key set in Step 2. Read it from an env var; never hardcode.
- **`BatchSpanProcessor`** batches spans before flush. For tests you may want `SimpleSpanProcessor` to see traces immediately.

### Where to place the setup

The setup must run **once, before any LLM call**. Common placements:

- A dedicated `tracing.py` (or `tracing.ts`) module imported from your app entrypoint
- The framework's startup hook (FastAPI startup event, Django app `ready()`, Next.js `instrumentation.ts`)
- `__main__.py` / top of the agent script for CLI tools

---

## Step 4: Send a Test Request + Verify Trace Ingestion

After wiring up the instrumentation:

1. Run a single LLM request through the application's normal entry point.
2. Wait ~10 seconds for the `BatchSpanProcessor` to flush.
3. Verify the trace landed in Scorable using the CLI:

```bash
scorable otel-trace list --since 5m --service-name <your-service-name>
```

You should see the trace listed with its root span name and a span count.

### Drilling in

```bash
scorable otel-trace spans <trace_id>
scorable otel-trace spans <trace_id> --output json | jq '.[].span.attributes'
```

The full payload shows you exactly what `gen_ai.*` attributes the framework emitted, which determines what your filter conditions can match against. Common attributes any spec-conformant instrumentor sets: `gen_ai.agent.name`, `gen_ai.request.model`, `gen_ai.operation.name`, `gen_ai.tool.name`, `gen_ai.usage.input_tokens`, .

### If no trace shows up

- **CLI works but no trace?** OTEL export error. Check the application's stderr for OTLP exporter logs (the SDK logs export failures by default).
- **Confirm the BatchSpanProcessor flushed.** For an immediate test, swap to `SimpleSpanProcessor` temporarily — flushes synchronously.
- **Confirm `service.name`.** `scorable otel-trace list --since 5m` (without a `--service-name` filter) should show *something*. If it does, your filter value is off; if it doesn't, the export isn't reaching Scorable.
- **Auth failure?** A 401 from the exporter is logged at error level. Re-verify with `scorable judge list`.

Do not move on until at least one trace from the application is visible.

---

## Step 5: Create an Evaluation Filter

A filter runs an evaluator or judge automatically against every matching trace. The filter `criteria` uses the same wire format as the CLI's `--filter` flag.

### Decide the target

Two options:

- **An existing evaluator** — if the user has one (built with `scorable evaluator create` or pre-existing in their org), use `--evaluator-id <uuid>`. Single score per trace.
- **A judge** — bundle of evaluators producing per-evaluator scores. Use `--judge-id <uuid>`. If no judge exists, generate one using the **scorable-integration** skill (it walks through `scorable judge generate` with proper intent prompts), then come back here with the resulting `judge_id`.

### Decide what to match

A solid first filter scopes to the service you just instrumented:

```bash
scorable otel-filter create \
  --name "<service-name>-<evaluator-name>" \
  --evaluator-id <uuid> \
  --filter-criteria '{"conditions":[{"column":"resource","type":"string","key":"service.name","operator":"=","value":"<your-service-name>"}]}' \
  --delay-seconds 10
```

Or via judge:

```bash
scorable otel-filter create \
  --name "<service-name>-quality-judge" \
  --judge-id <uuid> \
  --filter-criteria '{"conditions":[{"column":"resource","type":"string","key":"service.name","operator":"=","value":"<your-service-name>"}]}' \
  --delay-seconds 10
```

### Key parameters

- **`--filter-criteria`** — JSON conditions, AND-combined. Scope to your `service.name` at minimum; narrow further with `gen_ai.agent.name`, `gen_ai.request.model`, etc. as needed.
- **`--delay-seconds`** — wait this long after the most recent span lands before triggering evaluation. 10s is a safe default; bump higher (30–60s) for long-running agents whose final span arrives much later than the first.
- **`--sampling-rate`** — between 0.0 and 1.0; default 1.0 (every match). Use 0.1 for 10% sampling in production.

### Filter grammar reference

For the full grammar (operators, columns, attribute syntax, time windows), refer to:

```bash
scorable otel-filter create --help
scorable otel-trace list --help    # same column vocabulary
```

The `--help` of `otel-trace list` documents all filter columns, the GenAI semantic conventions, and operator semantics inline.

---

## Step 6: Verify the Filter Triggered

1. Send another test request through the instrumented application.
2. Wait `delay_seconds + ~5s` for the eval to run.
3. List traces and find the matching one:

```bash
scorable otel-trace list --since 5m --service-name <your-service-name>
```

4. Inspect spans on the trace. The evaluation result lands as a **child span** parented to the original root, named `evaluate <evaluator-name>`:

```bash
scorable otel-trace spans <trace_id>
```

5. For full attributes:

```bash
scorable otel-trace spans <trace_id> --output json
```

The eval span carries:

- **`gen_ai.evaluation.name`** — the evaluator/judge that ran
- **`gen_ai.evaluation.score.value`** — numeric score (0–1)
- **`gen_ai.evaluation.explanation`** — the justification text
- **`scorable.evaluation = true`** — Scorable's marker indicating this is an eval span (not user data)
- **`resource.service.name = scorable.evaluation`** — distinct service so eval spans don't accidentally match customer service-scoped filters

### Querying by evaluation result

Once filters start producing eval spans, the CLI can target them too:

```bash
# Every trace that received any evaluation
scorable otel-trace list --since 24h \
  --filter 'gen_ai.evaluation.name;string;gen_ai.evaluation.name;=;<evaluator-name>'

# Low-scoring runs from the last 24h, exported as CSV for review
scorable otel-trace list --since 24h --output csv \
  --filter 'gen_ai.evaluation.score.value;number;gen_ai.evaluation.score.value;<;0.5' > low-scores.csv
```

If no eval span appears within the expected window:

- Run `scorable otel-filter list` and confirm the filter is `is_active: true` and matches by `service.name`.
- Confirm the trace's `service.name` (in `resource.service.name` attribute) actually matches the filter value — typos here are the most common cause of silent non-firing.
- The filter only fires after `delay_seconds`. For long delays, just wait longer.
- Filters only evaluate traces ingested *after* the filter was created — older traces are not backfilled.

---

## Step 7: Provide Next Steps

After verification:

1. **Sampling for production** — recommend `--sampling-rate 0.1` (or lower) to control evaluator cost. Sampling rate is part of the filter; recreate the filter to change it.
2. **Multiple filters per service** — separate filters for separate concerns (one for truthfulness, one for toxicity, one for tool-call correctness) keep results readable. They run independently against the same trace.
3. **Frontend visibility** — traces and their eval child spans are visible at https://scorable.ai/traces. Scores show up on the trace detail view alongside the original spans.
4. **Everyday CLI usage**:
   - `scorable otel-trace list --since 1h --has-error` — traces that errored in the last hour
   - `scorable otel-trace list --since 7d --filter 'gen_ai.evaluation.score.value;number;gen_ai.evaluation.score.value;<;0.3'` — flag low-scoring traces from the past week
   - `scorable otel-trace spans <trace_id> --output json | jq` — drill into one trace for debugging
   - `scorable otel-filter list` — review currently active filters
5. **Documentation entry** — add a short README/CONTRIBUTING note describing:
   - The `service.name` value the application reports
   - Where the OTEL setup lives (`tracing.py` or equivalent)
   - The active filter(s) and what they evaluate
6. **Link to docs**:
   - OTEL setup: https://docs.scorable.ai/usage/otel
   - GenAI semantic conventions (the source of truth for span attribute names): https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/
   - OpenInference instrumentation libraries catalog: https://github.com/Arize-ai/openinference

---

## Key Implementation Notes

- **Instrumentation runs once, before LLM calls** — get this wrong and traces silently disappear.
- **Use a stable `service.name`** — the strongest filter target. Per-environment naming (`my-agent-prod`, `my-agent-staging`) is a good default.
- **OpenInference libraries are framework-specific** — pick the one matching your framework. Don't double-instrument; OpenAI auto-instrumentation alongside pydantic-ai's own tracing creates duplicate spans.
- **API key from env vars** — same key for the CLI and the OTEL exporter. Same security posture as the existing Scorable integration skill.
- **Eval spans carry `scorable.evaluation = true`** — handy for distinguishing your application's spans from Scorable-emitted eval spans when querying or building dashboards.
- **Filters apply to new traces only** — they don't backfill historical data, so create filters as part of the rollout, not after the fact.
- **For private custom judges, follow scorable-integration** — that skill covers `scorable judge generate` with proper intent prompts, then return here with the `judge_id` for filter creation.
