# Python OTEL Instrumentation Guide

This file shows the OpenInference instrumentation patterns for the most common Python LLM frameworks. Pair any of these with the OTLP exporter setup below.

## Universal OTLP Exporter Setup

Configures the trace pipeline. Run it once at application startup, **before any LLM call**.

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

Replace `my-agent` with a stable, descriptive name for the service.

---

## OpenAI SDK (direct calls)

```bash
pip install openinference-instrumentation-openai opentelemetry-exporter-otlp-proto-http
```

```python
from openinference.instrumentation.openai import OpenAIInstrumentor

# After the trace provider setup above
OpenAIInstrumentor().instrument()

# Now every openai SDK call (chat completions, responses, embeddings) is auto-traced.
import openai
client = openai.OpenAI()
client.chat.completions.create(...)
```

---

## openai-agents SDK

```bash
pip install openinference-instrumentation-openai-agents
```

```python
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor

OpenAIAgentsInstrumentor().instrument()

from agents import Agent, Runner
agent = Agent(name="my-agent", instructions="...")
result = Runner.run(agent, "user input")
```

---

## pydantic-ai

pydantic-ai has built-in OTEL support via `InstrumentationSettings` — no openinference layer needed; you just pass your tracer provider directly into the `Agent`.

```bash
pip install pydantic-ai-slim[openai] opentelemetry-exporter-otlp-proto-http
```

```python
import os
from pydantic_ai import Agent, InstrumentationSettings
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def build_provider() -> TracerProvider:
    exporter = OTLPSpanExporter(
        endpoint="https://api.scorable.ai/otel/v1/traces",
        headers={"Authorization": f"Api-Key {os.environ['SCORABLE_API_KEY']}"},
    )
    resource = Resource.create({"service.name": "my-agent"})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    return provider


agent = Agent(
    model="openai:gpt-5.5",
    instrument=InstrumentationSettings(tracer_provider=build_provider()),
)

result = agent.run_sync("user input")
```

---

## LangChain / LangGraph

```bash
pip install openinference-instrumentation-langchain
```

```python
from openinference.instrumentation.langchain import LangChainInstrumentor

LangChainInstrumentor().instrument()
```

Covers chains, agents, tools, retrievers, and LangGraph nodes.

---

## Anthropic SDK

```bash
pip install openinference-instrumentation-anthropic
```

```python
from openinference.instrumentation.anthropic import AnthropicInstrumentor

AnthropicInstrumentor().instrument()
```

---

## LlamaIndex

```bash
pip install openinference-instrumentation-llama-index
```

```python
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

LlamaIndexInstrumentor().instrument()
```

---

## Custom application attributes

For multi-tenant systems or production scoping, set custom attributes on a wrapping span around the LLM call. They land in Scorable as filterable span attributes.

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("agent.run") as span:
    span.set_attribute("app.tenant_id", deps.tenant_id)
    span.set_attribute("app.feature", "checkout-bot")
    span.set_attribute("app.environment", "prod")
    result = await agent.run(question, deps=deps)
```

Then create filters scoped to specific tenants or features:

```bash
scorable otel-filter create \
  --name "checkout-bot-truthfulness" \
  --evaluator-id <uuid> \
  --filter-criteria '{"conditions":[
    {"column":"app.feature","type":"string","key":"app.feature","operator":"=","value":"checkout-bot"},
    {"column":"app.environment","type":"string","key":"app.environment","operator":"=","value":"prod"}
  ]}'
```

---

## Tips

- **One instrumentor per framework** — don't double-instrument. `OpenAIInstrumentor` plus `LangChainInstrumentor` is fine because they cover different layers, but `OpenAIInstrumentor` plus pydantic-ai's own tracing creates duplicate spans.
- **`SimpleSpanProcessor` for testing** — flushes synchronously, easier to see traces immediately. Don't use in production (blocking export per span).
- **Forced flush on exit** — for short-lived scripts (CLI tools, batch jobs), call `provider.force_flush()` before the process exits or spans may not export in time.
- **Logging the OTLP transport** — set `OTEL_PYTHON_LOG_LEVEL=DEBUG` to see export attempts and failures in stderr while debugging.
