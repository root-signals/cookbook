"""OpenTelemetry wiring for the OTel evaluation example.

Builds a tracer provider that ships spans to Scorable's OTel ingest endpoint
(`/otel/v1/traces`) and attaches it to a Pydantic AI agent. Only
`evaluate_with_otel.py` uses this — the SDK example evaluates in-process and
emits no traces.

Configure via env (loaded from `.env` by `uv run --env-file`):

    SCORABLE_OTEL_ENDPOINT   e.g. https://api.scorable.ai/otel/v1/traces
    SCORABLE_OTEL_API_KEY    a Scorable API key
"""

from __future__ import annotations

import os
from typing import Any

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from pydantic_ai import Agent, InstrumentationSettings


def build_tracer_provider(service_name: str) -> TracerProvider:
    endpoint = os.environ.get("SCORABLE_OTEL_ENDPOINT")
    api_key = os.environ.get("SCORABLE_OTEL_API_KEY")
    if not endpoint or not api_key:
        raise RuntimeError(
            "Set SCORABLE_OTEL_ENDPOINT and SCORABLE_OTEL_API_KEY (see .env) to ship traces to Scorable."
        )

    exporter = OTLPSpanExporter(endpoint=endpoint, headers={"Authorization": f"Api-Key {api_key}"})
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(exporter))
    return provider


def instrument_agent(agent: Agent[Any, Any]) -> TracerProvider:
    """Attach a Scorable tracer provider to `agent` and return it.

    The caller must `provider.shutdown()` (or `force_flush()`) before the
    process exits, or the batched spans never leave.
    """
    provider = build_tracer_provider(agent.name or "agent")
    agent.instrument = InstrumentationSettings(tracer_provider=provider)
    return provider
