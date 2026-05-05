# TypeScript / JavaScript OTEL Instrumentation Guide

OpenInference provides framework-specific JS/TS instrumentation libraries on npm. Combine with the OpenTelemetry SDK and the OTLP HTTP exporter pointing at Scorable.

## Universal OTLP Exporter Setup

Run this once at application startup, **before any LLM call**.

```bash
npm install \
  @opentelemetry/api \
  @opentelemetry/sdk-trace-node \
  @opentelemetry/exporter-trace-otlp-http \
  @opentelemetry/resources \
  @opentelemetry/sdk-trace-base
```

```typescript
import { Resource } from "@opentelemetry/resources";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { BatchSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-http";

const exporter = new OTLPTraceExporter({
  url: "https://api.scorable.ai/otel/v1/traces",
  headers: { Authorization: `Api-Key ${process.env.SCORABLE_API_KEY}` },
});

const provider = new NodeTracerProvider({
  resource: new Resource({ "service.name": "my-agent" }),
});
provider.addSpanProcessor(new BatchSpanProcessor(exporter));
provider.register();
```

In Next.js, place this in `instrumentation.ts` at the project root and ensure `experimental.instrumentationHook = true` in `next.config.js`.

---

## OpenAI Node SDK

```bash
npm install @arizeai/openinference-instrumentation-openai @opentelemetry/instrumentation
```

```typescript
import { OpenAIInstrumentation } from "@arizeai/openinference-instrumentation-openai";
import { registerInstrumentations } from "@opentelemetry/instrumentation";

registerInstrumentations({
  instrumentations: [new OpenAIInstrumentation()],
});
```

Run this *after* the provider setup but *before* importing `openai`.

---

## LangChain.js

```bash
npm install @arizeai/openinference-instrumentation-langchain
```

```typescript
import { LangChainInstrumentation } from "@arizeai/openinference-instrumentation-langchain";
import { registerInstrumentations } from "@opentelemetry/instrumentation";

registerInstrumentations({
  instrumentations: [new LangChainInstrumentation()],
});
```

---

## Vercel AI SDK

The Vercel AI SDK has built-in `experimental_telemetry` support. No openinference layer is needed; the trace provider configured above picks up the spans automatically.

```typescript
import { generateText } from "ai";
import { openai } from "@ai-sdk/openai";

const result = await generateText({
  model: openai("gpt-5.5"),
  prompt: "...",
  experimental_telemetry: {
    isEnabled: true,
    functionId: "my-agent",
  },
});
```

---

## Custom application attributes

```typescript
import { trace } from "@opentelemetry/api";

const tracer = trace.getTracer("my-app");

await tracer.startActiveSpan("agent.run", async (span) => {
  span.setAttribute("app.tenant_id", tenantId);
  span.setAttribute("app.feature", "checkout-bot");
  span.setAttribute("app.environment", "prod");
  try {
    return await runAgent();
  } finally {
    span.end();
  }
});
```

Filter on these attributes the same way Python does — see the main SKILL.md for filter creation examples.

---

## See Also

- OpenInference catalog (full list of JS/TS instrumentation libraries): https://github.com/Arize-ai/openinference/tree/main/js
- OpenTelemetry JS docs: https://opentelemetry.io/docs/languages/js/
