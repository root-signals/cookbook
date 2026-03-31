import {
  generateAnswerFromQueryResult,
  generateIntentFromRequest,
  generateSqlFromSpec,
} from './pipeline.js';

export const LLM_MODES = {
  MOCK: 'mock',
  OPENAI: 'openai',
};

export const DEFAULT_OPENAI_MODEL = 'gpt-4.1-mini';
const OPENAI_COMPLETIONS_URL = 'https://api.openai.com/v1/chat/completions';

export function resolveRuntimeConfig(input = {}) {
  const mode =
    input.mode === LLM_MODES.OPENAI || input.llmMode === LLM_MODES.OPENAI ? LLM_MODES.OPENAI : LLM_MODES.MOCK;
  return {
    mode,
    apiKey: (input.apiKey || input.openaiKey || '').trim(),
    model: (input.model || input.openaiModel || DEFAULT_OPENAI_MODEL).trim() || DEFAULT_OPENAI_MODEL,
    temperature: Number.isFinite(input.temperature) ? input.temperature : 0.2,
  };
}

function extractTextFromCodeFence(value) {
  return (value || '')
    .trim()
    .replace(/^```(?:sql|json|markdown)?\s*\n?/i, '')
    .replace(/\n?\s*```$/s, '')
    .trim();
}

function parseJsonResponse(payload) {
  const strict = extractTextFromCodeFence(payload).trim();
  try {
    return JSON.parse(strict);
  } catch (error) {
    return null;
  }
}

function normalizeOpenAiResult(content) {
  const direct = parseJsonResponse(content);
  if (direct) return direct;
  if (!content) return null;
  try {
    const fenced = content.match(/```json([\s\S]*?)```/i);
    if (fenced?.[1]) return parseJsonResponse(fenced[1]);
  } catch {
    // ignore
  }
  return null;
}

async function callOpenAi({ apiKey, model, temperature, messages, response_format }) {
  const body = {
    model,
    messages,
    temperature,
  };
  if (response_format) {
    body.response_format = response_format;
  }

  const response = await fetch(OPENAI_COMPLETIONS_URL, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`OpenAI API returned ${response.status}: ${errorText}`);
  }

  const data = await response.json();
  const content = data?.choices?.[0]?.message?.content;
  if (typeof content !== 'string') {
    throw new Error('OpenAI response missing message content');
  }
  return content;
}

async function fallbackToMockStage1(request, config) {
  return generateIntentFromRequest(request, config);
}

async function fallbackToMockStage2(request, schema, config) {
  return generateSqlFromSpec(request, schema, config);
}

async function fallbackToMockStage3(query, resultset, config) {
  return generateAnswerFromQueryResult(query, resultset, config);
}

export async function generateStage1(request, runtimeConfig = {}) {
  const config = resolveRuntimeConfig(runtimeConfig);
  if (config.mode === LLM_MODES.MOCK) {
    return fallbackToMockStage1(request, config);
  }

  if (!config.apiKey) {
    throw new Error('OpenAI key is required when LLM mode is set to OpenAI.');
  }

  try {
    const prompt = `Convert this user request into a strict JSON query spec:
${request}

Rules:
- Return only JSON with keys intent, filters, aggregation, confidence.
- intent must be one of count_deals, sum_revenue, avg_deal_size, list_deals.
- aggregation must be one of count, sum, average.
- Use filters keys only when implied by request.
`;

  const content = await callOpenAi({
      apiKey: config.apiKey,
      model: config.model,
      temperature: config.temperature,
      response_format: { type: 'json_object' },
      messages: [
        {
          role: 'system',
          content: 'You are a strict JSON-only assistant for CRM query intent extraction.',
        },
        {
          role: 'user',
          content: prompt,
        },
      ],
    });
    const parsed = normalizeOpenAiResult(content);
    if (!parsed) {
      throw new Error('Stage 1 response was not valid JSON.');
    }
    return parsed;
  } catch (error) {
    throw error;
  }
}

export async function generateStage2(request, schema, runtimeConfig = {}) {
  const config = resolveRuntimeConfig(runtimeConfig);
  if (config.mode === LLM_MODES.MOCK) {
    return fallbackToMockStage2(request, schema, config);
  }

  if (!config.apiKey) {
    throw new Error('OpenAI key is required when LLM mode is set to OpenAI.');
  }

  try {
    const prompt = `Given this query spec and database schema, generate safe SQL.\n\nQuerySpec:\n${request}\n\nSchema:\n${schema}\n\nRules:\n- Return plain SQL only.\n- Use only SELECT statements.\n- Never use markdown or JSON formatting.\n- Prefer SQLite-compatible SQL.\n- If a date filter is needed, use expressions like:\n  - datetime('now','-7 days')\n  - datetime('now','-30 days')\n  - datetime('now','-1 month')\n  - date('now','weekday 1','-7 days')\n- Do not use INTERVAL, NOW(), or PostgreSQL-only date helpers.\n`;
    const content = await callOpenAi({
      apiKey: config.apiKey,
      model: config.model,
      temperature: config.temperature,
      messages: [
        {
          role: 'system',
          content: 'You are a SQL generator for SQLite. Output only SQL text.',
        },
        {
          role: 'user',
          content: prompt,
        },
      ],
    });
    return extractTextFromCodeFence(content);
  } catch (error) {
    throw error;
  }
}

export async function generateStage3(query, resultset, runtimeConfig = {}) {
  const config = resolveRuntimeConfig(runtimeConfig);
  if (config.mode === LLM_MODES.MOCK) {
    return fallbackToMockStage3(query, resultset, config);
  }

  if (!config.apiKey) {
    throw new Error('OpenAI key is required when LLM mode is set to OpenAI.');
  }

  try {
    const prompt = `Write a concise business answer from this query spec and result set.\n\nQuerySpec:\n${query}\n\nResultSet:\n${resultset}\n\nRules:\n- Return one plain sentence.`;
    const content = await callOpenAi({
      apiKey: config.apiKey,
      model: config.model,
      temperature: config.temperature,
      messages: [
        {
          role: 'system',
          content:
            'You are a sales operations analyst writing short user-facing summaries from SQL results.',
        },
        {
          role: 'user',
          content: prompt,
        },
      ],
    });
    return extractTextFromCodeFence(content);
  } catch (error) {
    throw error;
  }
}
