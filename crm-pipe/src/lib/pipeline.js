export const exampleScenario = {
  name: 'London deals closed',
  stage1Request: 'How many deals closed last week by the London team?',
  schema:
    'CREATE TABLE deals (id SERIAL PRIMARY KEY, status VARCHAR(20), closed_date TIMESTAMP, team_id INT, amount DECIMAL(10,2));\nCREATE TABLE teams (id SERIAL PRIMARY KEY, name VARCHAR(100), region VARCHAR(50));',
  resultset: [{ deal_count: 47 }],
};

function safeParseJson(input) {
  if (input == null) return null;
  if (typeof input === 'object') return input;
  if (typeof input !== 'string') return null;
  try {
    return JSON.parse(input.trim());
  } catch {
    return null;
  }
}

function firstJsonObject(input) {
  return typeof input === 'object' && input !== null ? input : safeParseJson(input);
}

export function parseRowsFromSpec(querySpecInput) {
  const value = firstJsonObject(querySpecInput);
  return value && typeof value === 'object' ? value : null;
}

function hasRegex(input, re) {
  return re.test((input || '').toLowerCase());
}

function inferIntent(requestText) {
  const text = (requestText || '').toLowerCase();
  if (/\b(how many|count|number of|total number)\b/.test(text)) return 'count_deals';
  if (/\b(sum|revenue|total amount)\b/.test(text)) return 'sum_revenue';
  if (/\b(average|avg|mean)\b/.test(text)) return 'avg_deal_size';
  return 'list_deals';
}

function inferTimeframe(requestText) {
  if (hasRegex(requestText, /\b(last week|past week|previous week)\b/)) return 'last_week';
  if (hasRegex(requestText, /\bthis week\b/)) return 'this_week';
  if (hasRegex(requestText, /\blast month\b/)) return 'last_month';
  if (hasRegex(requestText, /\blast 30 days\b/)) return 'last_30_days';
  return null;
}

function inferStatus(requestText) {
  if (hasRegex(requestText, /\b(closed|won|successful)\b/)) return 'closed';
  if (hasRegex(requestText, /\blost\b/)) return 'lost';
  if (hasRegex(requestText, /\bopen|active\b/)) return 'open';
  return null;
}

export function inferTeam(requestText) {
  const patterns = [
    /\bby\s+the\s+([A-Za-z][A-Za-z0-9 _-]{2,30})\s+team\b/i,
    /\bfor\s+the\s+([A-Za-z][A-Za-z0-9 _-]{2,30})\s+team\b/i,
    /\b(?:by|for)\s+([A-Za-z][A-Za-z0-9 _-]{2,30})\b/i,
  ];
  for (const regex of patterns) {
    const match = (requestText || '').match(regex);
    if (match?.[1]) return match[1].trim();
  }
  return null;
}

export function generateIntentFromRequest(requestText) {
  const text = (requestText || '').trim();
  const intent = inferIntent(text);
  const timeframe = inferTimeframe(text);
  const status = inferStatus(text);
  const team = inferTeam(text);
  return {
    intent,
    filters: {
      ...(status ? { status } : {}),
      ...(timeframe ? { timeframe } : {}),
      ...(team ? { team } : {}),
    },
    aggregation: intent === 'sum_revenue' ? 'sum' : intent === 'avg_deal_size' ? 'average' : 'count',
    confidence: Math.min(0.97, 0.62 + Math.min(text.length / 180, 0.25)),
  };
}

function extractTablesFromSchema(schemaText) {
  const schema = (schemaText || '').toLowerCase();
  return schema
    .match(/create\s+table\s+([a-z_][a-z0-9_]*)/g)
    ?.map((line) => line.replace(/create\s+table\s+/, '').trim()) || [];
}

export function generateSqlFromSpec(specInput, schemaText = '') {
  const spec = firstJsonObject(specInput);
  if (!spec || typeof spec !== 'object') return '';
  const tables = extractTablesFromSchema(schemaText);
  const hasDeals = tables.includes('deals');
  const hasTeams = tables.includes('teams');

  const filters = spec.filters || {};
  const aggregation = spec.aggregation || 'count';
  const timeframe = filters.timeframe;
  const status = filters.status;
  const team = filters.team;

  const metric =
    aggregation === 'sum'
      ? 'SUM(d.amount) AS value'
      : aggregation === 'average'
      ? 'AVG(d.amount) AS value'
      : 'COUNT(*) AS value';

  let sql = `SELECT ${metric} FROM ${hasDeals ? 'deals d' : 'source_data d'}`;
  if (team && hasTeams) {
    sql += ' JOIN teams t ON d.team_id = t.id';
  }

  const where = [];
  if (status) where.push(`d.status = '${status}'`);
  if (timeframe === 'last_week') where.push("d.closed_date >= NOW() - INTERVAL '7 days'");
  if (timeframe === 'this_week') where.push("d.closed_date >= DATE_TRUNC('week', NOW())");
  if (timeframe === 'last_month') where.push("d.closed_date >= NOW() - INTERVAL '1 month'");
  if (timeframe === 'last_30_days') where.push("d.closed_date >= NOW() - INTERVAL '30 days'");
  if (team && hasTeams) where.push(`t.name = '${team}'`);

  if (where.length) sql += ` WHERE ${where.join(' AND ')}`;
  return sql;
}

export function generateAnswerFromQueryResult(queryInput, resultsetInput) {
  const query = firstJsonObject(queryInput) || {};
  const rows = Array.isArray(safeParseJson(resultsetInput)) ? safeParseJson(resultsetInput) : [];
  const firstRow = rows[0] || {};
  const metricKeys = ['deal_count', 'value', 'amount', 'total', 'sum', 'count'];

  let metricValue = null;
  for (const key of metricKeys) {
    const candidate = firstRow[key];
    if (typeof candidate === 'number' && Number.isFinite(candidate)) {
      metricValue = candidate;
      break;
    }
  }
  if (metricValue === null) {
    for (const value of Object.values(firstRow)) {
      if (typeof value === 'number' && Number.isFinite(value)) {
        metricValue = value;
        break;
      }
    }
  }

  const filters = query.filters || {};
  const timeframe = filters.timeframe || 'recent period';
  const team = filters.team || 'the team';
  const metricName =
    query.aggregation === 'sum'
      ? 'total value'
      : query.aggregation === 'average'
      ? 'average deal size'
      : 'deal count';

  if (metricValue === null) {
    return `No numeric results were returned for ${metricName} for ${team} in ${timeframe.replace('_', ' ')}.`;
  }
  return `${team} had ${metricValue} ${metricName} for ${timeframe.replace('_', ' ')}.`;
}

export function runPipelineHeadless({ request, schema, resultset }) {
  const stage1Spec = generateIntentFromRequest(request);
  const stage1Response = JSON.stringify(stage1Spec, null, 2);
  const stage2Response = generateSqlFromSpec(stage1Response, schema);
  const parsedResultset = Array.isArray(resultset) ? resultset : safeParseJson(resultset) || [];
  const stage3Response = generateAnswerFromQueryResult(stage1Response, JSON.stringify(parsedResultset));

  return {
    stage1: {
      request,
      response: stage1Response,
    },
    stage2: {
      request: stage1Response,
      schema,
      response: stage2Response,
    },
    stage3: {
      query: stage1Response,
      resultset: JSON.stringify(parsedResultset),
      response: stage3Response,
    },
    payloads: {
      stage1: { request, response: stage1Response },
      stage2: { request: stage1Response, schema, response: stage2Response },
      stage3: { query: stage1Response, resultset: JSON.stringify(parsedResultset), response: stage3Response },
    },
  };
}
