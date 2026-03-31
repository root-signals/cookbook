import { parseRowsFromSpec } from './pipeline.js';

export const SQL_MODES = {
  MOCK: 'mock',
  SQLITE: 'sqlite',
  REMOTE: 'remote',
};

const DEFAULT_SQL_ENDPOINT = '/api/sql';

function toRows(payload) {
  if (Array.isArray(payload)) return payload;
  if (Array.isArray(payload?.rows)) return payload.rows;
  if (Array.isArray(payload?.resultset)) return payload.resultset;
  if (Array.isArray(payload?.data)) return payload.data;
  return [];
}

function parseDateOffsetDays(days) {
  const now = new Date();
  const past = new Date(now.getTime() + days * 24 * 60 * 60 * 1000);
  return past.toISOString();
}

const SAMPLE_ROWS = [
  { team: 'London', status: 'closed', closed_date: parseDateOffsetDays(-1), amount: 1200 },
  { team: 'London', status: 'closed', closed_date: parseDateOffsetDays(-2), amount: 860 },
  { team: 'London', status: 'open', closed_date: parseDateOffsetDays(-3), amount: 640 },
  { team: 'London', status: 'closed', closed_date: parseDateOffsetDays(-8), amount: 740 },
  { team: 'London', status: 'closed', closed_date: parseDateOffsetDays(-12), amount: 300 },
  { team: 'Chicago', status: 'closed', closed_date: parseDateOffsetDays(-1), amount: 930 },
  { team: 'Chicago', status: 'open', closed_date: parseDateOffsetDays(-1), amount: 500 },
];

function parseRowsFromSql(sql) {
  const text = (sql || '').toLowerCase();
  const now = new Date();
  const hasTeamFilter = /t\.name\s*=\s*'([^']+)'/.exec(text);
  const team = hasTeamFilter ? hasTeamFilter[1].toLowerCase() : null;

  const hasStatusFilter = /d\.status\s*=\s*'([^']+)'/.exec(text);
  const status = hasStatusFilter ? hasStatusFilter[1].toLowerCase() : null;

  const cutoff = (() => {
    if (/now\(\)\s*-\s*interval\s*'7 days'/.test(text)) {
      return new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
    }
    if (/date_trunc\('week',\s*now\(\)\)/.test(text)) {
      const monday = new Date(now);
      monday.setDate(now.getDate() - ((now.getDay() + 6) % 7));
      monday.setHours(0, 0, 0, 0);
      return monday;
    }
    if (/now\(\)\s*-\s*interval\s*'1 month'/.test(text) || /now\(\)\s*-\s*interval\s*'30 days'/.test(text)) {
      return new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
    }
    return null;
  })();

  return SAMPLE_ROWS.filter((row) => {
    if (team && row.team.toLowerCase() !== team) return false;
    if (status && row.status.toLowerCase() !== status) return false;
    if (cutoff && new Date(row.closed_date) < cutoff) return false;
    return true;
  });
}

export function executeSqlMock(sql, querySpec) {
  const normalized = (typeof sql === 'string' ? sql : '').toLowerCase();
  const spec = parseRowsFromSpec(querySpec);

  const teamFilter = (spec?.filters?.team || '').toLowerCase();
  let rows = SAMPLE_ROWS;
  if (teamFilter) {
    rows = rows.filter((row) => row.team.toLowerCase() === teamFilter);
  }
  if (spec?.filters?.status) {
    rows = rows.filter((row) => row.status.toLowerCase() === String(spec.filters.status).toLowerCase());
  }
  rows = parseRowsFromSql(sql).filter((r) => {
    if (teamFilter && r.team.toLowerCase() !== teamFilter) return false;
    if (spec?.filters?.status && r.status.toLowerCase() !== String(spec.filters.status).toLowerCase()) return false;
    return true;
  });

  if (/\bavg\s*\(/.test(normalized)) {
    const total = rows.reduce((acc, row) => acc + Number(row.amount), 0);
    const value = rows.length ? total / rows.length : 0;
    return [{ value }];
  }
  if (/\bsum\s*\(/.test(normalized)) {
    const total = rows.reduce((acc, row) => acc + Number(row.amount), 0);
    return [{ value: total }];
  }

  return [{ deal_count: rows.length }];
}

async function executeSqlRemote(sql, { endpoint, querySpec, headers = {}, payload = {} }) {
  const response = await fetch(endpoint, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...headers,
    },
    body: JSON.stringify({
      sql,
      querySpec,
      ...payload,
    }),
  });
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Remote SQL API ${response.status}: ${errorText}`);
  }
  const data = await response.json();
  if (Array.isArray(data)) return data;
  if (Array.isArray(data.resultset)) return data.resultset;
  if (Array.isArray(data.rows)) return data.rows;
  return [];
}

export async function executeSql(config = {}) {
  const mode =
    config.mode === SQL_MODES.MOCK
      ? SQL_MODES.MOCK
      : config.mode === SQL_MODES.REMOTE || config.mode === SQL_MODES.SQLITE
        ? config.mode
        : SQL_MODES.SQLITE;

  if (mode === SQL_MODES.REMOTE) {
    if (!config.endpoint) {
      throw new Error('Remote SQL endpoint required when SQL mode is remote.');
    }
    return executeSqlRemote(config.sql, {
      endpoint: config.endpoint,
      querySpec: config.querySpec,
      headers: config.headers,
      payload: config.payload,
    });
  }

  if (mode === SQL_MODES.SQLITE) {
    return executeSqlRemote(config.sql, {
      endpoint: config.endpoint || DEFAULT_SQL_ENDPOINT,
      querySpec: config.querySpec,
      headers: config.headers,
      payload: config.payload,
    });
  }

  return executeSqlMock(config.sql, config.querySpec);
}
