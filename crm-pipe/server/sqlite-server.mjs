import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import initSqlJs from 'sql.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PORT = Number(process.env.PORT || 5174);
const DB_PATH = path.join(__dirname, 'data', 'crm.sqlite');
const WASM_PATH = path.join(process.cwd(), 'node_modules', 'sql.js', 'dist', 'sql-wasm.wasm');

function nowIso(daysOffset = 0) {
  const date = new Date();
  date.setDate(date.getDate() + daysOffset);
  return date.toISOString().slice(0, 19).replace('T', ' ');
}

function getTeamsSeed() {
  return [
    { id: 1, name: 'London', region: 'EMEA' },
    { id: 2, name: 'Chicago', region: 'NA' },
    { id: 3, name: 'Singapore', region: 'APAC' },
  ];
}

function getDealsSeed() {
  return [
    { id: 1, team_id: 1, status: 'closed', amount: 1200.0, closed_date: nowIso(-1) },
    { id: 2, team_id: 1, status: 'closed', amount: 860.0, closed_date: nowIso(-2) },
    { id: 3, team_id: 1, status: 'open', amount: 640.0, closed_date: nowIso(-3) },
    { id: 4, team_id: 1, status: 'closed', amount: 740.0, closed_date: nowIso(-8) },
    { id: 5, team_id: 1, status: 'closed', amount: 300.0, closed_date: nowIso(-12) },
    { id: 6, team_id: 2, status: 'closed', amount: 930.0, closed_date: nowIso(-1) },
    { id: 7, team_id: 2, status: 'open', amount: 500.0, closed_date: nowIso(-1) },
    { id: 8, team_id: 3, status: 'closed', amount: 1120.0, closed_date: nowIso(-2) },
    { id: 9, team_id: 3, status: 'closed', amount: 910.0, closed_date: nowIso(-14) },
  ];
}

function toSqliteOffset(value, unit) {
  const safeValue = Number.isFinite(value) ? value : 0;
  const normalizedUnit = String(unit).toLowerCase();
  const sqliteUnit =
    normalizedUnit === 'week' || normalizedUnit === 'weeks'
      ? 'weeks'
      : normalizedUnit === 'day' || normalizedUnit === 'days'
        ? 'days'
        : normalizedUnit === 'month' || normalizedUnit === 'months'
          ? 'months'
          : normalizedUnit === 'year' || normalizedUnit === 'years'
            ? 'years'
            : normalizedUnit === 'hour' || normalizedUnit === 'hours'
              ? 'hours'
              : normalizedUnit === 'minute' || normalizedUnit === 'minutes'
                ? 'minutes'
                : normalizedUnit === 'second' || normalizedUnit === 'seconds'
                  ? 'seconds'
                  : 'days';
  return `${safeValue >= 0 ? '+' : '-'}${Math.abs(safeValue)} ${sqliteUnit}`;
}

function normalizeForSqlite(sql) {
  let text = String(sql || '')
    .replace(
      /NOW\(\)\s*-\s*INTERVAL\s*'(-?\d+)\s*(day|days|week|weeks|month|months|year|years|hour|hours|minute|minutes|second|seconds)'(?:\:\:interval)?/gi,
      (_match, count, unit) => `datetime('now','${toSqliteOffset(-Number(count), unit)}')`,
    )
    .replace(
      /NOW\(\)\s*\+\s*INTERVAL\s*'(-?\d+)\s*(day|days|week|weeks|month|months|year|years|hour|hours|minute|minutes|second|seconds)'(?:\:\:interval)?/gi,
      (_match, count, unit) => `datetime('now','${toSqliteOffset(Number(count), unit)}')`,
    )
    .replace(/DATE_TRUNC\('week',\s*NOW\(\)\)/gi, "DATE('now','weekday 1','-7 days')")
    .replace(/NOW\(\)/gi, "datetime('now')");
  return text;
}

function isSelectOnly(sql) {
  const trimmed = String(sql || '').trim();
  const lowered = trimmed.toLowerCase();
  if (!lowered.startsWith('select') && !lowered.startsWith('with')) {
    throw new Error('Only SELECT queries are allowed');
  }
  const semicolonIndex = trimmed.lastIndexOf(';');
  if (semicolonIndex !== -1 && semicolonIndex !== trimmed.length - 1) {
    throw new Error('Only one SQL statement is allowed');
  }
  return trimmed;
}

function rowsFromExecution(rows) {
  if (!rows || !rows.length) return [];
  const result = rows[0];
  const columns = result.columns || [];
  return (result.values || []).map((row) => {
    const obj = {};
    columns.forEach((col, idx) => {
      obj[col] = row[idx];
    });
    return obj;
  });
}

function setCors(res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,POST,OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
}

function sendJson(res, status, payload) {
  setCors(res);
  res.statusCode = status;
  res.setHeader('Content-Type', 'application/json');
  res.end(JSON.stringify(payload));
}

function readJsonBody(req) {
  return new Promise((resolve, reject) => {
    let body = '';
    req.on('data', (chunk) => {
      body += chunk;
      if (body.length > 1_000_000) {
        req.destroy();
        reject(new Error('Request payload too large'));
      }
    });
    req.on('end', () => {
      if (!body) {
        resolve({});
        return;
      }
      try {
        resolve(JSON.parse(body));
      } catch (error) {
        reject(error);
      }
    });
    req.on('error', reject);
  });
}

async function start() {
  if (!fs.existsSync(WASM_PATH)) {
    throw new Error(`Missing sql.js wasm file at ${WASM_PATH}`);
  }
  const sql = await initSqlJs({
    wasmBinary: fs.readFileSync(WASM_PATH),
  });
  fs.mkdirSync(path.dirname(DB_PATH), { recursive: true });

  let db;
  if (fs.existsSync(DB_PATH)) {
    db = new sql.Database(fs.readFileSync(DB_PATH));
  } else {
    db = new sql.Database();
  }

  db.exec(`
    CREATE TABLE IF NOT EXISTS teams (
      id INTEGER PRIMARY KEY,
      name TEXT NOT NULL UNIQUE,
      region TEXT
    );
    CREATE TABLE IF NOT EXISTS deals (
      id INTEGER PRIMARY KEY,
      team_id INTEGER NOT NULL,
      status TEXT NOT NULL,
      amount REAL NOT NULL,
      closed_date TEXT NOT NULL,
      FOREIGN KEY(team_id) REFERENCES teams(id)
    );
  `);

  const teamStmt = db.prepare('INSERT OR IGNORE INTO teams (id, name, region) VALUES (?, ?, ?)');
  getTeamsSeed().forEach((team) => {
    teamStmt.run([team.id, team.name, team.region]);
  });
  teamStmt.free();

  const dealStmt = db.prepare(
    'INSERT OR IGNORE INTO deals (id, team_id, status, amount, closed_date) VALUES (?, ?, ?, ?, ?)',
  );
  getDealsSeed().forEach((deal) => {
    dealStmt.run([deal.id, deal.team_id, deal.status, deal.amount, deal.closed_date]);
  });
  dealStmt.free();

  fs.writeFileSync(DB_PATH, Buffer.from(db.export()));

  const server = http.createServer(async (req, res) => {
    const { pathname } = new URL(req.url || '/', `http://${req.headers.host || 'localhost'}`);
    if (req.method === 'OPTIONS') {
      setCors(res);
      res.statusCode = 200;
      res.end();
      return;
    }

    if (pathname === '/api/health' && req.method === 'GET') {
      try {
        const counts = rowsFromExecution(
          db.exec(`
            SELECT
              (SELECT COUNT(*) FROM teams) AS team_count,
              (SELECT COUNT(*) FROM deals) AS deal_count,
              (SELECT ROUND(SUM(amount), 2) FROM deals) AS total_amount
          `),
        )[0] || {};
        sendJson(res, 200, {
          status: 'ok',
          path: DB_PATH,
          ...counts,
        });
      } catch (error) {
        sendJson(res, 500, { error: error.message });
      }
      return;
    }

    if (pathname === '/api/schema' && req.method === 'GET') {
      try {
        const rows = rowsFromExecution(
          db.exec(`SELECT name, sql FROM sqlite_master WHERE type='table' AND name IN ('deals', 'teams') ORDER BY name;`),
        );
        const schema = rows.map((row) => row.sql).filter(Boolean).join('\n');
        sendJson(res, 200, {
          schema,
          tables: rows.map((row) => row.name),
        });
      } catch (error) {
        sendJson(res, 500, { error: error.message });
      }
      return;
    }

    if (pathname === '/api/sql' && req.method === 'POST') {
      try {
        const body = await readJsonBody(req);
        const requestedSql = String(body.sql || '').trim();
        if (!requestedSql) {
          sendJson(res, 400, { error: 'Missing sql payload' });
          return;
        }

        const query = normalizeForSqlite(isSelectOnly(requestedSql));
        const rows = rowsFromExecution(db.exec(query));
        sendJson(res, 200, { rows });
      } catch (error) {
        sendJson(res, 500, { error: error.message });
      }
      return;
    }

    sendJson(res, 404, { error: 'Not found' });
  });

  server.listen(PORT, () => {
    console.log(`CRM SQLite API running on http://127.0.0.1:${PORT}`);
  });
}

start().catch((error) => {
  console.error('Failed to start SQLite API:', error.message);
  process.exit(1);
});
