import React, { useEffect, useMemo, useState } from 'react';
import {
  exampleScenario,
  generateAnswerFromQueryResult,
  generateIntentFromRequest,
  generateSqlFromSpec,
} from './lib/pipeline';
import { executeSql, SQL_MODES } from './lib/sqlExecutor';
import { generateStage1, generateStage2, generateStage3, LLM_MODES, resolveRuntimeConfig } from './lib/llmClient';

const localStoragePrefix = 'crm-pipeline';
const seedSchema = exampleScenario.schema;
const starterQuestion = exampleScenario.stage1Request;
const starterSpec = JSON.stringify(generateIntentFromRequest(starterQuestion), null, 2);
const starterRows = exampleScenario.resultset;
const starterSql = generateSqlFromSpec(starterSpec, seedSchema);
const starterAnswer = generateAnswerFromQueryResult(starterSpec, JSON.stringify(starterRows));

const sampleQuestions = [
  starterQuestion,
  'How many deals closed last week by the London team?',
  'What is the average deal size for open deals in Chicago?',
];

const defaultRuntime = {
  llmMode: LLM_MODES.MOCK,
  openaiKey: '',
  openaiModel: 'gpt-4.1-mini',
  sqlMode: SQL_MODES.SQLITE,
};

function readRuntimeFromStorage() {
  if (typeof window === 'undefined') return defaultRuntime;
  return {
    llmMode: window.localStorage.getItem(`${localStoragePrefix}.llmMode`) || defaultRuntime.llmMode,
    openaiKey: window.localStorage.getItem(`${localStoragePrefix}.openaiKey`) || defaultRuntime.openaiKey,
    openaiModel: window.localStorage.getItem(`${localStoragePrefix}.openaiModel`) || defaultRuntime.openaiModel,
    sqlMode: window.localStorage.getItem(`${localStoragePrefix}.sqlMode`) || defaultRuntime.sqlMode,
  };
}

function saveRuntimeToStorage(runtime) {
  if (typeof window === 'undefined') return;
  window.localStorage.setItem(`${localStoragePrefix}.llmMode`, runtime.llmMode);
  window.localStorage.setItem(`${localStoragePrefix}.openaiKey`, runtime.openaiKey);
  window.localStorage.setItem(`${localStoragePrefix}.openaiModel`, runtime.openaiModel);
  window.localStorage.setItem(`${localStoragePrefix}.sqlMode`, runtime.sqlMode);
}

function CopyButton({ text, label }) {
  return (
    <button type="button" onClick={() => navigator.clipboard?.writeText(text)}>
      {label}
    </button>
  );
}

function Card({ title, children }) {
  return (
    <section className="stage-panel">
      <h2>{title}</h2>
      {children}
    </section>
  );
}

function formatRows(rows) {
  return JSON.stringify(Array.isArray(rows) ? rows : [], null, 2);
}

function ResultTable({ rows }) {
  if (!Array.isArray(rows) || rows.length === 0) {
    return <p className="copy-status">No rows returned.</p>;
  }

  const columns = Object.keys(rows[0] || {});
  return (
    <div className="results">
      <div className="result-summary">{rows.length} row(s)</div>
      <div className="result-card">
        <table>
          <thead>
            <tr>
              {columns.map((column) => (
                <th key={column}>{column}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, rowIndex) => (
              <tr key={rowIndex}>
                {columns.map((column) => (
                  <td key={`${rowIndex}-${column}`}>{String(row[column] ?? '')}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function App() {
  const [question, setQuestion] = useState(starterQuestion);
  const [runtime, setRuntime] = useState(readRuntimeFromStorage());
  const [schema, setSchema] = useState(seedSchema);
  const [spec, setSpec] = useState(starterSpec);
  const [sql, setSql] = useState(starterSql);
  const [rows, setRows] = useState(starterRows);
  const [answer, setAnswer] = useState(starterAnswer);
  const [dbHealth, setDbHealth] = useState({ status: 'checking' });
  const [pipelineTrace, setPipelineTrace] = useState(null);
  const [status, setStatus] = useState('Ready');
  const [copyStatus, setCopyStatus] = useState('');
  const [isRunning, setIsRunning] = useState(false);

  const currentRuntime = useMemo(() => resolveRuntimeConfig(runtime), [runtime]);
  const sqlMode = useMemo(
    () => (dbHealth?.status === 'ok' ? runtime.sqlMode || SQL_MODES.SQLITE : SQL_MODES.MOCK),
    [dbHealth, runtime.sqlMode],
  );

  const payloads = useMemo(
    () => ({
      stage1: { request: question, response: spec },
      stage2: { request: spec, schema, response: sql },
      stage3: { query: spec, resultset: formatRows(rows), response: answer },
    }),
    [question, spec, schema, sql, rows, answer],
  );

  function setRuntimeField(field, value) {
    setRuntime((prev) => {
      const next = { ...prev, [field]: value };
      saveRuntimeToStorage(next);
      return next;
    });
  }

  function reportStatus(message, clearMs = 2100) {
    setStatus(message);
    if (clearMs > 0) {
      setTimeout(() => {
        setStatus((current) => (current === message ? 'Ready' : current));
      }, clearMs);
    }
  }

  function reportCopy(message) {
    setCopyStatus(message);
    setTimeout(() => setCopyStatus(''), 1300);
  }

  function copyAllPayloads() {
    navigator.clipboard?.writeText(JSON.stringify(payloads, null, 2));
    reportCopy('Evaluator payload copied');
  }

  function loadQuestion(nextQuestion) {
    setQuestion(nextQuestion);
  }

  async function runPipeline() {
    setIsRunning(true);
    setPipelineTrace(null);
    setCopyStatus('');

    try {
      const stage1Response = await generateStage1(question, currentRuntime);
      const stage1Text = JSON.stringify(stage1Response, null, 2);
      const stage2Response = await generateStage2(stage1Text, schema, currentRuntime);

      const queryRows = await executeSql({
        mode: sqlMode,
        sql: stage2Response,
        querySpec: stage1Text,
      });
      const safeRows = Array.isArray(queryRows) ? queryRows : [];

      const stage3Response = await generateStage3(stage1Text, formatRows(safeRows), currentRuntime);

      setSpec(stage1Text);
      setSql(stage2Response);
      setRows(safeRows);
      setAnswer(stage3Response);

      setPipelineTrace({
        mode: runtime,
        stage1Response: stage1Text,
        stage2Response,
        resultset: safeRows,
        stage3Response,
      });
      reportStatus('Pipeline complete');
    } catch (error) {
      setStatus(`Run failed: ${error.message}`);
    } finally {
      setIsRunning(false);
    }
  }

  useEffect(() => {
    let ignore = false;
    async function loadBackendState() {
      try {
        const health = await fetch('/api/health');
        if (!health.ok) {
          throw new Error('health endpoint error');
        }
        const data = await health.json();
        if (!ignore) {
          setDbHealth(data || {});
        }
      } catch {
        if (!ignore) {
          setDbHealth({ status: 'offline' });
        }
      }

      try {
        const schemaRes = await fetch('/api/schema');
        if (!schemaRes.ok) {
          throw new Error('schema endpoint error');
        }
        const data = await schemaRes.json();
        if (!ignore && data?.schema) {
          setSchema(data.schema);
        }
      } catch {
        if (!ignore) {
          setSchema(seedSchema);
        }
      }
    }

    loadBackendState();

    return () => {
      ignore = true;
    };
  }, []);

  return (
    <div className="app">
      <header className="header">
        <h1>CRM Query Pipeline Workbench</h1>
        <p>Ask a question, watch it flow through intent parsing, SQL generation, execution, and answer text.</p>
        <p className={`copy-status ${dbHealth?.status === 'ok' ? 'warn' : ''}`}>
          SQLite service: {dbHealth?.status === 'ok' ? 'connected' : 'using mocked fallback'}
          {dbHealth?.deal_count != null ? ` · ${dbHealth.deal_count} seeded deals` : ''}
        </p>
      </header>

      <section className="stage-panel">
        <h2>Live query workspace</h2>
        <label>
          Natural language request
          <textarea value={question} onChange={(event) => setQuestion(event.target.value)} />
        </label>

        <div className="card-actions">
          <button type="button" onClick={runPipeline} disabled={isRunning}>
            {isRunning ? 'Running...' : 'Run end-to-end'}
          </button>
          <label className="inline-control">
            LLM mode
            <select value={runtime.llmMode} onChange={(event) => setRuntimeField('llmMode', event.target.value)}>
              <option value={LLM_MODES.MOCK}>Mock pipeline</option>
              <option value={LLM_MODES.OPENAI}>OpenAI (GPT)</option>
            </select>
          </label>
          <label className="inline-control">
            OpenAI model
            <input
              value={runtime.openaiModel}
              onChange={(event) => setRuntimeField('openaiModel', event.target.value)}
              disabled={runtime.llmMode !== LLM_MODES.OPENAI}
            />
          </label>
          <label className="inline-control">
            OpenAI key
            <input
              value={runtime.openaiKey}
              onChange={(event) => setRuntimeField('openaiKey', event.target.value)}
              placeholder={runtime.llmMode === LLM_MODES.OPENAI ? 'sk-... (required)' : 'optional in mock'}
              type="password"
            />
          </label>
        </div>

        <div className="card-actions">
          {sampleQuestions.map((sample) => (
            <button key={sample} type="button" onClick={() => loadQuestion(sample)}>
              {sample}
            </button>
          ))}
        </div>
      </section>

      <main className="grid">
        <Card title="Parsed intent/specification">
          <pre>{spec}</pre>
          <div className="card-actions">
            <CopyButton text={spec} label="Copy spec" />
          </div>
        </Card>

        <Card title="Generated SQL against SQLite">
          <pre>{sql}</pre>
          <div className="card-actions">
            <CopyButton text={sql} label="Copy SQL" />
          </div>
        </Card>

        <Card title="Query result set">
          <ResultTable rows={rows} />
          <div className="card-actions">
            <CopyButton text={formatRows(rows)} label="Copy rows" />
          </div>
        </Card>

        <Card title="Final answer">
          <p>{answer || 'Run a query to generate an answer.'}</p>
          <div className="card-actions">
            <CopyButton text={answer} label="Copy answer" />
            <button type="button" onClick={copyAllPayloads}>
              Copy evaluator payloads
            </button>
          </div>
        </Card>
      </main>

      {pipelineTrace && (
        <section className="stage-panel">
          <h2>Execution trace</h2>
          <pre>{JSON.stringify(pipelineTrace, null, 2)}</pre>
        </section>
      )}

      <p className="copy-status">{status}</p>
      {copyStatus && <p className="copy-status">{copyStatus}</p>}
    </div>
  );
}
