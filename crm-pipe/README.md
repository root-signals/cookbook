# CRM Query Pipeline Workbench

A browser-based workbench that turns natural-language CRM questions into SQL, executes them, and returns a plain-English answer. The pipeline has three stages:

1. **Intent parsing** — extract intent, filters, and aggregation from a question
2. **SQL generation** — produce a SQLite query from the parsed spec
3. **Answer generation** — summarise the result set in one sentence

Works fully offline with a mock pipeline, or connect an OpenAI API key to use GPT for each stage.

## Quick start

```bash
npm install

# Terminal 1 — start the SQLite API server (seeds sample data automatically)
npm run db:serve

# Terminal 2 — start the Vite dev server
npm run dev
```

Open http://127.0.0.1:5173 in your browser.

## Headless test

Run the mock pipeline end-to-end without a browser:

```bash
node scripts/headless-test.mjs
```

## Using OpenAI mode

In the UI, switch **LLM mode** to "OpenAI (GPT)", paste your API key, and pick a model (default: `gpt-5.4-mini`). The key is stored in localStorage and never sent to any server other than OpenAI.
