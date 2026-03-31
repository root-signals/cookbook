#!/usr/bin/env node
import { exampleScenario } from '../src/lib/pipeline.js';
import { executeSql, SQL_MODES } from '../src/lib/sqlExecutor.js';
import { generateStage1, generateStage2, generateStage3, LLM_MODES } from '../src/lib/llmClient.js';

function printSection(label, body) {
  console.log(`\n${label}`);
  console.log('-'.repeat(label.length));
  console.log(body);
}

async function main() {
  const runtime = { mode: LLM_MODES.MOCK };
  const request = exampleScenario.stage1Request;
  const schema = exampleScenario.schema;

  const stage1 = await generateStage1(request, runtime);
  const stage1Response = JSON.stringify(stage1, null, 2);
  const stage2Response = await generateStage2(stage1Response, schema, runtime);
  const resultset = await executeSql({
    mode: SQL_MODES.MOCK,
    sql: stage2Response,
    querySpec: stage1Response,
  });
  const stage3Response = await generateStage3(stage1Response, JSON.stringify(resultset), runtime);

  const payloads = {
    stage1: { request, response: stage1Response },
    stage2: { request: stage1Response, schema, response: stage2Response },
    stage3: { query: stage1Response, resultset: JSON.stringify(resultset), response: stage3Response },
  };

  printSection('Stage 1 output', JSON.stringify({ request, response: stage1 }, null, 2));
  printSection('Stage 2 output', JSON.stringify({ request: stage1Response, schema, response: stage2Response }, null, 2));
  printSection('Stage 3 output', JSON.stringify({ query: stage1Response, resultset, response: stage3Response }, null, 2));
  printSection('Evaluator bundle', JSON.stringify(payloads, null, 2));
}

main().catch((error) => {
  console.error(error.message);
  process.exit(1);
});
