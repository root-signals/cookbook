# Other Languages Integration Guide

For languages without an official SDK, use the REST API directly via HTTP requests.

## cURL Template (simple)

```bash
curl 'https://api.scorable.ai/v1/judges/{judge_id}/execute/' \
  -H 'authorization: Api-Key your-api-key' \
  -H 'content-type: application/json' \
  --data-raw '{"response":"LLM output here","request":"User input here"}'
```

## cURL Template (multi-turn)

```bash
curl 'https://api.scorable.ai/v1/judges/{judge_id}/execute/' \
  -H 'authorization: Api-Key your-api-key' \
  -H 'content-type: application/json' \
  --data-raw '{
  "turns": [
    {
      "role": "user",
      "content": "Hello, I need help with my order"
    },
    {
      "role": "assistant",
      "content": "I'\''d be happy to help! What'\''s your order number?"
    },
    {
      "role": "user",
      "content": "It'\''s ORDER-12345"
    },
    {
      "role": "assistant",
      "content": "{\"order_number\": \"ORDER-12345\", \"status\": \"shipped\", \"eta\": \"Jan 20\"}",
      "tool_name": "order_lookup"
    },
    {
      "role": "assistant",
      "content": "I found your order. It is currently in transit and should arrive by Jan 20."
    }
  ]
}'
```

## Integration Points

- Insert evaluation code where LLM outputs are generated
- `response` parameter: The text you want to evaluate (required)
- `request` parameter: The input that prompted the response (optional but recommended)
- Use actual variables from your code, not static strings

## Result Format

```json
{
  "evaluator_results": [
    {
      "evaluator_name": "Accuracy",
      "score": 0.85,
      "justification": "The response correctly identifies..."
    }
  ]
}
```

#### RAG (Retrieval Augmented Generation)

**If you identify the application uses RAG (Retrieval Augmented Generation)**, you MUST include the `contexts` parameter.

```bash
curl 'https://api.scorable.ai/v1/judges/{judge_id}/execute/' \
  -H 'authorization: Api-Key your-api-key' \
  -H 'content-type: application/json' \
  --data-raw '{"response":"LLM output here","request":"User input here","contexts":["retrieved doc 1", "retrieved doc 2", ...]}'
```

### Optional parameters for execute call. Use ONLY if relevant to the evaluation.
- \`contexts\` parameter: If a RAG setup is used, a list of retrieved context snippets ["context snippet 1", "context snippet 2"] to evaluate the response against (optional)
- \`user_id\` parameter: The user id of the user who is interacting with the application (optional)
- \`tags\` parameter: Tag the evaluation for easier filtering and analysis, like production, development (optional)
- \`expected_output\` parameter: The expected output of the response (optional)

Each evaluator returns:
- `score`: Float between 0-1 (higher is better)
- `justification`: Natural language explanation

## Adapting to Your Language

Use this template with your language's HTTP client library