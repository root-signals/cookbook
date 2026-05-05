# TypeScript/JavaScript Integration Guide

## Installation

```bash
npm install @root-signals/scorable
```

## Basic Usage

```typescript
import { Scorable } from "@root-signals/scorable";

const client = new Scorable({ apiKey: "your-api-key" });

const result = await client.judges.execute(
  "judge-id-here",
  {
    request: "What is the refund policy?",
    response: "You can return items within 30 days.",
  }
);

// Access results
console.log(result.evaluator_results[0].score);
console.log(result.evaluator_results[0].justification);
```

## Multi-Turn Conversations

```typescript
const result = await client.judges.execute(
  "judge-id-here",
  {
    turns: [
      {
        role: 'user',
        content: 'Hello, I need help with my order'
      },
      {
        role: 'assistant',
        content: "I'd be happy to help! What's your order number?"
      },
      {
        role: 'user',
        content: "It's ORDER-12345"
      },
      {
        role: 'assistant',
        content: JSON.stringify({order_number: 'ORDER-12345', status: 'shipped', eta: 'Jan 20'}),
        tool_name: 'order_lookup' // Optional: name of tool used
      },
      {
        role: 'assistant',
        content: 'I found your order. It is currently in transit and should arrive by Jan 20.'
      }
    ]
  }
);
```

#### RAG (Retrieval Augmented Generation)

**If you identify the application uses RAG (Retrieval Augmented Generation)**, you MUST include the `contexts` parameter.

```typescript
const result = await client.judges.execute(
  "judge-id-here",
  {
    request: "User question",
    response: "LLM response",
    contexts: ["retrieved doc 1", "retrieved doc 2", ...], // REQUIRED for RAG
  }
);
```

Contexts parameter is available in all SDKs and in the API.

### Optional parameters for execute call. Use ONLY if relevant to the evaluation.
- \`contexts\` parameter: If a RAG setup is used, a list of retrieved context snippets ["context snippet 1", "context snippet 2"] to evaluate the response against (optional)
- \`user_id\` parameter: The user id of the user who is interacting with the application (optional)
- \`tags\` parameter: Tag the evaluation for easier filtering and analysis, like production, development (optional)
- \`expected_output\` parameter: The expected output of the response (optional)

## Result Format

```typescript
{
  evaluator_results: [
    {
      evaluator_name: "Accuracy",
      score: 0.85,
      justification: "The response correctly identifies..."
    }
  ]
}
```

Each evaluator returns:
- `score`: Float between 0-1 (higher is better)
- `justification`: Natural language explanation
