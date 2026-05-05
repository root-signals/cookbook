# Python Integration Guide

## Installation

```bash
pip install scorable
```

## Basic Usage

### Synchronous

```python
from scorable import Scorable

client = Scorable(api_key="your-api-key")
result = client.judges.run(
    judge_id="judge-id-here",
    request="INPUT to the LLM (optional)",
    response="OUTPUT from the LLM (required)"
)

# Access results
print(result.evaluator_results[0].score)  # Score: 0-1
print(result.evaluator_results[0].justification)  # String
```

### Async

```python
from scorable import Scorable

client = Scorable(api_key="your-api-key", run_async=True)
result = await client.judges.arun(
    judge_id="judge-id-here",
    request="INPUT to the LLM",
    response="OUTPUT from the LLM"
)
```

## Multi-Turn Conversations

For evaluating entire conversation flows:

```python
from scorable import Scorable
from scorable.multiturn import Turn

client = Scorable(api_key="your-api-key")
turns=[
    Turn(role="user", content="Hello, I need help with my order"),
    Turn(role="assistant", content="I'd be happy to help! What's your order number?"),
    Turn(role="user", content="It's ORDER-12345"),
    Turn(
        role="assistant",
        content="{'order_number': 'ORDER-12345', 'status': 'shipped', 'eta': 'Jan 20'}",
        tool_name="order_lookup",
    ),
    Turn(
        role="assistant",
        content="I found your order. It's currently in transit.",
    )
]
result = client.judges.run(judge_id="judge-id-here", turns=turns)
```

#### RAG (Retrieval Augmented Generation)

**If you identify the application uses RAG (Retrieval Augmented Generation)**, you MUST include the `contexts` parameter.

```python
eval_result = client.judges.run(
    judge_id="judge-id",
    request="User question",
    response="LLM response",
    contexts=["retrieved doc 1", "retrieved doc 2", ...]  # REQUIRED for RAG
)
```

### Optional parameters for execute call. Use ONLY if relevant to the evaluation.
- \`contexts\` parameter: If a RAG setup is used, a list of retrieved context snippets ["context snippet 1", "context snippet 2"] to evaluate the response against (optional)
- \`user_id\` parameter: The user id of the user who is interacting with the application (optional)
- \`tags\` parameter: Tag the evaluation for easier filtering and analysis, like production, development (optional)
- \`expected_output\` parameter: The expected output of the response (optional)

## Common Patterns

### Pattern 1: Development (100% evaluation)

```python
response = llm.generate(prompt)
eval_result = client.judges.run(judge_id, request=prompt, response=response)
log_evaluation(eval_result)
```

### Pattern 2: Production with Sampling (10% evaluation)

```python
import random

response = llm.generate(prompt)
if random.random() < 0.1:  # 10% sampling
    eval_result = client.judges.run(judge_id, request=prompt, response=response)
    store_evaluation_in_db(eval_result)
```

### Pattern 3: Batch Evaluation

See https://docs.scorable.ai/usage/cookbooks/batch-evaluation


## Integration Points

- Insert evaluation code where LLM outputs are generated (e.g., after OpenAI response calls)
- `response` parameter: The text you want to evaluate (required)
- `request` parameter: The input that prompted the response (optional but recommended)
- Use actual variables from your code, not static strings

## Result Format

Results are Pydantic models:

```python
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

Each evaluator returns:
- `score`: Float between 0-1 (higher is better)
- `justification`: Natural language explanation
