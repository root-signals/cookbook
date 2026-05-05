---
name: scorable-integration
description: Integrate Scorable LLM-as-a-Judge evaluators into applications with LLM interactions. Use when users want to add evaluation, guardrails, or quality monitoring to their LLM-powered applications. Also use when users mention Scorable, judges, LLM evaluation, or safeguarding applications.
---

# Add Scorable LLM-as-a-Judge to Your Application

These instructions guide you through creating LLM evaluation judges with Scorable and integrating them into your codebase.
Scorable is a tool for creating LLM-as-a-Judge based evaluators for safeguarding applications. Judge is the Scorable term for grouping evaluations from different metrics (Helpfulness, Policy Adherence, etc...)

## Execution Contract

You are responsible for completing Scorable setup and integration end-to-end in as few turns as possible.

- You MUST analyze the codebase for LLM interaction points.
- You MUST install and use Scorable CLI directly.
- You MUST execute judge generation commands yourself.
- You MUST integrate judge execution into code yourself.
- You MUST run verification checks after changes.
- You MUST update project documentation for usage.
- You MUST NOT delegate technical steps to the user except where explicitly required like setting up the API key if not using a temporary key.
- You MUST continue until implementation is complete or a hard blocker is reached.

## Overview

Your role is to:
1. **Analyze the codebase** to identify LLM interactions
2. **Create judges via the Scorable CLI** to evaluate those interactions (or use an existing judge ID if provided)
3. **Integrate judge execution** into the code at appropriate points
4. **Provide usage documentation** for the evaluation setup

**Note:** These instructions work for both creating new judges from scratch and integrating existing judges. If the user provides a judge ID, you can skip the judge creation step (Step 3) and proceed directly to integration (Step 4).

## Step 0: Explain the process

Before performing any analysis or technical steps, pause and clearly brief the user on what is about to happen.
Explain that you will:
- Analyze the codebase to identify LLM interactions
- Create judges via the Scorable CLI to evaluate those interactions
- Integrate judge execution into the code at appropriate points
- Provide usage documentation for the evaluation setup

---

## Step 1: Analyze the Application

Examine the codebase to understand:
- What LLM interactions exist (prompts, completions, agent calls)
- What the application does at each interaction point
- Which interactions are most critical to evaluate

If multiple LLM interactions exist, help the user prioritize. Recommend starting with the most critical one first.

---

## Step 2: Install Scorable CLI & Authenticate

First, install the Scorable CLI:

```bash
curl -sSL https://scorable.ai/cli/install.sh | sh
```

Or with npm:

```bash
npm install -g @root-signals/scorable-cli
```

Or run without installing via npx:

```bash
npx @root-signals/scorable-cli judge list
```

Then ask the user which authentication option they prefer:

### Option A: Permanent API Key (Recommended)

Direct them to: https://scorable.ai/api-key-setup to create an API key, then set it via the CLI:

```bash
scorable auth set-key
# paste the key when prompted

# or alternatively:
scorable auth set-key <your-api-key>
```

**Security:** Instruct the user to use environment variables or the project's secret management. Use existing `.env` files if available or ask user to save it as environment variable. Do not ask the user to paste the key into this session.

---

### Option B: Temporary API Key (Testing Only)

Get a free demo key (no registration required):

```bash
scorable auth demo-key
```

Warn the user appropriately that:
- Judges created with it will be public and visible to everyone
- The key only works for a limited time
- For private judges, they should create a permanent key at https://scorable.ai/register

Remember also the `api_token` field. It is used in the URL parameters for the judge URL, not in any other context.

---

### Option C: Existing API Key

If they have an account: https://scorable.ai/settings/api-keys

Set via CLI:

```bash
scorable auth set-key <your-api-key>
```

Or use an environment variable:

```bash
export SCORABLE_API_KEY="sk-your-api-key"
```

---

## Step 3: Generate a Judge

**Note:** If the user has already provided a judge ID (e.g., in their message), you can skip this step and proceed directly to Step 4 (Integration).

**Note:** After the user has authenticated, take control back and run the following commands yourself.

Generate a judge using the Scorable CLI with a detailed `intent` string.

### Intent String Guidelines:
- Describe the application context and what you're evaluating
- Mention the specific execution point (stage name)
- Include critical quality dimensions you care about
- Add examples, documentation links, mandatory tool calls, or policies if relevant
- Be specific and detailed (multiple sentences/paragraphs are good)
- Code level details like frameworks, libraries, etc. do not need to be mentioned

**Example:**
```bash
scorable judge generate \
  --intent "An email automation system that creates summary emails using an LLM based on database query results and user input. Evaluate the LLM output for: accuracy in summarizing data, appropriate tone for the audience, inclusion of all key information from queries, proper formatting, and absence of hallucinations. The system is used for customer-facing communications." \
  --visibility private \
  --reasoning-effort medium
```

Use `--visibility public` if using a temporary API key.

**Optional fields:**
- `enable_context_aware_evaluators`: Set to true if the application interaction uses RAG (document chunks) that are relevant and can be extracted to the evaluation (hallucinations, context drift, etc.).

Note that this can take up to 2 minutes to complete.

### Handling CLI Responses:

The CLI may return:

**1. `missing_context_from_system_goal`** - Additional context needed:
→ Ask the user for these details (if not evident from the code base), then re-run with the additional context:

```bash
scorable judge generate \
  --intent "..." \
  --judge-id <existing-judge-id> \
  --extra-contexts '{"target_audience":"Enterprise customers"}'
```

**2. `multiple_stages`** - Judge detected multiple evaluation points:
```json
{
  "error_code": "multiple_stages",
  "stages": ["Stage 1", "Stage 2", "Stage 3"]
}
```
→ Ask the user which stage to focus on, or if they have a custom stage name. Each judge evaluates one stage. You can create additional judges later for other stages. Re-run with `--stage "<stage name>"`.

**3. Success** - Judge created:
```json
{
  "judge_id": "abc123...",
  "evaluator_details": [...]
}
```
→ Proceed to integration.

---

## Step 4: Integrate Judge Execution

Add code to evaluate LLM outputs at the appropriate execution point(s). If the codebase is using a framework, check if there are integration instructions in Scorable docs (using curl is enough): https://docs.scorable.ai/llms.txt

### Language-Specific Integration

Choose the appropriate integration guide based on the codebase language:

- **Python**: See [references/python.md](references/python.md) for installation, sync/async usage, multi-turn conversations, and common patterns
- **TypeScript/JavaScript**: See [references/typescript.md](references/typescript.md) for npm installation and usage examples
- **Other languages**: See [references/other-languages.md](references/other-languages.md) for REST API integration via cURL template

### Integration Points

- Insert evaluation code where LLM outputs are generated (for example after an OpenAI responses call)
- `response` parameter: The text you want to evaluate (required)
- `request` parameter: The input that prompted the response (optional but recommended)
- Use actual variables from your code, not static strings

### Multi-Turn / Agent + Tool Calls evaluation

If a multi-turn conversation is detected, use the multi-turn format to evaluate the entire conversation flow. This may also include tool calls. Confirm from user if multi-turn evaluation would suit their needs. See language-specific guides for details.

### RAG, optional parameters, and result format

If the application uses RAG, you MUST pass a `contexts` parameter to the judge run. Optional parameters (`user_id`, `tags`, `expected_output`) and the exact result shape are documented in the language-specific reference files linked above.

---

## Step 5: Provide Next Steps

After integration:

1. **Ask about additional judges**: If multiple stages were identified, ask if the user wants to create judges for other stages
2. **Discuss evaluation strategy**:
   - Should every LLM call be evaluated or sampled (e.g., 10%)?
   - Should scores be stored in a database for analysis?
   - Should specific actions trigger based on scores (e.g., alerts for low scores)?
   - Batch evaluation vs real-time evaluation?
3. **Provide judge details**:
   - Judge URL: `https://scorable.ai/judge/{judge_id}`
     - If you used a temporary key, you must include its counterpart api_token base64 encoded in the url as a query parameter: https://scorable.ai/judge/{judge_id}?token={base64 encoded temporary api_token}
   - How to view results in the Scorable overview (https://scorable.ai/overview)
   - If temporary key was used, a note that it only works for a certain amount of time and they should create an account with a permanent key
4. **CLI usage**: Inform the user that judges and evaluators can be inspected, run, and managed — including execution logs — via the Scorable CLI
5. **Link to docs**: https://docs.scorable.ai
   - For agentic workflows with tool calls or multi-turn conversations, link to https://docs.scorable.ai/usage/usage/judges#multi-turn-conversations


---

## Key Implementation Notes

- **Install SDK first**: Check which dependency management system is used and install the appropriate package.
- **Store API keys securely**: Use environment variables, not hardcoded strings
- **Handle errors gracefully**: Evaluation failures shouldn't break your application
- **Start simple**: Evaluate one stage first, then expand
- **Sampling for production**: 5-10% sampling reduces costs while maintaining visibility
- **Non-blocking**: The evaluation should not block the main thread and not slow down the application
- **Common patterns**: See language-specific reference files for integration patterns (development, production sampling, batch evaluation)
