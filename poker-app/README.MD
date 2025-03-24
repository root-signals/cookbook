# 🃏 Poker CLI App

This is a simple CLI application that allows you to play poker against an LLM! 

It uses a custom evaluator made with Root Signals to judge the game stages and give a score and justifications for the score.

The evaluator ensures the quality of the game play does not decrease over time when changing the system prompt or the model.

## 🎲 How it works

The Root Signals evaluator takes the game state as input and outputs a score and justifications for the score.

The evaluator result and justification is then saved to the database. The data is used as additional context for the LLM to improve the game play. 📈

## 🚀 Setup

### Get Your API Key

[Sign up & create a key](https://app.rootsignals.ai/settings/api-keys) or [generate a temporary key](https://app.rootsignals.ai/demo-user)

```bash
uv sync
touch .env
# Add OPENAI_API_KEY to .env
# Add ROOTSIGNALS_API_KEY to .env
```

## Run

```bash
UV_ENV_FILE=.env uv run poker.py
```

