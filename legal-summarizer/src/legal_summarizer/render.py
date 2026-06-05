"""Markdown report renderer for a legal-summarizer run."""

from __future__ import annotations

from typing import Any


def render_markdown(output: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Legal Summarizer Run")
    lines.append("")
    lines.append(f"**Company URL:** {output['company_url']}")
    lines.append("")

    score = output.get("faithfulness")
    if score:
        lines.append(f"## Faithfulness to Citations: {score['score']:.3f}")
        lines.append("")
        if score.get("execution_log_id"):
            lines.append(f"**Execution log:** `{score['execution_log_id']}`")
            lines.append("")
        lines.append("### Justification")
        lines.append("")
        lines.append(score["justification"] or "_(no justification returned)_")
        lines.append("")
    elif output.get("faithfulness_error"):
        lines.append("## Faithfulness Evaluation Failed")
        lines.append("")
        lines.append(f"```\n{output['faithfulness_error']}\n```")
        lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(output["summary"])
    lines.append("")

    fetched = output.get("fetched_urls", [])
    if fetched:
        lines.append("## Fetched URLs")
        lines.append("")
        for url in fetched:
            lines.append(f"- {url}")
        lines.append("")

    return "\n".join(lines)
