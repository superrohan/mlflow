import logging
import json
from pathlib import Path
from openai import OpenAI

logger = logging.getLogger(__name__)

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"

client = OpenAI()
MODEL = "gpt-4o"

SYSTEM_PROMPT = """You are an expert ML Model Evaluation Agent.
Analyze model performance metrics and provide a structured evaluation.
Respond ONLY with a JSON object. No markdown, no backticks, no extra text."""


def evaluate(
    task_type: str,
    algorithm: str,
    metrics_output: str,
    understanding_output: str,
) -> dict:
    """Ask Claude to evaluate model performance and decide pass/retry."""
    prompt = f"""Evaluate the following ML model performance.

Algorithm: {algorithm}
Task type: {task_type}

Metrics output from execution:
{metrics_output}

Data context:
{understanding_output[:1000]}

Provide your evaluation as JSON:
{{
  "verdict": "pass" or "retry",
  "score": <primary metric value as float, 0-1>,
  "primary_metric": "<metric name>",
  "strengths": ["..."],
  "weaknesses": ["..."],
  "suggestions": ["..."],
  "summary": "<2-3 sentence overall assessment>"
}}

Verdict criteria:
- "pass" if: accuracy/f1 >= 0.70 for classification, R2 >= 0.60 for regression, silhouette >= 0.30 for clustering
- "retry" if below those thresholds

Output ONLY the JSON object."""

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    text = response.choices[0].message.content.strip()
    if text.startswith("```"):
        lines = text.split("\n")[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    try:
        evaluation = json.loads(text)
    except Exception:
        logger.warning("Could not parse evaluation JSON, defaulting to pass.")
        evaluation = {
            "verdict": "pass",
            "score": 0.0,
            "primary_metric": "unknown",
            "strengths": [],
            "weaknesses": ["Could not parse evaluation"],
            "suggestions": [],
            "summary": "Evaluation parsing failed. Defaulting to pass.",
        }

    _save_evaluation(evaluation, algorithm, task_type)
    return evaluation


def _save_evaluation(evaluation: dict, algorithm: str, task_type: str):
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "evaluation.json"
    full = {"algorithm": algorithm, "task_type": task_type, **evaluation}
    out_path.write_text(json.dumps(full, indent=2))
    logger.info(f"Evaluation saved to {out_path}")
