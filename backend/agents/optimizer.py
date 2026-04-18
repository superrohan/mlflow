"""Optimization Agent — generates improved ML code for iterations 1-3."""

import json
import logging
from pathlib import Path

from openai import OpenAI

logger = logging.getLogger(__name__)

GENERATED_CODE_DIR = Path(__file__).parent.parent / "generated_code"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"

client = OpenAI()
MODEL = "gpt-4o"

SYSTEM_PROMPT = """You are an expert ML Optimization Engineer. Write clean, executable Python code.

Rules:
- Output ONLY valid Python code. No markdown, no backticks, no explanations.
- ALWAYS use train_test_split(X, y, test_size=0.2, random_state=42).
- Fit ALL preprocessing (scalers, encoders) on TRAIN only — transform both sets.
- Compute ALL metrics on TEST SET ONLY.
- Wrap the final metrics JSON in METRICS_JSON_START / METRICS_JSON_END markers.
- Save the model with pickle to the exact path provided.
- Include self-check: reload model, re-predict on test, warn if metrics diverge.
"""

FIX_SYSTEM_PROMPT = """You are an expert Python debugger. Fix the provided code.
Output ONLY the complete fixed Python code. No explanations, no markdown, no backticks."""

STRATEGY_SELECTION_PROMPT = """You are an ML optimization strategist.
Respond ONLY with a JSON object — no markdown, no extra text:
{
  "algorithm": "GradientBoostingClassifier",
  "reason": "...",
  "hyperparameters": {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 5}
}"""


def _metrics_fields(task_type: str) -> str:
    if "classif" in task_type:
        return (
            '"accuracy": <float>, "f1": <float>, '
            '"precision": <float>, "recall": <float>'
        )
    if "regress" in task_type:
        return '"r2_score": <float>, "rmse": <float>, "mae": <float>'
    return '"silhouette_score": <float>'


def _scale_needed(algo: str) -> bool:
    return any(kw in algo for kw in ("Logistic", "Ridge", "SV", "Linear", "MLP", "KNN"))


def select_next_strategy(
    task_type: str,
    iteration: int,
    optimization_history: list,
    understanding_output: str,
    human_feedback: str = "",
) -> dict:
    """Ask GPT-4o to pick the next algorithm to try based on history."""
    history_summary = json.dumps(
        [
            {
                "iteration": h.get("iteration"),
                "algorithm": h.get("algorithm"),
                "score": h.get("primary_score"),
                "metrics": h.get("metrics", {}),
            }
            for h in optimization_history
        ],
        indent=2,
    )

    already_tried = [h.get("algorithm") for h in optimization_history]
    feedback_section = (
        f"\nHuman feedback / instructions:\n{human_feedback}"
        if human_feedback.strip() else ""
    )

    prompt = f"""You are optimizing an ML pipeline. Choose the NEXT algorithm to try for iteration {iteration}.

Task type: {task_type}
Already tried: {already_tried}

Results so far:
{history_summary}

Data characteristics:
{understanding_output[:1500]}
{feedback_section}

Reason about what has worked, what hasn't, and what would be most likely to improve performance.
Pick a DIFFERENT algorithm than those already tried. Consider ensembles, boosting, or tuned variants.
Respond ONLY with the JSON object."""

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=512,
        messages=[
            {"role": "system", "content": STRATEGY_SELECTION_PROMPT},
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
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Could not parse optimizer strategy JSON, using fallback.")
        if "classif" in task_type:
            return {
                "algorithm": "GradientBoostingClassifier",
                "reason": "Strong ensemble fallback",
                "hyperparameters": {
                    "n_estimators": 300, "learning_rate": 0.05,
                    "max_depth": 5, "random_state": 42,
                },
            }
        if "regress" in task_type:
            return {
                "algorithm": "GradientBoostingRegressor",
                "reason": "Strong ensemble fallback",
                "hyperparameters": {
                    "n_estimators": 300, "learning_rate": 0.05,
                    "max_depth": 5, "random_state": 42,
                },
            }
        return {
            "algorithm": "AgglomerativeClustering",
            "reason": "Alternative clustering fallback",
            "hyperparameters": {"n_clusters": 4},
        }


def generate_optimization_code(
    dataset_path: str,
    task_type: str,
    target_column: str,
    iteration: int,
    strategy: dict,
    optimization_history: list,
    human_feedback: str = "",
) -> str:
    """Return code string for the given iteration."""
    algo = strategy["algorithm"]
    hyperparams = json.dumps(strategy.get("hyperparameters", {}))
    reason = strategy.get("reason", "")
    model_path = str(OUTPUTS_DIR / f"model_iter{iteration}.pkl")
    stratify = "stratify=y, " if "classif" in task_type else ""
    scale = _scale_needed(algo)

    history_summary = json.dumps(
        [
            {
                "iteration": h.get("iteration"),
                "algorithm": h.get("algorithm"),
                "score": h.get("primary_score"),
            }
            for h in optimization_history
        ],
        indent=2,
    )

    self_check_metric = (
        "accuracy_score(y_test, sc_preds)"
        if "classif" in task_type
        else "r2_score(y_test, sc_preds)"
        if "regress" in task_type
        else "None"
    )
    original_metric_var = (
        "accuracy" if "classif" in task_type
        else "r2" if "regress" in task_type
        else "None"
    )

    feedback_section = (
        f"\nHuman feedback / instructions:\n{human_feedback}"
        if human_feedback.strip() else ""
    )

    prompt = f"""Write Python code for optimization iteration {iteration}.

Dataset: {dataset_path}
Task: {task_type}
{"Target: " + target_column if target_column else "Unsupervised"}
Algorithm: {algo}
Hyperparameters: {hyperparams}
Reason: {reason}
{feedback_section}

Previous iterations:
{history_summary}

Requirements (follow EXACTLY):
1. pandas read_csv
2. Preprocessing (fit on train only):
   - Drop cols with >50% missing
   - Fill numeric NaN appropriately (median for skewed, mean for symmetric)
   - LabelEncode object/category cols
   {"- Separate X and y (target='" + target_column + "')" if target_column else "- X = all columns"}
3. train_test_split(X, y, test_size=0.2, random_state=42, {stratify})
4. {"StandardScaler: fit X_train, transform X_train + X_test" if scale else "No scaling"}
5. Train {algo} with {hyperparams}
6. Evaluate on TEST SET ONLY:
   {_metrics_fields(task_type)}
7. pickle.dump model to: {model_path}
8. SELF-CHECK (data leakage guard):
   sc_model = pickle.load(open("{model_path}", "rb"))
   sc_preds = sc_model.predict({"X_test_scaled" if scale else "X_test"})
   sc_score = {self_check_metric}
   if abs(sc_score - {original_metric_var}) > 1e-6:
       import warnings; warnings.warn("SELF-CHECK MISMATCH iteration {iteration}")
9. Print metrics EXACTLY as:

print("METRICS_JSON_START")
print(json.dumps(metrics, indent=2, default=str))
print("METRICS_JSON_END")

metrics dict:
{{
  "algorithm": "{algo}",
  "iteration": {iteration},
  "task_type": "{task_type}",
  "model_path": "{model_path}",
  "train_samples": <int>,
  "test_samples": <int>,
  {_metrics_fields(task_type)}
}}

Output ONLY the Python code."""

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    code = _strip_markdown(response.choices[0].message.content.strip())
    logger.info("Optimizer generated code for iteration %d: %s", iteration, algo)
    return code


def fix_optimization_code(
    current_code: str, stderr: str, stdout: str, attempt: int
) -> str:
    logger.info("Optimizer fixing code (attempt %d)...", attempt)

    prompt = f"""Code failed.

ERROR:
{stderr}

STDOUT:
{stdout}

CODE:
{current_code}

Fix it. Output ONLY the complete fixed Python code."""

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": FIX_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return _strip_markdown(response.choices[0].message.content.strip())


def make_fix_callback(current_code_path: Path):
    def callback(stderr: str, stdout: str, attempt: int) -> str:
        return fix_optimization_code(
            current_code_path.read_text(), stderr, stdout, attempt
        )
    return callback


def _strip_markdown(code: str) -> str:
    if code.startswith("```"):
        lines = code.split("\n")[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines)
    return code.strip()


def run(
    dataset_path: str,
    task_type: str,
    target_column: str,
    iteration: int,
    optimization_history: list,
    understanding_output: str,
    human_feedback: str = "",
) -> dict:
    """Select next strategy via LLM, generate code, write it, return metadata dict."""
    strategy = select_next_strategy(
        task_type, iteration, optimization_history, understanding_output, human_feedback
    )
    logger.info(
        "Optimizer selected: %s — %s",
        strategy["algorithm"],
        strategy.get("reason", ""),
    )

    code = generate_optimization_code(
        dataset_path, task_type, target_column,
        iteration, strategy, optimization_history, human_feedback,
    )

    script_name = f"step3_ml_iter{iteration}.py"
    script_path = GENERATED_CODE_DIR / script_name
    script_path.write_text(code)
    logger.info("Written: %s", script_path)

    return {
        "script_name": script_name,
        "script_path": str(script_path),
        "code": code,
        "algorithm_info": strategy,
        "fix_callback": make_fix_callback(script_path),
    }
