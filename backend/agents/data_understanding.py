import logging
import json
from pathlib import Path
from openai import OpenAI

logger = logging.getLogger(__name__)

GENERATED_CODE_DIR = Path(__file__).parent.parent / "generated_code"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"

client = OpenAI()
MODEL = "gpt-4o"

SYSTEM_PROMPT = """You are an expert Data Understanding Agent. Your ONLY job is to write clean, executable Python code.

Rules:
- Output ONLY valid Python code. No markdown, no backticks, no explanations before/after code.
- The code must be completely self-contained and runnable.
- Use only standard Python libraries and pandas/numpy/scipy which are always available.
- Print a structured JSON summary to stdout at the end using: print(json.dumps(summary, indent=2))
- The summary dict must include: num_rows, num_cols, columns (list of {name, dtype, missing_count, missing_pct}), sample_rows (first 5 rows as list of dicts), numeric_stats (describe() output for numeric cols).
- Handle errors gracefully with try/except blocks.
"""

FIX_SYSTEM_PROMPT = """You are an expert Python debugger. Fix the provided code based on the error.
Output ONLY the complete fixed Python code. No explanations, no markdown, no backticks."""


def generate_understanding_code(dataset_path: str, task_type: str, target_column: str = None) -> str:
    """Ask Claude to generate step1_understanding.py."""
    prompt = f"""Write Python code to analyze the dataset at: {dataset_path}

Task type: {task_type}
{"Target column: " + target_column if target_column else ""}

The code must:
1. Load the CSV dataset using pandas
2. Sample up to 1000 rows (use df.sample(min(1000, len(df)), random_state=42))
3. Compute: shape, dtypes, missing values count and percentage per column
4. Compute numeric column statistics (df.describe())
5. Store first 5 rows as sample
6. Build a summary dict and print it as JSON: print(json.dumps(summary, indent=2, default=str))

The summary dict structure:
{{
  "dataset_path": "<path>",
  "task_type": "<type>",
  "target_column": "<col or null>",
  "num_rows": <int>,
  "num_cols": <int>,
  "columns": [{{"name": "...", "dtype": "...", "missing_count": 0, "missing_pct": 0.0}}],
  "numeric_stats": {{}},
  "sample_rows": []
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

    code = response.choices[0].message.content.strip()
    code = _strip_markdown(code)
    logger.info("Data Understanding Agent generated code.")
    return code


def fix_understanding_code(current_code: str, stderr: str, stdout: str, attempt: int) -> str:
    """Ask Claude to fix broken step1_understanding.py."""
    logger.info(f"Data Understanding Agent fixing code (attempt {attempt})...")

    prompt = f"""The following Python code failed to execute.

ERROR:
{stderr}

STDOUT (may be partial):
{stdout}

CURRENT CODE:
{current_code}

Fix the code so it runs correctly. Output ONLY the complete fixed Python code."""

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": FIX_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    fixed = response.choices[0].message.content.strip()
    fixed = _strip_markdown(fixed)
    logger.info("Data Understanding Agent returned fix.")
    return fixed


def make_fix_callback(current_code_path: Path):
    """Return a closure compatible with runner.run_script_with_retry."""
    def callback(stderr: str, stdout: str, attempt: int) -> str:
        current_code = current_code_path.read_text()
        fixed = fix_understanding_code(current_code, stderr, stdout, attempt)
        return fixed
    return callback


def _strip_markdown(code: str) -> str:
    """Remove ```python ... ``` fences if Claude added them despite instructions."""
    if code.startswith("```"):
        lines = code.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines)
    return code.strip()


def run(dataset_path: str, task_type: str, target_column: str = None) -> dict:
    """
    Main entry: generate code, write it, return dict with code + path.
    Actual execution is handled by the graph via runner.
    """
    code = generate_understanding_code(dataset_path, task_type, target_column)
    script_path = GENERATED_CODE_DIR / "step1_understanding.py"
    script_path.write_text(code)
    logger.info(f"Written: {script_path}")
    return {
        "script_name": "step1_understanding.py",
        "script_path": str(script_path),
        "code": code,
        "fix_callback": make_fix_callback(script_path),
    }
