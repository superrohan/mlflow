"""
LangGraph-based agentic pipeline with optimization loop.

Flow:
  data_understanding → execute_understanding
    → data_analysis → execute_analysis
    → [HUMAN APPROVAL with optional feedback]
    → ml_engineering → execute_ml
    → track_metrics                       ← parses metrics, updates best model
    → optimization_loop                   ← conditional: more iters OR finalize
        ↓ (iter < MAX_OPT_ITERATIONS)
      optimizer → execute_optimization → track_metrics (loop)
        ↓ (iter >= MAX_OPT_ITERATIONS)
      finalize_best → evaluation → END
"""

import json
import logging
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Any, Optional, TypedDict, Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import data_understanding, data_analyst, ml_engineer, evaluator, optimizer
from tools.runner import run_script_with_retry, parse_metrics_from_output, get_primary_score

logger = logging.getLogger(__name__)

GENERATED_CODE_DIR = Path(__file__).parent.parent / "generated_code"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
DB_PATH = Path(__file__).parent.parent / "outputs" / "checkpoints.db"

MAX_RETRIES = 10
MAX_OPT_ITERATIONS = 3


# ─────────────────────── Persistent checkpointer ──────────────

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
_conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
_checkpointer = SqliteSaver(_conn)


# ─────────────────────────────── State ────────────────────────

class PipelineState(TypedDict):
    # ── Inputs ──
    dataset_path: str
    task_type: str          # supervised_classification | supervised_regression | unsupervised
    target_column: Optional[str]

    # ── Step 1 — Data Understanding ──
    understanding_code: str
    understanding_output: str
    understanding_error: str
    understanding_retries: int

    # ── Step 2 — Data Analysis ──
    analysis_code: str
    analysis_output: str
    analysis_error: str
    analysis_retries: int

    # ── Human approval gate ──
    human_approved: bool
    human_feedback: str     # optional free-text feedback from user at approval gate

    # ── Step 3 — ML Engineering (baseline) ──
    ml_code: str
    ml_output: str
    ml_error: str
    ml_retries: int
    algorithm_info: dict

    # ── Optimization loop ──
    optimization_iteration: int     # 0 = baseline done, 1-3 = optimizer iterations
    optimization_history: list      # [{iteration, algorithm, metrics, primary_score}]
    best_model: dict                # {model_path, algorithm, primary_score, metrics, iteration}

    # ── Optimizer scratch (must be in TypedDict so SQLite checkpointer persists them) ──
    opt_script_name: str            # e.g. step3_ml_iter1.py
    opt_algorithm: str              # algorithm name for current iteration
    opt_code: str                   # generated code for current iteration
    opt_output: str                 # stdout from execute_optimization
    opt_error: str                  # stderr if failed

    # ── Evaluation ──
    evaluation: dict

    # ── UI streaming events ──
    events: list

    # ── Final status ──
    status: str     # running | awaiting_approval | completed | failed


# ─────────────────────────── Helpers ──────────────────────────

def _emit(state: PipelineState, event_type: str, message: str, data: Any = None) -> list:
    events = list(state.get("events", []))
    events.append({"type": event_type, "message": message, "data": data})
    logger.info("[EVENT] %s: %s", event_type, message)
    return events


def _primary_metric_key(task_type: str) -> str:
    if "classif" in task_type:
        return "f1"
    if "regress" in task_type:
        return "r2_score"
    return "silhouette_score"


def _try_parse_json(text: str) -> dict:
    """Attempt to parse JSON from stdout; return empty dict on failure."""
    try:
        return json.loads(text)
    except Exception:
        # stdout may contain log lines before JSON; try to find the JSON block
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except Exception:
                pass
    return {}


# ─────────────────────────── Nodes ────────────────────────────

def node_data_understanding(state: PipelineState) -> PipelineState:
    events = _emit(state, "agent_thinking", "Data Understanding Agent is generating code...")

    result = data_understanding.run(
        dataset_path=state["dataset_path"],
        task_type=state["task_type"],
        target_column=state.get("target_column"),
    )

    events = _emit(
        {"events": events}, "code_generated",
        "step1_understanding.py written.", data=result["code"],
    )
    return {**state, "understanding_code": result["code"], "events": events, "understanding_retries": 0}


def node_execute_understanding(state: PipelineState) -> PipelineState:
    retries = state.get("understanding_retries", 0)
    events = _emit(state, "executing", f"Running step1_understanding.py (attempt {retries + 1})...")

    script_path = GENERATED_CODE_DIR / "step1_understanding.py"
    result = run_script_with_retry(
        "step1_understanding.py",
        fix_callback=data_understanding.make_fix_callback(script_path),
        max_retries=MAX_RETRIES,
    )

    if result.success:
        events = _emit({"events": events}, "execution_success", "step1 succeeded.", data=result.stdout[:3000])
        return {**state, "understanding_output": result.stdout, "understanding_error": "", "events": events}

    events = _emit({"events": events}, "execution_error", f"step1 failed after {MAX_RETRIES} retries.", data=result.stderr)
    return {**state, "understanding_output": "", "understanding_error": result.stderr, "status": "failed", "events": events}


def node_data_analysis(state: PipelineState) -> PipelineState:
    events = _emit(state, "agent_thinking", "Data Analyst Agent is generating code...")

    result = data_analyst.run(
        dataset_path=state["dataset_path"],
        understanding_output=state["understanding_output"],
        task_type=state["task_type"],
        target_column=state.get("target_column"),
    )

    events = _emit({"events": events}, "code_generated", "step2_analysis.py written.", data=result["code"])
    return {**state, "analysis_code": result["code"], "events": events, "analysis_retries": 0}


def node_execute_analysis(state: PipelineState) -> PipelineState:
    retries = state.get("analysis_retries", 0)
    events = _emit(state, "executing", f"Running step2_analysis.py (attempt {retries + 1})...")

    script_path = GENERATED_CODE_DIR / "step2_analysis.py"
    result = run_script_with_retry(
        "step2_analysis.py",
        fix_callback=data_analyst.make_fix_callback(script_path),
        max_retries=MAX_RETRIES,
    )

    if result.success:
        events = _emit({"events": events}, "execution_success", "step2 succeeded.", data=result.stdout[:3000])
        # Emit structured analysis data for the UI approval panel
        analysis_data = _try_parse_json(result.stdout)
        if analysis_data:
            events = _emit({"events": events}, "analysis_data", "Analysis results ready.", data=analysis_data)
        events = _emit({"events": events}, "awaiting_approval",
                       "Analysis complete. Awaiting human approval to proceed to modeling.")
        return {**state, "analysis_output": result.stdout, "analysis_error": "", "status": "awaiting_approval", "events": events}

    events = _emit({"events": events}, "execution_error", f"step2 failed after {MAX_RETRIES} retries.", data=result.stderr)
    return {**state, "analysis_output": "", "analysis_error": result.stderr, "status": "failed", "events": events}


def node_human_approval(state: PipelineState) -> PipelineState:
    if state.get("human_approved"):
        feedback = state.get("human_feedback", "")
        msg = "Human approved. Proceeding to ML Engineering."
        if feedback:
            msg += f" Feedback: \"{feedback[:120]}\""
        events = _emit(state, "approved", msg)
        return {**state, "status": "running", "events": events}
    return {**state, "status": "awaiting_approval"}


def node_ml_engineering(state: PipelineState) -> PipelineState:
    events = _emit(state, "agent_thinking", "ML Engineer Agent is selecting algorithm and generating baseline code...")

    result = ml_engineer.run(
        dataset_path=state["dataset_path"],
        task_type=state["task_type"],
        target_column=state.get("target_column", ""),
        understanding_output=state["understanding_output"],
        analysis_output=state["analysis_output"],
        human_feedback=state.get("human_feedback", ""),
    )

    algo_info = result["algorithm_info"]
    events = _emit({"events": events}, "algorithm_selected",
                   f"Baseline: {algo_info['algorithm']} — {algo_info.get('reason', '')}", data=algo_info)
    events = _emit({"events": events}, "code_generated", "step3_ml.py (iteration 0) written.", data=result["code"])
    return {**state, "ml_code": result["code"], "algorithm_info": algo_info, "events": events, "ml_retries": 0}


def node_execute_ml(state: PipelineState) -> PipelineState:
    retries = state.get("ml_retries", 0)
    events = _emit(state, "executing", f"Running step3_ml.py — baseline (attempt {retries + 1})...")

    script_path = GENERATED_CODE_DIR / "step3_ml.py"
    result = run_script_with_retry(
        "step3_ml.py",
        fix_callback=ml_engineer.make_fix_callback(script_path),
        max_retries=MAX_RETRIES,
    )

    if result.success:
        events = _emit({"events": events}, "execution_success", "Baseline model trained.", data=result.stdout[:3000])
        return {**state, "ml_output": result.stdout, "ml_error": "", "events": events}

    events = _emit({"events": events}, "execution_error", f"Baseline failed after {MAX_RETRIES} retries.", data=result.stderr)
    return {**state, "ml_output": "", "ml_error": result.stderr, "status": "failed", "events": events}


def node_track_metrics(state: PipelineState) -> PipelineState:
    """Parse metrics from the most recently executed ML output, update best_model."""
    opt_iter = state.get("optimization_iteration", 0)

    if opt_iter == 0:
        raw_output = state.get("ml_output", "")
        algo = state.get("algorithm_info", {}).get("algorithm", "unknown")
    else:
        raw_output = state.get("opt_output", "")
        algo = state.get("opt_algorithm", "unknown")

    metrics = parse_metrics_from_output(raw_output)
    primary_score = get_primary_score(metrics)
    primary_key = _primary_metric_key(state["task_type"])

    history_entry = {
        "iteration": opt_iter,
        "algorithm": algo,
        "metrics": metrics,
        "primary_score": primary_score,
        "primary_metric": primary_key,
    }

    history = list(state.get("optimization_history", []))
    history.append(history_entry)

    best = dict(state.get("best_model", {}))
    if primary_score > best.get("primary_score", -1.0):
        best = {
            "model_path": metrics.get("model_path", str(OUTPUTS_DIR / f"model_iter{opt_iter}.pkl")),
            "algorithm": algo,
            "primary_score": primary_score,
            "primary_metric": primary_key,
            "metrics": metrics,
            "iteration": opt_iter,
        }
        events = _emit(state, "best_model_updated",
                       f"New best: {algo} — {primary_key}={primary_score:.4f} (iter {opt_iter})",
                       data=best)
    else:
        events = _emit(state, "iteration_tracked",
                       f"Iter {opt_iter}: {algo} — {primary_key}={primary_score:.4f} "
                       f"(best={best.get('primary_score', 0):.4f})")

    return {
        **state,
        "optimization_history": history,
        "best_model": best,
        "events": events,
    }


def node_optimizer(state: PipelineState) -> PipelineState:
    """Generate code for the next optimization iteration."""
    next_iter = state.get("optimization_iteration", 0) + 1
    events = _emit(
        state, "agent_thinking",
        f"Optimizer Agent selecting algorithm for iteration {next_iter}/{MAX_OPT_ITERATIONS}...",
    )

    result = optimizer.run(
        dataset_path=state["dataset_path"],
        task_type=state["task_type"],
        target_column=state.get("target_column", ""),
        iteration=next_iter,
        optimization_history=state.get("optimization_history", []),
        understanding_output=state.get("understanding_output", ""),
        human_feedback=state.get("human_feedback", ""),
    )

    algo_info = result["algorithm_info"]
    events = _emit({"events": events}, "code_generated",
                   f"step3_ml_iter{next_iter}.py written — {algo_info['algorithm']}",
                   data=result["code"])

    return {
        **state,
        "optimization_iteration": next_iter,
        "opt_script_name": result["script_name"],
        "opt_algorithm": algo_info["algorithm"],
        "opt_code": result["code"],
        "events": events,
    }


def node_execute_optimization(state: PipelineState) -> PipelineState:
    """Execute the current optimization iteration script."""
    script_name = state.get("opt_script_name", "")
    iteration = state.get("optimization_iteration", 1)

    if not script_name:
        err = f"opt_script_name is empty for iteration {iteration} — optimizer node may not have run."
        logger.error(err)
        events = _emit(state, "execution_error", err)
        return {**state, "opt_output": "", "opt_error": err, "status": "failed", "events": events}

    events = _emit(state, "executing", f"Running {script_name} (iteration {iteration})...")

    script_path = GENERATED_CODE_DIR / script_name
    result = run_script_with_retry(
        script_name,
        fix_callback=optimizer.make_fix_callback(script_path),
        max_retries=MAX_RETRIES,
    )

    if result.success:
        events = _emit({"events": events}, "execution_success",
                       f"Iteration {iteration} succeeded.", data=result.stdout[:3000])
        return {**state, "opt_output": result.stdout, "opt_error": "", "events": events}

    events = _emit({"events": events}, "execution_error",
                   f"Iteration {iteration} failed after {MAX_RETRIES} retries.", data=result.stderr)
    return {**state, "opt_output": "", "opt_error": result.stderr, "status": "failed", "events": events}


def node_finalize_best(state: PipelineState) -> PipelineState:
    """Copy best model to model.pkl and save optimization_history.json."""
    best = state.get("best_model", {})
    history = state.get("optimization_history", [])

    events = _emit(state, "agent_thinking", "Finalizing best model across all iterations...")

    best_path = best.get("model_path", "")
    final_path = str(OUTPUTS_DIR / "model.pkl")

    if best_path and Path(best_path).exists():
        shutil.copy2(best_path, final_path)
        logger.info("Copied best model %s → %s", best_path, final_path)
    else:
        logger.warning("Best model path not found: %s", best_path)

    history_path = OUTPUTS_DIR / "optimization_history.json"
    history_path.write_text(json.dumps(history, indent=2, default=str))

    summary_lines = [
        f"  iter {h['iteration']}: {h['algorithm']} — {h['primary_metric']}={h['primary_score']:.4f}"
        for h in history
    ]
    summary = "\n".join(summary_lines)

    events = _emit(
        {"events": events}, "optimization_complete",
        f"Optimization done. Best: {best.get('algorithm')} "
        f"({best.get('primary_metric')}={best.get('primary_score', 0):.4f})\n{summary}",
        data={"best_model": best, "history": history},
    )
    return {**state, "events": events}


def node_evaluation(state: PipelineState) -> PipelineState:
    events = _emit(state, "agent_thinking", "Evaluation Agent is analyzing best model performance...")

    best = state.get("best_model", {})
    algo = best.get("algorithm", state.get("algorithm_info", {}).get("algorithm", "unknown"))

    metrics_output = json.dumps(best.get("metrics", {})) or state.get("ml_output", "")

    evaluation = evaluator.evaluate(
        task_type=state["task_type"],
        algorithm=algo,
        metrics_output=metrics_output,
        understanding_output=state.get("understanding_output", ""),
    )

    verdict = evaluation.get("verdict", "pass")
    score = evaluation.get("score", 0.0)

    events = _emit(
        {"events": events}, "evaluation_done",
        f"Evaluation complete. Verdict: {verdict.upper()} | Score: {score:.3f}",
        data={**evaluation, "best_model": best, "optimization_history": state.get("optimization_history", [])},
    )
    events = _emit({"events": events}, "completed", "Pipeline finished successfully!")

    return {**state, "evaluation": evaluation, "status": "completed", "events": events}


# ─────────────────────────── Edges ────────────────────────────

def edge_after_understanding_exec(state: PipelineState) -> Literal["data_analysis", "__end__"]:
    return "__end__" if state.get("understanding_error") else "data_analysis"


def edge_after_analysis_exec(state: PipelineState) -> Literal["human_approval", "__end__"]:
    return "__end__" if state.get("analysis_error") else "human_approval"


def edge_after_human_approval(state: PipelineState) -> Literal["ml_engineering", "human_approval"]:
    return "ml_engineering" if state.get("human_approved") else "human_approval"


def edge_after_ml_exec(state: PipelineState) -> Literal["track_metrics", "__end__"]:
    return "__end__" if state.get("ml_error") else "track_metrics"


def edge_optimization_loop(state: PipelineState) -> Literal["optimizer", "finalize_best"]:
    completed = state.get("optimization_iteration", 0)
    if completed < MAX_OPT_ITERATIONS:
        return "optimizer"
    return "finalize_best"


def edge_after_opt_exec(state: PipelineState) -> Literal["track_metrics", "__end__"]:
    return "__end__" if state.get("opt_error") else "track_metrics"


def edge_after_track(state: PipelineState) -> Literal["optimization_loop", "__end__"]:
    return "__end__" if state.get("status") == "failed" else "optimization_loop"


# ─────────────────────────── Graph ────────────────────────────

def _build_workflow():
    workflow = StateGraph(PipelineState)

    workflow.add_node("data_understanding",     node_data_understanding)
    workflow.add_node("execute_understanding",  node_execute_understanding)
    workflow.add_node("data_analysis",          node_data_analysis)
    workflow.add_node("execute_analysis",       node_execute_analysis)
    workflow.add_node("human_approval",         node_human_approval)
    workflow.add_node("ml_engineering",         node_ml_engineering)
    workflow.add_node("execute_ml",             node_execute_ml)
    workflow.add_node("track_metrics",          node_track_metrics)
    workflow.add_node("optimization_loop",      lambda s: s)
    workflow.add_node("optimizer",              node_optimizer)
    workflow.add_node("execute_optimization",   node_execute_optimization)
    workflow.add_node("finalize_best",          node_finalize_best)
    workflow.add_node("evaluation",             node_evaluation)

    workflow.set_entry_point("data_understanding")

    workflow.add_edge("data_understanding", "execute_understanding")
    workflow.add_conditional_edges(
        "execute_understanding",
        edge_after_understanding_exec,
        {"data_analysis": "data_analysis", "__end__": END},
    )
    workflow.add_edge("data_analysis", "execute_analysis")
    workflow.add_conditional_edges(
        "execute_analysis",
        edge_after_analysis_exec,
        {"human_approval": "human_approval", "__end__": END},
    )
    workflow.add_conditional_edges(
        "human_approval",
        edge_after_human_approval,
        {"ml_engineering": "ml_engineering", "human_approval": "human_approval"},
    )
    workflow.add_edge("ml_engineering", "execute_ml")
    workflow.add_conditional_edges(
        "execute_ml",
        edge_after_ml_exec,
        {"track_metrics": "track_metrics", "__end__": END},
    )
    workflow.add_conditional_edges(
        "track_metrics",
        edge_after_track,
        {"optimization_loop": "optimization_loop", "__end__": END},
    )
    workflow.add_conditional_edges(
        "optimization_loop",
        edge_optimization_loop,
        {"optimizer": "optimizer", "finalize_best": "finalize_best"},
    )
    workflow.add_edge("optimizer", "execute_optimization")
    workflow.add_conditional_edges(
        "execute_optimization",
        edge_after_opt_exec,
        {"track_metrics": "track_metrics", "__end__": END},
    )
    workflow.add_edge("finalize_best", "evaluation")
    workflow.add_edge("evaluation", END)

    return workflow.compile(
        checkpointer=_checkpointer,
        interrupt_before=["human_approval"],
    )


def build_graph():
    return _build_workflow()


graph = build_graph()


def create_initial_state(
    dataset_path: str, task_type: str, target_column: str = None
) -> PipelineState:
    return PipelineState(
        dataset_path=dataset_path,
        task_type=task_type,
        target_column=target_column,
        understanding_code="",
        understanding_output="",
        understanding_error="",
        understanding_retries=0,
        analysis_code="",
        analysis_output="",
        analysis_error="",
        analysis_retries=0,
        human_approved=False,
        human_feedback="",
        ml_code="",
        ml_output="",
        ml_error="",
        ml_retries=0,
        algorithm_info={},
        optimization_iteration=0,
        optimization_history=[],
        best_model={},
        opt_script_name="",
        opt_algorithm="",
        opt_code="",
        opt_output="",
        opt_error="",
        evaluation={},
        events=[],
        status="running",
    )
