import os
import json
import subprocess
import sys
from datetime import datetime

# Setup sys.path for llm_agent imports
sys.path.append(os.path.join(os.path.dirname(__file__), "llm_agent"))

from llm_agent.state_agent import build_graph

def run_diagnostic_phase():
    """Execute diagnostic scripts."""
    scripts = [
        "phase2/metric_profiling.py",
        "phase2/sub_graph.py",
        "phase2/error_centric_log.py"
    ]
    
    print("\n--- Diagnostic Phase ---")
    for script in scripts:
        script_path = os.path.join(os.path.dirname(__file__), script)
        print(f"Running {script}...")
        try:
            subprocess.run([sys.executable, script_path], capture_output=True, text=True, check=True)
            print(f"[OK] {script}")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] {script}: {e.stderr}")

def load_initial_rankings():
    """Load rankings from Phase 1 logs."""
    log_dir = os.path.join(os.path.dirname(__file__), "log")
    
    metric_file = os.path.join(log_dir, "metric_rca_ranking.json")
    trace_file = os.path.join(log_dir, "trace_rca_ranking.json")
    
    metrics = []
    if os.path.exists(metric_file):
        with open(metric_file, "r") as f:
            metrics = [i["service"] for i in json.load(f).get("ranking", [])]
            
    traces = []
    if os.path.exists(trace_file):
        with open(trace_file, "r") as f:
            traces = [i["service"] for i in json.load(f).get("ranking", [])]
            
    return metrics, traces

def main():
    print("="*40)
    print("   RCA Pipeline Execution")
    print("="*40)
    
    run_diagnostic_phase()
    
    metrics, traces = load_initial_rankings()
    candidates = list(dict.fromkeys(metrics + traces)) or ["app-a", "app-b", "app-c"]
    
    state = {
        "initial_ranking_metrics": metrics,
        "initial_ranking_traces": traces,
        "current_ranking": candidates,
        "seen_candidates": [],
        "iteration_count": 0,
        "current_pod": "",
        "diagnostic_bundles": {},
        "summaries": {},
        "action": "Analyze Next",
        "next_candidate": candidates[0] if candidates else "",
        "final_report": ""
    }
    
    print("\n--- Reasoning Phase ---")
    app = build_graph()
    
    try:
        final_state = app.invoke(state)
        print("\n--- Final Report ---")
        print(final_state.get("final_report", "No report generated."))
    except Exception as e:
        print(f"[ERROR] Reasoning Loop: {e}")

    print("\n" + "="*40)
    print(f"Finished at {datetime.now().strftime('%H:%M:%S')}")
    print("="*40)

if __name__ == "__main__":
    main()


