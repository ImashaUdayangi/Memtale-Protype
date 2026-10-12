# main.py
import sys
import subprocess
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def run_module(module: str):
    """Run `python -m <module>` from project root."""
    print(f"\n▶ python -m {module}")
    subprocess.check_call([sys.executable, "-m", module], cwd=ROOT)

def require(path: str):
    """Fail fast if required file is missing."""
    p = ROOT / path
    if not p.exists():
        raise FileNotFoundError(
            f"\n Missing required file: {path}\n"
            f"Fix: generate it first, or place it in the expected path."
        )
    return p

def main():
    parser = argparse.ArgumentParser(description="Memtale-Co-Archivist Experiment Runner")
    parser.add_argument(
        "--mode",
        choices=["run", "rebuild"],
        default="run",
        help="run = use existing artifacts (datasets/queries/qrels) to reproduce results. "
             "rebuild = regenerate artifacts (includes manual judging step)."
    )
    args = parser.parse_args()

    print("\n==============================")
    print(" Memtale-Co-Archivist Prototype")
    print(f" Mode: {args.mode.upper()}")
    print("==============================\n")

    # ---------------------------
    # REBUILD MODE (developer)
    if args.mode == "rebuild":
        #  Rebuild pipeline
        run_module("src.data_build_empathetic")
        run_module("src.precompute_summaries")
        run_module("src.build_eval_queries_v3")

        # IMPORTANT: judging pool involves manual work.
        run_module("src.build_judging_pool_v3")
        print("\n Manual step required:")
        print("Fill judging_pool_blind.csv with relevance labels (2/1/0).")
        print("Then run: python -m src.convert_judged_to_qrels_v3\n")
        return

    # ---------------------------
    # DEMO MODE 
    # ---------------------------
    # Require frozen artifacts (no manual tasks)
    require("dataset/raw/empathetic_stories.csv")
    require("dataset/cache/summaries.csv")
    require("dataset/eval_v3/queries.csv")
    require("dataset/eval_v3/qrels.csv")

    # Run retrieval + evaluation + figures
    require("evaluation/run_two_stage_retrieval.py")
    require("evaluation/compute_metrics.py")
    require("evaluation/compute_emotional_fidelity.py")
    require("evaluation/compute_significance_test.py")
    require("evaluation/make_main_results_figure.py")
    require("evaluation/make_qualitative_rerank_example.py")

    run_module("evaluation.run_two_stage_retrieval")
    run_module("evaluation.compute_metrics")
    run_module("evaluation.compute_emotional_fidelity")
    run_module("evaluation.compute_significance_test")
    run_module("evaluation.make_main_results_figure")
    run_module("evaluation.make_qualitative_rerank_example")

    # Quick “what to show” checklist
    print("\n PROTOTYPE COMPLETED SUCCESSFULLY\n")
    print("Key outputs:")
    print(" - evaluation/results/evaluation_summary.csv")
    print(" - evaluation/results/main_results_bar_chart.png")
    print(" - evaluation/results/qualitative_rerank_example.csv\n")

if __name__ == "__main__":
    main()
