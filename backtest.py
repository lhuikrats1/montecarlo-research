"""Run the full research analysis and save this run under backtest/."""

import os
import runpy
from pathlib import Path


project_dir = Path(__file__).resolve().parent
research_script = project_dir / "portfolio_research.py"
if not research_script.is_file():
    raise SystemExit(f"Could not find the research script: {research_script}")

os.environ["MONTECARLO_RESULTS_DIR"] = str(project_dir / "backtest")
runpy.run_path(str(research_script), run_name="__main__")
