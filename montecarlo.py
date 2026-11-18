"""Run the multi-stock research dashboard and optional Gemini review."""

import runpy
from pathlib import Path


research_script = Path(__file__).with_name("portfolio_research.py")
if not research_script.is_file():
    raise SystemExit(f"Could not find the research script: {research_script}")

runpy.run_path(str(research_script), run_name="__main__")
