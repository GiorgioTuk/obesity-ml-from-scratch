"""Exports the figures of the executed notebook to images/ (used by the README).

Usage: python tools/export_figures.py
"""
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "obesity_ml_from_scratch.ipynb"
OUT_DIR = ROOT / "images"

# One name per figure, in the order they appear in the notebook
NAMES = [
    "01_data_distribution",
    "02_linear_regression_ridge_tuning",
    "03_gradient_descent_convergence",
    "04_neural_network_loss",
    "05_logistic_regression_coefficients",
    "06_knn_tuning",
    "07_decision_tree_tuning",
    "08_confusion_matrices",
    "09_final_comparison",
]

nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
images = [out["data"]["image/png"]
          for cell in nb["cells"] if cell["cell_type"] == "code"
          for out in cell.get("outputs", []) if "image/png" in out.get("data", {})]
if len(images) != len(NAMES):
    raise SystemExit(f"expected {len(NAMES)} figures, found {len(images)}: re-run the notebook or update NAMES")

OUT_DIR.mkdir(exist_ok=True)
for name, data in zip(NAMES, images):
    (OUT_DIR / f"{name}.png").write_bytes(base64.b64decode(data))
    print("saved", OUT_DIR / f"{name}.png")
