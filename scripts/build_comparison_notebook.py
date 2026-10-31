"""Produce a standalone, compressed-source experiment notebook for Kaggle."""
import base64
import json
import zlib
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sources = {p.name: p.read_text(encoding="utf-8") for p in sorted((root / "fashion_recs").glob("*.py"))}
blob = base64.b64encode(zlib.compress(json.dumps(sources).encode("utf-8"), 9)).decode("ascii")
bootstrap = f'''from pathlib import Path
import base64, json, sys, zlib
ROOT = Path('/kaggle/working/thread_source')
package = ROOT / 'fashion_recs'
package.mkdir(parents=True, exist_ok=True)
SOURCES = json.loads(zlib.decompress(base64.b64decode({blob!r})))
for name, content in SOURCES.items():
    (package / name).write_text(content, encoding='utf-8')
sys.path.insert(0, str(ROOT))
print('Loaded tested experiment sources', flush=True)
'''
config = '''TRAIN_USERS = 3000
EVAL_USERS = 5000
SEED = 42
OUTPUT = Path('/kaggle/working/thread_comparison')
input_root = Path('/kaggle/input')
candidates = list(input_root.glob('*/transactions_train.csv'))
candidates += list(input_root.glob('competitions/*/transactions_train.csv'))
candidates += list(input_root.glob('datasets/*/*/transactions_train.csv'))
candidates = sorted(set(candidates))
if len(candidates) != 1:
    raise RuntimeError('Attach the official H&M competition data')
DATA_DIR = candidates[0].parent
print('Data:', DATA_DIR, flush=True)
'''
experiment = '''from fashion_recs.data import load_data
from fashion_recs.experiment import run_comparison
from fashion_recs.pipeline import run
from fashion_recs.model import CandidateEngine
from fashion_recs.expanded import ExpandedCandidateEngine

transactions, articles, end, source = load_data(DATA_DIR, lookback_days=133)
print(source, flush=True)
comparison = run_comparison(transactions, articles, end, OUTPUT,
    train_users=TRAIN_USERS, eval_users=EVAL_USERS, seed=SEED, source=source)
selected_class = ExpandedCandidateEngine if comparison['selected_variant'] == 'expanded' else CandidateEngine
# Export uses the same 98-day windows and training/evaluation users as confirmation.
print('Comparison complete. Building private serving demo.', flush=True)
selected_report = run(transactions, articles, end, Path('/kaggle/working/thread_selected'),
    synthetic=False, train_users=TRAIN_USERS, eval_users=EVAL_USERS, seed=SEED,
    source=source, engine_class=selected_class, serving_method=comparison['selected_method'], history_days=98)
'''
display = '''import pandas as pd
from IPython.display import display
rows = []
for week in comparison['weeks']:
    for variant, metrics in week['variants'].items():
        rows.append({'week': week['start'], 'role': week['role'], 'variant': variant,
                     **metrics['map_at_12'], 'candidate_recall': metrics['candidate_recall']})
display(pd.DataFrame(rows))
print('Selected on earlier weeks:', comparison['selected_variant'], comparison['selected_method'])
print('Repeated confirmation:', comparison['confirmation_map_at_12'])
print(comparison['confirmation_caveat'])
'''
def cell(kind, text):
    value = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value
cells = [cell("markdown", """# THREAD — Expanded candidates and rolling evaluation

Three development weeks select between original/expanded candidates and
fixed-hybrid/learned scoring. Final week is a previously seen confirmation,
not a newly unseen test. Each feature window is limited to 98 prior days.
All comparisons use the same sampled purchasers and seed=42.
No submission or publication of customer histories is performed.
"""), cell("code", bootstrap), cell("code", config), cell("code", experiment), cell("code", display)]
notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                           "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
target = root / "notebooks" / "hm_rolling_comparison.ipynb"
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
for name, content in [("bootstrap", bootstrap), ("config", config), ("experiment", experiment), ("display", display)]:
    (root / "artifacts" / f"comparison_{name}.txt").write_text(content, encoding="utf-8")
print(target)
