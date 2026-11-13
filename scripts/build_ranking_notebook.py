"""Build the standalone ranking experiment notebook and editor cells."""
import base64
import json
import zlib
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sources = {p.name: p.read_text(encoding="utf-8") for p in sorted((root / "fashion_recs").glob("*.py"))}
encoded = base64.b64encode(zlib.compress(json.dumps(sources).encode(), 9)).decode()
bootstrap = f'''from pathlib import Path
import base64, json, sys, zlib
source_dir = Path('/kaggle/working/thread_ranking_source')
package = source_dir / 'fashion_recs'
package.mkdir(parents=True, exist_ok=True)
sources = json.loads(zlib.decompress(base64.b64decode({encoded!r})))
for filename, content in sources.items():
    (package / filename).write_text(content, encoding='utf-8')
sys.path.insert(0, str(source_dir))
import lightgbm
print('LightGBM:', lightgbm.__version__, flush=True)
'''
config = '''from fashion_recs.ranking_experiment import LOOKBACK_DAYS, run_ranking_comparison
from fashion_recs.data import load_data

input_root = Path('/kaggle/input')
files = sorted(set(path for pattern in ('*/transactions_train.csv', 'competitions/*/transactions_train.csv', 'datasets/*/*/transactions_train.csv')
                   for path in input_root.glob(pattern)))
if len(files) != 1:
    raise RuntimeError('Attach the official H&M competition data')
OUTPUT = Path('/kaggle/working/thread_ranking')
print('Data:', files[0].parent, 'Lookback:', LOOKBACK_DAYS, flush=True)
'''
run = '''transactions, articles, end, source = load_data(files[0].parent, lookback_days=LOOKBACK_DAYS)
print(source, flush=True)
ranking = run_ranking_comparison(transactions, articles, end, OUTPUT,
    train_users=3000, eval_users=5000, seed=42, source=source)
'''
display = '''import pandas as pd
from IPython.display import display
display(pd.DataFrame([{'week': w['start'], **w['map_at_12'],
                       'oracle': w['oracle_map_at_12'], 'candidate_recall': w['candidate_recall']}
                      for w in ranking['weeks']]))
print('Review rule:', ranking['rule'])
print('Review result:', ranking['review_gate'])
'''
final = '''print('Only aggregate evaluation artifacts were written. No submission or serving switch.')'''
intro = '''# THREAD — Same candidates, different ranking objectives

Fixed hybrid versus pointwise scoring and LambdaMART on four earlier H&M weeks.
Attach the official competition data. LightGBM is required; CPU is sufficient.
Each evaluation uses the preceding label week and 98 prior feature days.
Parameters are fixed before the run. Only aggregate comparison files are saved.
'''
def cell(kind, content):
    result = {"cell_type": kind, "metadata": {}, "source": content.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


cells = [("bootstrap", bootstrap), ("config", config), ("run", run), ("display", display), ("final", final)]
notebook = {"cells": [cell("markdown", intro), *[cell("code", code) for _, code in cells]],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
target = root / "notebooks" / "hm_ranking_comparison.ipynb"
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
(root / "artifacts").mkdir(exist_ok=True)
for name, code in cells:
    (root / "artifacts" / f"ranking_{name}.txt").write_text(code, encoding="utf-8")
print(target)
