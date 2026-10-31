"""Build a standalone notebook using exactly the tested package sources."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
files = {p.name: p.read_text(encoding="utf-8") for p in sorted((ROOT / "fashion_recs").glob("*.py"))}


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


bootstrap = '''from pathlib import Path
import json
import sys

ROOT = Path('/kaggle/working/thread_source')
package = ROOT / 'fashion_recs'
package.mkdir(parents=True, exist_ok=True)
SOURCES = json.loads(SOURCE_BLOB)
for name, content in SOURCES.items():
    (package / name).write_text(content, encoding='utf-8')
sys.path.insert(0, str(ROOT))
'''.replace("SOURCE_BLOB", repr(json.dumps(files)))

cells = [
    markdown("""# THREAD — H&M Fashion Recommendation Lab

Popularity → personal candidates → learned scoring → chronological evaluation.

**Before running:** accept the competition rules yourself, then attach the
official **H&M Personalized Fashion Recommendations** competition through
**Add Input**. CPU is sufficient; Internet can stay off. No image archive is read.

This notebook creates local outputs only. It does not publish anything or submit
predictions. No real-data result is claimed until this notebook finishes.
"""),
    code(bootstrap),
    markdown("""## Configuration

An initial reproducible run uses sampled customers to keep CPU cost reasonable.
The final observed week is reserved for evaluation. All methods use the same
evaluation customers. Set EVAL_USERS=0 to evaluate the full week; this costs more.
"""),
    code("""TRAIN_USERS = 3000
EVAL_USERS = 5000
SEED = 42
OUTPUT = Path('/kaggle/working/thread_artifacts')

input_root = Path('/kaggle/input')
# Avoid recursively walking the 100k-file image tree on Kaggle's remote mount.
candidates = list(input_root.glob('*/transactions_train.csv'))
candidates += list(input_root.glob('competitions/*/transactions_train.csv'))
candidates += list(input_root.glob('datasets/*/*/transactions_train.csv'))
candidates = sorted(set(candidates))
if len(candidates) != 1:
    raise RuntimeError(f'Expected one official H&M transactions file; found {len(candidates)}. Attach competition data.')
DATA_DIR = candidates[0].parent
print('Data:', DATA_DIR, flush=True)
"""),
    code("""from fashion_recs.data import load_data
from fashion_recs.pipeline import run

transactions, articles, end, source = load_data(DATA_DIR)
print(source)
report = run(transactions, articles, end, OUTPUT, synthetic=False,
             train_users=TRAIN_USERS, eval_users=EVAL_USERS, seed=SEED, source=source)
"""),
    code("""import pandas as pd
from IPython.display import display, Markdown
display(pd.DataFrame.from_dict(report['map_at_12'], orient='index', columns=['MAP@12']))
display(Markdown((OUTPUT / 'report.md').read_text(encoding='utf-8')))
print('Paired bootstrap interval:', report['learned_minus_popular_95pct_bootstrap_ci'])
print('Training feature max:', report['training_feature_max_date'])
print('Evaluation feature max:', report['evaluation_feature_max_date'])
"""),
    markdown("""## Optional submission CSV

The full customer list is large; per-customer scoring may take considerable time.
Leave this disabled for a first evaluation. Creation does not upload anything.
Check current rules before sharing data-derived model or demo artifacts.
"""),
    code("""CREATE_SUBMISSION = False
if CREATE_SUBMISSION:
    from fashion_recs.pipeline import submit
    submit(OUTPUT / 'model.pkl', DATA_DIR / 'sample_submission.csv', OUTPUT / 'submission.csv')
"""),
]
notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                           "language_info": {"name": "python", "version": "3.10"}}, "nbformat": 4, "nbformat_minor": 4}
destination = ROOT / "notebooks" / "hm_fashion_recommendations.ipynb"
destination.parent.mkdir(exist_ok=True)
destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(destination)
