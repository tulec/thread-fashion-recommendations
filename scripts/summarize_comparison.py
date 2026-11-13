"""Publish aggregate experiment results without customer-level artifacts."""
import argparse
import json
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("report", type=Path)
args = parser.parse_args()
report = json.loads(args.report.read_text(encoding="utf-8"))
weeks = report["weeks"]
if report["data_kind"] != "H&M competition data" or len(weeks) != 4:
    raise ValueError("Expected a completed real-data four-week report")
if [w["role"] for w in weeks] != ["development"] * 3 + ["previously_seen_confirmation"]:
    raise ValueError("Unexpected development/confirmation split")
pairs = [(v, m) for v in ["original", "expanded"] for m in ["hybrid", "learned"]]
winner = max(pairs, key=lambda p: np.mean([w["variants"][p[0]]["map_at_12"][p[1]] for w in weeks[:-1]]))
if winner != (report["selected_variant"], report["selected_method"]):
    raise ValueError("Winner does not match the frozen selection rule")
for week in weeks:
    for result in week["variants"].values():
        if result["feature_max_date"] >= week["start"] or result["training_feature_max_date"] >= week["training_labels_start"]:
            raise ValueError("Feature dates cross a label boundary")
lines = ["# Результаты сравнения по четырём неделям", "",
         "Три ранние недели использованы для выбора. Последняя — повторная контрольная",
         "проверка периода, который уже оценивался в первой версии проекта.", "",
         "| Неделя | Подбор | MAP@12 гибрида | MAP@12 модели | Полнота кандидатов |",
         "|---|---|---:|---:|---:|"]
for week in weeks:
    for variant, result in week["variants"].items():
        title = "Исходный" if variant == "original" else "Расширенный"
        scores = result["map_at_12"]
        lines.append(f"| {week['start']} | {title} | {scores['hybrid']:.6f} | {scores['learned']:.6f} | {result['candidate_recall']:.2%} |")
mean = lambda v, m: float(np.mean([w["variants"][v]["map_at_12"][m] for w in weeks[:-1]]))
lines += ["", "Средние MAP@12 трёх ранних недель:", ""]
for variant, method in pairs:
    lines.append(f"- `{variant}/{method}`: {mean(variant, method):.6f}.")
lines += ["", f"Выбран `{winner[0]}/{winner[1]}`. MAP@12 на повторной контрольной неделе: **{report['confirmation_map_at_12']:.6f}**.",
          "", "Выбор выполнен по ранним неделям, без изменения параметров по контрольному результату.",
          "Оба метода пересчитаны с одинаковым окном истории 98 дней. Поэтому первичное",
          "сравнение здесь — между строками одной недели, а не со старым баллом первой версии.",
          "", "Парные bootstrap-интервалы по каждой неделе, размер выборок и даты признаков",
          "сохранены в [машиночитаемом отчёте](comparison.json). Они отражают выборку",
          "покупателей одной недели, но не полную сезонную неопределённость.",
          "", "Полнота кандидатов усредняется по покупателям. Для новых профилей выдача остаётся",
          "популярностной. Выигрыш одного генератора может сопровождаться ухудшением ранжирования;",
          "эти показатели оцениваются отдельно.",
          "", "[Протокол и воспроизведение](COMPARISON.md)."]
(root / "docs" / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
(root / "docs" / "COMPARISON_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({"selected": winner, "confirmation": report["confirmation_map_at_12"]}, ensure_ascii=False))
