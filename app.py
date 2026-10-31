"""Local portfolio demo. Run: streamlit run app.py"""
import json
import os
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="THREAD / Fashion recommendations", page_icon="◈", layout="wide")
st.markdown("""<style>
.stApp {background:#f4f1eb;color:#182b26}
h1,h2,h3 {font-family:Georgia,serif!important;letter-spacing:-.035em}
h1 {font-size:3.8rem!important;line-height:1.02!important}
[data-testid="stMetric"] {background:#e8e7dd;border-radius:14px;padding:18px}
[data-testid="stSidebar"] {background:#e8e7dd}
.eyebrow {font-size:12px;letter-spacing:.2em;color:#64766b;margin-bottom:18px}
.product {border:1px solid #d5d8cc;border-radius:16px;overflow:hidden;background:#fffdf8;margin-bottom:20px}
.swatch {height:100px;display:flex;align-items:center;justify-content:center;font:italic 38px Georgia;color:#ffffffb0}
.details {padding:16px;min-height:140px}
.details h3 {font-size:21px!important;margin:0 0 8px}
.tag {font-size:11px;padding:5px 8px;border-radius:6px;background:#e8eddf;color:#355741;display:inline-block;margin-top:10px}
.small {font-size:12px;color:#6c756c}
</style>""", unsafe_allow_html=True)


@st.cache_data
def load(path, modified):
    return json.loads(Path(path).read_text(encoding="utf-8"))


default_path = "artifacts/real/demo.json" if Path("artifacts/real/demo.json").exists() else "artifacts/demo/demo.json"
source = Path(os.environ.get("FASHION_DEMO_PATH", default_path))
if not source.exists():
    st.title("THREAD")
    st.info("Сначала создайте демонстрационные данные: python -m fashion_recs.pipeline demo")
    st.stop()
data = load(str(source), source.stat().st_mtime_ns)
synthetic = data["data_kind"].startswith("SYNTHETIC")

with st.sidebar:
    st.markdown("### ◈ THREAD")
    st.caption("Рекомендации одежды · ML-портфолио")
    st.divider()
    user_names = [u["id"] for u in data["users"]]
    selected = st.selectbox("Профиль покупателя", ["Новый покупатель"] + user_names)
    method = st.radio("Подход", ["Персональная модель", "Популярные товары"])
    st.divider()
    st.caption("Горизонт прогноза: 7 дней")
    st.caption(f"История до {data['as_of']} (не включая дату)")
    st.caption("Профили обозначены условными номерами.")
    st.caption("Цветные карточки — условные обозначения, не фотографии товаров.")

st.markdown('<div class="eyebrow">THREAD / RECOMMENDATION LAB</div>', unsafe_allow_html=True)
st.title("Следующая покупка.")
st.write("Система подбирает 12 товаров по истории покупок и последним тенденциям спроса.")
if synthetic:
    st.warning("СИНТЕТИЧЕСКОЕ ДЕМО. Это искусственные покупки и товары. Метрики ниже относятся только к этому примеру, а не к данным H&M.")
else:
    st.info("Исследовательский проект на исторических данных H&M. Не связан с официальным магазином. Оценка — офлайн, не рост продаж.")

metrics = data["metrics"]["map_at_12"]
cols = st.columns(4)
cols[0].metric("MAP@12 · популярные", f"{metrics['popular']:.4f}")
cols[1].metric("MAP@12 · гибрид", f"{metrics['hybrid']:.4f}")
cols[2].metric("MAP@12 · модель", f"{metrics['learned']:.4f}")
cols[3].metric("Покупателей в проверке", f"{data['metrics']['evaluated_users']:,}")
st.caption("Метрики выше рассчитаны на отложенной неделе. Карточки ниже — прогноз модели после обновления истории до конца доступных данных.")
user = next((u for u in data["users"] if u["id"] == selected), None)
personalized = method == "Персональная модель" and user is not None
recs = user["recommendations"] if personalized else data["popular"]
st.divider()
st.subheader("Подборка для вас" if personalized else "Популярные товары")
if user is None:
    st.caption("Для нового покупателя используем популярность: личной истории пока нет.")

import html
colours = {"Black": "#363a35", "Cream": "#baa986", "Blue": "#617c92", "Olive": "#7a8261",
           "Burgundy": "#88565c", "Sand": "#ba9b79", "White": "#b5b5a9", "Dark Blue": "#40556e"}
columns = st.columns(4)
for position, article_id in enumerate(recs):
    item = data["articles"].get(article_id, {})
    name = html.escape(str(item.get("prod_name", article_id)))
    category = html.escape(str(item.get("product_type_name", "Одежда")))
    colour = str(item.get("colour_group_name", "Sand"))
    background = colours.get(colour, "#929b86")
    reason = user["reasons"].get(article_id, "Популярно сейчас") if personalized else "Популярно сейчас"
    columns[position % 4].markdown(
        f'<div class="product"><div class="swatch" style="background:{background}">{position + 1:02d}</div>'
        f'<div class="details"><div class="small">{category}</div><h3>{name}</h3>'
        f'<div class="small">{html.escape(article_id)} · {html.escape(colour)}</div>'
        f'<span class="tag">{html.escape(reason)}</span></div></div>', unsafe_allow_html=True)

if user:
    with st.expander("История покупок этого профиля"):
        st.dataframe([
            {"Артикул": a, "Товар": data["articles"].get(a, {}).get("prod_name", a),
             "Категория": data["articles"].get(a, {}).get("product_type_name", "")}
            for a in user["history"]
        ], hide_index=True, use_container_width=True)
with st.expander("Как проверяли качество"):
    st.write("Сначала формируем кандидатов: популярность за 7/28 дней, повторные покупки, соседние товары в истории. "
             "Затем модель градиентного бустинга оценивает вероятность следующей покупки.")
    st.write("Обучающая неделя предшествует проверочной. Признаки каждой недели используют только прошлое. "
             "Оценка включает покупателей без истории, если они покупали в проверочную неделю.")
    st.write("Подписи на карточках описывают доступные сигналы, а не причинное объяснение решения модели.")
    st.json(data["metrics"])

