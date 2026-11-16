# -*- coding: utf-8 -*-
"""
DASHBOARD OPERATOR - HỆ THỐNG IDS + XAI (TabNet + Attention + SHAP + LIME)
============================================================================
Chạy bằng:  python -m streamlit run DEMO.py

"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ==============================================================================
# CẤU HÌNH ĐƯỜNG DẪN (mặc định tương đối, có thể đổi ngay trong sidebar)
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR.parent / "data" / "processed"

CLASSES = [
    "attackfree", "dosattack", "floodattack", "fuzzyattack",
    "iec104starvationattack", "mitmattack", "ntpddosattack", "portscanattack",
]

LOCAL_SAMPLE_CLASSES = ["dosattack", "mitmattack", "portscanattack", "floodattack"]

# Ngưỡng số mẫu được coi là "quá ít để tin cậy thống kê" -> tô cảnh báo trong bảng
LOW_SUPPORT_THRESHOLD = 50

# Trang -> mục tương ứng trong luận văn (hiển thị ở trang Giới thiệu)
PAGE_TO_THESIS_SECTION = {
    "1. Tổng quan phân loại": "Mục 4.3 - Bảng 4.1, Hình 4.2 (ma trận nhầm lẫn)",
    "2. SHAP toàn cục": "Mục 4.4.1 - Hình 4.x (beeswarm, bar)",
    "3. SHAP theo lớp": "Mục 4.4.1 - Hình 4.7 (heatmap per-class)",
    "4. Giải thích cục bộ (SHAP + LIME)": "Mục 4.4.2, 4.4.3 - Bảng 4.10",
    "5. Mẫu bị dự đoán sai": "Mục 4.2 - Hình 4.2 (ma trận nhầm lẫn), phân tích lỗi",
    "6. Attention (TabNet nội tại)": "Mục 4.4.1 - Hình 4.3, 4.4, 4.5",
   # "7. Độ tin cậy mô hình": "Mục 4.3.1 (SMOTE), 4.3.2 (K-fold), 4.3.3 (baseline), 4.4.3 (chi phí)",
}


def build_files_dict(data_dir: Path) -> dict:
    """Tạo lại đường dẫn các file dữ liệu dựa theo thư mục dữ liệu hiện tại.

    LƯU Ý (đã đối chiếu với file thật trong D:\\Nln\\processed\\iec104):
    - Không có file `results.json` riêng. File `results_smote.json` do
      TABNET.ipynb xuất ra thực chất được huấn luyện trên dữ liệu GỐC
      (không SMOTE) - xem comment trong notebook: "dùng dữ liệu GỐC
      (không SMOTE)" - nên ở đây dùng luôn nó làm kết quả CHÍNH.
    - `results_smote_real.json` là kết quả SMOTE THẬT, do
      `train_tabnet_smote.ipynb` xuất ra (train trên X_train_smote.csv,
      val/test giữ nguyên bản gốc). Khớp đúng số liệu mục 4.3.1: Accuracy
      99,61%, Macro F1 0,9019 (tăng +0,0111 so với bản gốc 0,8908).
    - baseline_comparison_full.csv không tồn tại -> dùng final_model_comparison.csv
      (đã có đủ TabNet + toàn bộ baseline, đúng cột cần thiết).
    - Ảnh attention thực tế được lưu với tên att_*.png (không phải attention_*.png).
    - Ảnh SHAP per-class heatmap thực tế có dấu gạch dưới: shap_per_class_heatmap.png.
    """
    return {
        # Kết quả phân loại
        "results_json": data_dir / "results_smote.json",  # thực chất là kết quả GỐC (bug đặt tên trong notebook)
        "results_smote_json": data_dir / "results_smote_real.json",  # chưa tồn tại - chưa có lần chạy SMOTE thật
        "baseline_csv": data_dir / "final_model_comparison.csv",
        "kfold_tabnet_json": data_dir / "kfold_tabnet_results.json",
        "timing_json": data_dir / "timing_summary_full.json",

        # SHAP toàn cục / theo lớp
        "shap_beeswarm": data_dir / "shap_global_beeswarm.png",
        "shap_bar": data_dir / "shap_global_bar.png",
        "shap_perclass_heatmap": data_dir / "shap_per_class_heatmap.png",

        # Attention
        "attention_global": data_dir / "att_global.png",
        "attention_perstep": data_dir / "att_per_step.png",
        "attention_perclass": data_dir / "att_per_class.png",

        # Local top-features (tùy chọn, xem hướng dẫn ở Trang 4) - vẫn chưa có file này
        "local_top_features_json": data_dir / "local_top_features.json",

        # Ảnh SHAP waterfall cho MẪU BỊ NHẦM LẪN LỚN NHẤT (duy nhất 1 ảnh có sẵn)
        # File thật: err_mc_shap_waterfall.png, ứng với cặp nhầm lẫn có số mẫu
        # lớn nhất (portscanattack -> dosattack, xem Trang 5).
        "misclassified_top_waterfall": data_dir / "err_mc_shap_waterfall.png",
    }


# ==============================================================================
# HÀM TIỆN ÍCH
# ==============================================================================

def load_json_safe(path):
    if not Path(path).exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Không đọc được `{Path(path).name}`: {e}")
        return None


def show_image_safe(path, caption=""):
    if Path(path).exists():
        st.image(str(path), caption=caption, use_container_width=True)
        return True
    else:
        st.info(
            f"Chưa tìm thấy file ảnh: `{Path(path).name}`. "
            f"Kiểm tra lại thư mục dữ liệu ở sidebar bên trái."
        )
        return False


def metric_row(label, value, help_text=None):
    st.metric(label, value, help=help_text)


def fmt_pct(x, decimals=2):
    try:
        return f"{float(x) * 100:.{decimals}f}%"
    except (TypeError, ValueError):
        return "N/A"


def pick_column(df: pd.DataFrame, candidates: list[str]):
    """Chọn cột đầu tiên trong 'candidates' khớp CHÍNH XÁC (không dò mờ) với
    cột trong df. Trả về None nếu không cột nào khớp."""
    lower_map = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


# ------------------------------------------------------------------------------
# Bảng per-class: thêm hàng tổng hợp, làm tròn, tô cảnh báo mẫu ít
# ------------------------------------------------------------------------------

RENAME_COLS = {
    "precision": "Precision",
    "recall": "Recall",
    "f1_score": "F1-Score",
    "f1": "F1-Score",
    "auc": "AUC",
    "support": "Số mẫu",
}


def build_perclass_display_df(results: dict) -> pd.DataFrame | None:
    per_class = results.get("per_class_metrics", results.get("per_class"))
    if not per_class:
        return None

    df = pd.DataFrame(per_class).T
    df.index.name = "Lớp"

    total_support = df["support"].sum() if "support" in df.columns else None

    macro = results.get("macro_avg")
    weighted = results.get("weighted_avg")
    if macro:
        row = dict(macro)
        row["support"] = total_support
        df.loc["MACRO AVG"] = row
    if weighted:
        row = dict(weighted)
        row["support"] = total_support
        df.loc["WEIGHTED AVG"] = row

    df = df.rename(columns=RENAME_COLS)

    numeric_cols = [c for c in ["Precision", "Recall", "F1-Score", "AUC"] if c in df.columns]
    df[numeric_cols] = df[numeric_cols].astype(float).round(4)
    if "Số mẫu" in df.columns:
        df["Số mẫu"] = df["Số mẫu"].apply(lambda v: int(v) if pd.notna(v) else v)

    return df


def contrast_table_style(df):
    """Tô nền trắng chữ đen cho bảng - đối lập hoàn toàn với nền tối của
    dashboard để dễ đọc."""
    return df.style.set_properties(**{
        "background-color": "#FFFFFF",
        "color": "#111111",
    })


def render_perclass_table(df: pd.DataFrame):
    def highlight_low_support(row):
        support = row.get("Số mẫu", None)
        is_summary = row.name in ("MACRO AVG", "WEIGHTED AVG")
        if not is_summary and pd.notna(support) and support < LOW_SUPPORT_THRESHOLD:
            return ["background-color: #FBEBD1; color: #111111"] * len(row)
        if is_summary:
            return ["font-weight: 600; background-color: #E3F3EF; color: #111111"] * len(row)
        return ["background-color: #FFFFFF; color: #111111"] * len(row)

    try:
        styled = df.style.apply(highlight_low_support, axis=1)
        st.dataframe(styled, use_container_width=True)
    except Exception:
        # Phòng khi phiên bản Streamlit/pandas không hỗ trợ Styler -> hiện bảng thường
        st.dataframe(contrast_table_style(df), use_container_width=True)

    st.caption(
        f"Các hàng tô đỏ nhạt là lớp có dưới {LOW_SUPPORT_THRESHOLD} mẫu kiểm thử "
    )

    csv_bytes = df.to_csv().encode("utf-8-sig")  # utf-8-sig để Excel mở tiếng Việt không lỗi font
    st.download_button(
        "Tải bảng kết quả (CSV)",
        data=csv_bytes,
        file_name="ketqua_perclass.csv",
        mime="text/csv",
    )


# ------------------------------------------------------------------------------
# Ma trận nhầm lẫn vẽ trực tiếp từ JSON (không phụ thuộc ảnh PNG dựng sẵn)
# ------------------------------------------------------------------------------

def plot_confusion_matrix(cm, classes, show_percent=False):
    cm = np.array(cm, dtype=float)
    if show_percent:
        row_sums = cm.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.nan_to_num(cm / row_sums * 100)
        text = [[f"{v:.1f}%" for v in row] for row in z]
        colorbar_title = "% theo hàng"
    else:
        z = cm
        text = [[f"{int(v)}" for v in row] for row in cm]
        colorbar_title = "Số mẫu"

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=classes,
            y=classes,
            text=text,
            texttemplate="%{text}",
            colorscale=[[0, "#1A222B"], [0.55, "#1F6F70"], [1, "#4FD1C5"]],
            colorbar=dict(
                title=dict(text=colorbar_title, font=dict(color="#8B97A6")),
                tickfont=dict(color="#8B97A6"),
            ),
            hovertemplate="Thực tế: %{y}<br>Dự đoán: %{x}<br>Giá trị: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title="Nhãn dự đoán",
        yaxis_title="Nhãn thực tế",
        yaxis_autorange="reversed",
        height=560,
        margin=dict(t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Mono, monospace", color="#E7ECF2"),
        xaxis=dict(color="#8B97A6"),
        yaxis=dict(color="#8B97A6"),
    )
    return fig


def get_confusion_pairs(cm, classes):
    """Trả về danh sách (nhãn_thật, nhãn_dự_đoán, số_mẫu) cho các ô ngoài
    đường chéo, sắp xếp giảm dần theo số mẫu bị nhầm."""
    cm = np.array(cm)
    total_errors = int(cm.sum()) - int(np.trace(cm))
    pairs = []
    for i, true_cls in enumerate(classes):
        for j, pred_cls in enumerate(classes):
            if i != j and cm[i][j] > 0:
                pairs.append((true_cls, pred_cls, int(cm[i][j])))
    pairs.sort(key=lambda x: -x[2])
    return pairs, total_errors


# ==============================================================================
# CẤU HÌNH TRANG + SIDEBAR
# ==============================================================================

st.set_page_config(
    page_title="Dashboard IDS - TabNet + XAI",
    layout="wide",
)

# ------------------------------------------------------------------------------
# TOKEN HỆ THỐNG THIẾT KẾ — "Phòng điều khiển SCADA"
# Chủ đề: IDS cho giao thức công nghiệp IEC 104 (trạm biến áp/SCADA). Bảng màu
# và kiểu chữ mô phỏng màn hình HMI phòng điều khiển thay vì mặc định generic.
# ------------------------------------------------------------------------------
UI = {
    "bg": "#10161D",        # nền graphite-navy như panel điều khiển
    "panel": "#1A222B",     # bề mặt panel nổi
    "panel_raised": "#212B35",
    "line": "#2B3542",      # đường viền hairline kiểu sơ đồ đấu nối
    "text": "#E7ECF2",
    "muted": "#8B97A6",
    "cyan": "#4FD1C5",      # tín hiệu bình thường / dữ liệu
    "amber": "#E8A33D",     # cảnh báo / điểm nhấn cấu trúc
    "red": "#E5484D",       # lỗi / nhầm lẫn
}

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

.stApp {{
    background-color: {UI["bg"]} !important;
    color: {UI["text"]} !important;
    font-family: 'IBM Plex Sans', sans-serif;
}}

h1, .stApp h1 {{
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em;
    color: {UI["text"]} !important;
    border-bottom: 1px solid {UI["line"]};
    padding-bottom: 0.5rem;
}}
h2, h3, .stApp h2, .stApp h3, h4 {{
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
    color: {UI["text"]} !important;
}}

[data-testid="stMetricValue"] {{
    font-family: 'IBM Plex Mono', monospace !important;
    color: {UI["cyan"]} !important;
    font-weight: 600 !important;
}}
[data-testid="stMetricLabel"] {{
    font-family: 'IBM Plex Mono', monospace !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 0.72rem !important;
    color: {UI["muted"]} !important;
}}
[data-testid="stMetric"] {{
    background: {UI["panel"]};
    border: 1px solid {UI["line"]};
    border-left: 2px solid {UI["cyan"]};
    border-radius: 2px;
    padding: 0.75rem 1rem;
}}

section[data-testid="stSidebar"] {{
    background-color: #0C1116 !important;
    border-right: 1px solid {UI["line"]};
}}
section[data-testid="stSidebar"] * {{
    color: {UI["text"]} !important;
}}
section[data-testid="stSidebar"] [role="radiogroup"] label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    border-left: 2px solid transparent;
    padding-left: 0.5rem;
    transition: border-color 0.15s ease;
}}
section[data-testid="stSidebar"] [role="radiogroup"] label:hover {{
    border-left: 2px solid {UI["amber"]};
}}

[data-testid="stDataFrame"] {{
    border: 1px solid {UI["line"]} !important;
    border-radius: 2px;
}}
[data-testid="stDataFrame"] * {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.82rem !important;
}}

hr {{
    border: none !important;
    border-top: 1px solid {UI["line"]} !important;
    margin: 1.75rem 0 !important;
}}

button[data-baseweb="tab"] {{
    font-family: 'IBM Plex Mono', monospace !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-size: 0.78rem !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: {UI["cyan"]} !important;
    border-bottom-color: {UI["cyan"]} !important;
}}

.stDownloadButton button, .stButton button {{
    background-color: {UI["panel_raised"]} !important;
    color: {UI["cyan"]} !important;
    border: 1px solid {UI["cyan"]} !important;
    font-family: 'IBM Plex Mono', monospace !important;
    border-radius: 2px !important;
}}
.stDownloadButton button:hover, .stButton button:hover {{
    background-color: {UI["cyan"]} !important;
    color: {UI["bg"]} !important;
}}

[data-testid="stExpander"] {{
    background: {UI["panel"]};
    border: 1px solid {UI["line"]} !important;
    border-radius: 2px;
}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


st.sidebar.markdown(
    f"<div style='font-family:\"Space Grotesk\",sans-serif; font-weight:600; "
    f"font-size:1.35rem; color:{UI['text']};'>Dashboard Operator</div>"
    f"<div style='font-family:\"IBM Plex Mono\",monospace; font-size:0.72rem; "
    f"color:{UI['amber']}; letter-spacing:0.05em; margin:0.4rem 0 0.8rem 0;'>"
    f"● OFFLINE DISPLAY MODE</div>",
    unsafe_allow_html=True,
)
st.sidebar.caption(
    "Công cụ hiển thị OFFLINE các kết quả đã tính sẵn. "
)

with st.sidebar.expander("Thư mục dữ liệu", expanded=False):
    data_dir_str = st.text_input("Đường dẫn thư mục dữ liệu", value=str(DEFAULT_DATA_DIR))
    st.caption(
        "Mặc định là thư mục `data/` cạnh file app.py (đường dẫn tương đối, "
        "chạy được trên mọi máy). Đổi ở đây nếu dữ liệu của bạn nằm chỗ khác."
    )

DATA_DIR = Path(data_dir_str)
FILES = build_files_dict(DATA_DIR)

PAGE = st.sidebar.radio("Chọn mục", list(PAGE_TO_THESIS_SECTION.keys()))

st.sidebar.divider()
st.sidebar.caption("Nguồn dữ liệu: SANDI-2024 / IEC 104")
#st.sidebar.caption(f"Thư mục hiện tại: `{DATA_DIR}`")


# ==============================================================================
# TRANG 1 - TỔNG QUAN PHÂN LOẠI
# ==============================================================================

if PAGE == "1. Tổng quan phân loại":
    st.title("Tổng quan hiệu năng phân loại")
    st.caption(
        "Kết quả trên tập kiểm thử SANDI-2024/IEC 104. "

    )

    results = load_json_safe(FILES["results_json"])

    if results:
        macro = results.get("macro_avg", {})
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            metric_row("Accuracy", fmt_pct(results.get("accuracy", 0)))
        with col2:
            metric_row("Macro F1", f"{macro.get('f1_score', 0):.4f}")
        with col3:
            metric_row("Macro AUC", f"{macro.get('auc', 0):.4f}")
        with col4:
            weighted = results.get("weighted_avg", {})
            metric_row("Weighted F1", f"{weighted.get('f1_score', 0):.4f}")

        st.divider()
        st.subheader("Phân bố số mẫu theo lớp")
        per_class = results.get("per_class_metrics", results.get("per_class"))
        if per_class:
            support = {cls: v.get("support", 0) for cls, v in per_class.items()}
            fig_support = go.Figure(
                go.Bar(x=list(support.keys()), y=list(support.values()),
                       marker_color="#4FD1C5")
            )
            fig_support.update_yaxes(type="log", title="Số mẫu (thang log)", color="#8B97A6")
            fig_support.update_xaxes(color="#8B97A6")
            fig_support.update_layout(
                xaxis_title="Lớp tấn công",
                height=380,
                margin=dict(t=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="IBM Plex Mono, monospace", color="#E7ECF2"),
            )
            st.plotly_chart(fig_support, use_container_width=True)


        st.divider()
        st.subheader("Chi tiết theo từng lớp")
        df_display = build_perclass_display_df(results)
        if df_display is not None:
            render_perclass_table(df_display)
        else:
            st.warning(
                "Không tìm thấy dữ liệu per-class trong results.json. Kiểm tra "
                "lại cấu trúc key của file (cần 'per_class_metrics')."
            )
    else:
        st.error(
            f"Không tìm thấy file `{FILES['results_json'].name}` trong thư mục dữ liệu. "
            "Hãy chắc chắn đã chạy xong notebook huấn luyện và xuất results.json."
        )
        st.stop()

    st.divider()
    st.subheader("Ma trận nhầm lẫn (Confusion Matrix)")
    cm = results.get("confusion_matrix") if results else None
    if cm:
        show_percent = st.toggle("Hiển thị theo % (thay vì số lượng)", value=False)
        st.plotly_chart(plot_confusion_matrix(cm, CLASSES, show_percent), use_container_width=True)

    else:
        st.info(
            "results.json chưa có trường 'confusion_matrix'. Nếu muốn dùng ảnh "
            "PNG dựng sẵn thay thế, đặt file `confusion_matrix.png` vào thư mục "
            "dữ liệu."
        )
        show_image_safe(DATA_DIR / "confusion_matrix.png", "Confusion matrix trên tập kiểm thử")


# ==============================================================================
# TRANG 2 - SHAP TOÀN CỤC
# ==============================================================================

elif PAGE == "2. SHAP toàn cục":
    st.title("SHAP - Giải thích toàn cục")

    tab1, tab2 = st.tabs(["Beeswarm", "Bar chart"])
    with tab1:
        show_image_safe(FILES["shap_beeswarm"], "SHAP Beeswarm - toàn cục")
    with tab2:
        show_image_safe(FILES["shap_bar"], "SHAP Bar - Top đặc trưng quan trọng nhất")
        st.caption(
            "Xếp hạng đặc trưng theo giá trị |SHAP| trung bình, càng cao càng "
            "ảnh hưởng nhiều đến quyết định của mô hình, tính trên toàn bộ các lớp."
        )


# ==============================================================================
# TRANG 3 - SHAP THEO LỚP
# ==============================================================================

elif PAGE == "3. SHAP theo lớp":
    st.title("SHAP - Phân tích theo từng lớp tấn công")
    st.caption(
        "Mỗi lớp tấn công có thể phụ thuộc vào một nhóm đặc trưng khác nhau. "
        "Heatmap bên dưới cho thấy mức độ ảnh hưởng (|SHAP| trung bình) của "
        "top đặc trưng đối với từng lớp."
    )
    show_image_safe(FILES["shap_perclass_heatmap"], "SHAP per-class heatmap")


# ==============================================================================
# TRANG 4 - GIẢI THÍCH CỤC BỘ (SHAP + LIME)
# ==============================================================================

elif PAGE == "4. Giải thích cục bộ (SHAP + LIME)":
    st.title("Giải thích cục bộ cho 1 mẫu cụ thể")
    st.caption(
        "Chọn 1 lớp tấn công để xem giải thích SHAP Waterfall (bên trái) và "
        "LIME (bên phải) cho cùng 1 mẫu dự đoán đúng, đại diện cho lớp đó."
    )

    selected_class = st.selectbox("Chọn lớp tấn công", LOCAL_SAMPLE_CLASSES, index=0)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### SHAP Waterfall")
        shap_path = DATA_DIR / f"shap_waterfall_{selected_class}.png"
        show_image_safe(shap_path, f"SHAP Waterfall - {selected_class}")

    with col2:
        st.markdown("#### LIME Explanation")
        lime_path = DATA_DIR / f"lime_{selected_class}.png"
        show_image_safe(lime_path, f"LIME - {selected_class}")

    st.divider()



# ==============================================================================
# TRANG 5 - MẪU BỊ DỰ ĐOÁN SAI (MỚI)
# ==============================================================================

elif PAGE == "5. Mẫu bị dự đoán sai":
    st.title("Phân tích các mẫu bị dự đoán sai")
    st.caption(
        "Trang này minh họa trực tiếp bằng số liệu cho phần phân tích lỗi ở "
        "mục 4.2 của luận văn - đặc biệt là cặp nhầm lẫn lớn nhất "
        "portscanattack → dosattack (163 mẫu, 88,6% tổng lỗi)."
    )

    results = load_json_safe(FILES["results_json"])
    cm = results.get("confusion_matrix") if results else None

    if not cm:
        st.warning(
            "Cần có trường 'confusion_matrix' trong results.json để dùng trang này."
        )
        st.stop()

    pairs, total_errors = get_confusion_pairs(cm, CLASSES)

    if not pairs:
        st.success("Không có mẫu nào bị dự đoán sai trong ma trận nhầm lẫn này.")
        st.stop()

    st.subheader(f"Danh sách các cặp nhầm lẫn (tổng {total_errors} lỗi)")
    df_pairs = pd.DataFrame(pairs, columns=["Nhãn thật", "Nhãn dự đoán", "Số mẫu"])
    df_pairs["% tổng lỗi"] = (df_pairs["Số mẫu"] / total_errors * 100).round(1)
    st.dataframe(contrast_table_style(df_pairs), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Chi tiết cặp nhầm lẫn lớn nhất")

    # Notebook XAI (TABNET_XAI_LUANVAN_v2.ipynb, cell phân tích lỗi) chỉ tính SHAP
    # waterfall cho ĐÚNG 1 cặp nhầm lẫn: cặp có số mẫu lớn nhất (error_pairs[0]).
    # Vì get_confusion_pairs() ở trên sắp xếp cùng logic (giảm dần theo số mẫu),
    # cặp lớn nhất trong danh sách (index 0) chính là cặp duy nhất đã có ảnh sẵn
    # -> bỏ lựa chọn các cặp khác, chỉ hiển thị trực tiếp cặp này.
    true_cls, pred_cls, n_samples = pairs[0]

    st.markdown(
        f"TabNet dự đoán **{n_samples}** mẫu thuộc lớp **{true_cls}** thành "
        f"lớp **{pred_cls}** ({n_samples / total_errors * 100:.1f}% tổng số lỗi "
        f"phân loại của mô hình)."
    )

    waterfall_path = FILES["misclassified_top_waterfall"]
    show_image_safe(
        waterfall_path, f"SHAP Waterfall - mẫu {true_cls} bị đoán nhầm thành {pred_cls}"
    )


# ==============================================================================
# TRANG 6 - ATTENTION
# ==============================================================================

elif PAGE == "6. Attention (TabNet nội tại)":
    st.title("TabNet Attention - Giải thích nội tại (Tầng 1)")
    st.caption(
        "Attention được tính ngay trong quá trình suy luận của TabNet, gần "
        "như không tốn thêm chi phí tính toán (khác hẳn SHAP)."
    )

    tab1, tab2, tab3 = st.tabs(["Global importance", "Per-step", "Per-class"])
    with tab1:
        show_image_safe(FILES["attention_global"], "Attention - Global feature importance")
        st.caption("Mức độ mô hình chú ý trung bình đến từng đặc trưng, tính trên toàn bộ tập kiểm thử.")
    with tab2:
        show_image_safe(FILES["attention_perstep"], "Attention - Per-step (5 bước quyết định)")
        st.caption("TabNet ra quyết định qua nhiều bước tuần tự (n_steps=5), mỗi bước có thể chú ý vào nhóm đặc trưng khác nhau.")
    with tab3:
        show_image_safe(FILES["attention_perclass"], "Attention - Per-class heatmap")
        st.caption("Ví dụ CWR Flag Count nổi bật ở ntpddosattack, RST Flag Count nổi bật ở iec104starvationattack.")
