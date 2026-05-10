import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings(“ignore”)

# ── PAGE CONFIG ──────────────────────────────────────────────

st.set_page_config(
page_title=“Dự đoán giá xe máy cũ – HCM”,
page_icon=“🏍️”,
layout=“wide”,
)

# ── LOAD & CLEAN DATA ─────────────────────────────────────────

@st.cache_data
def load_data():
df = pd.read_excel(“dataxe.xlsx”)

```
# Strip trailing/leading spaces
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str).str.strip()

# ODO to numeric
df["Số km đã chạy (ODO)"] = pd.to_numeric(
    df["Số km đã chạy (ODO)"].astype(str).str.replace(",", "").str.replace(".", ""),
    errors="coerce",
)
df.dropna(subset=["Số km đã chạy (ODO)"], inplace=True)
df["Số km đã chạy (ODO)"] = df["Số km đã chạy (ODO)"].astype(int)

# Normalise khu vực (keep only district name, strip sub-ward notes)
df["Khu vực"] = (
    df["Khu vực bán (Hồ Chí Minh)"]
    .str.strip()
    .str.split("(")
    .str[0]
    .str.strip()
)

# Encode categoricals
le_hang   = LabelEncoder()
le_dong   = LabelEncoder()
le_khu    = LabelEncoder()
le_thay   = LabelEncoder()

df["hang_enc"]  = le_hang.fit_transform(df["Hãng xe"])
df["dong_enc"]  = le_dong.fit_transform(df["Dòng xe"])
df["khu_enc"]   = le_khu.fit_transform(df["Khu vực"])
df["thay_enc"]  = le_thay.fit_transform(df["Đã thay phụ tùng chưa ?"])

return df, le_hang, le_dong, le_khu, le_thay
```

@st.cache_resource
def train_models(df):
features = [“hang_enc”, “dong_enc”, “Năm sản xuất”,
“Số km đã chạy (ODO)”, “khu_enc”, “Tình trạng xe %”, “thay_enc”]
X = df[features]
y = df[“Giá bán”]

```
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest":     RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds  = model.predict(X_test)
    r2     = r2_score(y_test, preds)
    mae    = mean_absolute_error(y_test, preds)
    cv_r2  = cross_val_score(model, X, y, cv=5, scoring="r2").mean()
    results[name] = {
        "model":  model,
        "r2":     r2,
        "mae":    mae,
        "cv_r2":  cv_r2,
    }

best_name = max(results, key=lambda k: results[k]["cv_r2"])
return results, best_name, features
```

# ── LOAD ──────────────────────────────────────────────────────

df, le_hang, le_dong, le_khu, le_thay = load_data()
model_results, best_name, features = train_models(df)

# ── HEADER ────────────────────────────────────────────────────

st.markdown(
“””
<h1 style='text-align:center;color:#1a73e8;'>🏍️ Dự đoán giá xe máy cũ – TP.HCM</h1>
<p style='text-align:center;color:gray;'>Dữ liệu thực tế từ 241 xe đã qua sử dụng tại TP.HCM</p>
“””,
unsafe_allow_html=True,
)

# ── TABS ──────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs([“🔮 Dự đoán giá”, “📊 Phân tích dữ liệu”, “🤖 So sánh mô hình”])

# ════════════════════════════════════════════════════════════════

# TAB 1 – DỰ ĐOÁN

# ════════════════════════════════════════════════════════════════

with tab1:
st.subheader(“Nhập thông tin xe cần dự đoán”)

```
col1, col2, col3 = st.columns(3)

with col1:
    hang_list = sorted(df["Hãng xe"].unique())
    hang_chon = st.selectbox("🏭 Hãng xe", hang_list)

    dong_list = sorted(df[df["Hãng xe"] == hang_chon]["Dòng xe"].unique())
    dong_chon = st.selectbox("🏍️ Dòng xe", dong_list)

    nam_sx = st.number_input(
        "📅 Năm sản xuất",
        min_value=int(df["Năm sản xuất"].min()),
        max_value=int(df["Năm sản xuất"].max()),
        value=2018,
    )

with col2:
    odo = st.number_input(
        "🛣️ Số km đã chạy (ODO)",
        min_value=0,
        max_value=200000,
        value=20000,
        step=1000,
    )

    khu_list = sorted(df["Khu vực"].unique())
    khu_chon = st.selectbox("📍 Khu vực bán", khu_list)

with col3:
    tinh_trang = st.slider(
        "⭐ Tình trạng xe (%)",
        min_value=int(df["Tình trạng xe %"].min()),
        max_value=int(df["Tình trạng xe %"].max()),
        value=85,
    )

    thay_phu_tung = st.radio(
        "🔧 Đã thay phụ tùng chưa?",
        options=["Chưa thay", "Đã thay"],
        horizontal=True,
    )

    mo_hinh_chon = st.selectbox(
        "🤖 Mô hình dự đoán",
        options=list(model_results.keys()),
        index=list(model_results.keys()).index(best_name),
        help=f"Mô hình tốt nhất (CV R²): {best_name}",
    )

if st.button("🔮 Dự đoán giá xe", use_container_width=True, type="primary"):
    # Encode inputs
    try:
        hang_enc  = le_hang.transform([hang_chon])[0]
        dong_enc  = le_dong.transform([dong_chon])[0]
        khu_enc   = le_khu.transform([khu_chon])[0]
        thay_enc  = le_thay.transform([thay_phu_tung])[0]
    except Exception as e:
        st.error(f"Lỗi encode: {e}")
        st.stop()

    X_pred = pd.DataFrame(
        [[hang_enc, dong_enc, nam_sx, odo, khu_enc, tinh_trang, thay_enc]],
        columns=features,
    )

    model  = model_results[mo_hinh_chon]["model"]
    gia_du_doan = model.predict(X_pred)[0]

    # Show result
    st.markdown("---")
    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.metric("💰 Giá dự đoán", f"{gia_du_doan:,.0f} đ",
                  f"≈ {gia_du_doan/1_000_000:.1f} triệu")
    col_r2.metric("🤖 Mô hình sử dụng", mo_hinh_chon)
    col_r3.metric("📈 R² (test set)", f"{model_results[mo_hinh_chon]['r2']:.3f}")

    # Range estimate ±15%
    low = gia_du_doan * 0.85
    high = gia_du_doan * 1.15
    st.info(f"📊 Khoảng giá tham khảo: **{low/1e6:.1f}M – {high/1e6:.1f}M VNĐ** (±15%)")

    # Similar cars
    st.subheader("🔍 Xe tương tự trong dữ liệu")
    sim = df[
        (df["Hãng xe"] == hang_chon) &
        (df["Dòng xe"] == dong_chon) &
        (df["Năm sản xuất"].between(nam_sx - 2, nam_sx + 2))
    ][["Hãng xe", "Dòng xe", "Năm sản xuất",
       "Số km đã chạy (ODO)", "Tình trạng xe %", "Giá bán"]].copy()
    sim["Giá bán"] = sim["Giá bán"].apply(lambda x: f"{x:,} đ")
    if sim.empty:
        st.info("Không tìm thấy xe tương tự trong dữ liệu.")
    else:
        st.dataframe(sim.reset_index(drop=True), use_container_width=True)
```

# ════════════════════════════════════════════════════════════════

# TAB 2 – PHÂN TÍCH

# ════════════════════════════════════════════════════════════════

with tab2:
st.subheader(“📊 Tổng quan dữ liệu”)

```
m1, m2, m3, m4 = st.columns(4)
m1.metric("Tổng số xe", len(df))
m2.metric("Số hãng", df["Hãng xe"].nunique())
m3.metric("Số dòng xe", df["Dòng xe"].nunique())
m4.metric("Giá TB", f"{df['Giá bán'].mean()/1e6:.1f}M")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("**Phân bố số xe theo hãng**")
    brand_cnt = df["Hãng xe"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(brand_cnt.index, brand_cnt.values,
                  color=["#1a73e8", "#34a853", "#ea4335", "#fbbc04",
                         "#9c27b0", "#00bcd4", "#ff5722", "#607d8b"])
    ax.set_xlabel("Hãng xe")
    ax.set_ylabel("Số xe")
    ax.set_title("Số xe theo hãng")
    plt.xticks(rotation=30, ha="right")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)

with col_b:
    st.markdown("**Giá trung bình theo hãng (triệu VNĐ)**")
    avg_price = df.groupby("Hãng xe")["Giá bán"].mean().sort_values(ascending=False) / 1e6
    fig2, ax2 = plt.subplots(figsize=(5, 3.5))
    bars2 = ax2.barh(avg_price.index, avg_price.values, color="#1a73e8")
    ax2.set_xlabel("Giá TB (triệu)")
    ax2.set_title("Giá TB theo hãng")
    for bar in bars2:
        ax2.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                 f"{bar.get_width():.1f}M", va="center", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig2)

st.markdown("**Phân bố giá bán**")
fig3, ax3 = plt.subplots(figsize=(8, 3))
ax3.hist(df["Giá bán"] / 1e6, bins=30, color="#1a73e8", edgecolor="white")
ax3.set_xlabel("Giá bán (triệu VNĐ)")
ax3.set_ylabel("Số xe")
ax3.set_title("Phân bố giá xe máy cũ")
plt.tight_layout()
st.pyplot(fig3)

# Vespa explanation
st.markdown("---")
st.markdown("### ❓ Tại sao Vespa lại có giá 100+ triệu?")
vespa_df = df[df["Dòng xe"] == "Vespa"][
    ["Hãng xe", "Dòng xe", "Năm sản xuất",
     "Số km đã chạy (ODO)", "Tình trạng xe %", "Giá bán"]
].copy()
vespa_df["Giá bán"] = vespa_df["Giá bán"].apply(lambda x: f"{x:,} đ")
st.dataframe(vespa_df.reset_index(drop=True), use_container_width=True)
st.info(
    "🔍 **Giải thích:** Vespa (của Piaggio) là dòng xe tay ga cao cấp nhập khẩu từ Ý. "
    "Chiếc Vespa 2016 giá 109 triệu trong dữ liệu ở tình trạng 90%, km thấp – "
    "đây là mức giá thực tế trên thị trường do Vespa giữ giá tốt và có thương hiệu lâu đời. "
    "So sánh: Honda SH 2025-2026 cũng có giá 90–110 triệu trong dữ liệu."
)

# Raw data view
with st.expander("📋 Xem toàn bộ dữ liệu"):
    show_df = df[["Hãng xe", "Dòng xe", "Năm sản xuất",
                  "Số km đã chạy (ODO)", "Khu vực",
                  "Tình trạng xe %", "Giá bán", "Đã thay phụ tùng chưa ?"]].copy()
    show_df["Giá bán"] = show_df["Giá bán"].apply(lambda x: f"{x:,}")
    st.dataframe(show_df.reset_index(drop=True), use_container_width=True)
```

# ════════════════════════════════════════════════════════════════

# TAB 3 – SO SÁNH MÔ HÌNH

# ════════════════════════════════════════════════════════════════

with tab3:
st.subheader(“🤖 So sánh hiệu suất các mô hình”)

```
rows = []
for name, res in model_results.items():
    rows.append({
        "Mô hình": name,
        "R² (test)": round(res["r2"], 4),
        "CV R² (5-fold)": round(res["cv_r2"], 4),
        "MAE (VNĐ)": f"{res['mae']:,.0f}",
        "⭐ Tốt nhất?": "✅ Có" if name == best_name else "",
    })

result_df = pd.DataFrame(rows)
st.dataframe(result_df.set_index("Mô hình"), use_container_width=True)

# Bar chart
fig4, axes = plt.subplots(1, 2, figsize=(10, 4))
names  = [r["Mô hình"] for r in rows]
r2s    = [model_results[n]["r2"] for n in names]
cv_r2s = [model_results[n]["cv_r2"] for n in names]
colors = ["#34a853" if n == best_name else "#1a73e8" for n in names]

axes[0].bar(names, r2s, color=colors)
axes[0].set_title("R² – Test Set")
axes[0].set_ylim(0, 1)
axes[0].set_ylabel("R²")
for i, v in enumerate(r2s):
    axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

axes[1].bar(names, cv_r2s, color=colors)
axes[1].set_title("CV R² – Cross-Validation (5-fold)")
axes[1].set_ylim(0, 1)
axes[1].set_ylabel("CV R²")
for i, v in enumerate(cv_r2s):
    axes[1].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

plt.xticks(rotation=15)
plt.tight_layout()
st.pyplot(fig4)

st.success(
    f"✅ **Mô hình tốt nhất: {best_name}** "
    f"(CV R² = {model_results[best_name]['cv_r2']:.4f}) – "
    "được chọn dựa trên Cross-Validation 5-fold để tránh overfitting."
)

st.markdown("---")
st.markdown("### 📌 Giải thích các chỉ số")
st.markdown(
    """
    | Chỉ số | Ý nghĩa |
    |--------|---------|
    | **R² (test)** | Tỷ lệ phương sai giải thích được trên tập test (càng gần 1 càng tốt) |
    | **CV R² (5-fold)** | R² trung bình qua 5 lần cross-validation – đánh giá khả năng tổng quát hoá |
    | **MAE** | Sai số tuyệt đối trung bình (VNĐ) – sai bao nhiêu so với giá thực |
    """
)

st.markdown("---")
st.markdown("### ❓ Tại sao dữ liệu có Honda nhưng không có 'dòng xe Honda'?")
st.info(
    "**Honda** là *hãng xe* (nhà sản xuất), không phải tên dòng xe. "
    "Các dòng xe của Honda trong dữ liệu gồm: SH, Vision, Wave, Air Blade, PCX, Future, Lead, Dream, Vario, Winner, Sonic, ... "
    "Khi chọn hãng Honda, app sẽ tự động lọc ra đúng các dòng xe thuộc Honda."
)
```