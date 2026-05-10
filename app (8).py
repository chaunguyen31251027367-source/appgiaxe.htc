import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Du doan gia xe may cu - HCM",
    page_icon="https://em-content.zobj.net/source/google/387/motorcycle_1f3cd-fe0f.png",
    layout="wide",
)

COL_HANG  = "H\u00e3ng xe"
COL_DONG  = "D\u00f2ng xe"
COL_NAM   = "N\u0103m s\u1ea3n xu\u1ea5t"
COL_ODO   = "S\u1ed1 km \u0111\u00e3 ch\u1ea1y (ODO)"
COL_KHU   = "Khu v\u1ef1c b\u00e1n (H\u1ed3 Ch\u00ed Minh)"
COL_TINH  = "T\u00ecnh tr\u1ea1ng xe %"
COL_GIA   = "Gi\u00e1 b\u00e1n"
COL_THAY  = "\u0110\u00e3 thay ph\u1ee5 t\u00f9ng ch\u01b0a ?"

@st.cache_data
def load_data():
    df = pd.read_excel("dataxe.xlsx")
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype) == "string":
            df[col] = df[col].astype(str).str.strip()
    df["odo_num"] = pd.to_numeric(
        df[COL_ODO].astype(str).str.replace(",", "").str.replace(".", ""),
        errors="coerce",
    )
    df.dropna(subset=["odo_num"], inplace=True)
    df["odo_num"] = df["odo_num"].astype(int)
    df["khu_clean"] = (
        df[COL_KHU].str.strip().str.split("(").str[0].str.strip()
    )
    le_hang = LabelEncoder()
    le_dong = LabelEncoder()
    le_khu  = LabelEncoder()
    le_thay = LabelEncoder()
    df["hang_enc"] = le_hang.fit_transform(df[COL_HANG])
    df["dong_enc"] = le_dong.fit_transform(df[COL_DONG])
    df["khu_enc"]  = le_khu.fit_transform(df["khu_clean"])
    df["thay_enc"] = le_thay.fit_transform(df[COL_THAY])
    return df, le_hang, le_dong, le_khu, le_thay

@st.cache_resource
def train_models(_df):
    feats = ["hang_enc", "dong_enc", COL_NAM, "odo_num", "khu_enc", COL_TINH, "thay_enc"]
    X = _df[feats]
    y = _df[COL_GIA]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest":     RandomForestRegressor(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
    }
    res = {}
    for name, m in candidates.items():
        m.fit(X_tr, y_tr)
        preds = m.predict(X_te)
        cv    = cross_val_score(m, X, y, cv=5, scoring="r2").mean()
        res[name] = {
            "model": m,
            "r2":    r2_score(y_te, preds),
            "mae":   mean_absolute_error(y_te, preds),
            "cv_r2": cv,
        }
    best = max(res, key=lambda k: res[k]["cv_r2"])
    return res, best, feats

df, le_hang, le_dong, le_khu, le_thay = load_data()
res, best_name, feats = train_models(df)

st.markdown(
    "<h1 style='text-align:center;color:#1a73e8;'>Du doan gia xe may cu - TP.HCM</h1>"
    "<p style='text-align:center;color:gray;'>Du lieu thuc te 241 xe tai TP.HCM</p>",
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["Du doan gia", "Phan tich du lieu", "So sanh mo hinh"])

with tab1:
    st.subheader("Nhap thong tin xe")
    c1, c2, c3 = st.columns(3)
    with c1:
        hang_list = sorted(df[COL_HANG].unique())
        hang_sel  = st.selectbox("Hang xe", hang_list)
        dong_list = sorted(df[df[COL_HANG] == hang_sel][COL_DONG].unique())
        dong_sel  = st.selectbox("Dong xe", dong_list)
        nam_sx    = st.number_input(
            "Nam san xuat",
            min_value=int(df[COL_NAM].min()),
            max_value=int(df[COL_NAM].max()),
            value=2018,
        )
    with c2:
        odo = st.number_input("So km da chay (ODO)", min_value=0, max_value=200000, value=20000, step=1000)
        khu_list = sorted(df["khu_clean"].unique())
        khu_sel  = st.selectbox("Khu vuc ban", khu_list)
    with c3:
        tinh = st.slider(
            "Tinh trang xe (%)",
            min_value=int(df[COL_TINH].min()),
            max_value=int(df[COL_TINH].max()),
            value=85,
        )
        thay = st.radio("Da thay phu tung chua?", ["Chua thay", "Da thay"], horizontal=True)
        mo_hinh = st.selectbox(
            "Mo hinh du doan",
            list(res.keys()),
            index=list(res.keys()).index(best_name),
        )

    if st.button("Du doan gia xe", use_container_width=True, type="primary"):
        thay_input = "Chua thay" if thay == "Chua thay" else "Da thay"
        try:
            he = le_hang.transform([hang_sel])[0]
            de = le_dong.transform([dong_sel])[0]
            ke = le_khu.transform([khu_sel])[0]
            te = le_thay.transform([thay_input])[0]
        except Exception as ex:
            st.error(f"Loi encode: {ex}")
            st.stop()
        X_in = pd.DataFrame([[he, de, nam_sx, odo, ke, tinh, te]], columns=feats)
        gia  = res[mo_hinh]["model"].predict(X_in)[0]
        r1, r2, r3 = st.columns(3)
        r1.metric("Gia du doan", f"{gia:,.0f} VND", f"~ {gia/1e6:.1f} trieu")
        r2.metric("Mo hinh", mo_hinh)
        r3.metric("R2 test", f"{res[mo_hinh]['r2']:.3f}")
        st.info(f"Khoang tham khao: {gia*0.85/1e6:.1f}M - {gia*1.15/1e6:.1f}M VND (+-15%)")
        similar = df[
            (df[COL_HANG] == hang_sel) &
            (df[COL_DONG] == dong_sel) &
            (df[COL_NAM].between(nam_sx - 2, nam_sx + 2))
        ][[COL_HANG, COL_DONG, COL_NAM, "odo_num", COL_TINH, COL_GIA]].copy()
        if not similar.empty:
            st.subheader("Xe tuong tu trong du lieu")
            similar[COL_GIA] = similar[COL_GIA].apply(lambda x: f"{x:,} VND")
            st.dataframe(similar.reset_index(drop=True), use_container_width=True)

with tab2:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tong so xe", len(df))
    m2.metric("So hang", df[COL_HANG].nunique())
    m3.metric("So dong xe", df[COL_DONG].nunique())
    m4.metric("Gia TB", f"{df[COL_GIA].mean()/1e6:.1f}M")

    ca, cb = st.columns(2)
    with ca:
        st.markdown("**So xe theo hang**")
        bc = df[COL_HANG].value_counts()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.bar(bc.index, bc.values, color="#1a73e8")
        ax.set_ylabel("So xe")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
    with cb:
        st.markdown("**Gia TB theo hang (trieu VND)**")
        avg = df.groupby(COL_HANG)[COL_GIA].mean().sort_values() / 1e6
        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        ax2.barh(avg.index, avg.values, color="#34a853")
        ax2.set_xlabel("Trieu VND")
        plt.tight_layout()
        st.pyplot(fig2)

    st.markdown("**Phan bo gia ban**")
    fig3, ax3 = plt.subplots(figsize=(8, 3))
    ax3.hist(df[COL_GIA] / 1e6, bins=30, color="#1a73e8", edgecolor="white")
    ax3.set_xlabel("Gia ban (trieu VND)")
    ax3.set_ylabel("So xe")
    plt.tight_layout()
    st.pyplot(fig3)

    st.markdown("---")
    st.markdown("### Tai sao Vespa lai co gia 109 trieu?")
    vespa = df[df[COL_DONG] == "Vespa"][[COL_HANG, COL_DONG, COL_NAM, "odo_num", COL_TINH, COL_GIA]].copy()
    vespa[COL_GIA] = vespa[COL_GIA].apply(lambda x: f"{x:,} VND")
    st.dataframe(vespa.reset_index(drop=True), use_container_width=True)
    st.info(
        "Vespa la xe tay ga cao cap nhap khau tu Y cua Piaggio, giu gia rat tot. "
        "Chiec Vespa 2016 (90% tinh trang, 28.000km) gia 109 trieu la gia that tren thi truong. "
        "So sanh: Honda SH 2025-2026 trong du lieu cung co gia 90-110 trieu."
    )

    with st.expander("Xem toan bo du lieu"):
        show = df[[COL_HANG, COL_DONG, COL_NAM, "odo_num", "khu_clean", COL_TINH, COL_GIA, COL_THAY]].copy()
        show[COL_GIA] = show[COL_GIA].apply(lambda x: f"{x:,}")
        st.dataframe(show.reset_index(drop=True), use_container_width=True)

with tab3:
    st.subheader("So sanh hieu suat cac mo hinh")
    rows = []
    for name, r in res.items():
        rows.append({
            "Mo hinh":    name,
            "R2 (test)":  round(r["r2"], 4),
            "CV R2":      round(r["cv_r2"], 4),
            "MAE (VND)":  f"{r['mae']:,.0f}",
            "Tot nhat":   "YES" if name == best_name else "",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Mo hinh"), use_container_width=True)

    names  = list(res.keys())
    r2s    = [res[n]["r2"] for n in names]
    cvs    = [res[n]["cv_r2"] for n in names]
    colors = ["#34a853" if n == best_name else "#1a73e8" for n in names]
    fig4, (ax4, ax5) = plt.subplots(1, 2, figsize=(10, 4))
    ax4.bar(names, r2s, color=colors)
    ax4.set_title("R2 - Test Set")
    ax4.set_ylim(0, 1)
    for i, v in enumerate(r2s):
        ax4.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax5.bar(names, cvs, color=colors)
    ax5.set_title("CV R2 - Cross Validation 5-fold")
    ax5.set_ylim(0, 1)
    for i, v in enumerate(cvs):
        ax5.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    plt.xticks(rotation=15)
    plt.tight_layout()
    st.pyplot(fig4)

    st.success(f"Mo hinh tot nhat: {best_name} (CV R2 = {res[best_name]['cv_r2']:.4f})")

    st.markdown("---")
    st.markdown("### Tai sao co hang Honda nhung khong co dong xe Honda?")
    st.info(
        "Honda la TEN HANG (nha san xuat), khong phai ten dong xe. "
        "Cac dong xe Honda trong du lieu: SH, Vision, Wave, Air Blade, PCX, Future, Lead, Dream, Vario, Winner, Sonic... "
        "Khi ban chon hang Honda, app se tu dong hien dung cac dong xe cua Honda."
    )
