# -*- coding: utf-8 -*-
"""
MyOldMotorbike — Used Motorbike Price Estimator · Ho Chi Minh City
FINAL VERSION
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="MyOldMotorbike | Price Estimator",
    page_icon="🏍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────
# COLOR PALETTE  (Pink × Black)
# ──────────────────────────────────────────────────────
P = {
    "pink":   "#FF2D78",
    "pink2":  "#FF69B4",
    "pink3":  "#FFB3D9",
    "black":  "#080810",
    "dark":   "#111120",
    "dark2":  "#1A1A2E",
    "card":   "#16162A",
    "text":   "#F0E6FF",
    "muted":  "#9988BB",
}

PINK_SEQ = ["#2D0015", "#650030", "#A80050", "#FF2D78", "#FF69B4", "#FFB3D9"]

# ──────────────────────────────────────────────────────
# GLOBAL CSS
# ──────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    color: {P['text']};
}}
.stApp {{
    background: radial-gradient(ellipse at top left, #1a0a18 0%, {P['black']} 60%);
}}

/* ── Sidebar ─────────────────────────────── */
[data-testid="stSidebar"] {{
    background: {P['dark']} !important;
    border-right: 1px solid rgba(255,45,120,0.2);
}}
[data-testid="stSidebar"] * {{ color: {P['text']} !important; }}

/* ── Hero ────────────────────────────────── */
.hero-title {{
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.6rem;
    letter-spacing: 4px;
    background: linear-gradient(90deg, {P['pink']}, {P['pink2']}, {P['pink3']}, {P['pink2']}, {P['pink']});
    background-size: 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 4s linear infinite;
    line-height: 1.05;
}}
@keyframes shimmer {{ from{{background-position:0%}} to{{background-position:300%}} }}
.hero-sub {{
    color: {P['muted']};
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 6px;
}}

/* ── Metric cards ────────────────────────── */
.metric-card {{
    background: {P['card']};
    border: 1px solid rgba(255,45,120,0.22);
    border-radius: 14px;
    padding: 18px 14px;
    text-align: center;
    transition: transform .2s, border-color .2s, box-shadow .2s;
}}
.metric-card:hover {{
    transform: translateY(-4px);
    border-color: {P['pink']};
    box-shadow: 0 8px 24px rgba(255,45,120,0.18);
}}
.metric-label {{
    color: {P['muted']};
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 8px;
}}
.metric-value {{
    color: {P['pink']};
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.9rem;
    letter-spacing: 1px;
}}
.metric-delta {{
    color: {P['pink3']};
    font-size: 0.72rem;
    font-weight: 600;
    margin-top: 4px;
}}

/* ── Section headers ─────────────────────── */
.section-header {{
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.3rem;
    letter-spacing: 3px;
    color: {P['pink']};
    border-bottom: 1px solid rgba(255,45,120,0.25);
    padding-bottom: 6px;
    margin: 28px 0 16px;
}}

/* ── Prediction box ──────────────────────── */
.pred-box {{
    background: linear-gradient(140deg, #2a001a 0%, #5a0035 55%, #8c0050 100%);
    border: 1px solid {P['pink']};
    border-radius: 20px;
    padding: 36px 20px;
    text-align: center;
    box-shadow: 0 0 48px rgba(255,45,120,0.28), inset 0 1px 0 rgba(255,105,180,.15);
}}
.pred-label {{
    color: {P['pink3']};
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 3px;
    margin-bottom: 10px;
}}
.pred-price {{
    color: #fff;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3rem;
    letter-spacing: 1px;
    text-shadow: 0 0 24px rgba(255,45,120,.5);
}}
.pred-range {{
    color: {P['pink3']};
    font-size: 0.8rem;
    font-weight: 600;
    margin-top: 10px;
}}

/* ── Chips ───────────────────────────────── */
.chip {{
    display: inline-block;
    background: rgba(255,45,120,.12);
    border: 1px solid rgba(255,45,120,.38);
    border-radius: 20px;
    padding: 4px 14px;
    color: {P['pink2']};
    font-size: 0.76rem;
    font-weight: 700;
    margin: 3px;
}}

/* ── Streamlit widgets ───────────────────── */
div[data-baseweb="select"] > div {{
    background: {P['card']} !important;
    border: 1px solid rgba(255,45,120,.3) !important;
}}
div[data-baseweb="input"] > div {{
    background: {P['card']} !important;
    border: 1px solid rgba(255,45,120,.3) !important;
}}
input, textarea {{ color: {P['text']} !important; }}
.stSelectbox label, .stSlider label,
.stRadio label, .stNumberInput label {{
    color: {P['text']} !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
}}

/* ── Button ──────────────────────────────── */
div.stButton > button {{
    background: linear-gradient(135deg, {P['pink']}, {P['pink2']});
    color: #fff;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 2px;
    border: none;
    border-radius: 12px;
    padding: 14px 32px;
    width: 100%;
    transition: transform .15s, box-shadow .15s;
}}
div.stButton > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 22px rgba(255,45,120,.4);
}}

/* ── Dataframe ───────────────────────────── */
[data-testid="stDataFrame"] {{
    border: 1px solid rgba(255,45,120,.18) !important;
    border-radius: 10px;
    overflow: hidden;
}}

::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: {P['dark']}; }}
::-webkit-scrollbar-thumb {{ background: {P['pink']}; border-radius: 3px; }}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────
# PLOTLY BASE THEME
# ──────────────────────────────────────────────────────
PT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.02)",
    font=dict(color=P["text"], family="Inter"),
    title_font=dict(color=P["pink2"], family="Inter", size=13),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)",
               zerolinecolor="rgba(255,255,255,0.07)", color=P["muted"]),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)",
               zerolinecolor="rgba(255,255,255,0.07)", color=P["muted"]),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=P["text"])),
    margin=dict(t=44, b=16, l=10, r=10),
    colorway=[P["pink"], P["pink2"], P["pink3"], "#CC0055", "#FF99CC", "#AA0044"],
)

# ──────────────────────────────────────────────────────
# DATA — load & clean
# ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_excel("dataxe.xlsx", sheet_name="DuLieuXe_Cleaned")
    df.columns = ["Brand", "Model", "Year", "ODO", "District",
                  "Condition", "Price", "PartsReplaced"]

    # ODO: already integers in new data, but keep safe
    df["ODO"] = pd.to_numeric(
        df["ODO"].astype(str).str.replace(",", "."), errors="coerce"
    ).fillna(0).astype(int)

    # Brand
    df["Brand"] = df["Brand"].str.strip().str.capitalize()
    df["Brand"] = df["Brand"].replace({"Sym": "SYM"})

    # Model — normalise common variants
    df["Model"] = df["Model"].str.strip()
    df["Model"] = df["Model"].replace({
        "Air blade": "Air Blade", "Airblade": "Air Blade", "AirBlade": "Air Blade",
        "Honda AirBlade": "Air Blade",
        "LEAD": "Lead",
        "Winner X": "Winner", "Winner V1": "Winner",
        "Visson": "Vision",
        "Grande smartkey": "Grande Smartkey",
    })

    # PartsReplaced — strip space, unify to English
    df["PartsReplaced"] = df["PartsReplaced"].str.strip()
    replaced_map = {
        "Da thay": "Replaced", "Đã thay": "Replaced",
        "Chua thay": "Not Replaced", "Chưa thay": "Not Replaced",
    }
    df["PartsReplaced"] = df["PartsReplaced"].replace(replaced_map)
    # Catch anything that slipped through
    df["PartsReplaced"] = df["PartsReplaced"].apply(
        lambda v: "Replaced"
        if str(v).lower().startswith(("replaced", "da", "đã"))
        else "Not Replaced"
    )

    # District — strip invisible chars & sub-ward suffixes
    df["District"] = (
        df["District"]
        .astype(str)
        .str.replace(r"\xa0.*", "", regex=True)
        .str.strip()
    )

    df["Age"]    = 2026 - df["Year"]
    df["PriceM"] = df["Price"] / 1_000_000
    return df


# ──────────────────────────────────────────────────────
# MODEL — train once, cache
# ──────────────────────────────────────────────────────
@st.cache_resource
def train_models(_df):
    df2 = _df.copy()
    cat_cols = ["Brand", "Model", "District", "PartsReplaced"]
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df2[cat_cols] = encoder.fit_transform(df2[cat_cols])

    FEATS = ["Brand", "Model", "ODO", "District", "Condition", "PartsReplaced", "Age"]
    X, y  = df2[FEATS], df2["Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler   = StandardScaler()
    num_cols = ["ODO", "Condition", "Age"]
    X_tr, X_te = X_train.copy(), X_test.copy()
    X_tr[num_cols] = scaler.fit_transform(X_tr[num_cols])
    X_te[num_cols] = scaler.transform(X_te[num_cols])

    candidates = {
        "Linear Regression":  LinearRegression(),
        "Random Forest":      RandomForestRegressor(
                                  n_estimators=300, max_depth=None,
                                  min_samples_leaf=2, random_state=42),
        "Gradient Boosting":  GradientBoostingRegressor(
                                  n_estimators=300, learning_rate=0.07,
                                  max_depth=5, random_state=42),
    }

    results, fitted = {}, {}
    for name, m in candidates.items():
        m.fit(X_tr, y_train)
        yp = m.predict(X_te)
        results[name] = {
            "MAE":  mean_absolute_error(y_test, yp),
            "RMSE": np.sqrt(mean_squared_error(y_test, yp)),
            "R2":   r2_score(y_test, yp),
        }
        fitted[name] = m

    best_name = max(results, key=lambda k: results[k]["R2"])
    return encoder, scaler, fitted[best_name], best_name, results, FEATS


# ──────────────────────────────────────────────────────
# LOAD DATA + MODEL
# ──────────────────────────────────────────────────────
with st.spinner("Loading data & training models…"):
    df = load_data()
    encoder, scaler, best_model, best_name, model_results, FEAT_COLS = train_models(df)

# ──────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center; padding:20px 0 8px'>
        <div style='font-size:2.6rem'>🏍️</div>
        <div style='font-family:"Bebas Neue",sans-serif; font-size:1.35rem;
                    letter-spacing:2px; color:{P["pink"]}; margin-top:4px;'>
            MyOldMotorbike
        </div>
        <div style='font-size:0.65rem; color:{P["muted"]}; letter-spacing:2.5px;
                    text-transform:uppercase; margin-top:3px;'>
            HCMC · Used Bike Pricer
        </div>
    </div>
    <hr style='border-color:rgba(255,45,120,.2); margin:10px 0;'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "nav",
        ["📊  Overview",
         "🔍  Market Analysis",
         "🤖  ML Models",
         "🏷️  Price Estimator"],
        label_visibility="collapsed",
    )

    st.markdown(f"""
    <hr style='border-color:rgba(255,45,120,.15); margin:16px 0 10px;'>
    <div style='font-size:0.7rem; color:{P["muted"]}; line-height:2.1;'>
        <b style='color:{P["pink2"]}'>Dataset</b>&nbsp; {len(df)} bikes · HCMC<br>
        <b style='color:{P["pink2"]}'>Features</b>&nbsp; 7 input variables<br>
        <b style='color:{P["pink2"]}'>Best model</b>&nbsp; {best_name}<br>
        <b style='color:{P["pink2"]}'>R² score</b>&nbsp; {model_results[best_name]["R2"]:.4f}
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════
if page == "📊  Overview":

    st.markdown('<div class="hero-title">DATA OVERVIEW</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="hero-sub">Used motorbike listings · Ho Chi Minh City · 2024–2026</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # KPI cards
    cols = st.columns(5)
    kpis = [
        ("Total Bikes",  f"{len(df):,}",                   "records"),
        ("Brands",       str(df["Brand"].nunique()),        "manufacturers"),
        ("Models",       str(df["Model"].nunique()),        "unique models"),
        ("Avg Price",    f"{df['PriceM'].mean():.1f} M",   "VND"),
        ("Highest",      f"{df['PriceM'].max():.0f} M",    "VND"),
    ]
    for col, (label, val, sub) in zip(cols, kpis):
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">{label}</div>'
                f'<div class="metric-value">{val}</div>'
                f'<div class="metric-delta">{sub}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-header">Distributions</div>', unsafe_allow_html=True)

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        fig = px.histogram(df, x="PriceM", nbins=30,
                           title="Price Distribution (Million VND)",
                           labels={"PriceM": "Price (M VND)"},
                           color_discrete_sequence=[P["pink"]])
        fig.update_traces(opacity=0.82)
        fig.add_vline(x=df["PriceM"].mean(), line_dash="dash", line_color=P["pink3"],
                      annotation_text=f"Avg {df['PriceM'].mean():.1f}M",
                      annotation_font_color=P["pink3"])
        fig.update_layout(**PT)
        st.plotly_chart(fig, use_container_width=True)

    with r1c2:
        bc = df["Brand"].value_counts().reset_index()
        bc.columns = ["Brand", "Count"]
        fig = px.bar(bc, x="Brand", y="Count",
                     title="Bikes by Brand",
                     color="Count", color_continuous_scale=PINK_SEQ)
        fig.update_layout(**PT, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        fig = px.histogram(df, x="Year", nbins=20,
                           title="Manufacturing Year",
                           labels={"Year": "Year"},
                           color_discrete_sequence=[P["pink2"]])
        fig.update_layout(**PT)
        st.plotly_chart(fig, use_container_width=True)

    with r2c2:
        fig = px.histogram(df, x="ODO", nbins=25,
                           title="Odometer (km)",
                           labels={"ODO": "km"},
                           color_discrete_sequence=[P["pink3"]])
        fig.update_layout(**PT)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Sample Data</div>', unsafe_allow_html=True)
    show = df[["Brand","Model","Year","ODO","District",
               "Condition","PartsReplaced","Price"]].copy()
    show["Price (VND)"] = show["Price"].apply(lambda x: f"{x:,.0f}")
    st.dataframe(show.drop(columns="Price").head(20),
                 use_container_width=True, height=400)


# ══════════════════════════════════════════════════════
# PAGE 2 — MARKET ANALYSIS
# ══════════════════════════════════════════════════════
elif page == "🔍  Market Analysis":

    st.markdown('<div class="hero-title">MARKET ANALYSIS</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Price insights from HCMC used motorbike listings</div>',
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(f"<hr><b style='color:{P['pink2']}'>Filters</b>", unsafe_allow_html=True)
        sel_brands = st.multiselect(
            "Brand", sorted(df["Brand"].unique()),
            default=sorted(df["Brand"].unique()),
        )
        pmin = int(df["PriceM"].min())
        pmax = int(df["PriceM"].max()) + 1
        price_range = st.slider("Price range (M VND)", pmin, pmax, (pmin, pmax))

    filt = df[
        df["Brand"].isin(sel_brands) &
        (df["PriceM"] >= price_range[0]) &
        (df["PriceM"] <= price_range[1])
    ]

    st.markdown(
        f"<div class='chip'>🔎 {len(filt)} bikes</div>"
        f"<div class='chip'>💰 Avg {filt['PriceM'].mean():.1f}M VND</div>"
        f"<div class='chip'>📍 {filt['District'].nunique()} districts</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.box(filt, x="Brand", y="PriceM", color="Brand",
                     title="Price Range by Brand",
                     labels={"Brand": "", "PriceM": "Price (M VND)"})
        fig.update_layout(**PT, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.scatter(filt, x="ODO", y="PriceM",
                         color="Brand", size="Condition",
                         hover_data=["Model", "Year"],
                         title="Odometer vs Price",
                         labels={"ODO": "km", "PriceM": "Price (M VND)"})
        fig.update_layout(**PT)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        f2 = filt.copy()
        f2["Age Group"] = pd.cut(
            f2["Age"], bins=[0, 3, 6, 10, 15, 50],
            labels=["≤3 yrs", "4–6 yrs", "7–10 yrs", "11–15 yrs", ">15 yrs"],
        )
        avg_age = f2.groupby("Age Group", observed=True)["PriceM"].mean().reset_index()
        fig = px.bar(avg_age, x="Age Group", y="PriceM",
                     title="Avg Price by Bike Age",
                     labels={"Age Group": "", "PriceM": "Avg Price (M VND)"},
                     color="PriceM", color_continuous_scale=PINK_SEQ)
        fig.update_layout(**PT, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        top = (filt.groupby("Model")["PriceM"].mean()
                   .nlargest(10).reset_index())
        top.columns = ["Model", "Avg Price (M VND)"]
        fig = px.bar(top, x="Avg Price (M VND)", y="Model", orientation="h",
                     title="Top 10 Models by Avg Price",
                     color="Avg Price (M VND)", color_continuous_scale=PINK_SEQ)
        fig.update_layout(**PT, coloraxis_showscale=False,
                          yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Price Heatmap — Brand × Condition</div>',
                unsafe_allow_html=True)
    if len(filt) > 5:
        pivot = filt.pivot_table(
            values="PriceM", index="Brand",
            columns="Condition", aggfunc="mean",
        )
        fig = px.imshow(pivot, text_auto=".1f",
                        title="Avg Price (M VND) by Brand × Condition Score",
                        color_continuous_scale="RdPu",
                        labels={"color": "M VND"})
        fig.update_layout(**PT)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Listings by District — Top 15</div>',
                unsafe_allow_html=True)
    dist = filt["District"].value_counts().nlargest(15).reset_index()
    dist.columns = ["District", "Count"]
    fig = px.bar(dist, x="Count", y="District", orientation="h",
                 title="Number of Listings per District",
                 color="Count", color_continuous_scale=PINK_SEQ)
    fig.update_layout(**PT, coloraxis_showscale=False,
                      yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════
# PAGE 3 — ML MODELS
# ══════════════════════════════════════════════════════
elif page == "🤖  ML Models":

    st.markdown('<div class="hero-title">ML MODEL COMPARISON</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Regression algorithms benchmarked on 20% hold-out test set</div>',
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Results table
    rows = [
        {
            "Model":       name,
            "MAE (VND)":   f"{v['MAE']:,.0f}",
            "RMSE (VND)":  f"{v['RMSE']:,.0f}",
            "R² Score":    f"{v['R2']:.4f}",
            "Accuracy":    f"{max(v['R2'], 0)*100:.1f}%",
            "":            "🏆" if name == best_name else "",
        }
        for name, v in model_results.items()
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.markdown(
        f"<div class='chip'>🏆 Best: {best_name}</div>"
        f"<div class='chip'>R² = {model_results[best_name]['R2']:.4f}</div>"
        f"<div class='chip'>MAE = {model_results[best_name]['MAE']/1e6:.2f}M VND</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    names     = list(model_results.keys())
    r2_vals   = [model_results[n]["R2"]  for n in names]
    mae_vals  = [model_results[n]["MAE"] for n in names]
    bar_clrs  = [P["pink"] if n == best_name else "#4A1030" for n in names]

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Bar(
            x=names, y=r2_vals,
            text=[f"{v:.4f}" for v in r2_vals], textposition="outside",
            marker_color=bar_clrs,
            marker_line_color=P["pink"], marker_line_width=1,
        ))
        fig.update_layout(title="R² Score — higher is better",
                          yaxis_range=[0, 1.1], **PT)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = go.Figure(go.Bar(
            x=names, y=[v / 1e6 for v in mae_vals],
            text=[f"{v/1e6:.2f}M" for v in mae_vals], textposition="outside",
            marker_color=bar_clrs,
            marker_line_color=P["pink"], marker_line_width=1,
        ))
        fig.update_layout(title="MAE (M VND) — lower is better", **PT)
        st.plotly_chart(fig, use_container_width=True)

    # Radar
    st.markdown('<div class="section-header">Multi-Dimensional Comparison</div>',
                unsafe_allow_html=True)
    max_mae  = max(v["MAE"]  for v in model_results.values())
    max_rmse = max(v["RMSE"] for v in model_results.values())
    radar_clr = [P["pink"], P["pink2"], P["pink3"]]
    fig = go.Figure()
    for i, (name, m) in enumerate(model_results.items()):
        vals = [
            max(m["R2"], 0),
            1 - m["MAE"]  / max_mae,
            1 - m["RMSE"] / max_rmse,
            max(m["R2"], 0),          # close polygon
        ]
        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=["R²", "MAE (inv)", "RMSE (inv)", "R²"],
            fill="toself", name=name,
            line_color=radar_clr[i],
            opacity=0.72,
        ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1],
                            gridcolor="rgba(255,255,255,0.07)", color=P["muted"]),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.07)", color=P["pink2"]),
        ),
        title="Radar — R², MAE⁻¹, RMSE⁻¹",
        **PT,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    st.markdown('<div class="section-header">Feature Importance</div>',
                unsafe_allow_html=True)
    if hasattr(best_model, "feature_importances_"):
        feat_labels = ["Brand", "Model", "ODO (km)", "District",
                       "Condition %", "Parts Replaced", "Age (yrs)"]
        fi = pd.DataFrame({
            "Feature":    feat_labels,
            "Importance": best_model.feature_importances_,
        }).sort_values("Importance", ascending=True)
        fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
                     title=f"Feature Importance — {best_name}",
                     color="Importance", color_continuous_scale=PINK_SEQ)
        fig.update_layout(**PT, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance is not available for this model type.")

    # Model notes
    st.markdown('<div class="section-header">About the Models</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    notes = [
        ("Linear Regression", "Baseline model. Assumes a linear relationship between features and price. Fast but limited for non-linear patterns."),
        ("Random Forest",      "Ensemble of decision trees. Captures non-linear patterns, robust to outliers, handles mixed feature types well."),
        ("Gradient Boosting",  "Sequential tree boosting. Often achieves the best accuracy by correcting prior tree errors, at higher training cost."),
    ]
    for col, (title, desc) in zip([c1, c2, c3], notes):
        with col:
            st.markdown(
                f"<div class='metric-card' style='text-align:left; padding:16px 18px;'>"
                f"<div class='metric-label'>{title}</div>"
                f"<div style='font-size:0.78rem; color:{P['muted']}; margin-top:6px; line-height:1.6;'>{desc}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════
# PAGE 4 — PRICE ESTIMATOR
# ══════════════════════════════════════════════════════
elif page == "🏷️  Price Estimator":

    st.markdown('<div class="hero-title">PRICE ESTIMATOR</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Enter your bike details — get an AI-powered price estimate</div>',
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col_form, col_result = st.columns([1.1, 1], gap="large")

    # ── Input form ────────────────────────────────────
    with col_form:
        st.markdown('<div class="section-header">Bike Details</div>', unsafe_allow_html=True)

        brand = st.selectbox("🏢 Brand", sorted(df["Brand"].unique()))

        # Only show models that belong to the chosen brand (fixes the missing Honda bug)
        model_opts = sorted(df[df["Brand"] == brand]["Model"].unique())
        model_sel  = st.selectbox("🏍️ Model", model_opts)

        ca, cb = st.columns(2)
        with ca:
            year = st.selectbox("📅 Year", sorted(df["Year"].unique(), reverse=True))
        with cb:
            odo = st.number_input("🛣️ Odometer (km)",
                                  min_value=0, max_value=300_000,
                                  value=25_000, step=500)

        district = st.selectbox("📍 District", sorted(df["District"].unique()))

        condition = st.select_slider(
            "⭐ Condition",
            options=[70, 75, 80, 85, 90, 99],
            value=90,
            format_func=lambda x: {
                70: "70% — Needs repair",
                75: "75% — Heavy wear",
                80: "80% — Average",
                85: "85% — Good",
                90: "90% — Very good",
                99: "99% — Like new",
            }[x],
        )

        parts = st.radio(
            "🔧 Parts replaced?",
            ["Not Replaced", "Replaced"],
            horizontal=True,
        )

        predict_btn = st.button("⚡  ESTIMATE PRICE", type="primary")

    # ── Result panel ──────────────────────────────────
    with col_result:
        st.markdown('<div class="section-header">Prediction Result</div>',
                    unsafe_allow_html=True)

        if predict_btn:
            age = 2026 - year

            inp = pd.DataFrame([{
                "Brand":         brand,
                "Model":         model_sel,
                "ODO":           float(odo),
                "District":      district,
                "Condition":     condition,
                "PartsReplaced": parts,
                "Age":           age,
            }])

            # Encode
            inp[["Brand", "Model", "District", "PartsReplaced"]] = encoder.transform(
                inp[["Brand", "Model", "District", "PartsReplaced"]]
            )
            # Scale
            inp[["ODO", "Condition", "Age"]] = scaler.transform(
                inp[["ODO", "Condition", "Age"]]
            )

            predicted = best_model.predict(inp[FEAT_COLS])[0]
            low, high = predicted * 0.87, predicted * 1.13

            st.markdown(f"""
            <div class="pred-box">
                <div class="pred-label">Estimated Market Price</div>
                <div class="pred-price">{predicted/1e6:.1f} Million VND</div>
                <div class="pred-range">
                    Typical range &nbsp;·&nbsp;
                    {low/1e6:.1f}M &nbsp;—&nbsp; {high/1e6:.1f}M VND
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Similar bikes
            similar = df[
                (df["Brand"]  == brand) &
                (df["Model"]  == model_sel) &
                (abs(df["Age"] - age) <= 3)
            ][["Model", "Year", "ODO", "Condition", "Price"]].head(6)

            if not similar.empty:
                st.markdown(
                    f"<b style='color:{P['pink2']}'>🔍 Similar bikes in dataset:</b>",
                    unsafe_allow_html=True,
                )
                sim = similar.copy()
                sim.columns = ["Model", "Year", "ODO (km)", "Condition %", "Price (VND)"]
                sim["Price (VND)"] = sim["Price (VND)"].apply(lambda x: f"{x:,.0f}")
                st.dataframe(sim, use_container_width=True, hide_index=True)
            else:
                st.info("No closely matching bikes found in the dataset.")

            # Market position gauge  (mode="gauge+number" — no delta, avoids reference bug)
            pct = float((df["Price"].values < predicted).mean() * 100)
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pct,
                number={"suffix": "%",
                        "font": {"color": P["pink"], "family": "Bebas Neue", "size": 34}},
                title={"text": "Market Position (percentile)",
                       "font": {"color": P["pink2"], "family": "Inter", "size": 12}},
                gauge={
                    "axis":  {"range": [0, 100],
                               "tickcolor": P["muted"],
                               "tickfont":  {"color": P["muted"]}},
                    "bar":   {"color": P["pink"]},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0,   33], "color": "rgba(80,200,100,0.10)"},
                        {"range": [33,  66], "color": "rgba(255,45,120,0.08)"},
                        {"range": [66, 100], "color": "rgba(220,50,50,0.12)"},
                    ],
                    "threshold": {
                        "line":  {"color": P["pink3"], "width": 3},
                        "value": pct,
                    },
                },
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": P["text"], "family": "Inter"},
                height=230,
                margin=dict(t=36, b=0, l=16, r=16),
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.markdown(f"""
            <div style='text-align:center; padding:64px 16px;
                        border: 2px dashed rgba(255,45,120,.18);
                        border-radius: 18px; color:{P["muted"]};'>
                <div style='font-size:3rem'>🏍️</div>
                <div style='font-size:0.88rem; font-weight:600; margin-top:14px;'>
                    Fill in the bike details<br>
                    and press <span style='color:{P["pink"]}'>ESTIMATE PRICE</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Quick analysis (only shown after prediction) ──
    if predict_btn:
        st.markdown('<div class="section-header">Quick Brand & Model Analysis</div>',
                    unsafe_allow_html=True)
        brand_df = df[df["Brand"] == brand]
        model_df = df[df["Model"] == model_sel]

        ca2, cb2 = st.columns(2)
        with ca2:
            fig = px.histogram(brand_df, x="PriceM", nbins=20,
                               title=f"Price Distribution — {brand}",
                               labels={"PriceM": "Price (M VND)"},
                               color_discrete_sequence=[P["pink"]])
            fig.add_vline(x=predicted / 1e6, line_color=P["pink3"],
                          line_dash="dash",
                          annotation_text="Your bike",
                          annotation_font_color=P["pink3"])
            fig.update_layout(**PT)
            st.plotly_chart(fig, use_container_width=True)

        with cb2:
            if len(model_df) > 3:
                fig = px.scatter(model_df, x="Age", y="PriceM",
                                 color="Condition",
                                 title=f"Age vs Price — {model_sel}",
                                 labels={"Age": "Age (yrs)", "PriceM": "Price (M VND)"},
                                 color_continuous_scale="RdPu")
                fig.add_hline(y=predicted / 1e6, line_color=P["pink3"],
                              line_dash="dash",
                              annotation_text="Your estimate",
                              annotation_font_color=P["pink3"])
                fig.update_layout(**PT)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"Not enough data points for {model_sel} to render the chart.")


# ──────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────
st.markdown(f"""
<hr style='border-color:rgba(255,45,120,.12); margin:52px 0 16px;'>
<div style='text-align:center; color:{P["muted"]}; font-size:0.72rem;
            padding-bottom:24px; line-height:2.2;'>
    🏍️ <b style='color:{P["pink2"]}'> MyOldMotorbike</b>
    &nbsp;·&nbsp; Machine Learning Project
    &nbsp;·&nbsp; Ho Chi Minh City &nbsp;·&nbsp; 2026<br>
    {len(df)} listings &nbsp;·&nbsp; Random Forest + Gradient Boosting
    &nbsp;·&nbsp; Built with Streamlit &amp; Plotly
</div>
""", unsafe_allow_html=True)
