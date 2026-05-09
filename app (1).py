# -*- coding: utf-8 -*-
"""
🏍️ MyOldMotorbike — Ứng dụng Định Giá Xe Máy Cũ TP.HCM
Phiên bản nâng cấp — Dự án Machine Learning
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import io
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MyOldMotorbike | Định Giá Xe Máy Cũ",
    page_icon="🏍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Nunito:wght@400;600;700;800&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d1a 0%, #1a1a2e 100%);
    border-right: 1px solid rgba(255,165,0,0.2);
}
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }

/* Hero title */
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.8rem;
    letter-spacing: 4px;
    background: linear-gradient(90deg, #FF6B00, #FFD700, #FF6B00);
    background-size: 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 3s infinite linear;
    line-height: 1.1;
    margin-bottom: 0;
}
@keyframes shimmer {
    0% { background-position: 0% }
    100% { background-position: 200% }
}
.hero-sub {
    color: #aaaacc;
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 4px;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, rgba(255,107,0,0.12) 0%, rgba(255,215,0,0.06) 100%);
    border: 1px solid rgba(255,107,0,0.3);
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
}
.metric-card:hover {
    transform: translateY(-3px);
    border-color: rgba(255,107,0,0.6);
}
.metric-label {
    color: #aaaacc;
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 6px;
}
.metric-value {
    color: #FFD700;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2rem;
    letter-spacing: 1px;
}
.metric-delta {
    color: #66ffaa;
    font-size: 0.82rem;
    font-weight: 700;
    margin-top: 4px;
}

/* Section headers */
.section-header {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.8rem;
    letter-spacing: 3px;
    color: #FF8C00;
    border-bottom: 2px solid rgba(255,107,0,0.3);
    padding-bottom: 8px;
    margin: 28px 0 20px 0;
}

/* Prediction result box */
.pred-box {
    background: linear-gradient(135deg, #FF6B00 0%, #FF8C00 50%, #FFD700 100%);
    border-radius: 20px;
    padding: 36px 24px;
    text-align: center;
    box-shadow: 0 8px 40px rgba(255,107,0,0.4);
}
.pred-label {
    color: rgba(255,255,255,0.85);
    font-size: 0.9rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
}
.pred-price {
    color: white;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.2rem;
    letter-spacing: 2px;
    line-height: 1.1;
    text-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
.pred-range {
    color: rgba(255,255,255,0.8);
    font-size: 0.88rem;
    font-weight: 600;
    margin-top: 8px;
}

/* Info chips */
.chip {
    display: inline-block;
    background: rgba(255,107,0,0.15);
    border: 1px solid rgba(255,107,0,0.4);
    border-radius: 20px;
    padding: 4px 14px;
    color: #FFB347;
    font-size: 0.82rem;
    font-weight: 700;
    margin: 3px;
}

/* Model comparison table */
.model-row-best {
    background: rgba(255,215,0,0.1) !important;
    font-weight: 700;
}

/* Streamlit overrides */
.stSelectbox label, .stSlider label, .stRadio label { color: #ccccee !important; font-weight: 600 !important; }
[data-testid="stMetricValue"] { color: #FFD700 !important; font-family: 'Bebas Neue', sans-serif !important; font-size: 2rem !important; }
[data-testid="stMetricLabel"] { color: #aaaacc !important; font-size: 0.82rem !important; text-transform: uppercase; letter-spacing: 1px; }
div.stButton > button {
    background: linear-gradient(135deg, #FF6B00, #FFD700);
    color: #0f0f1a;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.1rem;
    letter-spacing: 2px;
    border: none;
    border-radius: 12px;
    padding: 12px 32px;
    width: 100%;
    transition: transform 0.15s, box-shadow 0.15s;
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(255,107,0,0.5);
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────
@st.cache_data
def load_and_clean():
    df = pd.read_excel('dataxe.xlsx', sheet_name='DuLieuXe_Cleaned')
    df.columns = ['Hang_xe', 'Dong_xe', 'Nam_sx', 'ODO', 'Khu_vuc', 'Tinh_trang', 'Gia_ban', 'Thay_phu_tung']
    df['ODO'] = df['ODO'].astype(str).str.replace(',', '.').astype(float)
    df['Hang_xe'] = df['Hang_xe'].str.strip().str.capitalize()
    df['Hang_xe'] = df['Hang_xe'].replace({'Sym': 'SYM'})
    df['Dong_xe'] = df['Dong_xe'].str.strip()
    df['Dong_xe'] = df['Dong_xe'].replace({
        'Air blade': 'Air Blade', 'Airblade': 'Air Blade', 'AirBlade': 'Air Blade',
        'Honda AirBlade': 'Air Blade', 'LEAD': 'Lead',
        'Winner X': 'Winner', 'Winner V1': 'Winner', 'Visson': 'Vision'
    })
    df['Thay_phu_tung'] = df['Thay_phu_tung'].str.strip()
    # Normalize Khu_vuc: extract main district
    df['Khu_vuc_clean'] = df['Khu_vuc'].str.replace(r'\xa0.*', '', regex=True).str.strip()
    df['Tuoi_xe'] = 2026 - df['Nam_sx']
    return df

@st.cache_resource
def train_model(df):
    df2 = df.copy()
    categorical_cols = ['Hang_xe', 'Dong_xe', 'Khu_vuc_clean', 'Thay_phu_tung']
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df2[categorical_cols] = encoder.fit_transform(df2[categorical_cols])

    X = df2[['Hang_xe', 'Dong_xe', 'ODO', 'Khu_vuc_clean', 'Tinh_trang', 'Thay_phu_tung', 'Tuoi_xe']]
    y = df2['Gia_ban']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    num_cols = ['ODO', 'Tinh_trang', 'Tuoi_xe']
    X_train_s = X_train.copy()
    X_test_s  = X_test.copy()
    X_train_s[num_cols] = scaler.fit_transform(X_train_s[num_cols])
    X_test_s[num_cols]  = scaler.transform(X_test_s[num_cols])

    # Benchmark models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest':     RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, random_state=42),
    }
    results = {}
    fitted  = {}
    for name, m in models.items():
        m.fit(X_train_s, y_train)
        yp = m.predict(X_test_s)
        results[name] = {
            'MAE':  mean_absolute_error(y_test, yp),
            'RMSE': np.sqrt(mean_squared_error(y_test, yp)),
            'R2':   r2_score(y_test, yp),
        }
        fitted[name] = m

    best_name = max(results, key=lambda k: results[k]['R2'])
    best_model = fitted[best_name]

    return encoder, scaler, best_model, best_name, results, X.columns.tolist()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 8px 0;'>
        <span style='font-size:2.8rem'>🏍️</span>
        <div style='font-family: Bebas Neue, sans-serif; font-size:1.5rem; letter-spacing:3px; color:#FFD700;'>
            MyOldMotorbike
        </div>
        <div style='font-size:0.72rem; color:#888; letter-spacing:2px; text-transform:uppercase;'>
            Định Giá Xe Máy Cũ · HCM
        </div>
    </div>
    <hr style='border-color:rgba(255,165,0,0.2); margin: 12px 0;'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "🧭 Điều hướng",
        ["📊 Tổng Quan Dữ Liệu", "🔍 Phân Tích Thị Trường", "🤖 So Sánh Mô Hình", "🏷️ Định Giá Xe"],
        label_visibility="collapsed"
    )

    st.markdown("""
    <hr style='border-color:rgba(255,165,0,0.15); margin:20px 0 12px 0;'>
    <div style='font-size:0.75rem; color:#666; text-align:center; line-height:1.8;'>
        <b style='color:#FF8C00'>Dữ liệu</b>: 243 xe cũ tại TP.HCM<br>
        <b style='color:#FF8C00'>Mô hình</b>: Random Forest & Gradient Boosting<br>
        <b style='color:#FF8C00'>Năm</b>: 2024–2026
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
with st.spinner("Đang tải dữ liệu..."):
    df = load_and_clean()
    encoder, scaler, best_model, best_name, model_results, feature_cols = train_model(df)

PLOTLY_THEME = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(255,255,255,0.03)',
    font=dict(color='#ccccee', family='Nunito'),
    title_font=dict(color='#FFD700', family='Nunito', size=15),
    xaxis=dict(gridcolor='rgba(255,255,255,0.07)', zerolinecolor='rgba(255,255,255,0.1)'),
    yaxis=dict(gridcolor='rgba(255,255,255,0.07)', zerolinecolor='rgba(255,255,255,0.1)'),
    colorway=['#FF6B00','#FFD700','#FF8C69','#FFA07A','#4FC3F7','#81C784','#CE93D8'],
)
ORANGE_SEQ = px.colors.sequential.Oranges


# ═══════════════════════════════════════════════
# PAGE 1: TỔNG QUAN DỮ LIỆU
# ═══════════════════════════════════════════════
if page == "📊 Tổng Quan Dữ Liệu":

    st.markdown('<div class="hero-title">TỔNG QUAN DỮ LIỆU</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Bộ dữ liệu xe máy cũ tại TP. Hồ Chí Minh</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        ("Tổng Xe", f"{len(df):,}", "243 quan sát"),
        ("Hãng Xe", str(df['Hang_xe'].nunique()), "8 thương hiệu"),
        ("Dòng Xe", str(df['Dong_xe'].nunique()), "Đa dạng"),
        ("Giá TB", f"{df['Gia_ban'].mean()/1e6:.1f}M", "VNĐ"),
        ("Giá Cao Nhất", f"{df['Gia_ban'].max()/1e6:.0f}M", "VNĐ"),
    ]
    for col, (label, val, delta) in zip([c1,c2,c3,c4,c5], kpis):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{val}</div>
                <div class="metric-delta">{delta}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">PHÂN PHỐI DỮ LIỆU</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Price distribution
        fig = px.histogram(df, x='Gia_ban', nbins=30,
                           title="📈 Phân Phối Giá Bán",
                           labels={'Gia_ban': 'Giá bán (VNĐ)'},
                           color_discrete_sequence=['#FF6B00'])
        fig.update_traces(opacity=0.85)
        fig.add_vline(x=df['Gia_ban'].mean(), line_dash='dash', line_color='#FFD700',
                      annotation_text=f"TB: {df['Gia_ban'].mean()/1e6:.1f}M",
                      annotation_font_color='#FFD700')
        fig.update_layout(**PLOTLY_THEME)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Brand distribution
        brand_count = df['Hang_xe'].value_counts().reset_index()
        brand_count.columns = ['Hãng', 'Số lượng']
        fig = px.bar(brand_count, x='Hãng', y='Số lượng',
                     title="🏢 Phân Bố Theo Hãng Xe",
                     color='Số lượng', color_continuous_scale=ORANGE_SEQ)
        fig.update_layout(**PLOTLY_THEME, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Year distribution
        fig = px.histogram(df, x='Nam_sx', nbins=20,
                           title="📅 Năm Sản Xuất",
                           labels={'Nam_sx': 'Năm sản xuất'},
                           color_discrete_sequence=['#FFD700'])
        fig.update_layout(**PLOTLY_THEME)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        # ODO distribution
        fig = px.histogram(df, x='ODO', nbins=25,
                           title="🛣️ Số Km Đã Đi (ODO)",
                           labels={'ODO': 'Số km'},
                           color_discrete_sequence=['#FF8C69'])
        fig.update_layout(**PLOTLY_THEME)
        st.plotly_chart(fig, use_container_width=True)

    # Data table
    st.markdown('<div class="section-header">DỮ LIỆU MẪU</div>', unsafe_allow_html=True)
    display_df = df[['Hang_xe','Dong_xe','Nam_sx','ODO','Khu_vuc_clean','Tinh_trang','Thay_phu_tung','Gia_ban']].copy()
    display_df.columns = ['Hãng','Dòng xe','Năm SX','ODO (km)','Khu vực','Tình trạng %','Phụ tùng','Giá bán (VNĐ)']
    display_df['Giá bán (VNĐ)'] = display_df['Giá bán (VNĐ)'].apply(lambda x: f"{x:,.0f}")
    st.dataframe(display_df.head(20), use_container_width=True, height=400)


# ═══════════════════════════════════════════════
# PAGE 2: PHÂN TÍCH THỊ TRƯỜNG
# ═══════════════════════════════════════════════
elif page == "🔍 Phân Tích Thị Trường":

    st.markdown('<div class="hero-title">PHÂN TÍCH THỊ TRƯỜNG</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Insight từ dữ liệu xe cũ TP.HCM</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Filter sidebar
    with st.sidebar:
        st.markdown("### 🔧 Bộ lọc")
        sel_brands = st.multiselect("Hãng xe", df['Hang_xe'].unique(), default=list(df['Hang_xe'].unique()))
        price_range = st.slider("Khoảng giá (triệu VNĐ)",
                                 int(df['Gia_ban'].min()//1e6),
                                 int(df['Gia_ban'].max()//1e6)+1,
                                 (0, int(df['Gia_ban'].max()//1e6)+1))

    filtered = df[
        df['Hang_xe'].isin(sel_brands) &
        (df['Gia_ban'] >= price_range[0]*1e6) &
        (df['Gia_ban'] <= price_range[1]*1e6)
    ]
    st.markdown(f"<div class='chip'>📊 {len(filtered)} xe được lọc</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Box plot: price by brand
        fig = px.box(filtered, x='Hang_xe', y='Gia_ban',
                     color='Hang_xe',
                     title="💰 Phân Phối Giá theo Hãng",
                     labels={'Hang_xe': 'Hãng xe', 'Gia_ban': 'Giá bán (VNĐ)'},
                     color_discrete_sequence=['#FF6B00','#FFD700','#FF8C69','#FFA07A','#FFCC80','#FFB347','#FFC107','#FF9800'])
        fig.update_layout(**PLOTLY_THEME, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Scatter: ODO vs Price colored by brand
        fig = px.scatter(filtered, x='ODO', y='Gia_ban',
                         color='Hang_xe', size='Tinh_trang',
                         hover_data=['Dong_xe','Nam_sx'],
                         title="🔗 ODO vs Giá Bán",
                         labels={'ODO': 'Số km', 'Gia_ban': 'Giá (VNĐ)'},
                         color_discrete_sequence=['#FF6B00','#FFD700','#FF8C69','#FFA07A','#FFCC80','#FFB347','#FFC107','#FF9800'])
        fig.update_layout(**PLOTLY_THEME)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Avg price by age bucket
        filtered2 = filtered.copy()
        filtered2['Tuoi_nhom'] = pd.cut(filtered2['Tuoi_xe'],
                                         bins=[0,3,6,10,15,50],
                                         labels=['≤3 năm','4–6 năm','7–10 năm','11–15 năm','> 15 năm'])
        avg_age = filtered2.groupby('Tuoi_nhom', observed=True)['Gia_ban'].mean().reset_index()
        fig = px.bar(avg_age, x='Tuoi_nhom', y='Gia_ban',
                     title="📉 Giá TB theo Tuổi Xe",
                     labels={'Tuoi_nhom': 'Tuổi xe', 'Gia_ban': 'Giá TB (VNĐ)'},
                     color='Gia_ban', color_continuous_scale=ORANGE_SEQ)
        fig.update_layout(**PLOTLY_THEME, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        # Top 10 models by avg price
        top_models = filtered.groupby('Dong_xe')['Gia_ban'].mean().nlargest(10).reset_index()
        top_models.columns = ['Dòng xe', 'Giá TB']
        fig = px.bar(top_models, x='Giá TB', y='Dòng xe',
                     orientation='h',
                     title="🏆 Top 10 Dòng Xe Giá Cao Nhất (TB)",
                     color='Giá TB', color_continuous_scale=ORANGE_SEQ)
        fig.update_layout(**PLOTLY_THEME, coloraxis_showscale=False, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    # Heatmap: brand × condition
    st.markdown('<div class="section-header">TÌNH TRẠNG XE & GIÁ BÁN</div>', unsafe_allow_html=True)
    pivot = filtered.pivot_table(values='Gia_ban', index='Hang_xe', columns='Tinh_trang', aggfunc='mean')
    fig = px.imshow(pivot/1e6, text_auto='.1f',
                    title="🗺️ Giá TB (triệu VNĐ) theo Hãng × Tình Trạng",
                    color_continuous_scale='Oranges',
                    labels={'color': 'Triệu VNĐ'})
    fig.update_layout(**PLOTLY_THEME)
    st.plotly_chart(fig, use_container_width=True)

    # District bar
    st.markdown('<div class="section-header">PHÂN BỐ THEO KHU VỰC</div>', unsafe_allow_html=True)
    dist_count = filtered['Khu_vuc_clean'].value_counts().nlargest(15).reset_index()
    dist_count.columns = ['Khu vực', 'Số xe']
    fig = px.bar(dist_count, x='Số xe', y='Khu vực', orientation='h',
                 title="📍 Top 15 Khu Vực Có Nhiều Xe Nhất",
                 color='Số xe', color_continuous_scale=ORANGE_SEQ)
    fig.update_layout(**PLOTLY_THEME, coloraxis_showscale=False, yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════
# PAGE 3: SO SÁNH MÔ HÌNH
# ═══════════════════════════════════════════════
elif page == "🤖 So Sánh Mô Hình":

    st.markdown('<div class="hero-title">SO SÁNH MÔ HÌNH ML</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Đánh giá hiệu suất các thuật toán hồi quy</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Results table
    rows = []
    for name, metrics in model_results.items():
        rows.append({
            'Mô hình': name,
            'MAE (VNĐ)': f"{metrics['MAE']:,.0f}",
            'RMSE (VNĐ)': f"{metrics['RMSE']:,.0f}",
            'R² Score': f"{metrics['R2']:.4f}",
            'Độ chính xác': f"{metrics['R2']*100:.1f}%",
        })
    results_df = pd.DataFrame(rows)

    st.markdown('<div class="section-header">KẾT QUẢ ĐÁNH GIÁ</div>', unsafe_allow_html=True)
    st.dataframe(results_df, use_container_width=True, hide_index=True)
    st.markdown(f"<div class='chip'>🏆 Mô hình tốt nhất: {best_name} — R²={model_results[best_name]['R2']:.4f}</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    names = list(model_results.keys())
    r2_vals  = [model_results[n]['R2']  for n in names]
    mae_vals = [model_results[n]['MAE'] for n in names]

    with col1:
        fig = go.Figure(go.Bar(
            x=names, y=r2_vals,
            marker_color=['#FFD700' if n == best_name else '#FF6B00' for n in names],
            text=[f"{v:.4f}" for v in r2_vals],
            textposition='outside',
        ))
        fig.update_layout(title="📊 R² Score (càng cao càng tốt)", yaxis_range=[0, 1.05], **PLOTLY_THEME)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure(go.Bar(
            x=names, y=[v/1e6 for v in mae_vals],
            marker_color=['#FFD700' if n == best_name else '#FF6B00' for n in names],
            text=[f"{v/1e6:.2f}M" for v in mae_vals],
            textposition='outside',
        ))
        fig.update_layout(title="📉 MAE — Sai Số TB (triệu VNĐ, càng thấp càng tốt)", **PLOTLY_THEME)
        st.plotly_chart(fig, use_container_width=True)

    # Radar chart
    st.markdown('<div class="section-header">RADAR — ĐA CHIỀU ĐÁNH GIÁ</div>', unsafe_allow_html=True)
    max_mae  = max(v['MAE']  for v in model_results.values())
    max_rmse = max(v['RMSE'] for v in model_results.values())
    colors_radar = ['#FF6B00','#FFD700','#FF8C69']
    fig = go.Figure()
    for i, (name, metrics) in enumerate(model_results.items()):
        r2   = metrics['R2']
        acc_mae  = 1 - metrics['MAE']  / max_mae
        acc_rmse = 1 - metrics['RMSE'] / max_rmse
        fig.add_trace(go.Scatterpolar(
            r=[r2, acc_mae, acc_rmse, r2],
            theta=['R²', 'MAE (inv)', 'RMSE (inv)', 'R²'],
            fill='toself', name=name,
            line_color=colors_radar[i],
            fillcolor=colors_radar[i].replace(')', ',0.15)').replace('rgb', 'rgba') if 'rgb' in colors_radar[i] else colors_radar[i],
            opacity=0.7,
        ))
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[0,1], gridcolor='rgba(255,255,255,0.1)', color='#aaa'),
            angularaxis=dict(gridcolor='rgba(255,255,255,0.1)', color='#FFD700'),
        ),
        title="🕸️ So sánh đa chiều",
        **PLOTLY_THEME,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    st.markdown('<div class="section-header">ĐỘ QUAN TRỌNG ĐẶC TRƯNG</div>', unsafe_allow_html=True)
    if hasattr(best_model, 'feature_importances_'):
        feat_labels = ['Hãng xe','Dòng xe','ODO','Khu vực','Tình trạng','Phụ tùng','Tuổi xe']
        importances = best_model.feature_importances_
        fi_df = pd.DataFrame({'Feature': feat_labels, 'Importance': importances}).sort_values('Importance', ascending=True)
        fig = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                     title=f"🔬 Feature Importance — {best_name}",
                     color='Importance', color_continuous_scale=ORANGE_SEQ)
        fig.update_layout(**PLOTLY_THEME, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════
# PAGE 4: ĐỊNH GIÁ XE
# ═══════════════════════════════════════════════
elif page == "🏷️ Định Giá Xe":

    st.markdown('<div class="hero-title">ĐỊNH GIÁ XE CỦA BẠN</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Nhập thông tin xe — AI sẽ ước tính giá bán phù hợp</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col_form, col_result = st.columns([1.1, 1], gap="large")

    with col_form:
        st.markdown('<div class="section-header">THÔNG TIN XE</div>', unsafe_allow_html=True)

        hang_xe = st.selectbox("🏢 Hãng xe", sorted(df['Hang_xe'].unique()))
        dong_xe_options = sorted(df[df['Hang_xe'] == hang_xe]['Dong_xe'].unique())
        dong_xe = st.selectbox("🏍️ Dòng xe", dong_xe_options)

        c1, c2 = st.columns(2)
        with c1:
            nam_sx = st.selectbox("📅 Năm sản xuất", sorted(df['Nam_sx'].unique(), reverse=True))
        with c2:
            odo = st.number_input("🛣️ Số km đã đi", min_value=0, max_value=200000, value=25000, step=1000)

        khu_vuc_opts = sorted(df['Khu_vuc_clean'].unique())
        khu_vuc = st.selectbox("📍 Khu vực", khu_vuc_opts)

        tinh_trang = st.select_slider(
            "⭐ Tình trạng xe (%)",
            options=[70, 75, 80, 85, 90, 99],
            value=90,
            format_func=lambda x: {70:"70% — Cần sửa", 75:"75% — Khá cũ", 80:"80% — Trung bình",
                                     85:"85% — Khá tốt", 90:"90% — Tốt", 99:"99% — Như mới"}[x]
        )

        thay_pt = st.radio("🔧 Đã thay phụ tùng chưa?",
                           ["Chưa thay", "Đã thay"], horizontal=True)

        predict_btn = st.button("⚡ DỰ ĐOÁN GIÁ NGAY", type="primary")

    with col_result:
        st.markdown('<div class="section-header">KẾT QUẢ DỰ ĐOÁN</div>', unsafe_allow_html=True)

        if predict_btn:
            tuoi_xe = 2026 - nam_sx
            input_raw = pd.DataFrame([{
                'Hang_xe': hang_xe, 'Dong_xe': dong_xe,
                'ODO': float(odo), 'Khu_vuc_clean': khu_vuc,
                'Tinh_trang': tinh_trang,
                'Thay_phu_tung': thay_pt,
                'Tuoi_xe': tuoi_xe,
            }])

            input_enc = input_raw.copy()
            input_enc[['Hang_xe','Dong_xe','Khu_vuc_clean','Thay_phu_tung']] = encoder.transform(
                input_enc[['Hang_xe','Dong_xe','Khu_vuc_clean','Thay_phu_tung']]
            )
            num_cols = ['ODO','Tinh_trang','Tuoi_xe']
            input_enc[num_cols] = scaler.transform(input_enc[num_cols])
            input_enc = input_enc[feature_cols]

            predicted = best_model.predict(input_enc)[0]
            low  = predicted * 0.87
            high = predicted * 1.13

            st.markdown(f"""
            <div class="pred-box">
                <div class="pred-label">Giá Ước Tính</div>
                <div class="pred-price">{predicted/1e6:.1f} Triệu VNĐ</div>
                <div class="pred-range">Khoảng: {low/1e6:.1f}M — {high/1e6:.1f}M VNĐ</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Comparable cars
            st.markdown("**🔍 Xe tương tự trong dữ liệu:**")
            similar = df[
                (df['Hang_xe'] == hang_xe) &
                (df['Dong_xe'] == dong_xe) &
                (abs(df['Tuoi_xe'] - tuoi_xe) <= 3)
            ][['Dong_xe','Nam_sx','ODO','Tinh_trang','Gia_ban']].head(5)

            if not similar.empty:
                similar_show = similar.copy()
                similar_show.columns = ['Dòng xe','Năm SX','ODO (km)','Tình trạng','Giá bán (VNĐ)']
                similar_show['Giá bán (VNĐ)'] = similar_show['Giá bán (VNĐ)'].apply(lambda x: f"{x:,.0f}")
                st.dataframe(similar_show, use_container_width=True, hide_index=True)
            else:
                st.info("Chưa có xe tương tự trong bộ dữ liệu.")

            # Gauge chart
            all_prices = df['Gia_ban'].values
            percentile = (all_prices < predicted).mean() * 100
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=percentile,
                title={'text': "Vị trí giá trong thị trường", 'font': {'color': '#FFD700', 'family': 'Nunito'}},
                number={'suffix': "% phần trăm vị", 'font': {'color': '#FFD700', 'family': 'Nunito'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#888'},
                    'bar': {'color': '#FF6B00'},
                    'bgcolor': 'rgba(0,0,0,0)',
                    'bordercolor': 'rgba(255,165,0,0.3)',
                    'steps': [
                        {'range': [0,25],  'color': 'rgba(100,200,100,0.2)'},
                        {'range': [25,75], 'color': 'rgba(255,165,0,0.15)'},
                        {'range': [75,100],'color': 'rgba(255,80,80,0.2)'},
                    ],
                    'threshold': {'line': {'color': '#FFD700', 'width': 3}, 'value': percentile}
                }
            ))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color':'#ccc','family':'Nunito'}, height=220)
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.markdown("""
            <div style='text-align:center; padding:60px 20px; border:2px dashed rgba(255,107,0,0.25); border-radius:16px; color:#666;'>
                <div style='font-size:3rem'>🏍️</div>
                <div style='font-size:1rem; font-weight:600; margin-top:12px;'>
                    Nhập thông tin xe và nhấn<br><span style='color:#FF8C00'>DỰ ĐOÁN GIÁ NGAY</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Insight below
    st.markdown('<div class="section-header">PHÂN TÍCH NHANH DÒNG XE NÀY</div>', unsafe_allow_html=True)
    if predict_btn:
        brand_data = df[df['Hang_xe'] == hang_xe]
        model_data = df[df['Dong_xe'] == dong_xe]

        ca, cb = st.columns(2)
        with ca:
            fig = px.histogram(brand_data, x='Gia_ban',
                               title=f"Giá bán — {hang_xe}",
                               color_discrete_sequence=['#FF6B00'], nbins=20)
            fig.add_vline(x=predicted, line_color='#FFD700', line_dash='dash',
                          annotation_text="Xe bạn", annotation_font_color='#FFD700')
            fig.update_layout(**PLOTLY_THEME)
            st.plotly_chart(fig, use_container_width=True)

        with cb:
            if len(model_data) > 3:
                fig = px.scatter(model_data, x='Tuoi_xe', y='Gia_ban',
                                 color='Tinh_trang',
                                 title=f"Tuổi xe vs Giá — {dong_xe}",
                                 labels={'Tuoi_xe':'Tuổi xe', 'Gia_ban':'Giá (VNĐ)'},
                                 color_continuous_scale=ORANGE_SEQ)
                fig.add_hline(y=predicted, line_color='#FFD700', line_dash='dash',
                              annotation_text="Dự đoán", annotation_font_color='#FFD700')
                fig.update_layout(**PLOTLY_THEME)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Không đủ dữ liệu dòng xe này để vẽ biểu đồ.")
    else:
        st.markdown("<div style='color:#666; font-style:italic;'>Kết quả sẽ hiển thị sau khi dự đoán.</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<hr style='border-color:rgba(255,165,0,0.15); margin:40px 0 16px 0;'>
<div style='text-align:center; color:#555; font-size:0.8rem; padding-bottom:20px;'>
    🏍️ <b style='color:#FF8C00'>MyOldMotorbike</b> · Dự án Machine Learning · TP. Hồ Chí Minh · 2026<br>
    Dữ liệu: 243 xe cũ · Mô hình: Random Forest + Gradient Boosting · Made with Streamlit & Plotly
</div>
""", unsafe_allow_html=True)
