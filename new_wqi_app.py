
import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import uuid
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
import warnings
warnings.filterwarnings('ignore')

# ================= UNIQUE KEY =================
def chart_key():
    return str(uuid.uuid4())

# ================= MODEL FEATURES =================
MODEL_FEATURES = ['DO', 'pH', 'ORP', 'Cond', 'Temp', 'WQI']

# ================= STATUS MAP =================
STATUS_MAP = {
    0: "Excellent",
    1: "Good",
    2: "Fair",
    3: "Poor",
    4: "Very Poor"
}

WQI_THRESHOLDS = {
    0: (75, 100),
    1: (50, 74.9),
    2: (25, 49.9),
    3: (10, 24.9),
    4: (0,  9.9),
}

STATUS_COLOR = {
    "Excellent": "#00b894",
    "Good":      "#0984e3",
    "Fair":      "#f39c12",
    "Poor":      "#e17055",
    "Very Poor": "#d63031"
}

# ================= PAGE =================
st.set_page_config(page_title="HydroGuard AI", layout="wide")

# ================= THEME =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Poppins', sans-serif !important;
    background: linear-gradient(120deg, #d4f1f9, #a6e3f5, #6fd3f7) !important;
}

p, span, label, div, li, td, th {
    font-family: 'Poppins', sans-serif !important;
    color: #1a1a2e !important;
    font-size: 15px !important;
}

h1 { font-size: 2.4rem !important; font-weight: 800 !important; color: #0d3b66 !important; }
h2 { font-size: 1.8rem !important; font-weight: 700 !important; color: #0d3b66 !important; }
h3 { font-size: 1.4rem !important; font-weight: 700 !important; color: #0d3b66 !important; }
h4 { font-size: 1.1rem !important; font-weight: 700 !important; color: #0d3b66 !important; }

[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label {
    font-size: 15px !important;
    font-weight: 700 !important;
    color: #0d3b66 !important;
}

.stNumberInput > div > div > input {
    font-size: 16px !important;
    font-weight: 600 !important;
    color: #1a1a2e !important;
    background: white !important;
    border: 2px solid #74b9ff !important;
    border-radius: 8px !important;
}

.stButton > button {
    background: linear-gradient(135deg, #0984e3, #0652dd) !important;
    color: white !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    border: none !important;
    padding: 12px 30px !important;
    font-family: 'Poppins', sans-serif !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0652dd, #023dd4) !important;
    transform: translateY(-1px) !important;
}
.stButton > button p { color: white !important; font-size: 16px !important; font-weight: 700 !important; }

.stDownloadButton > button {
    background: linear-gradient(135deg, #00b894, #00896e) !important;
    color: white !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    border: none !important;
}
.stDownloadButton > button p { color: white !important; }

.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.6) !important;
    border-radius: 12px !important;
    padding: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    font-size: 15px !important;
    font-weight: 700 !important;
    color: #0d3b66 !important;
    font-family: 'Poppins', sans-serif !important;
    padding: 10px 22px !important;
    border-radius: 8px !important;
}
.stTabs [aria-selected="true"] {
    background: white !important;
    color: #0984e3 !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
}

.kpi-card {
    background: white !important;
    padding: 20px !important;
    border-radius: 15px !important;
    text-align: center !important;
    box-shadow: 0px 6px 16px rgba(0,0,0,0.12) !important;
    border-left: 5px solid #0984e3 !important;
}
.kpi-card h4 {
    color: #636e72 !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    margin: 0 0 8px 0 !important;
}
.kpi-card h2 {
    color: #0d3b66 !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
    margin: 0 !important;
}

.insight-box {
    background: white !important;
    padding: 14px 18px !important;
    border-radius: 12px !important;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08) !important;
    margin-bottom: 10px !important;
    border-left: 4px solid #74b9ff !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    color: #2d3436 !important;
}
.insight-box b { color: #0984e3 !important; }

[data-testid="stDataFrame"] { border-radius: 12px !important; overflow: hidden !important; }

[data-testid="stAlert"] p { font-size: 15px !important; font-weight: 600 !important; color: inherit !important; }

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.7) !important;
    border: 2px dashed #74b9ff !important;
    border-radius: 12px !important;
}
            
/* ===== FORCE INPUT WHITE (Fix dark theme issue) ===== */

div[data-baseweb="input"] input,
div[data-baseweb="base-input"] input,
input[type="number"],
input[type="text"],
textarea {
    background-color: white !important;
    color: #1a1a2e !important;
    border-radius: 8px !important;
}

/* ===== FIX BUTTON DARK MODE TEXT ===== */

button[kind="primary"],
button[kind="secondary"] {
    color: white !important;
}
            
/* ===== FIX FILE UPLOADER BROWSE BUTTON ===== */

[data-testid="stFileUploader"] button {
    background: linear-gradient(135deg, #0984e3, #0652dd) !important;
    color: white !important;
    border-radius: 8px !important;
    border: none !important;
    font-weight: 600 !important;
}

/* Hover */
[data-testid="stFileUploader"] button:hover {
    background: linear-gradient(135deg, #0652dd, #023dd4) !important;
}

</style>
""", unsafe_allow_html=True)

# ================= BUILT-IN ENSEMBLE MODEL =================
@st.cache_resource
def build_ensemble_model():
    np.random.seed(42)
    N = 6000
    rows = []
    for label, (wlo, whi) in WQI_THRESHOLDS.items():
        n = N // 5
        wqi  = np.random.uniform(wlo, whi, n)
        do   = np.clip(np.random.normal(3 + (wqi/100)*7, 1.2, n), 0.5, 15)
        ph   = np.clip(np.random.normal(7.0 - (label-2)*0.3, 0.7, n), 4, 10)
        orp  = np.clip(np.random.normal(100 + (wqi/100)*200, 40, n), -100, 450)
        cond = np.clip(np.random.normal(350 + label*50, 80, n), 50, 1500)
        temp = np.clip(np.random.normal(25, 5, n), 5, 45)
        for i in range(n):
            rows.append([do[i], ph[i], orp[i], cond[i], temp[i], wqi[i], label])
    df_train = pd.DataFrame(rows, columns=MODEL_FEATURES + ['label'])
    X, y = df_train[MODEL_FEATURES].values, df_train['label'].values
    rf = RandomForestClassifier(n_estimators=300, max_depth=12,
                                min_samples_leaf=2, class_weight='balanced',
                                random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=150, max_depth=5,
                                    learning_rate=0.08, random_state=42)
    ensemble = VotingClassifier(estimators=[('rf', rf), ('gb', gb)],
                                voting='soft', weights=[2, 1])
    ensemble.fit(X, y)
    return ensemble

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    try:
        with open("wqi_status_model.pkl", "rb") as f:
            return pickle.load(f), "pkl"
    except:
        return build_ensemble_model(), "ensemble"

model, model_source = load_model()

# ================= CLEAN DATASET =================
import re as _re

def _coerce_cell(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s.lower() in ('', 'na', 'n/a', 'nan', 'null', 'none', '?', '??', '-', '--', 'error', '#n/a', '#value!'):
        return np.nan
    cleaned = _re.sub(r'[^\d.\-eE]', '', s)
    try:
        return float(cleaned)
    except:
        return np.nan

def clean_dataset(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    missing = [f for f in MODEL_FEATURES if f not in df.columns]
    if missing:
        st.error(f"❌ Missing Required Columns: {missing}. Your file must contain: {MODEL_FEATURES}")
        st.stop()
    for col in MODEL_FEATURES:
        df[col] = df[col].apply(_coerce_cell)
    df = df[MODEL_FEATURES]
    df = df.dropna(how="all")
    for col in MODEL_FEATURES:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        df[col] = df[col].clip(lower=q1 - 3*iqr, upper=q3 + 3*iqr)
    df = df.fillna(df.median())
    return df

# ================= CHART THEME =================
def style_fig(fig, height=420):
    fig.update_layout(
        height=height,
        font=dict(family="Poppins, sans-serif", size=14, color="#1a1a2e"),
        title_font=dict(family="Poppins, sans-serif", size=17, color="#0d3b66"),
        paper_bgcolor="rgba(255,255,255,0.85)",
        plot_bgcolor="rgba(255,255,255,0.85)",
        legend=dict(font=dict(size=13, color="#2d3436"), bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(l=20, r=20, t=55, b=20),
    )
    fig.update_xaxes(tickfont=dict(size=13, color="#2d3436", family="Poppins"),
                     title_font=dict(size=14, color="#0d3b66", family="Poppins"),
                     gridcolor="rgba(0,0,0,0.06)")
    fig.update_yaxes(tickfont=dict(size=13, color="#2d3436", family="Poppins"),
                     title_font=dict(size=14, color="#0d3b66", family="Poppins"),
                     gridcolor="rgba(0,0,0,0.06)")
    return fig

# ================= HEADER =================
st.title("💧 HydroGuard AI Dashboard")
st.markdown("### Intelligent Water Quality Analytics System")

if model_source == "ensemble":
    st.info("ℹ️ Using built-in Ensemble model (RF + GradientBoost). Place `wqi_status_model.pkl` in the same folder to use your trained model.")

if "bulk_df" not in st.session_state:
    st.session_state.bulk_df = None

# ================= TABS =================
t1, t2, t3 = st.tabs(["💧 Single Prediction", "📂 Bulk Prediction", "📊 WQI Dashboard"])

# ======================================================
# ================= SINGLE =============================
# ======================================================
with t1:
    st.subheader("Manual Water Sample Prediction")

    DEFAULTS = {'DO': 7.0, 'pH': 7.0, 'ORP': 200.0, 'Cond': 350.0, 'Temp': 27.0, 'WQI': 70.0}
    inputs = {}

    col1, col2, col3 = st.columns(3)
    with col1:
        inputs['DO'] = st.number_input("Dissolved Oxygen (DO)", 0.0, 15.0, DEFAULTS['DO'])
    with col2:
        inputs['pH'] = st.number_input("pH Level", 0.0, 14.0, DEFAULTS['pH'])
    with col3:
        inputs['ORP'] = st.number_input("ORP", -500.0, 500.0, DEFAULTS['ORP'])

    col4, col5, col6 = st.columns(3)
    with col4:
        inputs['Cond'] = st.number_input("Conductivity", 0.0, 2000.0, DEFAULTS['Cond'])
    with col5:
        inputs['Temp'] = st.number_input("Temperature", 0.0, 50.0, DEFAULTS['Temp'])
    with col6:
        inputs['WQI'] = st.number_input("Water Quality Index", 0.0, 100.0, DEFAULTS['WQI'])

    predict_btn = st.button("Predict Water Quality")

    result_placeholder = st.empty()

    if predict_btn:

        # ===== Create feature dataframe =====
        X = pd.DataFrame({
            "DO":[inputs["DO"]],
            "pH":[inputs["pH"]],
            "ORP":[inputs["ORP"]],
            "Cond":[inputs["Cond"]],
            "Temp":[inputs["Temp"]],
            "WQI":[inputs["WQI"]],
        })

        # ===== ML Prediction =====
        prediction = int(model.predict(X)[0])
        probabilities = model.predict_proba(X)[0]

        # ===== Hybrid Risk Rule Engine =====
        risk_score = 0

        # DO Risk
        if inputs["DO"] < 3:
            risk_score += 3
        elif inputs["DO"] < 5:
            risk_score += 1

        # Conductivity Risk
        if inputs["Cond"] > 900:
            risk_score += 3
        elif inputs["Cond"] > 500:
            risk_score += 1

        # Temperature Risk
        if inputs["Temp"] > 35:
            risk_score += 2
        elif inputs["Temp"] > 29:
            risk_score += 1

        # pH Risk
        if inputs["pH"] > 9 or inputs["pH"] < 5:
            risk_score += 2
        elif inputs["pH"] > 8:
            risk_score += 1

        # WQI Risk
        if inputs["WQI"] < 25:
            risk_score += 3
        elif inputs["WQI"] < 55:
            risk_score += 1
        # ===== Final Safe Decision =====
        if risk_score >= 5:
            prediction = 4
        elif risk_score >= 3:
            prediction = max(prediction,3)
        elif risk_score >= 1:
            prediction = max(prediction,2)

        label = STATUS_MAP[prediction]

        with result_placeholder.container():

            st.markdown("### 🎯 Prediction Result")

            if prediction >= 3:
                st.error(f"Water Quality: {label}")
            elif prediction == 2:
                st.warning(f"Water Quality: {label}")
            else:
                st.success(f"Water Quality: {label}")

            # # ===== Confidence Chart =====
            # st.markdown("### 📊 Model Confidence")

            # prob_df = pd.DataFrame({
            #     "Status":[STATUS_MAP[i] for i in range(len(probabilities))],
            #     "Probability":(probabilities*100).round(1)
            # })

            # fig = go.Figure(go.Bar(
            #     x=prob_df["Status"],
            #     y=prob_df["Probability"],
            #     text=prob_df["Probability"].astype(str)+"%",
            #     textposition="outside",
            #     marker_color=['#00b894','#0984e3','#f39c12','#e17055','#d63031']
            # ))

            # fig.update_layout(
            #     yaxis_title="Probability %",
            #     yaxis=dict(range=[0,100])
            # )

            # st.plotly_chart(fig,use_container_width=True)

# ======================================================
# ================= BULK ===============================
# ======================================================
with t2:

    st.subheader("📂 Bulk Water Quality Scanner")

    sample_df = pd.DataFrame({
        "DO":   [7.2, 6.8, 5.5],
        "pH":   [7.1, 6.5, 8.2],
        "ORP":  [210, 190, 170],
        "Cond": [320, 300, 410],
        "Temp": [26,  28,  30],
        "WQI":  [75,  68,  55]
    })

    # ===== 3 COLUMN CARD LAYOUT =====
    c1, c2, c3 = st.columns(3)

    # ================= SAMPLE CARD =================
    with c1:
        st.markdown("#### 📥 Sample File")

        format_choice = st.selectbox(
            "Format",
            ["CSV", "Excel", "JSON", "SQLite"],
            key="sample_format"
        )

        if st.button("Download", use_container_width=True):

            if format_choice == "CSV":
                st.download_button(
                    "Download CSV",
                    sample_df.to_csv(index=False).encode(),
                    "sample.csv",
                    use_container_width=True
                )

            elif format_choice == "Excel":
                from io import BytesIO
                buffer = BytesIO()
                sample_df.to_excel(buffer, index=False)
                st.download_button(
                    "Download Excel",
                    buffer.getvalue(),
                    "sample.xlsx",
                    use_container_width=True
                )

            elif format_choice == "JSON":
                st.download_button(
                    "Download JSON",
                    sample_df.to_json(orient="records"),
                    "sample.json",
                    use_container_width=True
                )

            elif format_choice == "SQLite":
                import sqlite3
                conn = sqlite3.connect("sample.db")
                sample_df.to_sql("water_quality", conn, if_exists="replace", index=False)
                conn.close()
                with open("sample.db", "rb") as f:
                    st.download_button(
                        "Download DB",
                        f,
                        "sample.db",
                        use_container_width=True
                    )

    # ================= GOOGLE DRIVE CARD =================
    with c2:
        st.markdown("#### 🔗 Google Drive")

        import gdown, tempfile

        drive_link = st.text_input("Paste Link", key="drive")

        if st.button("Fetch", use_container_width=True):

            try:
                file_id = drive_link.split("/d/")[1].split("/")[0]
                url = f"https://drive.google.com/uc?id={file_id}"

                temp = tempfile.NamedTemporaryFile(delete=False)
                gdown.download(url, temp.name, quiet=False)

                df = pd.read_csv(temp.name)
                st.session_state.uploaded_df = df

                st.success("Loaded")
            except:
                st.error("Invalid Drive Link")

    # ================= UPLOAD CARD =================
    with c3:
        st.markdown("#### 📤 Upload File")

        file = st.file_uploader(
            "CSV / Excel / JSON / DB",
            type=["csv", "xlsx", "json", "db"],
            label_visibility="collapsed"
        )

        if file:

            try:
                if file.name.endswith(".csv"):
                    df = pd.read_csv(file)

                elif file.name.endswith(".xlsx"):
                    df = pd.read_excel(file)

                elif file.name.endswith(".json"):
                    df = pd.read_json(file)

                elif file.name.endswith(".db"):
                    import sqlite3
                    with open("temp.db", "wb") as f:
                        f.write(file.getbuffer())
                    conn = sqlite3.connect("temp.db")
                    df = pd.read_sql("SELECT * FROM water_quality", conn)

                st.session_state.uploaded_df = df
                st.success("Uploaded")

            except:
                st.error("File Error")

    st.markdown("---")

    # ================= PREDICT BUTTON =================
    if "uploaded_df" in st.session_state:

        if st.button("🚀 Run Bulk Prediction", use_container_width=True):

            df = st.session_state.uploaded_df

            X = clean_dataset(df)
            preds = model.predict(X)

            result_df = X.copy()
            result_df["Predicted Status"] = [
                STATUS_MAP.get(int(p), str(p)) for p in preds
            ]

            st.session_state.bulk_df = result_df

            st.success("Prediction Completed")

            st.markdown("### 📋 Prediction Output Preview")

            show_cols = MODEL_FEATURES + ["Predicted Status"]

            st.dataframe(
            result_df[show_cols].head(20),
            use_container_width=True
            )

            fig = px.pie(
                result_df,
                names="Predicted Status",
                hole=0.4,
                color="Predicted Status",
                color_discrete_map=STATUS_COLOR
            )
            fig = style_fig(fig)
            st.plotly_chart(fig, use_container_width=True)

            st.download_button(
                "⬇️ Download Result",
                result_df.to_csv(index=False).encode(),
                "result.csv",
                use_container_width=True
            )

# ======================================================
# ================= DASHBOARD ==========================
# ======================================================
with t3:
    st.subheader("📊 WQI Analytics Dashboard")

    if st.session_state.bulk_df is not None:
        df = st.session_state.bulk_df.copy()

        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(f'<div class="kpi-card"><h4>Average pH</h4><h2>{df["pH"].mean():.2f}</h2></div>',
                    unsafe_allow_html=True)
        k2.markdown(f'<div class="kpi-card"><h4>Average DO (mg/L)</h4><h2>{df["DO"].mean():.2f}</h2></div>',
                    unsafe_allow_html=True)
        k3.markdown(f'<div class="kpi-card"><h4>Avg Temperature (°C)</h4><h2>{df["Temp"].mean():.1f}</h2></div>',
                    unsafe_allow_html=True)
        k4.markdown(f'<div class="kpi-card"><h4>Total Samples</h4><h2>{len(df)}</h2></div>',
                    unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        k5, k6 = st.columns(2)
        most_common = df["Predicted Status"].mode()[0]
        k5.markdown(f'<div class="kpi-card"><h4>Most Common Status</h4><h2>{most_common}</h2></div>',
                    unsafe_allow_html=True)
        k6.markdown(f'<div class="kpi-card"><h4>Average WQI</h4><h2>{df["WQI"].mean():.1f}</h2></div>',
                    unsafe_allow_html=True)

        st.markdown("---")

        # Pie
        st.markdown('<div class="insight-box"><b>Insight:</b> Distribution of samples across quality levels.</div>',
                    unsafe_allow_html=True)
        fig1 = px.pie(df, names="Predicted Status",
                      title="Overall Water Quality Situation",
                      color="Predicted Status",
                      color_discrete_map=STATUS_COLOR,
                      hole=0.4)
        fig1 = style_fig(fig1)
        fig1.update_traces(textfont_size=14, textfont_family="Poppins")
        st.plotly_chart(fig1, use_container_width=True, key=chart_key())

        # Bar count
        count_df = df["Predicted Status"].value_counts().reset_index()
        count_df.columns = ["Status", "Count"]
        count_df["Color"] = count_df["Status"].map(STATUS_COLOR)
        st.markdown('<div class="insight-box"><b>Insight:</b> Higher counts in poor categories indicate environmental risk.</div>',
                    unsafe_allow_html=True)
        fig2 = go.Figure(go.Bar(
            x=count_df["Status"], y=count_df["Count"],
            marker_color=count_df["Color"],
            text=count_df["Count"],
            textposition="outside",
            textfont=dict(size=15, color="#0d3b66", family="Poppins"),
        ))
        fig2.update_layout(title="Risk Level Comparison", yaxis_title="Sample Count",
                           yaxis=dict(range=[0, count_df["Count"].max() * 1.2]))
        fig2 = style_fig(fig2)
        st.plotly_chart(fig2, use_container_width=True, key=chart_key())

        # Grouped avg params
        avg_df = df.groupby("Predicted Status")[["pH", "DO", "Temp"]].mean().reset_index()
        st.markdown('<div class="insight-box"><b>Insight:</b> Parameter behaviour across different quality levels.</div>',
                    unsafe_allow_html=True)
        fig3 = px.bar(avg_df, x="Predicted Status", y=["pH", "DO", "Temp"],
                      barmode="group", title="Average Parameter Analysis",
                      color_discrete_sequence=["#0984e3", "#00b894", "#e17055"])
        fig3 = style_fig(fig3)
        fig3.update_traces(texttemplate='%{y:.1f}', textposition='outside',
                           textfont=dict(size=13, family="Poppins"))
        st.plotly_chart(fig3, use_container_width=True, key=chart_key())

        # Date trend
        if "Date" in df.columns:
            try:
                df["Date"] = pd.to_datetime(df["Date"])
                df["Month"] = df["Date"].dt.month_name()
                trend = df.groupby("Month")["WQI"].mean().reset_index()
                st.markdown('<div class="insight-box"><b>Insight:</b> Seasonal trend of water quality.</div>',
                            unsafe_allow_html=True)
                fig4 = px.line(trend, x="Month", y="WQI", markers=True,
                               title="Seasonal WQI Trend", line_shape="spline")
                fig4.update_traces(line=dict(width=3, color="#0984e3"),
                                   marker=dict(size=10, color="#0d3b66"))
                fig4 = style_fig(fig4)
                st.plotly_chart(fig4, use_container_width=True, key=chart_key())
            except:
                pass

        # Temp histogram
        st.markdown('<div class="insight-box"><b>Insight:</b> Temperature distribution in water samples.</div>',
                    unsafe_allow_html=True)
        fig5 = px.histogram(df, x="Temp", nbins=20, title="Temperature Spread",
                            color_discrete_sequence=["#0984e3"])
        fig5.update_traces(marker_line_width=1.5, marker_line_color="white")
        fig5 = style_fig(fig5)
        st.plotly_chart(fig5, use_container_width=True, key=chart_key())

        # DO vs WQI scatter
        st.markdown('<div class="insight-box"><b>Insight:</b> Relationship between Dissolved Oxygen and Water Quality Index.</div>',
                    unsafe_allow_html=True)
        fig6 = px.scatter(df, x="DO", y="WQI",
                          color="Predicted Status",
                          color_discrete_map=STATUS_COLOR,
                          title="DO vs WQI Relationship",
                          opacity=0.8)
        fig6.update_traces(marker=dict(size=9, line=dict(width=1, color="white")))
        fig6 = style_fig(fig6)
        st.plotly_chart(fig6, use_container_width=True, key=chart_key())

        # Correlation heatmap
        st.markdown('<div class="insight-box"><b>Insight:</b> Correlation between all water parameters.</div>',
                    unsafe_allow_html=True)
        corr = df.select_dtypes("number").corr().round(2)
        fig7 = px.imshow(corr, text_auto=True,
                         color_continuous_scale="RdBu_r",
                         title="Feature Correlation Matrix",
                         zmin=-1, zmax=1)
        fig7.update_traces(textfont=dict(size=14, family="Poppins", color="#1a1a2e"))
        fig7 = style_fig(fig7, height=500)
        st.plotly_chart(fig7, use_container_width=True, key=chart_key())

    else:
        st.info("Run Bulk Prediction First to see the dashboard.")

# ================= FOOTER =================
st.markdown("""
<hr>
<center style="color:#0d3b66; font-size:15px; font-weight:600;">
💧 HydroGuard AI • Smart Water Intelligence System <br>
<b>Created by Tirth</b>
</center>
""", unsafe_allow_html=True)

