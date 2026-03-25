import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import sqlite3
import uuid

# ================= UNIQUE KEY =================
def chart_key():
    return str(uuid.uuid4())

# ================= MODEL FEATURES =================
MODEL_FEATURES = ['DO','pH','ORP','Cond','Temp','WQI']

# ================= STATUS MAP =================
STATUS_MAP = {
    0:"Very Poor ❌",
    1:"Poor ⚠️",
    2:"Fair 🙂",
    3:"Good ✅",
    4:"Excellent 🌟"
}

# ================= PAGE =================
st.set_page_config(page_title="HydroGuard AI", layout="wide")

# ================= THEME =================
st.markdown("""
<style>
.stApp {
background: linear-gradient(120deg,#d4f1f9,#a6e3f5,#6fd3f7);
}
.kpi-card{
background:white;
padding:20px;
border-radius:15px;
text-align:center;
box-shadow:0px 6px 16px rgba(0,0,0,0.1);
}
.insight-box{
background:white;
padding:15px;
border-radius:12px;
box-shadow:0px 4px 12px rgba(0,0,0,0.1);
margin-bottom:10px;
}
</style>
""",unsafe_allow_html=True)

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    with open("wqi_status_model.pkl","rb") as f:
        return pickle.load(f)

model = load_model()

# ================= HEADER =================
st.title("💧 HydroGuard AI Dashboard")
st.markdown("### Intelligent Water Quality Analytics System")

if "bulk_df" not in st.session_state:
    st.session_state.bulk_df=None

# ================= TABS =================
t1,t2,t3 = st.tabs(["💧 Single Prediction","📂 Bulk Prediction","📊 WQI Dashboard"])

# ======================================================
# ================= SINGLE ==============================
# ======================================================
with t1:

    st.subheader("Manual Water Sample Prediction")

    inputs={}
    cols=st.columns(3)

    for i,f in enumerate(MODEL_FEATURES):
        inputs[f]=cols[i%3].number_input(f)

    if st.button("Predict Water Quality"):

        X=pd.DataFrame([inputs])
        pred=model.predict(X)[0]
        label=STATUS_MAP.get(pred,pred)

        if pred<=1:
            st.error(f"Water Quality : {label}")
        elif pred==2:
            st.warning(f"Water Quality : {label}")
        else:
            st.success(f"Water Quality : {label}")

        probs=model.predict_proba(X)[0]

        prob_df=pd.DataFrame({
            "Status":[STATUS_MAP[i] for i in range(len(probs))],
            "Probability":probs
        })

        st.markdown('<div class="insight-box"><b>Insight:</b> Model confidence for each water quality class.</div>',unsafe_allow_html=True)

        fig=px.bar(prob_df,x="Status",y="Probability",
                   color="Probability",
                   title="Prediction Confidence")
        st.plotly_chart(fig,use_container_width=True,key=chart_key())

# ======================================================
# ================= BULK ================================
# ======================================================
with t2:

    file=st.file_uploader("Upload CSV / Excel / JSON / SQLite",
                          type=["csv","xlsx","json","db"])

    if file:

        try:
            if file.name.endswith(".csv"):
                df=pd.read_csv(file)

            elif file.name.endswith(".xlsx"):
                df=pd.read_excel(file)

            elif file.name.endswith(".json"):
                df=pd.read_json(file)

            elif file.name.endswith(".db"):
                with open("temp.db","wb") as f:
                    f.write(file.getbuffer())
                conn=sqlite3.connect("temp.db")
                df=pd.read_sql("SELECT * FROM water_quality",conn)

            st.success("Dataset Loaded")
            st.dataframe(df.head())

            if st.button("Run Bulk Prediction"):

                X=df.copy()

                for c in X.columns:
                    if c.lower() in ["date","status"]:
                        X=X.drop(columns=[c])

                X=X.select_dtypes(include=["int64","float64"])

                for f in MODEL_FEATURES:
                    if f not in X.columns:
                        X[f]=0

                X=X[MODEL_FEATURES]

                preds=model.predict(X)

                df["Predicted_Status"]=[STATUS_MAP.get(p,p) for p in preds]

                st.session_state.bulk_df=df

                st.success("Prediction Completed")

                st.markdown('<div class="insight-box"><b>Insight:</b> Overall water quality distribution after prediction.</div>',unsafe_allow_html=True)

                fig=px.pie(df,names="Predicted_Status",
                           title="Water Quality Distribution")
                st.plotly_chart(fig,use_container_width=True,key=chart_key())

                csv=df.to_csv(index=False).encode("utf-8")

                st.download_button("Download Results",
                                   csv,
                                   "water_predictions.csv",
                                   "text/csv")

        except Exception as e:
            st.error(e)

# ======================================================
# ================= DASHBOARD ===========================
# ======================================================
with t3:

    st.subheader("📊 WQI Analytics Dashboard")

    if st.session_state.bulk_df is not None:

        df=st.session_state.bulk_df.copy()

        k1,k2,k3,k4=st.columns(4)

        k1.markdown(f'<div class="kpi-card"><h4>Average pH</h4><h2>{round(df["pH"].mean(),2)}</h2></div>',unsafe_allow_html=True)
        k2.markdown(f'<div class="kpi-card"><h4>Average DO</h4><h2>{round(df["DO"].mean(),2)}</h2></div>',unsafe_allow_html=True)
        k3.markdown(f'<div class="kpi-card"><h4>Average Temperature</h4><h2>{round(df["Temp"].mean(),2)}</h2></div>',unsafe_allow_html=True)
        k4.markdown(f'<div class="kpi-card"><h4>Total Samples</h4><h2>{len(df)}</h2></div>',unsafe_allow_html=True)

        st.markdown("---")

        st.markdown('<div class="insight-box"><b>Insight:</b> Shows distribution of samples across quality levels.</div>',unsafe_allow_html=True)

        fig1=px.pie(df,names="Predicted_Status",
                    title="Overall Water Quality Situation")
        st.plotly_chart(fig1,use_container_width=True,key=chart_key())

        count_df=df["Predicted_Status"].value_counts().reset_index()
        count_df.columns=["Status","Count"]

        st.markdown('<div class="insight-box"><b>Insight:</b> Higher counts in poor categories indicate environmental risk.</div>',unsafe_allow_html=True)

        fig2=px.bar(count_df,x="Status",y="Count",
                    color="Status",
                    title="Risk Level Comparison")
        st.plotly_chart(fig2,use_container_width=True,key=chart_key())

        avg_df=df.groupby("Predicted_Status")[["pH","DO","Temp"]].mean().reset_index()

        st.markdown('<div class="insight-box"><b>Insight:</b> Parameter behaviour across different quality levels.</div>',unsafe_allow_html=True)

        fig3=px.bar(avg_df,x="Predicted_Status",
                    y=["pH","DO","Temp"],
                    barmode="group",
                    title="Average Parameter Analysis")
        st.plotly_chart(fig3,use_container_width=True,key=chart_key())

        if "Date" in df.columns:
            df["Date"]=pd.to_datetime(df["Date"])
            df["Month"]=df["Date"].dt.month_name()

            trend=df.groupby("Month")["WQI"].mean().reset_index()

            st.markdown('<div class="insight-box"><b>Insight:</b> Seasonal trend of water quality.</div>',unsafe_allow_html=True)

            fig4=px.line(trend,x="Month",y="WQI",
                         title="Seasonal WQI Trend")
            st.plotly_chart(fig4,use_container_width=True,key=chart_key())

        st.markdown('<div class="insight-box"><b>Insight:</b> Temperature distribution in water samples.</div>',unsafe_allow_html=True)

        fig5=px.histogram(df,x="Temp",
                          title="Temperature Spread")
        st.plotly_chart(fig5,use_container_width=True,key=chart_key())

        st.markdown('<div class="insight-box"><b>Insight:</b> Relationship between Dissolved Oxygen and Water Quality.</div>',unsafe_allow_html=True)

        fig6=px.scatter(df,x="DO",y="WQI",
                        color="Predicted_Status",
                        title="DO vs WQI Relationship")
        st.plotly_chart(fig6,use_container_width=True,key=chart_key())

        st.markdown('<div class="insight-box"><b>Insight:</b> Correlation between all water parameters.</div>',unsafe_allow_html=True)

        corr=df.select_dtypes("number").corr()

        fig7=px.imshow(corr,text_auto=True,
                       title="Feature Correlation Intelligence")
        st.plotly_chart(fig7,use_container_width=True,key=chart_key())

    else:
        st.info("Run Bulk Prediction First")

# ================= FOOTER =================
st.markdown("""
<hr>
<center>
💧 HydroGuard AI • Smart Water Intelligence System <br>
<b>Created by Tirth</b>
</center>
""",unsafe_allow_html=True)