from pathlib import Path
import os, io, sqlite3, hashlib, secrets, json, time
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import joblib

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
DATA_PATH = DATA_DIR / "california_housing.csv"
MODEL_PATH = APP_DIR / "models"
MODEL_PATH.mkdir(exist_ok=True)
DB_PATH = APP_DIR / "house_price_lab.db"

st.set_page_config(page_title="House Price Intelligence Lab", page_icon="🏡", layout="wide")
st.markdown("""
<style>
.block-container {padding-top:1.4rem; padding-bottom:2rem; max-width:1500px}
.hero {padding:1.6rem 1.8rem;border-radius:20px;background:linear-gradient(120deg,#10243a,#246b73);color:white;margin-bottom:1rem}
.hero h1 {color:white;font-size:2.15rem;margin-bottom:.25rem}
.hero p {color:#e2f3f4;margin:0}
.small-note {font-size:.85rem;color:#64748b}
div[data-testid="stMetric"] {border:1px solid rgba(120,130,150,.25);padding:12px;border-radius:14px}
</style>
""", unsafe_allow_html=True)

FEATURES = [
"Professional Home Dashboard","Responsive UI and Theme","Navigation Categories","Dashboard KPI Cards",
"Property Cards and Images","Property Search and Filters","English/Telugu UI labels","House Price Prediction",
"Multiple Feature Inputs","Bulk CSV Prediction","Asking vs Predicted Price","Over/Underpricing Flags",
"Price per Square Foot","Prediction Reliability Warnings","What-If Simulator","Prediction Explanation",
"Model Comparison","Best Model Selection","Hyperparameter Tuning","Cross-Validation",
"Feature Importance","SHAP Explainability","Preprocessing Pipeline","Feature Engineering",
"Data Leakage Checks","Outlier Detection","Model Drift Monitoring","Automated Model Training",
"Model Version Tracking","Area-wise Price Analysis","Interactive Property Map","Historical Price Analysis",
"Price Trend Forecasting (data-limited)","Neighborhood Comparison","Affordability Score","Mortgage/EMI Calculator",
"Buy-vs-Rent Calculator","Rental Yield and ROI","Renovation Scenario Simulator","Data Quality Checker",
"SQL Database Integration","User Login Demo","Saved Favorites","Persistent Prediction History",
"PDF Valuation Report","Excel/CSV Export","AI Property Assistant (rule-based)","REST API with FastAPI",
"Automated Tests and CI Template","Deployment Readiness"
]

@st.cache_data(show_spinner=False)
def load_data():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        try:
            bunch = fetch_california_housing(as_frame=True)
            df = bunch.frame.copy()
            df["MedHouseVal_USD"] = df["MedHouseVal"] * 100_000
            df.to_csv(DATA_PATH, index=False)
        except Exception as exc:
            raise RuntimeError("Housing dataset not found and download failed. Run `python download_dataset.py` while connected to the internet.") from exc
    if "MedHouseVal_USD" not in df.columns and "MedHouseVal" in df.columns:
        df["MedHouseVal_USD"] = df["MedHouseVal"] * 100_000
    return df

@st.cache_resource(show_spinner=False)
def build_models(df):
    feature_cols = ["MedInc","HouseAge","AveRooms","AveBedrms","Population","AveOccup","Latitude","Longitude"]
    X = df[feature_cols].copy()
    y = df["MedHouseVal"]  # units of $100,000; original dataset target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    candidates = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=120, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "Extra Trees": ExtraTreesRegressor(n_estimators=120, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42, max_iter=120)
    }
    results, fitted = [], {}
    for name, model in candidates.items():
        pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results.append({"Model":name,"MAE_USD":mean_absolute_error(y_test,pred)*100000,
                        "RMSE_USD":mean_squared_error(y_test,pred)**0.5*100000,
                        "R2":r2_score(y_test,pred)})
        fitted[name] = pipe
    metrics = pd.DataFrame(results).sort_values("RMSE_USD")
    best_name = metrics.iloc[0]["Model"]
    return feature_cols, X_train, X_test, y_train, y_test, fitted, metrics, best_name

def db_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""CREATE TABLE IF NOT EXISTS predictions(
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, username TEXT,
        MedInc REAL, HouseAge REAL, AveRooms REAL, AveBedrms REAL,
        Population REAL, AveOccup REAL, Latitude REAL, Longitude REAL,
        predicted_usd REAL, model_name TEXT)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS favorites(
        id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, label TEXT, details TEXT, created_at TEXT)""")
    conn.commit()
    return conn

def add_prediction(row, price, model_name):
    conn=db_conn()
    vals=[datetime.now().isoformat(timespec="seconds"),st.session_state.get("username","guest")]
    vals += [float(row[c]) for c in ["MedInc","HouseAge","AveRooms","AveBedrms","Population","AveOccup","Latitude","Longitude"]]
    vals += [float(price),model_name]
    conn.execute("INSERT INTO predictions(created_at,username,MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup,Latitude,Longitude,predicted_usd,model_name) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", vals)
    conn.commit(); conn.close()

def get_history():
    conn=db_conn()
    df=pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC",conn)
    conn.close()
    return df

def make_pdf_bytes(title, details):
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        buf=io.BytesIO(); c=canvas.Canvas(buf,pagesize=letter)
        width,height=letter; y=height-60
        c.setFont("Helvetica-Bold",18); c.drawString(50,y,title); y-=35
        c.setFont("Helvetica",10)
        for line in details:
            if y<70: c.showPage(); y=height-60; c.setFont("Helvetica",10)
            c.drawString(50,y,str(line)[:110]); y-=18
        c.save(); return buf.getvalue()
    except Exception as e:
        return None

try:
    df=load_data()
    feature_cols, X_train, X_test, y_train, y_test, models, metrics, best_model_name=build_models(df)
except Exception as e:
    st.error(str(e))
    st.info("Run `python download_dataset.py` in your terminal, then restart the app.")
    st.stop()

if "username" not in st.session_state: st.session_state.username="guest"
if "authenticated" not in st.session_state: st.session_state.authenticated=False
if "favorites" not in st.session_state: st.session_state.favorites=[]

with st.sidebar:
    st.markdown("## 🏡 House Intelligence")
    st.caption("AI valuation • Market analytics • ML lab")
    page=st.radio("Workspace",[
        "Overview","Price Prediction","Bulk Prediction","Market Analytics","Interactive Map",
        "Model Lab","Explainable AI","Data Quality","Drift Monitoring","Forecasting",
        "Investment Calculator","Property Comparison","Favorites & History","PDF Report",
        "Data Explorer","Login Demo","API & Deployment","Feature Checklist","About Dataset"
    ])
    st.divider()
    st.caption(f"Dataset rows: {len(df):,}")
    st.caption(f"Selected model: {best_model_name}")
    st.caption("Currency: USD (historical dataset)")

st.markdown('<div class="hero"><h1>House Price Intelligence Lab</h1><p>Professional ML valuation, property analytics, and investment scenario workspace</p></div>',unsafe_allow_html=True)
st.warning("Dataset limitation: California Housing is based on 1990 census block groups. It estimates historical area-level median values, not current individual home listings or Indian market prices. Values and outputs are illustrative for ML learning.", icon="⚠️")

if page=="Overview":
    st.subheader("Executive overview")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Observations",f"{len(df):,}")
    c2.metric("Input features",len(feature_cols))
    c3.metric("Best model",best_model_name)
    c4.metric("Best model R²",f"{metrics.iloc[0]['R2']:.3f}")
    left,right=st.columns([1.4,1])
    with left:
        fig=px.histogram(df,x="MedHouseVal_USD",nbins=45,title="Historical district median values",labels={"MedHouseVal_USD":"USD"})
        st.plotly_chart(fig,use_container_width=True)
    with right:
        st.dataframe(metrics,use_container_width=True,hide_index=True)
    st.markdown("### Quick actions")
    st.write("Use the sidebar to run predictions, inspect model metrics, explore geographic data, validate quality, and calculate investment scenarios.")
    st.caption("The California dataset has latitude/longitude for census areas; markers are not exact property addresses.")

elif page=="Price Prediction":
    st.subheader("Property-area valuation")
    st.caption("Inputs correspond to California census block-group attributes, not a typical individual property listing.")
    c1,c2,c3,c4=st.columns(4)
    with c1: medinc=st.number_input("Median income (10k USD units)",0.5,20.0,3.5,.1)
    with c2: age=st.number_input("Median house age (years)",1.0,60.0,25.0,1.0)
    with c3: rooms=st.number_input("Average rooms / household",1.0,20.0,5.0,.1)
    with c4: beds=st.number_input("Average bedrooms / household",0.2,10.0,1.0,.1)
    c1,c2,c3,c4=st.columns(4)
    with c1: pop=st.number_input("Population",1.0,50000.0,1200.0,50.0)
    with c2: occ=st.number_input("Average occupancy",0.5,20.0,3.0,.1)
    with c3: lat=st.number_input("Latitude",32.0,42.5,34.1,.1)
    with c4: lon=st.number_input("Longitude",-125.0,-114.0,-118.2,.1)
    model_name=st.selectbox("Model",list(models.keys()),index=list(models.keys()).index(best_model_name))
    if st.button("Estimate area median value",type="primary"):
        row=pd.DataFrame([[medinc,age,rooms,beds,pop,occ,lat,lon]],columns=feature_cols)
        pred=float(models[model_name].predict(row)[0])*100000
        st.success(f"Estimated historical area median value: ${pred:,.0f}")
        st.caption("This is not a current individual property valuation. Use the estimate only as an ML demonstration.")
        st.metric("Model estimate",f"${pred:,.0f}")
        try:
            pi=permutation_importance(models[model_name],X_test,y_test,n_repeats=3,random_state=42,scoring="neg_mean_absolute_error")
            imp=pd.DataFrame({"Feature":feature_cols,"Importance":pi.importances_mean}).sort_values("Importance",ascending=False)
            st.plotly_chart(px.bar(imp,x="Importance",y="Feature",orientation="h",title="Permutation importance (validation sample)"),use_container_width=True)
        except Exception: pass
        add_prediction(row.iloc[0],pred,model_name)
        st.session_state["last_report"]={"price":pred,"model":model_name,"row":row.iloc[0].to_dict()}
        st.info("Prediction saved to local SQLite history.")

elif page=="Bulk Prediction":
    st.subheader("Bulk CSV prediction")
    st.write("Upload a CSV containing these columns:",", ".join(feature_cols))
    file=st.file_uploader("Upload property-area CSV",type=["csv"])
    if file:
        up=pd.read_csv(file)
        missing=[c for c in feature_cols if c not in up.columns]
        if missing: st.error("Missing required columns: "+", ".join(missing))
        else:
            clean=up[feature_cols].apply(pd.to_numeric,errors="coerce")
            if st.button("Run bulk predictions",type="primary"):
                clean=clean.fillna(df[feature_cols].median())
                up["Predicted_Median_Value_USD"]=models[best_model_name].predict(clean)*100000
                st.dataframe(up.head(100),use_container_width=True)
                st.download_button("Download prediction CSV",up.to_csv(index=False).encode(), "bulk_predictions.csv","text/csv")

elif page=="Market Analytics":
    st.subheader("Market analytics")
    c1,c2=st.columns(2)
    with c1:
        xcol=st.selectbox("Group / compare by",["HouseAge","MedInc","Latitude","Longitude","AveRooms","AveOccup"])
    with c2:
        bins=st.slider("Number of groups",4,20,10)
    temp=df[[xcol,"MedHouseVal_USD"]].copy()
    temp["Group"]=pd.qcut(temp[xcol],q=bins,duplicates="drop")
    agg=temp.groupby("Group",observed=True)["MedHouseVal_USD"].agg(["median","mean","count"]).reset_index()
    agg["Group"]=agg["Group"].astype(str)
    st.plotly_chart(px.bar(agg,x="Group",y="median",hover_data=["mean","count"],title=f"Median value by {xcol} group"),use_container_width=True)
    st.plotly_chart(px.scatter(df.sample(min(2500,len(df)),random_state=42),x="Longitude",y="Latitude",color="MedHouseVal_USD",size="MedHouseVal_USD",title="Geographic price pattern (sample)"),use_container_width=True)

elif page=="Interactive Map":
    st.subheader("Interactive property-area map")
    st.caption("Each point represents a census block group, not an individual property.")
    n=st.slider("Map sample size",500,5000,2000,500)
    mapdf=df.sample(min(n,len(df)),random_state=42)
    fig = px.scatter_map(
            mapdf,
            lat="Latitude",
            lon="Longitude",
            color="MedHouseVal_USD",
            size="MedHouseVal_USD",
            hover_data=["MedInc", "HouseAge", "MedHouseVal_USD"],
            zoom=4,
            height=620,
            color_continuous_scale="Viridis",
            map_style="open-street-map",
        )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig,use_container_width=True)

elif page=="Model Lab":
    st.subheader("Model comparison and evaluation")
    st.dataframe(metrics.style.format({"MAE_USD":"${:,.0f}","RMSE_USD":"${:,.0f}","R2":"{:.3f}"}),use_container_width=True,hide_index=True)
    a,b=st.columns(2)
    with a: st.plotly_chart(px.bar(metrics,x="Model",y="RMSE_USD",title="RMSE — lower is better"),use_container_width=True)
    with b: st.plotly_chart(px.bar(metrics,x="Model",y="R2",title="R² — higher is better"),use_container_width=True)
    st.caption("Models use the same 80/20 split and random_state=42. Scores are historical benchmark results, not a guarantee of live-market accuracy.")
    if st.button("Run 3-fold cross-validation on Random Forest"):
        with st.spinner("Running cross-validation..."):
            cv= cross_val_score(models["Random Forest"],df[feature_cols],df["MedHouseVal"],cv=3,scoring="neg_mean_absolute_error",n_jobs=-1)
        st.write("Fold MAE (USD):", [round(abs(v)*100000) for v in cv])
        st.metric("Mean CV MAE",f"${abs(cv.mean())*100000:,.0f}")

elif page=="Explainable AI":
    st.subheader("Explainable AI")
    st.write("This page provides permutation-based global feature importance. SHAP values are available if the optional `shap` package is installed.")
    imp=permutation_importance(models[best_model_name],X_test,y_test,n_repeats=4,random_state=42,scoring="neg_mean_absolute_error")
    impdf=pd.DataFrame({"Feature":feature_cols,"Importance":imp.importances_mean}).sort_values("Importance")
    st.plotly_chart(px.bar(impdf,x="Importance",y="Feature",orientation="h",title="Global permutation importance"),use_container_width=True)
    try:
        import shap
        st.success("SHAP is installed. Use a small sample for explanations.")
        sample=X_test.sample(min(80,len(X_test)),random_state=42)
        model=models[best_model_name]
        underlying=model.named_steps["model"]
        transformed=model.named_steps["imputer"].transform(sample)
        explainer=shap.Explainer(underlying,transformed)
        sv=explainer(transformed)
        import matplotlib.pyplot as plt
        shap.summary_plot(
            sv,
            features=transformed,
            feature_names=feature_cols,
            show=False
        )
        st.pyplot(plt.gcf())
        plt.close()
    except Exception:
        st.info("Install optional SHAP support with `pip install shap`. SHAP is optional; permutation importance works without it.")

elif page=="Data Quality":
    st.subheader("Data quality checker")
    missing=df.isna().sum().rename("Missing").to_frame()
    missing["Missing %"]=(missing["Missing"]/len(df)*100).round(2)
    duplicates=int(df.duplicated().sum())
    st.metric("Duplicate rows",f"{duplicates:,}")
    st.dataframe(missing,use_container_width=True)
    quality=pd.DataFrame({
        "Check":["Rows","Columns","Duplicate rows","Missing cells","Negative target values","Infinite numeric values"],
        "Result":[len(df),len(df.columns),duplicates,int(df.isna().sum().sum()),int((df["MedHouseVal"]<0).sum()),int(np.isinf(df.select_dtypes("number")).sum().sum())]
    })
    st.dataframe(quality,use_container_width=True,hide_index=True)
    st.caption("This report checks the bundled source dataset. Uploaded data is validated on the Bulk Prediction page.")

elif page=="Drift Monitoring":
    st.subheader("Model monitoring and drift detection")
    st.caption("Upload new unlabeled feature data to compare feature distributions with the training baseline.")
    upload=st.file_uploader("Upload recent data CSV (feature columns required)",type=["csv"],key="driftfile")
    if upload:
        new=pd.read_csv(upload)
        missing=[c for c in feature_cols if c not in new.columns]
        if missing: st.error("Missing columns: "+", ".join(missing))
        else:
            rows=[]
            for col in feature_cols:
                a=df[col].dropna(); b=pd.to_numeric(new[col],errors="coerce").dropna()
                denom=max(float(a.std()),1e-9)
                shift=abs(float(b.mean())-float(a.mean()))/denom if len(b) else np.nan
                rows.append({"Feature":col,"Baseline mean":a.mean(),"New mean":b.mean() if len(b) else np.nan,"Mean shift / baseline SD":shift,"Flag":bool(shift>0.5) if np.isfinite(shift) else True})
            result=pd.DataFrame(rows)
            st.dataframe(result,use_container_width=True,hide_index=True)
            st.caption("The 0.5-standard-deviation threshold is a simple heuristic, not a universal drift test.")

elif page=="Forecasting":
    st.subheader("Price trend forecasting")
    st.warning("The bundled dataset has no observation dates and cannot validate a true time-series forecast. This page demonstrates a trend view of historical geographic groups, not a forecast into future calendar years.")
    agebins=pd.cut(df["HouseAge"],bins=10)
    trend=df.assign(AgeGroup=agebins).groupby("AgeGroup",observed=True)["MedHouseVal_USD"].median().reset_index()
    trend["AgeGroup"]=trend["AgeGroup"].astype(str)
    st.plotly_chart(px.line(trend,x="AgeGroup",y="MedHouseVal_USD",markers=True,title="Historical value by housing-age band (not a time forecast)"),use_container_width=True)
    st.info("To add a validated forecast, connect dated monthly or quarterly housing index/sales data and evaluate with a time-based holdout.")

elif page=="Investment Calculator":
    st.subheader("Investment and rental yield calculator")
    c1,c2,c3=st.columns(3)
    with c1: price=st.number_input("Purchase price (USD)",min_value=1000.0,value=300000.0,step=5000.0)
    with c2: rent=st.number_input("Monthly rent (USD)",min_value=0.0,value=1800.0,step=100.0)
    with c3: expenses=st.number_input("Annual operating expenses (USD)",min_value=0.0,value=4000.0,step=250.0)
    c1,c2,c3=st.columns(3)
    with c1: down=st.slider("Down payment (%)",0,100,20)
    with c2: rate=st.number_input("Annual loan interest (%)",0.0,25.0,6.5,.25)
    with c3: years=st.number_input("Loan term (years)",1,40,30)
    gross=rent*12/price*100
    net=(rent*12-expenses)/price*100
    loan=price*(1-down/100)
    r=rate/1200; n=years*12
    emi=(loan*r*(1+r)**n/((1+r)**n-1)) if r else loan/n
    a,b,c,d=st.columns(4)
    a.metric("Gross rental yield",f"{gross:.2f}%")
    b.metric("Net yield before financing",f"{net:.2f}%")
    c.metric("Monthly EMI",f"${emi:,.0f}")
    d.metric("Annual cash flow after EMI",f"${rent*12-expenses-emi*12:,.0f}")
    st.caption("Scenario calculator only. Excludes taxes, vacancy, closing costs, insurance variations, and local rules. Inputs are user assumptions.")

elif page=="Property Comparison":
    st.subheader("Compare property-area profiles")
    choices=list(df.index)
    selected=st.multiselect("Choose up to 4 dataset records",choices,default=choices[:2],max_selections=4,format_func=lambda i:f"Record {i} — ${df.loc[i,'MedHouseVal_USD']:,.0f}")
    if selected:
        compare=df.loc[selected,feature_cols+["MedHouseVal_USD"]].copy()
        compare.index=[f"Record {i}" for i in selected]
        st.dataframe(compare,use_container_width=True)
        st.plotly_chart(px.bar(compare.reset_index(),x="index",y="MedHouseVal_USD",title="Historical area median value comparison"),use_container_width=True)

elif page=="Favorites & History":
    st.subheader("Prediction history and favorites")
    hist=get_history()
    if hist.empty: st.info("No predictions saved yet. Make a prediction to populate history.")
    else:
        st.dataframe(hist,use_container_width=True,hide_index=True)
        st.download_button("Export prediction history CSV",hist.to_csv(index=False).encode(),"prediction_history.csv","text/csv")
    st.markdown("### Save a favorite scenario")
    label=st.text_input("Favorite label",placeholder="e.g. Coastal area scenario")
    details=st.text_area("Notes / property details")
    if st.button("Save favorite"):
        conn=db_conn()
        conn.execute("INSERT INTO favorites(username,label,details,created_at) VALUES(?,?,?,?)",(st.session_state.username,label,details,datetime.now().isoformat(timespec="seconds")))
        conn.commit(); fav=pd.read_sql_query("SELECT * FROM favorites ORDER BY id DESC",conn); conn.close()
        st.success("Favorite saved.")
    conn=db_conn(); fav=pd.read_sql_query("SELECT * FROM favorites ORDER BY id DESC",conn); conn.close()
    st.dataframe(fav,use_container_width=True,hide_index=True)

elif page=="PDF Report":
    st.subheader("Downloadable valuation report")
    last=st.session_state.get("last_report")
    if last:
        lines=[f"Generated: {datetime.now().isoformat(timespec='seconds')}",f"Model: {last['model']}",f"Estimated historical area median value: ${last['price']:,.0f}","Dataset: California Housing; 1990 census-derived census block groups.","Limitation: not a current individual property valuation or Indian-market estimate.","Input features:"]+[f"{k}: {v}" for k,v in last["row"].items()]
    else:
        lines=["No recent prediction in this session. This sample report summarizes the dataset and its limitations.","Dataset: California Housing; 1990 census-derived census block groups."]
    pdf=make_pdf_bytes("House Price Intelligence — Valuation Summary",lines)
    if pdf:
        st.download_button("Download PDF valuation report",pdf,"house_valuation_report.pdf","application/pdf",type="primary")
    else: st.error("PDF support unavailable. Install requirements.txt again.")

elif page=="Data Explorer":
    st.subheader("Dataset explorer")
    columns=st.multiselect("Columns",list(df.columns),default=list(df.columns[:5]))
    maxrows=st.slider("Rows to display",10,500,100,10)
    st.dataframe(df[columns].head(maxrows),use_container_width=True)
    st.download_button("Download dataset CSV",df.to_csv(index=False).encode(),"california_housing.csv","text/csv")
    st.write("Summary statistics")
    st.dataframe(df.describe().T,use_container_width=True)

elif page=="Login Demo":
    st.subheader("Login and account demo")
    st.warning("This is a demonstration only. It does not implement production-grade password authentication. Do not enter a real password.")
    username=st.text_input("Demo username",value=st.session_state.username)
    if st.button("Start demo session"):
        st.session_state.username=username.strip() or "guest"
        st.session_state.authenticated=True
        st.success(f"Demo session active for {st.session_state.username}.")
    if st.session_state.authenticated:
        st.info(f"Current demo user: {st.session_state.username}")
        if st.button("Log out"):
            st.session_state.username="guest"; st.session_state.authenticated=False
            st.rerun()
    st.caption("Production version should use a secure auth provider or properly salted password hashing, session protection, rate limits, and secrets management.")

elif page=="API & Deployment":
    st.subheader("REST API and deployment")
    st.code("uvicorn api:app --reload",language="bash")
    st.write("Open `http://127.0.0.1:8000/docs` to inspect the API after starting it.")
    st.markdown("**Suggested deployment checklist**")
    st.write("- Set `DATABASE_URL` through deployment secrets.")
    st.write("- Pin dependencies and run tests.")
    st.write("- Use HTTPS and secure authentication.")
    st.write("- Do not commit `.env`, database passwords, or user data.")
    st.write("- Explain the dataset limitations in the live demo.")
    st.download_button("Download deployment checklist", "\n".join(["House Price Intelligence deployment checklist","1. Configure secrets via environment variables","2. Install requirements.txt","3. Run pytest","4. Start FastAPI with uvicorn","5. Start Streamlit","6. Enable HTTPS and authentication","7. Monitor logs and errors","8. Document dataset limitations"]), "deployment_checklist.txt")

elif page=="Feature Checklist":
    st.subheader("50-feature portfolio checklist")
    st.caption("The app contains working implementations, demonstrations, and planned production upgrades. Review the notes before describing a feature as production-ready.")
    for start in range(0,len(FEATURES),10):
        st.markdown(f"### Features {start+1}–{min(start+10,len(FEATURES))}")
        for i,feature in enumerate(FEATURES[start:start+10],start+1):
            st.checkbox(f"{i}. {feature}",value=(i in [1,8,17,19,22,24,27,28,31,32,38,40,41,44,45,46,48]),key=f"feature_{i}")
    st.caption("A checkbox is a progress tracker only; it does not prove production implementation.")

elif page=="About Dataset":
    st.subheader("Dataset source and limitations")
    st.markdown("""
**Dataset:** California Housing (scikit-learn)  
**Observation unit:** California census block groups  
**Source context:** Derived from 1990 U.S. Census data  
**Rows:** 20,640  
**Input features:** 8  
**Target:** `MedHouseVal`, median house value in units of $100,000  

Official source: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html

**Not suitable for:** current market valuations, individual property sale estimates, or Indian house price predictions without relevant local data.
""")
    st.dataframe(df.head(10),use_container_width=True)
    st.download_button("Download current dataset",df.to_csv(index=False).encode(),"california_housing.csv","text/csv")

st.divider()
st.caption("House Price Intelligence Lab • Educational portfolio project • Validate all outputs before real-world use.")
