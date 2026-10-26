from pathlib import Path
from textwrap import dedent

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

st.set_page_config(page_title="FactoryLens | Manufacturing Intelligence", page_icon="🏭", layout="wide", initial_sidebar_state="expanded")


def H(markup: str) -> None:
    """Render custom HTML safely without indentation becoming a Markdown code block."""
    st.markdown(dedent(markup).strip(), unsafe_allow_html=True)


H("""
<style>
.stApp{background:#0B1220;color:#E5E7EB}.block-container{max-width:1450px;padding-top:1.5rem;padding-bottom:3rem}#MainMenu,footer,header{visibility:hidden}
[data-testid="stSidebar"]{background:#0A1020;border-right:1px solid #1F2937}h1,h2,h3,h4{color:#F8FAFC!important}
p,label,.stMarkdown,.stCaption{color:#CBD5E1}
.hero{background:linear-gradient(135deg,#111827,#0F172A 60%,#10233C);border:1px solid #243244;border-radius:20px;padding:28px 30px;margin-bottom:22px}.hero-kicker,.section-kicker{color:#38BDF8;font-size:.7rem;font-weight:850;letter-spacing:.15em;text-transform:uppercase}.hero-title{color:#F8FAFC;font-size:2.35rem;font-weight:900;line-height:1.05}.hero-subtitle{color:#94A3B8;font-size:1rem;margin-top:10px}.status{display:inline-block;margin-top:15px;padding:7px 12px;border-radius:999px;background:#0C2638;border:1px solid #174A63;color:#7DD3FC;font-size:.72rem;font-weight:800}
.kpi{background:#111827;border:1px solid #243244;border-radius:16px;padding:18px;min-height:112px}.kpi-label{color:#94A3B8;font-size:.68rem;text-transform:uppercase;letter-spacing:.1em;font-weight:800}.kpi-value{color:#F8FAFC;font-size:1.72rem;font-weight:900;margin-top:8px}.kpi-note{color:#64748B;font-size:.75rem;margin-top:7px}
.panel{background:#111827;border:1px solid #243244;border-radius:18px;padding:20px}.panel-title{color:#F8FAFC;font-size:.98rem;font-weight:850;margin-bottom:9px}.panel-copy{color:#CBD5E1;font-size:.87rem;line-height:1.6}.muted{color:#64748B;font-size:.75rem;line-height:1.45}.badge{display:inline-block;border-radius:999px;padding:5px 10px;font-size:.66rem;font-weight:850;letter-spacing:.06em;margin-bottom:9px}.blue{color:#7DD3FC;background:#082638;border:1px solid #16465E}.red{color:#FCA5A5;background:#35141A;border:1px solid #6A2730}.amber{color:#FCD34D;background:#36270B;border:1px solid #6C4C10}
.risk{background:#111827;border:1px solid #243244;border-radius:18px;padding:24px;text-align:center}.prob{font-size:3.3rem;font-weight:900;color:#F8FAFC}.high{color:#F87171}.medium{color:#FBBF24}.low{color:#4ADE80}.recommend{background:#0F172A;border:1px solid #243244;border-radius:14px;padding:15px;color:#CBD5E1;font-size:.83rem;text-align:left}.derived{background:#0F172A;border:1px solid #243244;border-radius:14px;padding:12px}.derived-label{color:#64748B;font-size:.66rem;text-transform:uppercase;letter-spacing:.08em;font-weight:800}.derived-value{color:#E2E8F0;font-size:1rem;font-weight:850;margin-top:4px}
.stTabs [data-baseweb="tab-list"]{gap:7px}.stTabs [data-baseweb="tab"]{background:#0F172A;border:1px solid #243244;border-radius:10px;padding:8px 12px;color:#94A3B8}.stTabs [aria-selected="true"]{background:#10283A;color:#7DD3FC!important;border-color:#1B5874}
</style>
""")

BASE = Path(__file__).resolve().parent
DATA = BASE / "MyFactorylens_dataset.csv"
MODEL = BASE / "factorylens_model.pkl"
FEATURES = BASE / "factorylens_features.pkl"


@st.cache_data

def load_data(path):
    return pd.read_csv(path)


@st.cache_resource

def load_model(path):
    return joblib.load(path)


@st.cache_data

def load_features(path):
    return list(joblib.load(path))


@st.cache_resource

def explainer_for(model):
    return shap.TreeExplainer(model)


def wear_state(x):
    if x <= 100:
        return 0, "New"
    if x <= 160:
        return 1, "Mid-life"
    if x <= 200:
        return 2, "Aging"
    return 3, "Critical"


def model_matrix(raw, features):
    out = pd.DataFrame(index=raw.index)
    out["air_temp_k"] = raw["air_temp_k"].astype(float)
    out["process_temp_k"] = raw["process_temp_k"].astype(float)
    out["rotational_speed_rpm"] = raw["rotational_speed_rpm"].astype(float)
    out["torque_nm"] = raw["torque_nm"].astype(float)
    out["tool_wear_min"] = raw["tool_wear_min"].astype(float)
    out["power_kw"] = raw["torque_nm"] * raw["rotational_speed_rpm"] / 9549
    out["temp_diff"] = raw["process_temp_k"] - raw["air_temp_k"]
    out["tool_wear_category_num"] = raw["tool_wear_min"].apply(lambda x: wear_state(x)[0]).astype(float)
    for c in ["H", "L", "M"]: out[f"product_type_{c}"] = (raw["product_type"].astype(str) == c).astype(int)
    for c in ["Evening", "Morning", "Night"]: out[f"shift_{c}"] = (raw["shift"].astype(str) == c).astype(int)
    for c in ["Expert", "Junior", "Senior"]: out[f"operator_experience_{c}"] = (raw["operator_experience"].astype(str) == c).astype(int)
    return out[features]


def shap_array(values):
    if isinstance(values, list): return np.asarray(values[1] if len(values) > 1 else values[0])
    arr = np.asarray(values)
    return arr[:, :, 1] if arr.ndim == 3 else arr


def style(ax, title=None, ylabel=None):
    if title: ax.set_title(title, color="#F8FAFC", fontsize=11, fontweight="bold", loc="left", pad=10)
    if ylabel: ax.set_ylabel(ylabel, color="#94A3B8", fontsize=8)
    ax.tick_params(colors="#94A3B8", labelsize=8); ax.grid(axis="y", alpha=.12, color="#64748B"); ax.set_axisbelow(True)
    ax.spines[["top","right","left"]].set_visible(False); ax.spines["bottom"].set_color("#334155")
    ax.set_facecolor("#111827")


missing = [str(p) for p in [DATA, MODEL, FEATURES] if not p.exists()]
if missing:
    st.error("Required FactoryLens files are missing.")
    st.code("\n".join(missing)); st.stop()

try:
    df = load_data(str(DATA)).copy(); model = load_model(str(MODEL)); features = load_features(str(FEATURES))
except Exception as e:
    st.error(f"Could not load FactoryLens files: {e}"); st.stop()

df["power_kw"] = df["torque_nm"] * df["rotational_speed_rpm"] / 9549
df["temp_diff"] = df["process_temp_k"] - df["air_temp_k"]
df["tool_wear_category_num"] = df["tool_wear_min"].apply(lambda x: wear_state(x)[0])
df["failure_label"] = df["machine_failure"].map({0:"No Failure",1:"Failure"})

with st.sidebar:
    st.markdown("### 🏭 FACTORYLENS"); st.caption("Manufacturing Intelligence Platform"); st.markdown("---")
    st.markdown("**Dashboard Modules**"); st.markdown("• Operations Overview"); st.markdown("• Risk Driver Analysis"); st.markdown("• Failure Risk Predictor"); st.markdown("• Explainable ML")
    st.markdown("---"); st.markdown("**Model**"); st.markdown("Random Forest Classifier"); st.markdown("17 predictive features")
    st.markdown("---"); st.markdown("**Data**"); st.markdown(f"{len(df):,} machine observations"); st.markdown("AI4I 2020 + operational enrichment")
    st.markdown("---"); st.caption("Portfolio prototype. Shift, operator experience and machine age are simulated operational enrichment.")

H("""
<div class="hero"><div class="hero-kicker">MANUFACTURING INTELLIGENCE</div><div class="hero-title">🏭 FactoryLens</div><div class="hero-subtitle">Predict · Diagnose · Prevent machine failures with data-driven operations analytics.</div><div class="status">● MODEL CONNECTED &nbsp; | &nbsp; RANDOM FOREST &nbsp; | &nbsp; 17 FEATURES</div></div>
<div class="section-kicker">01 · OPERATIONS OVERVIEW</div><h2>Plant Health Snapshot</h2><div class="muted">A compact view of failure exposure, shift risk, and machine condition.</div>
""")

shift_rates = df.groupby("shift")["machine_failure"].mean() * 100
worst = shift_rates.idxmax(); critical_pct = (df["machine_age_category"] == "Critical").mean() * 100
vals = [("Total Observations",f"{len(df):,}","Machine records analyzed"),("Overall Failure Rate",f"{df.machine_failure.mean()*100:.2f}%","Historical failure incidence"),("Highest Risk Shift",str(worst),f"{shift_rates[worst]:.2f}% failure rate"),("Critical Machines",f"{critical_pct:.1f}%","Highest tool-wear state")]
k = st.columns(4)
for col,(label,value,note) in zip(k,vals):
    with col: H(f"<div class='kpi'><div class='kpi-label'>{label}</div><div class='kpi-value'>{value}</div><div class='kpi-note'>{note}</div></div>")

st.markdown("<div style='height:18px'></div>",unsafe_allow_html=True)
c1,c2 = st.columns([1.65,1],gap="large")
with c1:
    H("<div class='panel'><div class='panel-title'>Failure Rate by Shift</div>")
    order=[x for x in ["Morning","Evening","Night"] if x in shift_rates.index]; s=shift_rates.reindex(order)
    fig,ax=plt.subplots(figsize=(6.2,3.1)); bars=ax.bar(s.index,s.values,color=["#334155","#38BDF8","#475569"],width=.58)
    for b,v in zip(bars,s.values): ax.text(b.get_x()+b.get_width()/2,v+.15,f"{v:.2f}%",ha="center",va="bottom",fontsize=9,color="#E2E8F0",fontweight="bold")
    style(ax,ylabel="Failure Rate (%)"); fig.patch.set_facecolor("#111827"); fig.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig); H("</div>")
with c2:
    H(f"<div class='panel'><div class='badge blue'>OPERATIONAL SIGNAL</div><div class='panel-title'>Shift concentration</div><div class='panel-copy'><strong style='color:#F8FAFC'>{worst}</strong> has the highest observed failure rate at <strong style='color:#7DD3FC'>{shift_rates[worst]:.2f}%</strong>.</div><div style='height:14px'></div><div class='derived'><div class='derived-label'>Lowest-risk shift</div><div class='derived-value'>{shift_rates.idxmin()} · {shift_rates.min():.2f}%</div></div></div>")

st.markdown("<div style='height:28px'></div>",unsafe_allow_html=True)
H("<div class='section-kicker'>02 · RISK DRIVER ANALYSIS</div><h2>What Drives Machine Failure?</h2><div class='muted'>Explore the operational dimensions from the FactoryLens analysis.</div>")
tabs=st.tabs(["Shift","Tool Wear","Product Type","Operator","Temperature","Machine Age"])


def bar_tab(tab,data,x,y,title,ylabel,colors):
    with tab:
        a,b=st.columns([1.45,1],gap="large")
        with a:
            fig,ax=plt.subplots(figsize=(5.8,3.15)); bars=ax.bar(data[x].astype(str),data[y],color=colors,width=.58)
            for z,v in zip(bars,data[y]): ax.text(z.get_x()+z.get_width()/2,v+max(data[y])*.03,f"{v:.2f}%",ha="center",va="bottom",fontsize=9,color="#E2E8F0",fontweight="bold")
            style(ax,title,ylabel); fig.patch.set_facecolor("#111827"); fig.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)
        return b

with tabs[0]:
    d=shift_rates.reset_index(name="Rate"); b=st.columns([1.45,1])[1]
    with st.columns([1.45,1])[0]:
        fig,ax=plt.subplots(figsize=(5.8,3.15)); ax.bar(d.shift,d.Rate,color=["#475569","#38BDF8","#64748B"],width=.58); style(ax,"Failure Rate by Shift","Failure Rate (%)"); fig.patch.set_facecolor("#111827"); fig.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)
    with b: H(f"<div class='panel'><div class='badge blue'>SHIFT ANALYSIS</div><div class='panel-title'>Highest observed exposure</div><div class='panel-copy'><strong style='color:#F8FAFC'>{worst}</strong> records the highest failure rate at <strong style='color:#7DD3FC'>{shift_rates[worst]:.2f}%</strong>.</div></div>")

with tabs[1]:
    g=df.groupby("failure_label")["tool_wear_min"].median().reindex(["No Failure","Failure"]); a,b=st.columns([1.45,1],gap="large")
    with a:
        fig,ax=plt.subplots(figsize=(5.8,3.15)); ax.bar(g.index,g.values,color=["#475569","#F87171"],width=.58); style(ax,"Median Tool Wear by Machine Status","Tool Wear (minutes)"); fig.patch.set_facecolor("#111827"); fig.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)
    with b: H(f"<div class='panel'><div class='badge red'>WEAR SIGNAL</div><div class='panel-title'>Tool wear separates failures</div><div class='panel-copy'>Failed machines show a median tool-wear level approximately <strong style='color:#FCA5A5'>{g['Failure']-g['No Failure']:.0f} minutes higher</strong> than non-failures.</div></div>")

for tab,col,title,badge,badgecls in [(tabs[2],"product_type","Failure Rate by Product Type","PRODUCT EXPOSURE","red"),(tabs[3],"operator_experience","Failure Rate by Operator Experience","WORKFORCE SIGNAL","amber")]:
    d=df.groupby(col)["machine_failure"].mean().mul(100).reset_index(name="Rate").sort_values("Rate",ascending=False); a,b=st.columns([1.45,1],gap="large")
    with a:
        fig,ax=plt.subplots(figsize=(5.8,3.15)); ax.bar(d[col].astype(str),d.Rate,color=["#F87171","#38BDF8","#64748B"],width=.58); style(ax,title,"Failure Rate (%)"); fig.patch.set_facecolor("#111827"); fig.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)
    with b: H(f"<div class='panel'><div class='badge {badgecls}'>{badge}</div><div class='panel-title'>Highest observed exposure</div><div class='panel-copy'><strong style='color:#F8FAFC'>{d.iloc[0][col]}</strong> has the highest observed failure rate at <strong style='color:#FCD34D'>{d.iloc[0].Rate:.2f}%</strong>.</div></div>")

with tabs[4]:
    t=df.groupby("failure_label")["temp_diff"].median().reindex(["No Failure","Failure"]); a,b=st.columns([1.45,1],gap="large")
    with a:
        fig,ax=plt.subplots(figsize=(5.8,3.15)); ax.bar(t.index,t.values,color=["#475569","#38BDF8"],width=.58); style(ax,"Median Process-Air Temperature Difference","Temperature Difference (K)"); fig.patch.set_facecolor("#111827"); fig.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)
    with b: H("<div class='panel'><div class='badge blue'>THERMAL SIGNAL</div><div class='panel-title'>Temperature separation matters</div><div class='panel-copy'>Failure observations show a lower median process-air temperature difference, consistent with the project's heat-dissipation interpretation.</div></div>")

with tabs[5]:
    order=["New","Mid-life","Aging","Critical"]; d=df.groupby("machine_age_category",observed=True).machine_failure.mean().mul(100).reindex(order).dropna().reset_index(name="Rate"); a,b=st.columns([1.45,1],gap="large")
    with a:
        fig,ax=plt.subplots(figsize=(6.3,3.15)); ax.bar(d.machine_age_category,d.Rate,color=["#475569","#FBBF24","#F59E0B","#F87171"],width=.58); style(ax,"Failure Rate by Machine Age Category","Failure Rate (%)"); fig.patch.set_facecolor("#111827"); fig.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)
    with b: H(f"<div class='panel'><div class='badge red'>AGE / WEAR RISK</div><div class='panel-title'>Critical state needs attention</div><div class='panel-copy'>Critical machines show a failure rate of <strong style='color:#FCA5A5'>{float(d.loc[d.machine_age_category=='Critical','Rate'].iloc[0]):.2f}%</strong>.</div></div>")

st.markdown("<div style='height:30px'></div>",unsafe_allow_html=True)
H("<div class='section-kicker'>03 · PREDICTIVE MAINTENANCE</div><h2>Failure Risk Predictor</h2><div class='muted'>Adjust machine conditions, estimate failure probability, and inspect the factors influencing the prediction.</div>")
a,b=st.columns([1.05,1],gap="large")
with a:
    H("<div class='panel'><div class='panel-title'>Machine Parameters</div>")
    air=st.slider("Air Temperature (K)",295.0,305.0,300.0,.1); proc=st.slider("Process Temperature (K)",305.0,315.0,310.0,.1)
    rpm=st.slider("Rotational Speed (RPM)",1168,2886,1500,10); tq=st.slider("Torque (Nm)",3.8,76.6,40.0,.1); wear=st.slider("Tool Wear (minutes)",0,253,100,1)
    product=st.selectbox("Product Type",["L","M","H"]); shift=st.selectbox("Shift",["Morning","Evening","Night"]); op=st.selectbox("Operator Experience",["Junior","Senior","Expert"])
    H("</div>")

raw=pd.DataFrame([{"air_temp_k":air,"process_temp_k":proc,"rotational_speed_rpm":rpm,"torque_nm":tq,"tool_wear_min":wear,"product_type":product,"shift":shift,"operator_experience":op}]); x=model_matrix(raw,features); prob=float(model.predict_proba(x)[0,1]); pct=prob*100; wear_num,wear_label=wear_state(wear); power=float(x.power_kw.iloc[0]); gap=float(x.temp_diff.iloc[0])
if prob>.35: level,cls,action,fill="HIGH RISK","high","Schedule immediate inspection within 24 hours.","#EF4444"
elif prob>.15: level,cls,action,fill="MEDIUM RISK","medium","Monitor closely and inspect within 72 hours.","#F59E0B"
else: level,cls,action,fill="LOW RISK","low","Continue production with the next scheduled maintenance.","#22C55E"
with b:
    H(f"<div class='risk'><div class='muted'>CURRENT FAILURE PROBABILITY</div><div class='prob'>{pct:.1f}%</div><div style='height:9px;background:#1F2937;border-radius:99px;overflow:hidden'><div style='height:100%;width:{max(2,min(100,pct))}%;background:{fill}'></div></div><div class='{cls}' style='font-weight:900;letter-spacing:.1em;margin:14px 0'>{level}</div><div class='recommend'><strong style='color:#F8FAFC'>Recommended action</strong><br>{action}</div></div>")
    q=st.columns(3)
    with q[0]: H(f"<div class='derived'><div class='derived-label'>Mechanical Power</div><div class='derived-value'>{power:.2f} kW</div></div>")
    with q[1]: H(f"<div class='derived'><div class='derived-label'>Thermal Gap</div><div class='derived-value'>{gap:.2f} K</div></div>")
    with q[2]: H(f"<div class='derived'><div class='derived-label'>Machine State</div><div class='derived-value'>{wear_label}</div></div>")

st.markdown("<div style='height:22px'></div>",unsafe_allow_html=True); s1,s2=st.columns([1.45,1],gap="large")
with s1:
    H("<div class='panel'><div class='panel-title'>Why did FactoryLens make this prediction?</div><div class='muted'>Positive SHAP values push toward failure. Negative values push away.</div>")
    try:
        sv=shap_array(explainer_for(model).shap_values(x))[0]; e=pd.DataFrame({"Feature":features,"Contribution":sv}).assign(A=lambda z:z.Contribution.abs()).nlargest(7,"A").sort_values("Contribution")
        fig,ax=plt.subplots(figsize=(6.8,3.9)); ax.barh(e.Feature,e.Contribution,color=["#F87171" if v>0 else "#38BDF8" for v in e.Contribution]); lim=max(float(e.Contribution.abs().max())*1.25,.01); ax.set_xlim(-lim,lim); ax.axvline(0,color="#64748B",lw=1); ax.tick_params(axis="y",colors="#CBD5E1",labelsize=8); ax.tick_params(axis="x",colors="#64748B",labelsize=8); ax.grid(axis="x",alpha=.12,color="#64748B"); ax.spines[["top","right","left"]].set_visible(False); ax.spines["bottom"].set_color("#334155"); ax.set_facecolor("#111827"); fig.patch.set_facecolor("#111827"); fig.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)
        st.caption("Red = increases modelled failure risk. Blue = decreases it for this machine profile.")
    except Exception as e: st.warning(f"SHAP explanation could not be rendered: {e}")
    H("</div>")
with s2:
    H("<div class='panel'><div class='panel-title'>Global Failure Drivers</div><div class='muted'>Mean absolute SHAP value across up to 1,000 sampled machines.</div>")
    try:
        sample=model_matrix(df,features).sample(min(1000,len(df)),random_state=42); sv=np.abs(shap_array(explainer_for(model).shap_values(sample))).mean(axis=0); imp=pd.DataFrame({"Feature":features,"SHAP":sv}).sort_values("SHAP").tail(7)
        fig,ax=plt.subplots(figsize=(5.5,3.9)); ax.barh(imp.Feature,imp.SHAP,color="#38BDF8"); ax.tick_params(axis="y",colors="#CBD5E1",labelsize=8); ax.tick_params(axis="x",colors="#64748B",labelsize=8); ax.grid(axis="x",alpha=.12,color="#64748B"); ax.spines[["top","right","left"]].set_visible(False); ax.spines["bottom"].set_color("#334155"); ax.set_facecolor("#111827"); fig.patch.set_facecolor("#111827"); fig.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)
        st.caption(f"Top global driver: {imp.iloc[-1].Feature}.")
    except Exception as e: st.warning(f"Global SHAP profile could not be rendered: {e}")
    H("</div>")

with st.expander("Model and project notes"):
    st.markdown("""**FactoryLens uses a saved Random Forest classifier trained on 17 engineered / encoded features.**\n\nThe predictive layer uses `power_kw = torque × RPM / 9549`, `temp_diff = process temperature − air temperature`, and ordered tool-wear state encoding.\n\nThe dashboard combines historical failure patterns, a live probability estimate, and local/global SHAP explanations.\n\nThe project uses the AI4I 2020 predictive maintenance dataset with simulated operational enrichment for shift, operator experience, machine age category, and maintenance status.""")

st.caption("FactoryLens · Manufacturing Operations Analytics · Portfolio decision-support prototype")