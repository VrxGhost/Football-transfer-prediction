import streamlit as st  
import pickle
import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go 

#its for to page configuration like page icon and page title layout etc
st.set_page_config(
    page_title = "Football prediction🔮",
    page_icon = "🔮",
    layout = "wide",
    initial_sidebar_state = "collapsed",
)

#its for the main title or the 1st heading of the website
st.title("_Football Prediction :green[system]_ 🔮⚽")

#loading the model

@st.cache_resource
def load_model():
    model_path = "model.pkl"
    if not os.path.exists(model_path):
        return None, "model missing"
    with open(model_path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict):
        return obj.get("xgboost", list(obj.values())[0]), "ok"
    return obj,"ok"

model, model_status = load_model()

#Exacting encoded columns 

POSITION_COLS = [
    "position_Defender",
    "position_Goalkeeper",
    "position_Midfield",
    "position_Missing",
]
FOOT_COLS = [
    "foot_left",
    "foot_right"
]
NUMERIC_FEATURES = [
    "age",
    "height_in_cm",
    "contract_years_left",
    "goals_per_match",
    "assists_per_match",
    "matches_played",
]

ALL_FEATURES = NUMERIC_FEATURES + POSITION_COLS + FOOT_COLS

def build_input(age,height,contract_years,goals_pm,assists_pm,matches,position,foot):
    row = {f:0.0 for f in ALL_FEATURES}
    row["age"] = age
    row["height_in_cm"] = height
    row["contract_years_left"] = contract_years
    row["goals_per_match"] = goals_pm
    row["assists_per_match"] = assists_pm
    row["matches_played"] = matches

    pos_map = {
        "Attacker": None,
        "Defender": "position_Defender",
        "Goalkeeper": "position_Goalkeeper",
        "Midfielder": "position_Midfield",
    }
    if pos_map.get(position):
        row[pos_map[position]] = 1.0

    foot_map = {
        "Right": None,
        "Left": "foot_left",
        "Both": "foot_both",
    }
    if foot_map.get(foot):
        row[foot_map[foot]] = 1.0
    return pd.DataFrame([row])[ALL_FEATURES]

def tier_label(val_eur):
    if val_eur >= 80_000_000:
        return "World class","#ffd700", "#2a2000"
    elif val_eur >= 40_000_000:
        return "Elite", "#BCC6CC", "#0d2a1a"
    elif val_eur >= 15_000_000:
        return "High Value", "#E4953C", "#0a1f2a"
    elif val_eur >= 5_000_000:
        return "Mid-Market", "#00e676", "#1a0f2a"
    else:
        return "Emerging", "#35D4E9", "#2a1000"

def fmt_eur(val):
    if val >= 1_000_000:
        return f"€ {val/1_000_000:.1f}M"
    return f"€{val/1_000_000:.0f}K"

def make_gauge(predicted, low, high):
    max_val = 120_000_000
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = predicted,
        number = {"prefix": "€", "suffix": "", "valueformat":",0f", "font":{"size":20,"color":"#e8eaf0","family": "DM Sans"}},
        gauge={
            "axis": {"range": [0, max_val], "tickwidth": 0,
                     "tickcolor": "#1c2133",
                     "tickvals": [0, 20e6, 40e6, 80e6, 120e6],
                     "ticktext": ["€0", "€20M", "€40M", "€80M", "€120M+"],
                     "tickfont": {"color": "#7c8296", "size": 10}},
            "bar":  {"color": "#00e676", "thickness": 0.25},
            "bgcolor": "#111827",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 5e6],   "color": "#0f172a"},
                {"range": [5e6, 15e6], "color": "#0f1f2a"},
                {"range": [15e6, 40e6], "color": "#0f2218"},
                {"range": [40e6, 80e6], "color": "#0a2a14"},
                {"range": [80e6, max_val], "color": "#0d3318"},
            ],
            "threshold": {
                "line": {"color": "#4ade80", "width": 2},
                "thickness": 0.75,
                "value": predicted,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0a0e1a",
        margin=dict(l=20, r=20, t=20, b=10),
        height=220,
        font={"family": "DM Sans", "color": "#e8eaf0"},
    )
    return fig
 
 
def make_feature_bar(age, height, contract_years, goals_pm, assists_pm, matches):
    labels = ["Age", "Height", "Contract\nyears left",
              "Goals/match", "Assists/match", "Matches\nplayed"]
    raw    = [age, height, contract_years, goals_pm, assists_pm, matches]
    maxes  = [40,  210,    10,             1.2,       0.8,        600]
    pcts   = [min(v / m, 1.0) * 100 for v, m in zip(raw, maxes)]
 
    fig = go.Figure(go.Bar(
        x=labels, y=pcts,
        marker_color=["#00e676" if p >= 60 else "#1c2133" if p <= 20 else "#1a3a28"
                      for p in pcts],
        marker_line_width=0,
        text=[f"{v:.2f}" if isinstance(v, float) else str(v) for v in raw],
        textposition="outside",
        textfont={"color": "#7c8296", "size": 11},
    ))
    fig.update_layout(
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#111827",
        margin=dict(l=10, r=10, t=10, b=10),
        height=220,
        xaxis=dict(showgrid=False, tickfont=dict(color="#7c8296", size=10)),
        yaxis=dict(showgrid=False, visible=False, range=[0, 130]),
        bargap=0.35,
        showlegend=False,
    )
    return fig

col_left, col_right = st.columns([1.1, 1], gap="large")
 
with col_left:
    age = st.slider("age_sl", 15, 40, 24, label_visibility="collapsed")
    height = st.slider("height_sl", 155, 205, 181, label_visibility="collapsed")
    contract_years = st.slider("contract_sl", 0.0, 8.0, 2.0, step=0.5, label_visibility="collapsed")
    position = st.selectbox("pos_sel", ["Attacker", "Midfielder", "Defender", "Goalkeeper"], label_visibility="collapsed")
    foot = st.selectbox("foot_sel", ["Right", "Left", "Both"], label_visibility="collapsed")
with col_right:
    goals_pm = st.slider("goals_sl", 0.0, 1.2, 0.25, step=0.01, label_visibility="collapsed")
    assists_pm = st.slider("assists_sl", 0.0, 0.8, 0.12, step=0.01, label_visibility="collapsed")
    matches = st.slider("matches_sl", 0, 600, 120, step=5, label_visibility="collapsed")
    predict_btn = st.button("PREDICT VALUE", use_container_width=True)

if predict_btn or True:   # show live preview always
    X_input = build_input(
        age, height, contract_years,
        goals_pm, assists_pm, matches,
        position, foot
    )
 
    if model is not None:
        raw_pred   = float(model.predict(X_input)[0])
        predicted  = max(raw_pred, 0)
        low        = predicted * 0.78
        high       = predicted * 1.22
        tier, tier_color, tier_bg = tier_label(predicted)
 
        res_col, chart_col = st.columns([1, 1.2], gap="large")
 
        with res_col:
            st.markdown('<p class="section-label">Estimated market value</p>', unsafe_allow_html=True)
with res_col:
            st.markdown("### Estimated Market Value")
            st.markdown(f"# {fmt_eur(predicted)}")
            st.markdown(f"Range: {fmt_eur(low)} — {fmt_eur(high)}")
            tier, tier_color, tier_bg = tier_label(predicted)
            st.markdown(f"**{tier}**")

with chart_col:
            st.plotly_chart(make_gauge(predicted, low, high), use_container_width=True)
            st.plotly_chart(make_feature_bar(age, height, contract_years, goals_pm, assists_pm, matches), use_container_width=True)