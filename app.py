import datetime as dt

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================================
# Page config
# ============================================================================
st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# Palette (professional light theme: neutral greys, navy + blue accent)
# ============================================================================
BG, CARD, CARD2, LINE = "#F4F6F9", "#FFFFFF", "#F8FAFC", "#E2E8F0"
TEXT, MUTED = "#1E293B", "#64748B"
NAVY, BLUE, SLATE = "#0F2A4A", "#2563EB", "#94A3B8"
GREEN, AMBER, RED = "#16A34A", "#D97706", "#DC2626"

# ============================================================================
# Styling (explicit colors, so it renders correctly regardless of browser theme)
# ============================================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, .stApp, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #F4F6F9; color: #1E293B; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 3rem; padding-bottom: 2rem; max-width: 1220px; }

.stApp h1, .stApp h2, .stApp h3, .stApp h4 { font-family: 'Inter', sans-serif; color: #0F2A4A; letter-spacing: -.01em; }
.stApp p, .stApp label, .stApp li { color: #1E293B; }
[data-testid="stWidgetLabel"] p { color: #475569 !important; font-weight: 600; font-size: .86rem; }
[data-testid="stCaptionContainer"] p { color: #64748B !important; }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #E2E8F0; }
section[data-testid="stSidebar"] h3 { color: #0F2A4A !important; }

/* ---------- bordered containers become cards ---------- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF; border: 1px solid #E2E8F0 !important; border-radius: 12px; padding: .4rem .6rem;
    box-shadow: 0 1px 2px rgba(15,42,74,.05);
}

/* ---------- inputs ---------- */
div[data-baseweb="input"], div[data-baseweb="base-input"], div[data-baseweb="select"] > div {
    border: 1px solid #CBD5E1 !important; border-radius: 8px !important;
}
/* every layer inside an input/select: white background, dark text */
div[data-baseweb="input"], div[data-baseweb="input"] *,
div[data-baseweb="base-input"], div[data-baseweb="base-input"] *,
div[data-baseweb="select"], div[data-baseweb="select"] *,
[data-testid="stNumberInput"] input, [data-testid="stSelectbox"] input {
    background-color: #FFFFFF !important; color: #1E293B !important; -webkit-text-fill-color: #1E293B !important;
}
div[data-baseweb="input"]:focus-within, div[data-baseweb="select"] > div:focus-within {
    border-color: #2563EB !important; box-shadow: 0 0 0 3px rgba(37,99,235,.15) !important;
}
div[data-baseweb="select"] svg { fill: #64748B !important; }
[data-testid="stNumberInput"] button { background-color: #F1F5F9 !important; border: none !important; color: #1E293B !important; }
[data-testid="stNumberInput"] button svg { fill: #1E293B !important; }
input::placeholder { color: #94A3B8 !important; -webkit-text-fill-color: #94A3B8 !important; }
div[data-baseweb="popover"] ul, div[data-baseweb="menu"] { background: #FFFFFF !important; }
div[data-baseweb="popover"] li { color: #1E293B !important; }
div[data-baseweb="popover"] li:hover { background: #EFF4FB !important; }

/* slider */
div[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] { background: #2563EB !important; }
div[data-testid="stSlider"] [data-testid="stThumbValue"] { color: #2563EB !important; }
div[data-testid="stSlider"] [data-testid="stTickBarMin"],
div[data-testid="stSlider"] [data-testid="stTickBarMax"] { color: #64748B !important; }

/* ---------- buttons ---------- */
.stButton > button, .stDownloadButton > button {
    border-radius: 8px; font-weight: 600; border: 1px solid #CBD5E1; background: #FFFFFF; width: 100%;
}
.stButton > button p, .stDownloadButton > button p { color: #1E293B !important; font-weight: 600; }
.stButton > button:hover, .stDownloadButton > button:hover { border-color: #2563EB; background: #F1F6FE; }
button[kind="primary"], [data-testid="stBaseButton-primary"] { background: #0F2A4A !important; border: none !important; }
button[kind="primary"] p, [data-testid="stBaseButton-primary"] p { color: #FFFFFF !important; }
button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover { background: #1E40AF !important; }

/* ---------- hero ---------- */
.hero { padding: .2rem 0 1.2rem 0; }
.hero .tag {
    display: inline-block; font-size: .78rem; font-weight: 600; color: #1D4ED8;
    background: #EAF1FE; border: 1px solid #C7D8F8;
    padding: .25rem .8rem; border-radius: 6px; margin-bottom: .8rem; letter-spacing: .02em;
}
.hero h1 { font-size: 2.4rem; font-weight: 800; line-height: 1.15; margin: 0 0 .6rem 0; color: #0F2A4A !important; }
.hero h1 span { color: #2563EB !important; }
.hero p { color: #64748B !important; font-size: 1.02rem; max-width: 680px; margin: 0; }

/* ---------- section headers ---------- */
.sec-title { font-size: 1.02rem; font-weight: 700; color: #0F2A4A !important; margin: .3rem 0 .1rem 0; }
.sec-sub { color: #64748B !important; font-size: .85rem; margin-bottom: .6rem; }

/* ---------- result card ---------- */
.result {
    background: #FFFFFF;
    border: 1px solid #E2E8F0; border-top: 4px solid #0F2A4A; border-radius: 12px;
    padding: 1.4rem 1.5rem 1.3rem 1.5rem; text-align: center;
    box-shadow: 0 1px 3px rgba(15,42,74,.08);
}
.result .r-label { color: #64748B !important; font-size: .88rem; font-weight: 600; }
.result .r-big { font-size: 2.9rem; font-weight: 800; color: #0F2A4A !important; line-height: 1.05; margin-top: -.3rem; }
.result .r-sub { color: #2563EB !important; font-weight: 600; margin-top: .35rem; }
.result .scale { display: flex; justify-content: space-between; color: #94A3B8 !important; font-size: .74rem; margin: -.1rem 1.6rem 0 1.6rem; }

.kpis { display: grid; grid-template-columns: repeat(3, 1fr); gap: .7rem; margin-top: .9rem; }
.kpi { background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: .75rem .9rem; text-align: left; }
.kpi .k-l { color: #64748B !important; font-size: .75rem; font-weight: 600; }
.kpi .k-v { color: #0F2A4A !important; font-size: 1.12rem; font-weight: 700; margin-top: .1rem; }

.chip { display: inline-block; padding: .22rem .75rem; border-radius: 6px; font-size: .78rem; font-weight: 600; margin: .15rem .25rem; }
.chip-low  { background: #E8F6EE; color: #15803D !important; border: 1px solid #BBE5CB; }
.chip-mid  { background: #FDF3E3; color: #B45309 !important; border: 1px solid #F3D9A8; }
.chip-high { background: #FCEAEA; color: #B91C1C !important; border: 1px solid #F3C2C2; }

/* ---------- drivers / tips ---------- */
.factor-row { margin-bottom: 1rem; }
.f-name { display: flex; justify-content: space-between; font-size: .9rem; margin-bottom: .3rem; color: #1E293B !important; }
.f-sub { color: #64748B !important; }
.bar-bg { background: #E9EEF5; border-radius: 4px; height: 8px; overflow: hidden; }
.bar-fill { height: 8px; border-radius: 4px; }
.tip { display: flex; gap: .8rem; align-items: flex-start; padding: .75rem 0; border-bottom: 1px solid #EDF1F6; }
.tip:last-child { border-bottom: none; }
.t-icon { font-size: 1.2rem; line-height: 1.3; }
.t-text { color: #334155 !important; font-size: .92rem; line-height: 1.5; }
.t-text b { color: #0F2A4A; }

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"] { gap: .4rem; border-bottom: 1px solid #E2E8F0; }
.stTabs [data-baseweb="tab"] p { font-weight: 600; color: #64748B !important; }
.stTabs [aria-selected="true"] p { color: #0F2A4A !important; }
.stTabs [data-baseweb="tab-highlight"] { background-color: #2563EB !important; }

div[data-testid="stMetric"] { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: .9rem 1.1rem; box-shadow: 0 1px 2px rgba(15,42,74,.05); }
div[data-testid="stMetricLabel"] p { color: #64748B !important; }
div[data-testid="stMetricValue"] { color: #0F2A4A !important; }
div[data-testid="stAlert"] { border-radius: 10px; }
div[data-testid="stAlert"] p { color: #1E293B !important; }
hr { border-color: #E2E8F0 !important; }
.foot { text-align: center; color: #94A3B8 !important; font-size: .84rem; margin-top: 2.4rem; line-height: 1.7; }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================================
# Model (unchanged logic)
# ============================================================================
@st.cache_resource
def load_model():
    return joblib.load("insurance_model.pkl")


model = load_model()

REGION_MAPPING = {"Southeast": 0, "Southwest": 1, "Northeast": 2, "Northwest": 3}
DATASET_AVG = 13270.0  # approximate average charge in the public insurance dataset
GAUGE_MAX = 50000.0


def predict(age, sex, bmi, children, smoker, region) -> float:
    """Same encoding and feature order as the original app."""
    sex_encoded = 0 if sex == "Male" else 1
    smoker_encoded = 0 if smoker == "Yes" else 1
    x = np.array([[age, sex_encoded, bmi, children, smoker_encoded, REGION_MAPPING[region]]])
    return float(model.predict(x)[0])


def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight", "chip-mid"
    if bmi < 25:
        return "Healthy weight", "chip-low"
    if bmi < 30:
        return "Overweight", "chip-mid"
    return "Obese", "chip-high"


def cost_band(cost):
    if cost < 8000:
        return "Low cost band", "chip-low", GREEN
    if cost < 20000:
        return "Moderate cost band", "chip-mid", AMBER
    return "High cost band", "chip-high", RED


def gauge_svg(cost, color):
    frac = max(0.0, min(cost / GAUGE_MAX, 1.0))
    length = 282.74
    return (
        '<svg viewBox="0 0 220 125" width="100%" style="max-width:360px;">'
        '<path d="M 20 110 A 90 90 0 0 1 200 110" fill="none" stroke="#E2E8F0" stroke-width="16" stroke-linecap="round"/>'
        f'<path d="M 20 110 A 90 90 0 0 1 200 110" fill="none" stroke="{color}" stroke-width="16" '
        f'stroke-linecap="round" stroke-dasharray="{length * frac:.1f} {length}"/></svg>'
    )


def style_chart(chart):
    return (
        chart.configure(background=CARD)
        .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor="#EDF1F6", domainColor=LINE, tickColor=LINE)
        .configure_legend(labelColor=TEXT)
        .configure_view(stroke=None)
    )


# ============================================================================
# Session state + callbacks
# ============================================================================
DEFAULTS = dict(age=30, sex="Male", bmi=25.0, children=0, smoker="No", region="Southeast",
                h_cm=170.0, w_kg=70.0)
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("saved", [])

PRESETS = {
    "Young & healthy": dict(age=24, sex="Female", bmi=22.0, children=0, smoker="No", region="Northeast"),
    "Family, 40s": dict(age=42, sex="Male", bmi=27.5, children=3, smoker="No", region="Southwest"),
    "Smoker, 50s": dict(age=54, sex="Male", bmi=31.0, children=1, smoker="Yes", region="Southeast"),
}


def apply_preset(name):
    for k, v in PRESETS[name].items():
        st.session_state[k] = v


def use_helper_bmi():
    h, w = st.session_state.h_cm, st.session_state.w_kg
    st.session_state.bmi = float(min(60.0, max(10.0, round(w / ((h / 100) ** 2), 1))))


def save_scenario(inp, cost):
    n = len(st.session_state.saved) + 1
    st.session_state.saved.append(
        {
            "Scenario": f"#{n}",
            "Time": dt.datetime.now().strftime("%H:%M:%S"),
            "Age": inp["age"], "Sex": inp["sex"], "BMI": round(inp["bmi"], 1),
            "Children": inp["children"], "Smoker": inp["smoker"], "Region": inp["region"],
            "Estimated cost ($)": round(cost, 2),
        }
    )


# ============================================================================
# Sidebar
# ============================================================================
with st.sidebar:
    st.markdown("### 🧮 BMI helper")
    st.caption("Enter height and weight, then send the result to the form.")
    st.number_input("Height (cm)", 100.0, 250.0, step=1.0, key="h_cm")
    st.number_input("Weight (kg)", 30.0, 250.0, step=0.5, key="w_kg")
    helper_bmi = st.session_state.w_kg / ((st.session_state.h_cm / 100) ** 2)
    cat, chip = bmi_category(helper_bmi)
    st.markdown(
        f'<div class="kpi"><div class="k-l">Calculated BMI</div><div class="k-v">{helper_bmi:.1f}</div>'
        f'<span class="chip {chip}" style="margin-left:0">{cat}</span></div>',
        unsafe_allow_html=True,
    )
    st.write("")
    st.button("Use this BMI in the form", on_click=use_helper_bmi, type="primary")

    st.divider()
    st.markdown("### ℹ️ About")
    st.caption(
        "A machine learning model trained on historical insurance records predicts yearly "
        "medical charges from six inputs. Results update live as you change the inputs."
    )

# ============================================================================
# Hero
# ============================================================================
st.markdown(
    '<div class="hero"><span class="tag">ML-POWERED ESTIMATE</span>'
    "<h1>Know your <span>medical insurance cost</span> before you buy.</h1>"
    "<p>Adjust the details and watch the estimate update instantly. Then see what drives the "
    "price and test what-if scenarios.</p></div>",
    unsafe_allow_html=True,
)

# ============================================================================
# Main: inputs (left) + live result (right)
# ============================================================================
left, right = st.columns([1, 1.1], gap="large")

with left:
    with st.container(border=True):
        st.markdown('<div class="sec-title">Your profile</div><div class="sec-sub">Change any value and the estimate updates.</div>', unsafe_allow_html=True)

        st.caption("Quick start")
        pc = st.columns(3)
        for col, name in zip(pc, PRESETS):
            col.button(name, key=f"preset_{name}", on_click=apply_preset, args=(name,))

        st.slider("Age", 18, 100, key="age")
        a, b = st.columns(2)
        a.selectbox("Sex", ["Male", "Female"], key="sex")
        b.selectbox("Smoking status", ["Yes", "No"], key="smoker")
        a.number_input("BMI", min_value=10.0, max_value=60.0, step=0.1, key="bmi")
        b.number_input("Number of children", min_value=0, max_value=10, key="children")
        st.selectbox("Region", ["Southeast", "Southwest", "Northeast", "Northwest"], key="region")

inp = dict(
    age=st.session_state.age, sex=st.session_state.sex, bmi=st.session_state.bmi,
    children=st.session_state.children, smoker=st.session_state.smoker, region=st.session_state.region,
)
cost = predict(**inp)
band_txt, band_chip, band_color = cost_band(cost)
bmi_txt, bmi_chip = bmi_category(inp["bmi"])
diff = (cost - DATASET_AVG) / DATASET_AVG * 100
direction = "above" if diff >= 0 else "below"

with right:
    st.markdown(
        '<div class="result"><div class="r-label">Estimated yearly insurance cost</div>'
        + gauge_svg(cost, band_color)
        + f'<div class="scale"><span>$0</span><span>${GAUGE_MAX/1000:.0f}k+</span></div>'
        + f'<div class="r-big">${cost:,.2f}</div>'
        + f'<div class="r-sub">≈ ${cost/12:,.2f} per month</div>'
        + f'<div style="margin-top:.7rem;"><span class="chip {band_chip}">{band_txt}</span>'
        + f'<span class="chip {bmi_chip}">BMI: {bmi_txt}</span></div>'
        + '<div class="kpis">'
        + f'<div class="kpi"><div class="k-l">Vs average</div><div class="k-v">{abs(diff):.0f}% {direction}</div></div>'
        + f'<div class="kpi"><div class="k-l">Daily cost</div><div class="k-v">${cost/365:,.2f}</div></div>'
        + f'<div class="kpi"><div class="k-l">Typical average</div><div class="k-v">${DATASET_AVG:,.0f}</div></div>'
        + "</div></div>",
        unsafe_allow_html=True,
    )
    st.write("")
    r1, r2 = st.columns(2)
    r1.button("Save to compare", on_click=save_scenario, args=(inp, cost), type="primary")
    report = (
        "Medical Insurance Cost Estimate\n"
        f"Generated: {dt.datetime.now():%Y-%m-%d %H:%M}\n\n"
        f"Age: {inp['age']}\nSex: {inp['sex']}\nBMI: {inp['bmi']:.1f} ({bmi_txt})\n"
        f"Children: {inp['children']}\nSmoker: {inp['smoker']}\nRegion: {inp['region']}\n\n"
        f"Estimated yearly cost: ${cost:,.2f}\nEstimated monthly cost: ${cost/12:,.2f}\n\n"
        "This is a model estimate, not an insurance quotation.\n"
    )
    r2.download_button("Download estimate", report, file_name="insurance_estimate.txt", mime="text/plain")

# ============================================================================
# Tabs
# ============================================================================
st.write("")
tab_ins, tab_what, tab_cmp = st.tabs(["📊 Insights", "🔀 What-if analysis", "⚖️ Compare scenarios"])

# ---------------- Insights ----------------
with tab_ins:
    baselines = {
        "Smoking": (dict(smoker="No"), "non-smoker"),
        "BMI": (dict(bmi=24.0), "BMI of 24"),
        "Age": (dict(age=max(18, inp["age"] - 10)), "10 years younger"),
        "Children": (dict(children=0), "no children"),
    }
    impacts = {k: cost - predict(**{**inp, **chg}) for k, (chg, _) in baselines.items()}

    c1, c2 = st.columns(2, gap="large")
    with c1:
        max_abs = max(max(abs(v) for v in impacts.values()), 1)
        rows = ""
        for k, v in sorted(impacts.items(), key=lambda kv: -abs(kv[1])):
            pct = max(abs(v) / max_abs * 100, 2)
            color = RED if v > 0.5 else BLUE
            sign = "+" if v > 0.5 else ("−" if v < -0.5 else "")
            rows += (
                '<div class="factor-row">'
                f'<div class="f-name"><span>{k} <span class="f-sub">vs {baselines[k][1]}</span></span>'
                f"<b>{sign}${abs(v):,.0f}</b></div>"
                f'<div class="bar-bg"><div class="bar-fill" style="width:{pct:.0f}%; background:{color};"></div></div></div>'
            )
        with st.container(border=True):
            st.markdown(
                '<div class="sec-title">What drives this number</div>'
                '<div class="sec-sub">Change in cost if each factor moved to a healthier or baseline value.</div>' + rows,
                unsafe_allow_html=True,
            )

    with c2:
        tips = []
        if inp["smoker"] == "Yes" and impacts["Smoking"] > 0:
            tips.append(("🚭", f"Smoking is the biggest lever. As a non-smoker, the estimate would drop by about <b>${impacts['Smoking']:,.0f}</b> a year."))
        if inp["bmi"] >= 25:
            tips.append(("🥗", f"Your BMI is {inp['bmi']:.1f} ({bmi_txt.lower()}). Moving toward a healthy range could change the estimate by about <b>${abs(impacts['BMI']):,.0f}</b>."))
        if inp["bmi"] < 18.5:
            tips.append(("⚖️", "Your BMI is below the healthy range. Talk to a doctor about a healthy target weight."))
        if inp["age"] >= 50:
            tips.append(("🩺", "Costs rise with age in this model, so regular check-ups and preventive care matter more from here."))
        if inp["children"] >= 3:
            tips.append(("👨‍👩‍👧‍👦", "Larger families can change the estimate. Compare scenarios to see the effect of dependants."))
        if not tips:
            tips.append(("✅", "Your inputs sit in a lower-cost range. Keep up healthy habits to stay there."))
        tips_html = "".join(f'<div class="tip"><div class="t-icon">{i}</div><div class="t-text">{t}</div></div>' for i, t in tips)
        with st.container(border=True):
            st.markdown(
                '<div class="sec-title">Personalised insights</div><div class="sec-sub">Based on your inputs and the model.</div>' + tips_html,
                unsafe_allow_html=True,
            )

# ---------------- What-if ----------------
with tab_what:
    yes_cost = predict(**{**inp, "smoker": "Yes"})
    no_cost = predict(**{**inp, "smoker": "No"})
    m1, m2, m3 = st.columns(3)
    m1.metric("If smoker", f"${yes_cost:,.0f}")
    m2.metric("If non-smoker", f"${no_cost:,.0f}", delta=f"${no_cost - yes_cost:,.0f}", delta_color="inverse")
    m3.metric("Smoking premium", f"${yes_cost - no_cost:,.0f}")

    st.markdown('<div class="sec-title" style="margin-top:1rem">Cost by age</div>', unsafe_allow_html=True)
    ages = list(range(18, 101, 2))
    age_df = pd.DataFrame(
        {
            "Age": ages * 2,
            "Group": ["Smoker"] * len(ages) + ["Non-smoker"] * len(ages),
            "Cost": [predict(**{**inp, "age": a, "smoker": "Yes"}) for a in ages]
            + [predict(**{**inp, "age": a, "smoker": "No"}) for a in ages],
        }
    )
    st.altair_chart(
        style_chart(
            alt.Chart(age_df).mark_line(strokeWidth=2.5).encode(
                x=alt.X("Age:Q"), y=alt.Y("Cost:Q", title="Estimated cost ($)"),
                color=alt.Color("Group:N", scale=alt.Scale(domain=["Smoker", "Non-smoker"], range=[RED, BLUE]),
                                legend=alt.Legend(title=None, orient="top")),
                tooltip=["Age", "Group", alt.Tooltip("Cost:Q", format="$,.0f")],
            ).properties(height=300)
        ),
        use_container_width=True, theme=None,
    )

    g1, g2 = st.columns(2, gap="large")
    with g1:
        st.markdown('<div class="sec-title">Cost by BMI</div>', unsafe_allow_html=True)
        bmis = [round(b, 1) for b in np.arange(15, 50.5, 1.0)]
        bmi_df = pd.DataFrame({"BMI": bmis, "Cost": [predict(**{**inp, "bmi": b}) for b in bmis]})
        st.altair_chart(
            style_chart(
                alt.Chart(bmi_df).mark_area(line={"color": NAVY, "strokeWidth": 2.5}, color=BLUE, opacity=0.15).encode(
                    x="BMI:Q", y=alt.Y("Cost:Q", title="Estimated cost ($)"),
                    tooltip=["BMI", alt.Tooltip("Cost:Q", format="$,.0f")],
                ).properties(height=260)
            ),
            use_container_width=True, theme=None,
        )
    with g2:
        st.markdown('<div class="sec-title">Cost by region</div>', unsafe_allow_html=True)
        reg_df = pd.DataFrame({"Region": list(REGION_MAPPING), "Cost": [predict(**{**inp, "region": r}) for r in REGION_MAPPING]})
        st.altair_chart(
            style_chart(
                alt.Chart(reg_df).mark_bar(cornerRadiusEnd=4, color=NAVY).encode(
                    x=alt.X("Cost:Q", title="Estimated cost ($)"), y=alt.Y("Region:N", sort="-x", title=None),
                    tooltip=["Region", alt.Tooltip("Cost:Q", format="$,.0f")],
                ).properties(height=260)
            ),
            use_container_width=True, theme=None,
        )

# ---------------- Compare ----------------
with tab_cmp:
    if not st.session_state.saved:
        st.info("No saved scenarios yet. Set up a profile and select **Save to compare**, then change a value and save again.")
    else:
        df = pd.DataFrame(st.session_state.saved)
        st.markdown('<div class="sec-title">Saved scenarios</div>', unsafe_allow_html=True)
        st.altair_chart(
            style_chart(
                alt.Chart(df).mark_bar(cornerRadiusEnd=4, color=NAVY).encode(
                    x=alt.X("Estimated cost ($):Q", title="Estimated yearly cost ($)"),
                    y=alt.Y("Scenario:N", sort=None, title=None),
                    tooltip=list(df.columns),
                ).properties(height=max(120, 46 * len(df)))
            ),
            use_container_width=True, theme=None,
        )
        st.dataframe(df, use_container_width=True, hide_index=True)
        d1, d2 = st.columns(2)
        d1.download_button("Download comparison (CSV)", df.to_csv(index=False), file_name="scenarios.csv", mime="text/csv")
        if d2.button("Clear saved scenarios"):
            st.session_state.saved = []
            st.rerun()

st.markdown(
    '<div class="foot">This model provides an estimate and is not an actual insurance quotation.'
    "<br>Made by Priyanshu Singh (2427010573)</div>",
    unsafe_allow_html=True,
)