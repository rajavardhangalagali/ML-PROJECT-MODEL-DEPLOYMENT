import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
matplotlib.use('Agg')

st.set_page_config(
    page_title="MoodMeal",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
# LUXURY FINE-DINING EDITORIAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;0,700;1,300;1,400&family=Jost:wght@300;400;500;600&display=swap');

:root {
    --cream:    #f7f2e8;
    --parchment:#ede6d4;
    --forest:   #1a2e1a;
    --sage:     #3d5c3d;
    --moss:     #5a7a5a;
    --gold:     #c8a96e;
    --gold-lt:  #dfc08a;
    --rust:     #9e4c2c;
    --ink:      #1c1a16;
    --muted:    #7a6e5a;
}

* { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Jost', sans-serif;
    background: var(--cream) !important;
    color: var(--ink);
}

.stApp {
    background: var(--cream) !important;
    min-height: 100vh;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] { display: none; }

/* ── MASTHEAD ── */
.masthead {
    background: var(--forest);
    padding: 2.8rem 4rem 2.2rem;
    display: flex; align-items: flex-end; justify-content: space-between;
    border-bottom: 3px solid var(--gold);
    position: relative; overflow: hidden;
}
.masthead::before {
    content: '';
    position: absolute; inset: 0;
    background: repeating-linear-gradient(
        -45deg,
        transparent, transparent 40px,
        rgba(200,169,110,0.03) 40px, rgba(200,169,110,0.03) 41px
    );
}
.masthead-left { position: relative; z-index: 1; }
.masthead-eyebrow {
    font-family: 'Jost', sans-serif; font-weight: 500;
    font-size: 0.65rem; letter-spacing: 0.35em; text-transform: uppercase;
    color: var(--gold); margin-bottom: 0.5rem;
}
.masthead-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 4.5rem; font-weight: 300; line-height: 0.9;
    color: var(--cream); letter-spacing: -0.02em;
}
.masthead-title em {
    font-style: italic; color: var(--gold-lt);
}
.masthead-sub {
    font-family: 'Jost', sans-serif; font-weight: 300;
    font-size: 0.8rem; letter-spacing: 0.18em; color: var(--moss);
    margin-top: 0.7rem; text-transform: uppercase;
}
.masthead-right {
    position: relative; z-index: 1; text-align: right;
}
.masthead-badge {
    border: 1px solid var(--gold);
    padding: 0.6rem 1.2rem; display: inline-block;
    font-size: 0.65rem; letter-spacing: 0.2em; color: var(--gold);
    text-transform: uppercase; font-family: 'Jost', sans-serif;
}
.masthead-algo {
    font-family: 'Cormorant Garamond', serif; font-style: italic;
    font-size: 1.1rem; color: rgba(247,242,232,0.4); margin-top: 0.4rem;
}

/* ── TABS ── */
.tab-bar {
    background: var(--parchment);
    border-bottom: 1px solid rgba(200,169,110,0.4);
    display: flex; padding: 0 4rem;
}
.tab-item {
    padding: 1rem 0; margin-right: 3rem;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.2em;
    text-transform: uppercase; color: var(--muted); cursor: pointer;
    border-bottom: 2px solid transparent;
    transition: all 0.2s;
}
.tab-item.active {
    color: var(--forest); border-bottom-color: var(--gold);
}

/* Override Streamlit tabs */
div[data-testid="stTabs"] > div:first-child {
    background: var(--parchment) !important;
    border-bottom: 1px solid rgba(200,169,110,0.4) !important;
    padding: 0 3rem !important; gap: 0 !important;
}
button[data-baseweb="tab"] {
    font-family: 'Jost', sans-serif !important;
    font-size: 0.7rem !important; font-weight: 600 !important;
    letter-spacing: 0.22em !important; text-transform: uppercase !important;
    color: var(--muted) !important; padding: 1rem 0 !important;
    margin-right: 2.5rem !important; background: transparent !important;
    border: none !important; border-bottom: 2px solid transparent !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--forest) !important;
    border-bottom: 2px solid var(--gold) !important;
}
div[data-testid="stTabsContent"] {
    background: var(--cream) !important;
    padding: 3rem 4rem !important;
}

/* ── INPUT SECTION ── */
.section-rule {
    display: flex; align-items: center; gap: 1.2rem;
    margin: 0 0 2rem;
}
.section-rule-line {
    flex: 1; height: 1px; background: rgba(200,169,110,0.35);
}
.section-rule-text {
    font-family: 'Cormorant Garamond', serif; font-style: italic;
    font-size: 1.05rem; color: var(--muted); white-space: nowrap;
}

/* ── SELECT BOXES ── */
div[data-testid="stSelectbox"] > label {
    font-family: 'Jost', sans-serif !important;
    font-size: 0.65rem !important; font-weight: 600 !important;
    letter-spacing: 0.22em !important; text-transform: uppercase !important;
    color: var(--muted) !important; margin-bottom: 0.4rem !important;
}
div[data-testid="stSelectbox"] > div > div {
    background: white !important;
    border: 1px solid rgba(200,169,110,0.5) !important;
    border-radius: 2px !important;
    font-family: 'Jost', sans-serif !important;
    font-size: 0.9rem !important; color: var(--ink) !important;
    box-shadow: 0 1px 4px rgba(26,46,26,0.06) !important;
}
div[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: var(--forest) !important;
    box-shadow: 0 0 0 2px rgba(26,46,26,0.08) !important;
}

/* ── BUTTON ── */
.stButton > button {
    background: var(--forest) !important;
    color: var(--cream) !important;
    border: none !important; border-radius: 2px !important;
    padding: 0.9rem 0 !important; width: 100% !important;
    font-family: 'Jost', sans-serif !important;
    font-size: 0.72rem !important; font-weight: 600 !important;
    letter-spacing: 0.3em !important; text-transform: uppercase !important;
    cursor: pointer !important; margin-top: 1.5rem !important;
    transition: background 0.2s !important;
}
.stButton > button:hover {
    background: var(--sage) !important;
}

/* ── MEAL RESULT CARD ── */
.result-wrapper {
    display: grid; grid-template-columns: 1fr 1.6fr; gap: 0;
    border: 1px solid rgba(200,169,110,0.4);
    margin: 2.5rem 0; overflow: hidden;
    box-shadow: 0 8px 40px rgba(26,46,26,0.12);
}
.result-left {
    background: var(--forest);
    padding: 3rem 2.5rem; display: flex; flex-direction: column;
    justify-content: center; align-items: center; text-align: center;
    position: relative;
}
.result-left::after {
    content: '';
    position: absolute; inset: 16px;
    border: 1px solid rgba(200,169,110,0.2);
    pointer-events: none;
}
.result-emoji  { font-size: 4.5rem; margin-bottom: 1.2rem; line-height: 1; }
.result-category {
    font-family: 'Jost', sans-serif; font-size: 0.6rem;
    font-weight: 600; letter-spacing: 0.35em; text-transform: uppercase;
    color: var(--gold); margin-bottom: 0.5rem;
}
.result-meal {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.4rem; font-weight: 300; line-height: 1.1;
    color: var(--cream); margin-bottom: 1rem;
}
.result-right {
    background: white; padding: 2.8rem 3rem;
    display: flex; flex-direction: column; justify-content: center;
}
.result-desc {
    font-family: 'Cormorant Garamond', serif; font-style: italic;
    font-size: 1.25rem; color: var(--muted); line-height: 1.7;
    border-left: 2px solid var(--gold); padding-left: 1.2rem;
    margin-bottom: 2rem;
}
.tags-label {
    font-size: 0.6rem; font-weight: 600; letter-spacing: 0.25em;
    text-transform: uppercase; color: var(--muted); margin-bottom: 0.8rem;
}
.tags-row { display: flex; flex-wrap: wrap; gap: 0.5rem; }
.tag-pill {
    background: var(--parchment); border: 1px solid rgba(200,169,110,0.4);
    padding: 0.3rem 0.9rem; font-size: 0.78rem;
    color: var(--forest); font-family: 'Jost', sans-serif; font-weight: 500;
}

/* ── METRICS ROW ── */
.metrics-row {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 1px;
    background: rgba(200,169,110,0.3); border: 1px solid rgba(200,169,110,0.3);
    margin: 2rem 0;
}
.metric-cell {
    background: white; padding: 1.5rem 2rem; text-align: center;
}
.metric-cell-label {
    font-size: 0.6rem; font-weight: 600; letter-spacing: 0.25em;
    text-transform: uppercase; color: var(--muted); margin-bottom: 0.4rem;
}
.metric-cell-value {
    font-family: 'Cormorant Garamond', serif; font-size: 2rem;
    font-weight: 600; color: var(--forest);
}

/* ── DATASET TABLE ── */
div[data-testid="stDataFrame"] {
    border: 1px solid rgba(200,169,110,0.3) !important;
    border-radius: 0 !important;
}

/* ── MENU GRID ── */
.menu-grid {
    display: grid; grid-template-columns: repeat(2, 1fr); gap: 1px;
    background: rgba(200,169,110,0.25);
    border: 1px solid rgba(200,169,110,0.25);
    margin-top: 1.5rem;
}
.menu-cell {
    background: white; padding: 1.8rem 2rem;
    display: flex; align-items: flex-start; gap: 1.2rem;
}
.menu-cell-emoji { font-size: 2.2rem; flex-shrink: 0; margin-top: 0.1rem; }
.menu-cell-name {
    font-family: 'Cormorant Garamond', serif; font-size: 1.2rem;
    font-weight: 600; color: var(--forest); margin-bottom: 0.3rem;
}
.menu-cell-desc { font-size: 0.83rem; color: var(--muted); line-height: 1.5; }

/* ── CHAPTER HEADING ── */
.chapter {
    font-family: 'Cormorant Garamond', serif; font-style: italic;
    font-size: 0.85rem; color: var(--gold); letter-spacing: 0.15em;
    text-transform: uppercase; margin-bottom: 0.3rem;
}
.chapter-title {
    font-family: 'Cormorant Garamond', serif; font-size: 2rem;
    font-weight: 300; color: var(--forest); margin-bottom: 0.5rem;
}
.chapter-rule {
    height: 1px; background: rgba(200,169,110,0.4); margin-bottom: 2rem;
}

div[data-testid="stMetric"] {
    background: white !important; border: 1px solid rgba(200,169,110,0.3) !important;
    border-radius: 0 !important; padding: 1.2rem 1.5rem !important;
}
div[data-testid="stMetric"] label {
    font-size: 0.62rem !important; letter-spacing: 0.2em !important;
    text-transform: uppercase !important; color: var(--muted) !important;
    font-family: 'Jost', sans-serif !important; font-weight: 600 !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.9rem !important; color: var(--forest) !important;
}

/* ── RESPONSIVE STYLES ── */
@media (max-width: 768px) {
    .masthead { flex-direction: column; align-items: flex-start; padding: 2rem; }
    .masthead-right { margin-top: 1.5rem; text-align: left; }
    .masthead-title { font-size: 3rem; }
    .result-wrapper { grid-template-columns: 1fr; }
    .result-left { padding: 2rem 1.5rem; }
    .metrics-row { grid-template-columns: 1fr; }
    .menu-grid { grid-template-columns: 1fr; }
    .tab-bar { padding: 0 1.5rem; flex-wrap: wrap; }
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════
MEAL_INFO = {
    "Light Breakfast":  ("🥐", "Croissant, fruit bowl, green tea, granola — light and energising."),
    "Heavy Breakfast":  ("🍳", "Full cooked breakfast, parathas, eggs, juice — hearty start."),
    "Snacks":           ("🍿", "Samosas, sandwiches, fries, fruit chaat — quick bites."),
    "Fast Food":        ("🍔", "Burgers, pizza, wraps, fried chicken — quick & satisfying."),
    "Light Dinner":     ("🥗", "Salad, soup, grilled veggies, dal — easy on the stomach."),
    "Heavy Dinner":     ("🍛", "Biryani, paneer curry, naan, dessert — a full feast."),
    "Comfort Food":     ("🍜", "Khichdi, mac & cheese, warm soup, dal rice — soul-soothing."),
}

@st.cache_resource
def train_model():
    df = pd.read_csv("dataset.csv")
    le = {}
    
    # 1. Target encoding using LabelEncoder
    le["meal_recommendation"] = LabelEncoder()
    y = le["meal_recommendation"].fit_transform(df["meal_recommendation"])
    
    features = ["mood","time_of_day","weather","hunger_level","age_group"]
    
    # 2. One-Hot Encoding for the features instead of LabelEncoding
    X = pd.get_dummies(df[features])
    
    # Save column names so we can align inputs during prediction
    train_cols = X.columns
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Hyperparameter tuning to prevent overfitting on tiny datasets
    model = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=4, random_state=42)
    
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, le, df, acc, train_cols

model, le, df, accuracy, train_cols = train_model()

# ══════════════════════════════════════════════════════════════════════════════
# MASTHEAD
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="masthead">
  <div class="masthead-left">
    <div class="masthead-eyebrow">✦ Fine Dining Intelligence ✦</div>
    <div class="masthead-title">Mood<em>Meal</em></div>
    <div class="masthead-sub">Restaurant Meal Recommender &nbsp;·&nbsp; ML Mini Project</div>
  </div>
  <div class="masthead-right">
    <div class="masthead-badge">Random Forest Classifier</div>
    <div class="masthead-algo">mood-aware recommendations</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(["  ✦  Recommend My Meal  ", "  ◈  Dataset & Model  ", "  ❧  Menu Guide  ", "  ✿  About Us  "])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 – RECOMMEND
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("""
    <div class="chapter">Chapter I</div>
    <div class="chapter-title">Tell us about this moment</div>
    <div class="chapter-rule"></div>
    """, unsafe_allow_html=True)

    col1, spacer, col2 = st.columns([1, 0.08, 1])

    with col1:
        st.markdown('<div style="font-size:0.6rem;font-weight:600;letter-spacing:0.25em;text-transform:uppercase;color:#7a6e5a;margin-bottom:1.5rem">Your Current State</div>', unsafe_allow_html=True)
        mood       = st.selectbox("How are you feeling?",     ["Happy","Sad","Stressed","Excited","Tired","Anxious"])
        time_of_day= st.selectbox("What time of day is it?",  ["Morning","Afternoon","Evening","Night"])
        weather    = st.selectbox("What's the weather like?", ["Sunny","Cloudy","Rainy"])

    with col2:
        st.markdown('<div style="font-size:0.6rem;font-weight:600;letter-spacing:0.25em;text-transform:uppercase;color:#7a6e5a;margin-bottom:1.5rem">Your Appetite & Profile</div>', unsafe_allow_html=True)
        hunger     = st.selectbox("How hungry are you?",      ["Low","Medium","High"])
        age_group  = st.selectbox("Your age group?",          ["Teen","Young","Adult","Old"])
        st.markdown("<br>", unsafe_allow_html=True)
        btn = st.button("✦  Find My Perfect Meal")

    if btn:
        inp_df = pd.DataFrame([{
            "mood": mood,
            "time_of_day": time_of_day,
            "weather": weather,
            "hunger_level": hunger,
            "age_group": age_group,
        }])
        
        # One-Hot Encode the input
        X_inp = pd.get_dummies(inp_df)
        
        # Align with training columns (fill missing columns with False/0 depending on pandas version)
        X_inp = X_inp.reindex(columns=train_cols, fill_value=0)
        pred_enc = model.predict(X_inp)[0]
        meal = le["meal_recommendation"].inverse_transform([pred_enc])[0]
        probs = model.predict_proba(X_inp)[0]
        classes = le["meal_recommendation"].inverse_transform(model.classes_)
        emoji, desc = MEAL_INFO.get(meal, ("🍽️", "A perfect meal for you."))
        confidence = round(max(probs) * 100, 1)

        st.markdown(f"""
        <div class="result-wrapper">
          <div class="result-left">
            <div class="result-emoji">{emoji}</div>
            <div class="result-category">Recommended</div>
            <div class="result-meal">{meal}</div>
            <div style="font-size:0.65rem;letter-spacing:0.2em;color:rgba(200,169,110,0.6);text-transform:uppercase;font-family:'Jost',sans-serif">{confidence}% confidence</div>
          </div>
          <div class="result-right">
            <div style="font-size:0.6rem;font-weight:600;letter-spacing:0.3em;text-transform:uppercase;color:#7a6e5a;margin-bottom:1rem">Chef's Note</div>
            <div class="result-desc">{desc}</div>
            <div class="tags-label">Based on your inputs</div>
            <div class="tags-row">
              <span class="tag-pill">{mood}</span>
              <span class="tag-pill">{time_of_day}</span>
              <span class="tag-pill">{weather}</span>
              <span class="tag-pill">{hunger} Hunger</span>
              <span class="tag-pill">{age_group}</span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence chart
        st.markdown("""
        <div style="margin-top:2rem">
          <div class="chapter" style="margin-bottom:0.3rem">Confidence Breakdown</div>
          <div style="height:1px;background:rgba(200,169,110,0.3);margin-bottom:1.5rem"></div>
        </div>
        """, unsafe_allow_html=True)

        prob_df = pd.DataFrame({"Meal": classes, "Confidence": probs}).sort_values("Confidence", ascending=True)
        fig, ax = plt.subplots(figsize=(8, 3.5))
        fig.patch.set_facecolor('#ffffff')
        ax.set_facecolor('#ffffff')
        bar_colors = ['#1a2e1a' if m == meal else '#e8e0d0' for m in prob_df["Meal"]]
        bars = ax.barh(prob_df["Meal"], prob_df["Confidence"]*100,
                       color=bar_colors, edgecolor='none', height=0.55)
        for bar, val in zip(bars, prob_df["Confidence"]*100):
            if val > 2:
                ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                        f'{val:.1f}%', va='center', fontsize=8,
                        color='#7a6e5a', fontfamily='sans-serif')
        ax.set_xlabel("Confidence (%)", color='#7a6e5a', fontsize=9, labelpad=8)
        ax.tick_params(colors='#7a6e5a', labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#e8e0d0')
        ax.spines['bottom'].set_color('#e8e0d0')
        ax.set_xlim(0, max(prob_df["Confidence"]*100) * 1.2)
        plt.tight_layout()
        st.pyplot(fig, width='stretch')
        plt.close()
        
        components.html(
            """
            <script>
            setTimeout(function() {
                var parentDoc = window.parent.document;
                parentDoc.documentElement.style.scrollBehavior = 'smooth';
                var elements = parentDoc.getElementsByClassName('result-wrapper');
                if (elements.length > 0) {
                    elements[elements.length - 1].scrollIntoView({behavior: 'smooth', block: 'center'});
                }
            }, 300);
            </script>
            """,
            height=0
        )

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 – DATASET & MODEL
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("""
    <div class="chapter">Chapter II</div>
    <div class="chapter-title">Data, Training & Performance</div>
    <div class="chapter-rule"></div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Algorithm",  "Random Forest")
    c2.metric("Accuracy",   f"{accuracy*100:.1f}%")
    c3.metric("Total Records", f"{len(df)}")

    st.markdown('<div style="margin-top:2.5rem;font-size:0.6rem;font-weight:600;letter-spacing:0.25em;text-transform:uppercase;color:#7a6e5a;margin-bottom:0.8rem">Full Dataset</div>', unsafe_allow_html=True)
    st.dataframe(df, width='stretch', height=300)

    st.markdown("<br>", unsafe_allow_html=True)
    ch1, ch2 = st.columns(2, gap="large")

    with ch1:
        st.markdown('<div style="font-size:0.6rem;font-weight:600;letter-spacing:0.25em;text-transform:uppercase;color:#7a6e5a;margin-bottom:0.8rem">Meal Distribution</div>', unsafe_allow_html=True)
        dist = df["meal_recommendation"].value_counts()
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        fig2.patch.set_facecolor('#ffffff')
        ax2.set_facecolor('#ffffff')
        palette = ['#1a2e1a','#3d5c3d','#5a7a5a','#c8a96e','#dfc08a','#9e4c2c','#b8bfb0']
        wedges, texts, autotexts = ax2.pie(
            dist.values, labels=dist.index, autopct='%1.0f%%',
            colors=palette, startangle=140,
            wedgeprops=dict(linewidth=1.5, edgecolor='white'),
            textprops={'fontsize': 8, 'color': '#1c1a16'}
        )
        for at in autotexts:
            at.set_fontsize(7); at.set_color('white')
        ax2.set_title("Records per Category", fontsize=9, color='#7a6e5a',
                      fontfamily='sans-serif', pad=10)
        plt.tight_layout()
        st.pyplot(fig2, width='stretch')
        plt.close()

    with ch2:
        st.markdown('<div style="font-size:0.6rem;font-weight:600;letter-spacing:0.25em;text-transform:uppercase;color:#7a6e5a;margin-bottom:0.8rem">Feature Importance</div>', unsafe_allow_html=True)
        raw_importances = model.feature_importances_
        feat_imp_dict = {"mood": 0, "time_of_day": 0, "weather": 0, "hunger_level": 0, "age_group": 0}
        for col, imp in zip(train_cols, raw_importances):
            for feat in feat_imp_dict.keys():
                if col.startswith(feat):
                    feat_imp_dict[feat] += imp
        features = list(feat_imp_dict.keys())
        importances = np.array(list(feat_imp_dict.values()))
        fi_df = pd.DataFrame({"Feature": features, "Importance": importances*100}).sort_values("Importance")

        fig3, ax3 = plt.subplots(figsize=(5, 4))
        fig3.patch.set_facecolor('#ffffff')
        ax3.set_facecolor('#ffffff')
        colors_fi = ['#e8e0d0','#d4c9b0','#c8a96e','#3d5c3d','#1a2e1a']
        ax3.barh(fi_df["Feature"], fi_df["Importance"],
                 color=colors_fi, edgecolor='none', height=0.5)
        for i, (val, feat) in enumerate(zip(fi_df["Importance"], fi_df["Feature"])):
            ax3.text(val + 0.3, i, f'{val:.1f}%', va='center', fontsize=8, color='#7a6e5a')
        ax3.set_xlabel("Importance (%)", color='#7a6e5a', fontsize=9)
        ax3.set_title("Which inputs matter most?", fontsize=9, color='#7a6e5a', pad=10)
        ax3.tick_params(colors='#7a6e5a', labelsize=9)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.spines['left'].set_color('#e8e0d0')
        ax3.spines['bottom'].set_color('#e8e0d0')
        ax3.set_xlim(0, fi_df["Importance"].max() * 1.25)
        plt.tight_layout()
        st.pyplot(fig3, width='stretch')
        plt.close()

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 – MENU GUIDE
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("""
    <div class="chapter">Chapter III</div>
    <div class="chapter-title">Our Seven Courses</div>
    <div class="chapter-rule"></div>
    """, unsafe_allow_html=True)

    items = list(MEAL_INFO.items())
    for i in range(0, len(items), 2):
        cols = st.columns(2, gap="small")
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(items):
                meal_name, (emoji, desc) = items[idx]
                col.markdown(f"""
                <div class="menu-cell" style="border:1px solid rgba(200,169,110,0.25);">
                  <div class="menu-cell-emoji">{emoji}</div>
                  <div>
                    <div class="menu-cell-name">{meal_name}</div>
                    <div class="menu-cell-desc">{desc}</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:3rem;padding:1.5rem 2rem;border-top:1px solid rgba(200,169,110,0.3);
                display:flex;justify-content:space-between;align-items:center;">
      <div style="font-family:'Cormorant Garamond',serif;font-style:italic;font-size:0.9rem;color:#7a6e5a">
        MoodMeal &nbsp;·&nbsp; ML Mini Project
      </div>
      <div style="font-size:0.62rem;letter-spacing:0.2em;text-transform:uppercase;color:rgba(200,169,110,0.6);font-family:'Jost',sans-serif">
        Random Forest Classification &nbsp;·&nbsp; Python + Streamlit
      </div>
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 – ABOUT US
# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("""
    <div class="chapter">Chapter IV</div>
    <div class="chapter-title">The Makers & The Mission</div>
    <div class="chapter-rule"></div>
    
    <div style="background:white; padding: 4rem 3rem; border: 1px solid rgba(200,169,110,0.3); text-align: center; margin-top: 1rem;">
        <div style="font-family: 'Jost', sans-serif; font-size: 0.8rem; font-weight: 600; letter-spacing: 0.3em; text-transform: uppercase; color: var(--gold); margin-bottom: 0.8rem;">
            AIML Mini Project
        </div>
        <div style="font-family: 'Cormorant Garamond', serif; font-size: 2.8rem; color: var(--forest); margin-bottom: 2rem;">
            Engineered by the Next Generation
        </div>
        <div style="display:flex; justify-content: center; gap: 3rem; margin-bottom: 2.5rem;">
            <div style="text-align:center;">
                <div style="font-family: 'Cormorant Garamond', serif; font-size: 1.6rem; color: var(--ink); font-weight: 600;">Rajavardhan S G</div>
                <div style="font-family: 'Jost', sans-serif; font-size: 0.85rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.3rem;">Project Lead / Developer</div>
            </div>
            <div style="width: 1px; background: rgba(200,169,110,0.3);"></div>
            <div style="text-align:center;">
                <div style="font-family: 'Cormorant Garamond', serif; font-size: 1.6rem; color: var(--ink); font-weight: 600;">Darshan G K</div>
                <div style="font-family: 'Jost', sans-serif; font-size: 0.85rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.3rem;">Co-Developer</div>
            </div>
        </div>
        <div style="font-family: 'Jost', sans-serif; font-size: 1.05rem; color: var(--muted); line-height: 1.8; max-width: 750px; margin: 0 auto; border-top: 1px solid rgba(200,169,110,0.2); padding-top: 2rem;">
            <strong>MoodMeal</strong> was developed as our core mini-project for the Artificial Intelligence & Machine Learning (AIML) curriculum. 
            Our aim was to move beyond theory and build a functioning, end-to-end classification product that solves a real-world dilemma: <em>what should I eat right now?</em> <br><br>
            By designing a custom dataset, tuning a Random Forest Classifier, and deploying a responsive Streamlit UI, 
            we demonstrated full-stack Machine Learning capabilities. Thank you to our professors and mentors guiding us through the world of AI!
        </div>
    </div>
    """, unsafe_allow_html=True)
