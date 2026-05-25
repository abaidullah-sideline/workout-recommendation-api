"""BetterMe — Streamlit frontend for the Workout Recommendation API."""

import requests
import streamlit as st

API_URL = "https://workout-recommendation-api.vercel.app"

# ── Page config (must be the very first Streamlit call) ───────────────────────
st.set_page_config(
    page_title="BetterMe",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 4rem !important;
    max-width: 1100px !important;
}

/* ── Background ── */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg, #08081a 0%, #0f1629 55%, #081a11 100%);
}
[data-testid="stHeader"] { background: transparent !important; }

/* ── Base text ── */
html, body, p, li, span, label, div { color: #e2e8f0; }

/* ── Cards ── */
.fp-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 16px;
}
.fp-card-accent {
    background: linear-gradient(135deg,
        rgba(79,142,247,0.12) 0%,
        rgba(0,212,170,0.07) 100%);
    border: 1px solid rgba(79,142,247,0.25);
    border-radius: 16px;
    padding: 22px 26px;
    margin-bottom: 16px;
}

/* ── Hero ── */
.fp-hero-title {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(130deg, #4f8ef7 20%, #00d4aa 80%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 8px;
}
.fp-hero-sub {
    font-size: 1.1rem;
    color: #94a3b8;
    margin-bottom: 2rem;
    max-width: 600px;
}

/* ── Stat boxes ── */
.fp-stat {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 14px 16px;
    text-align: center;
}
.fp-stat-label {
    font-size: 0.68rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 5px;
}
.fp-stat-value {
    font-size: 1rem;
    font-weight: 700;
    color: #e2e8f0;
}

/* ── Exercise ── */
.fp-ex-name {
    font-size: 0.96rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 4px;
}
.fp-ex-meta {
    font-size: 0.76rem;
    color: #64748b;
}

/* ── Section label ── */
.fp-section-label {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: #4f8ef7;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(79,142,247,0.18);
    margin-bottom: 14px;
}

/* ── AI summary ── */
.fp-ai-summary {
    border-left: 3px solid #4f8ef7;
    padding: 14px 20px;
    background: rgba(79,142,247,0.06);
    border-radius: 0 12px 12px 0;
    color: #cbd5e1;
    font-style: italic;
    line-height: 1.7;
    font-size: 0.97rem;
}

/* ── Rest day ── */
.fp-rest {
    background: rgba(148,163,184,0.05);
    border: 1px solid rgba(148,163,184,0.12);
    border-radius: 16px;
    padding: 40px 24px;
    text-align: center;
    color: #64748b;
}

/* ── Feature list item ── */
.fp-feature {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    margin-bottom: 14px;
}
.fp-feature-icon {
    font-size: 1.25rem;
    min-width: 28px;
}
.fp-feature-text {
    font-size: 0.9rem;
    color: #94a3b8;
    line-height: 1.4;
}
.fp-feature-title {
    font-weight: 600;
    color: #e2e8f0;
    display: block;
    margin-bottom: 2px;
}

/* ── Primary button ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #4f8ef7, #00d4aa) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.6rem 1.8rem !important;
    transition: opacity 0.2s !important;
}
[data-testid="stButton"] > button:hover { opacity: 0.88 !important; }

/* ── Form inputs ── */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}
[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(255,255,255,0.03);
    border-radius: 14px;
    padding: 4px;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    border-radius: 10px !important;
    color: #94a3b8 !important;
    font-weight: 500 !important;
    padding: 8px 14px !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: rgba(79,142,247,0.18) !important;
    color: #4f8ef7 !important;
    font-weight: 700 !important;
}

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.07) !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for _k, _v in [("page", "profile"), ("data", None)]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Constants ─────────────────────────────────────────────────────────────────
GOAL_ICONS = {
    "Weight Loss": "🔥",
    "Muscle Gain": "💪",
    "Strength": "🏋️",
    "Endurance/Stamina": "🏃",
}

GOAL_PHRASES = {
    "Weight Loss":       "I want to lose weight",
    "Muscle Gain":       "I want to build muscle",
    "Strength":          "I want to get stronger",
    "Endurance/Stamina": "I want to improve my endurance",
}

CONDITION_PHRASES = {
    "Heart Disease":   "I have heart disease",
    "Hypertension":    "I have high blood pressure",
    "Asthma":          "I have asthma",
    "Lower Back Pain": "I have lower back pain",
    "Knee Issues":     "I have knee issues",
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def call_api(query: str):
    try:
        r = requests.post(
            f"{API_URL}/recommend",
            json={"query": query},
            timeout=45,
        )
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.Timeout:
        return None, "The request timed out. Please try again in a moment."
    except requests.exceptions.HTTPError as exc:
        try:
            detail = exc.response.json().get("detail", str(exc))
        except Exception:
            detail = str(exc)
        return None, detail
    except Exception as exc:
        return None, str(exc)


def build_query(goal, level, timeframe, gender, age, bmi, condition, days) -> str:
    parts = [GOAL_PHRASES.get(goal, f"My goal is {goal}")]
    parts.append(f"in {timeframe.lower()}")
    parts.append(f"I am {level.lower()} level")
    if gender and gender != "Prefer not to say":
        parts.append(f"I am {gender.lower()}")
    if age:
        parts.append(f"I am {int(age)} years old")
    if bmi:
        parts.append(f"my BMI is {bmi:.1f}")
    if condition and condition != "None":
        parts.append(CONDITION_PHRASES.get(condition, f"I have {condition.lower()}"))
    if days:
        parts.append(f"I can work out {days} days per week")
    return ", ".join(parts)


def stat_html(label: str, value: str) -> str:
    return (
        f'<div class="fp-stat">'
        f'<div class="fp-stat-label">{label}</div>'
        f'<div class="fp-stat-value">{value}</div>'
        f'</div>'
    )


# ── Profile page ──────────────────────────────────────────────────────────────
def profile_page():
    # Hero
    st.markdown('<div class="fp-hero-title">BetterMe</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="fp-hero-sub">'
        "Your AI-powered personal trainer. Fill in your profile and get a full "
        "weekly workout plan — complete with exercise GIFs and YouTube guides."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    col_form, col_info = st.columns([3, 2], gap="large")

    # ── Left: form ──
    with col_form:
        with st.form("profile_form"):
            st.markdown("#### 🎯 Primary Goal")
            goal = st.radio(
                "goal",
                list(GOAL_ICONS.keys()),
                horizontal=True,
                label_visibility="collapsed",
            )

            st.markdown("#### ⚡ Activity Level")
            level = st.radio(
                "level",
                ["Beginner", "Intermediate", "Advanced"],
                horizontal=True,
                label_visibility="collapsed",
            )

            st.markdown("#### 📅 Plan Duration")
            timeframe = st.select_slider(
                "timeframe",
                options=["4 Weeks", "8 Weeks", "12 Weeks", "6 Months"],
                value="12 Weeks",
                label_visibility="collapsed",
            )

            st.markdown("---")
            with st.expander("➕  Optional: age, BMI, medical conditions, frequency"):
                r1c1, r1c2, r1c3 = st.columns(3)
                with r1c1:
                    gender = st.selectbox(
                        "Gender",
                        ["Prefer not to say", "Male", "Female"],
                    )
                with r1c2:
                    age = st.number_input(
                        "Age",
                        min_value=16, max_value=80,
                        value=None, placeholder="e.g. 28", step=1,
                    )
                with r1c3:
                    bmi = st.number_input(
                        "BMI",
                        min_value=10.0, max_value=60.0,
                        value=None, placeholder="e.g. 24.5", format="%.1f",
                    )

                r2c1, r2c2 = st.columns(2)
                with r2c1:
                    condition = st.selectbox(
                        "Medical Condition",
                        ["None", "Heart Disease", "Hypertension", "Asthma",
                         "Lower Back Pain", "Knee Issues"],
                    )
                with r2c2:
                    days_opt = st.select_slider(
                        "Workout days / week",
                        options=["Any", "2", "3", "4", "5", "6"],
                        value="Any",
                    )

            submitted = st.form_submit_button(
                "🚀  Get My Personalised Plan",
                use_container_width=True,
            )

    # ── Right: feature overview ──
    with col_info:
        st.markdown('<div class="fp-card-accent">', unsafe_allow_html=True)
        st.markdown("### What you'll receive")
        features = [
            ("📋", "Weekly Schedule", "A full Mon–Sun workout plan tailored to your goal and level."),
            ("🎬", "Exercise GIFs", "Every exercise has an animated demo so your form stays perfect."),
            ("📹", "Video Guides", "2 curated YouTube videos per training day for deeper guidance."),
            ("🤖", "AI Summary", "A personalised explanation of why this plan suits you."),
            ("🛌", "Rest Days", "Built-in recovery days — no guesswork about when to rest."),
        ]
        for icon, title, desc in features:
            st.markdown(
                f'<div class="fp-feature">'
                f'<div class="fp-feature-icon">{icon}</div>'
                f'<div class="fp-feature-text">'
                f'<span class="fp-feature-title">{title}</span>{desc}'
                f'</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="fp-card" style="margin-top:0">', unsafe_allow_html=True)
        st.markdown(
            "**Supported goals**\n\n"
            "🔥 Weight Loss &nbsp; 💪 Muscle Gain\n\n"
            "🏋️ Strength &nbsp; 🏃 Endurance / Stamina",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Submit handler ──
    if submitted:
        days_val = None if days_opt == "Any" else int(days_opt)
        query = build_query(goal, level, timeframe, gender, age, bmi, condition, days_val)
        with st.spinner(f"Building your {goal.lower()} plan…  ⏳"):
            data, err = call_api(query)

        if err:
            st.error(f"⚠️  {err}")
        else:
            st.session_state.data = data
            st.session_state.page = "dashboard"
            st.rerun()


# ── Dashboard ─────────────────────────────────────────────────────────────────
def dashboard_page():
    data = st.session_state.data

    # ── Top nav ──
    nav_left, nav_right = st.columns([1, 6])
    with nav_left:
        if st.button("← New Plan"):
            st.session_state.page = "profile"
            st.session_state.data = None
            st.rerun()

    st.markdown("---")

    # ── Plan title ──
    icon = GOAL_ICONS.get(data["fitness_goal"], "🏋️")
    st.markdown(
        f'<div class="fp-hero-title" style="font-size:2.2rem">'
        f'{icon}  {data["fitness_goal"]} Plan</div>',
        unsafe_allow_html=True,
    )

    # ── Stats row ──
    stats: list[tuple[str, str]] = [
        ("Duration",     data.get("target_timeframe") or "—"),
        ("Level",        data.get("activity_level")   or "—"),
        ("Kcal / week",  f"{data['est_calories_burned']:,}"),
    ]
    if data.get("gender"):
        stats.append(("Gender", data["gender"]))
    if data.get("age_group"):
        stats.append(("Age Group", data["age_group"]))
    if data.get("bmi_category"):
        stats.append(("BMI Category", data["bmi_category"]))
    if data.get("medical_conditions"):
        stats.append(("Condition", data["medical_conditions"]))
    if data.get("workout_frequency"):
        stats.append(("Frequency", data["workout_frequency"]))

    stat_cols = st.columns(len(stats))
    for col, (label, value) in zip(stat_cols, stats):
        col.markdown(stat_html(label, value), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── AI summary ──
    st.markdown(
        f'<div class="fp-ai-summary">🤖 &nbsp;{data["ai_summary"]}</div>',
        unsafe_allow_html=True,
    )

    if data.get("condition_note"):
        st.info(f"ℹ️  {data['condition_note']}")

    st.markdown("---")
    st.markdown("### 📅 Weekly Schedule")

    # ── Day tabs ──
    schedule = data["schedule"]
    tab_labels = [
        f"{'🛌' if d['is_rest'] else '💪'} {d['day'][:3]}"
        for d in schedule
    ]
    tabs = st.tabs(tab_labels)

    for tab, day in zip(tabs, schedule):
        with tab:
            render_day(day)

    st.markdown("---")
    _, btn_col, _ = st.columns([2, 1, 2])
    with btn_col:
        if st.button("🔄  Different Plan", use_container_width=True):
            st.session_state.page = "profile"
            st.session_state.data = None
            st.rerun()


# ── Day renderer ──────────────────────────────────────────────────────────────
def render_day(day: dict):
    # Rest day
    if day["is_rest"]:
        st.markdown(
            '<div class="fp-rest">'
            '<div style="font-size:2.8rem;margin-bottom:12px">🛌</div>'
            '<div style="font-size:1.2rem;font-weight:700;color:#94a3b8">Rest Day</div>'
            '<div style="margin-top:10px;font-size:0.9rem;max-width:380px;margin-left:auto;margin-right:auto">'
            "Recovery is as important as training. Rest, hydrate, and let your "
            "muscles repair for the next session."
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # Day header
    focus = day.get("focus") or "Training"
    st.markdown(
        f'<div style="font-size:1.3rem;font-weight:700;color:#e2e8f0;margin-bottom:2px">'
        f'💪 &nbsp;{day["day"]}</div>'
        f'<div style="font-size:0.88rem;color:#94a3b8;margin-bottom:18px">{focus}</div>',
        unsafe_allow_html=True,
    )

    exercises = day.get("exercises", [])
    videos    = day.get("videos", [])

    # ── Exercises ──
    if exercises:
        st.markdown(
            '<div class="fp-section-label">Exercises</div>',
            unsafe_allow_html=True,
        )
        for ex in exercises:
            col_gif, col_text = st.columns([1, 3], gap="medium")

            with col_gif:
                if ex.get("gif_url"):
                    st.image(ex["gif_url"], use_container_width=True)
                else:
                    st.markdown(
                        '<div style="height:110px;background:rgba(255,255,255,0.04);'
                        'border:1px solid rgba(255,255,255,0.07);border-radius:10px;'
                        'display:flex;align-items:center;justify-content:center;'
                        'font-size:2rem;color:#4f8ef7">🏋️</div>',
                        unsafe_allow_html=True,
                    )

            with col_text:
                muscle = ex.get("muscle_group") or "General"
                st.markdown(
                    f'<div style="padding-top:12px">'
                    f'<div class="fp-ex-name">{ex["name"]}</div>'
                    f'<div class="fp-ex-meta">🎯 {muscle}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                '<hr style="margin:6px 0;border-color:rgba(255,255,255,0.05)">',
                unsafe_allow_html=True,
            )

    # ── Videos ──
    if videos:
        st.markdown(
            '<div class="fp-section-label" style="margin-top:24px">'
            '📹 &nbsp;Video Guides</div>',
            unsafe_allow_html=True,
        )
        video_cols = st.columns(len(videos), gap="medium")
        for vcol, vid in zip(video_cols, videos):
            with vcol:
                st.video(vid["youtube_url"])
                st.caption(vid["title"])


# ── Router ────────────────────────────────────────────────────────────────────
if st.session_state.page == "profile":
    profile_page()
else:
    dashboard_page()
