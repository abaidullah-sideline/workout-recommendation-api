"""
Workout Recommendation API
POST /recommend  — natural-language query → structured workout plan + GIFs + videos
GET  /health     — liveness probe
"""
import asyncio
import json
import os
import random
import re
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types as genai_types
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# ── Environment ───────────────────────────────────────────────────────────────
_SUPABASE_BASE = os.environ["SUPABASE_URL"].rstrip("/")
SUPABASE_REST  = f"{_SUPABASE_BASE}/rest/v1"
SUPABASE_KEY   = os.environ["SUPABASE_SERVICE_KEY"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

_gemini = genai.Client(api_key=GEMINI_API_KEY)
_GEMINI_MODEL = "gemini-2.5-flash"

DB_HEADERS = {
    "apikey":        SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type":  "application/json",
}

# ── Allowed category values (mirror DB CHECK constraints) ─────────────────────
VALID: dict[str, list[str]] = {
    "fitness_goal":       ["Weight Loss", "Muscle Gain", "Strength", "Endurance/Stamina"],
    "target_timeframe":   ["4 Weeks", "8 Weeks", "12 Weeks", "6 Months"],
    "activity_level":     ["Beginner", "Intermediate", "Advanced"],
    "gender":             ["Male", "Female", "Any"],
    "age_group":          ["18-25", "26-35", "36-50", "50+"],
    "bmi_category":       ["Underweight", "Normal", "Overweight", "Obese"],
    "medical_conditions": [
        "Heart Disease", "Hypertension", "Asthma",
        "Lower Back Pain", "Knee Issues", "None",
    ],
}

# Muscle groups present in the exercises table
VALID_MUSCLE_GROUPS = ["Back", "Cardio", "Chest", "Lower Arms", "Lower Legs",
                       "Shoulders", "Upper Arms", "Upper Legs", "Waist"]

# Most → least important for progressive filter relaxation
_FILTER_PRIORITY = [
    "fitness_goal", "activity_level", "target_timeframe",
    "medical_conditions", "gender", "bmi_category", "age_group",
]

# Maps fitness goal → exercise primary_goal values in DB
_EXERCISE_GOAL_MAP: dict[str, str] = {
    "Weight Loss":       "fat_loss",
    "Muscle Gain":       "muscle_gain",
    "Strength":          "muscle_gain",
    "Endurance/Stamina": "fat_loss",
}

# Maps fitness goal → youtube_videos.category values in DB
_VIDEO_CAT_MAP: dict[str, list[str]] = {
    "Weight Loss":       ["fat_loss", "meal_prep"],
    "Muscle Gain":       ["muscle_gain", "bulking"],
    "Strength":          ["muscle_gain", "recomp"],
    "Endurance/Stamina": ["endurance", "pre_workout"],
}

# Conditions not in DB schema → nearest DB value (for filtering)
# Raw condition still passed to AI summary for personalization
_CONDITION_FALLBACK: dict[str, str] = {
    "diabetes":           "None",
    "type 2 diabetes":    "None",
    "type 1 diabetes":    "None",
    "obesity":            "None",
    "arthritis":          "Knee Issues",
    "osteoporosis":       "None",
    "copd":               "Asthma",
    "anxiety":            "None",
    "depression":         "None",
    "high blood pressure": "Hypertension",
}

# ── Rate limiter ─────────────────────────────────────────────────────────────
# NOTE: slowapi uses in-memory storage by default.  On Vercel each warm
# serverless instance has its own counter.  For global enforcement use an
# Upstash Redis backend: Limiter(..., storage_uri="redis://...")
limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Workout Recommendation API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url=None,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
    max_age=600,
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=10, max_length=500)

    @field_validator("query")
    @classmethod
    def sanitize(cls, v: str) -> str:
        v = re.sub(r"[<>{}\[\]\\]", "", v)
        return v.strip()


class Exercise(BaseModel):
    id: str
    name: str
    gif_url: Optional[str] = None
    muscle_group: Optional[str] = None


class Video(BaseModel):
    id: int
    title: str
    youtube_url: str
    channel_name: Optional[str] = None


class RecommendationResponse(BaseModel):
    plan_id: str
    fitness_goal: str
    target_timeframe: Optional[str] = None
    activity_level: Optional[str] = None
    # Only populated when user explicitly provided the value
    gender: Optional[str] = None
    age_group: Optional[str] = None
    bmi_category: Optional[str] = None
    medical_conditions: Optional[str] = None
    gym_workout_plan: str
    est_calories_burned: int
    exercises: list[Exercise]
    videos: list[Video]
    ai_summary: str
    matched_filters: dict[str, str]
    workout_frequency: Optional[str] = None
    target_muscle_groups: Optional[list[str]] = None
    condition_note: Optional[str] = None


# ── BMI helper ────────────────────────────────────────────────────────────────
def _classify_bmi(value: float) -> str:
    if value < 18.5:
        return "Underweight"
    elif value < 25.0:
        return "Normal"
    elif value < 30.0:
        return "Overweight"
    return "Obese"


# ── Gemini: extract structured params ────────────────────────────────────────
def _extract_prompt(query: str) -> str:
    return f"""You are a fitness profile extractor. Parse this query and return a JSON object.

User query: "{query}"

Return a JSON object with these fields:

{{
  "fitness_goal":            one of {VALID["fitness_goal"]} or null,
  "target_timeframe":        one of {VALID["target_timeframe"]} or null,
  "activity_level":          one of {VALID["activity_level"]} or null,
  "gender":                  one of {VALID["gender"]} or null — ONLY if explicitly stated,
  "age_group":               one of {VALID["age_group"]} or null — ONLY if user states their age,
  "bmi_value":               numeric BMI if the user gives a number (e.g. "BMI 23" → 23.0), else null,
  "bmi_category":            one of {VALID["bmi_category"]} or null — ONLY if user explicitly names a category,
  "medical_conditions":      one of {VALID["medical_conditions"]} — map to closest match; default "None",
  "raw_medical_conditions":  exact health condition text from query, or null if none mentioned,
  "target_muscle_groups":    list from {VALID_MUSCLE_GROUPS} matching muscles user mentions, or [],
  "workout_days_per_week":   integer if user specifies frequency (e.g. "5 days a week" → 5), else null
}}

Rules:
- Do NOT infer age_group unless user states their age explicitly.
- Do NOT infer gender unless user states it explicitly.
- Do NOT infer bmi_category unless user states it explicitly — use bmi_value for numbers.
- For medical_conditions: map known conditions to closest DB value:
    diabetes / high blood sugar → "None" (capture raw)
    high blood pressure → "Hypertension"
    bad knees / knee pain → "Knee Issues"
    bad back / back pain → "Lower Back Pain"
    breathing problems → "Asthma"
    heart issues → "Heart Disease"
    If no condition mentioned → "None"
- target_muscle_groups: chest focus / pecs / bench → ["Chest"];
    back / lats / pulling → ["Back"];
    shoulders / delts → ["Shoulders"];
    legs / quads / glutes → ["Upper Legs", "Lower Legs"];
    arms / biceps / triceps → ["Upper Arms", "Lower Arms"];
    cardio → ["Cardio"];
    full body → []

Mapping hints for fitness_goal:
- lose weight / burn fat / slim / cut → "Weight Loss"
- build muscle / bulk / gain mass / hypertrophy → "Muscle Gain"
- get stronger / powerlifting / 1RM → "Strength"
- endurance / stamina / cardio / marathon / run → "Endurance/Stamina"

Return ONLY valid JSON. No markdown, no explanation."""


async def extract_params(query: str) -> dict:
    """Call Gemini to parse query; whitelist-validate all DB filter values."""
    try:
        resp = await _gemini.aio.models.generate_content(
            model=_GEMINI_MODEL,
            contents=_extract_prompt(query),
            config=genai_types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=512,
                response_mime_type="application/json",
                thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
            ),
        )
        raw: dict = json.loads(resp.text)
    except Exception:
        return {}

    result: dict = {}

    # Whitelist-validate DB filter fields
    for field in ("fitness_goal", "target_timeframe", "activity_level",
                  "gender", "age_group", "medical_conditions"):
        val = raw.get(field)
        if val in VALID[field]:
            result[field] = val

    # Code-based BMI classification overrides LLM bmi_category
    bmi_val = raw.get("bmi_value")
    if isinstance(bmi_val, (int, float)) and 10 < bmi_val < 60:
        result["bmi_category"] = _classify_bmi(float(bmi_val))
    elif raw.get("bmi_category") in VALID["bmi_category"]:
        result["bmi_category"] = raw["bmi_category"]

    # Pass-through extra context (not DB filters)
    result["raw_medical_conditions"] = raw.get("raw_medical_conditions")
    result["target_muscle_groups"] = [
        m for m in (raw.get("target_muscle_groups") or [])
        if m in VALID_MUSCLE_GROUPS
    ]
    freq = raw.get("workout_days_per_week")
    result["workout_days_per_week"] = int(freq) if isinstance(freq, (int, float)) else None

    return result


# ── Gemini: personalized summary ─────────────────────────────────────────────
async def generate_summary(plan: dict, query: str, extra: dict) -> str:
    freq_note = (f"{extra['workout_days_per_week']} days/week as requested"
                 if extra.get("workout_days_per_week") else "structured weekly sessions")
    condition_note = (f"Your specific condition ({extra['raw_medical_conditions']}) has been "
                      "taken into account in tailoring this plan."
                      if extra.get("raw_medical_conditions") else "")
    muscle_note = (f"Exercises focus on {', '.join(extra['target_muscle_groups'])}."
                   if extra.get("target_muscle_groups") else "")

    prompt = (
        f'User asked: "{query}"\n\n'
        f"Matched plan: {plan['target_timeframe']} {plan['fitness_goal']} program "
        f"for a {plan['activity_level']} ({plan['age_group']}), "
        f"medical notes: {plan['medical_conditions']}, "
        f"~{plan['est_calories_burned']} kcal/week, {freq_note}.\n"
        f"{condition_note} {muscle_note}\n\n"
        "Write 2-3 friendly, encouraging sentences explaining why this plan suits the user. "
        "Be specific. Do NOT make medical claims or guarantee results."
    )
    try:
        resp = await _gemini.aio.models.generate_content(
            model=_GEMINI_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(temperature=0.7, max_output_tokens=200),
        )
        return resp.text.strip()
    except Exception:
        return (
            f"This {plan['target_timeframe']} {plan['fitness_goal']} plan is tailored "
            f"for {plan['activity_level']} level and accounts for your individual profile."
        )


# ── Supabase helpers ──────────────────────────────────────────────────────────
async def search_plans(params: dict) -> tuple[list[dict], dict[str, str]]:
    """Query workout_recommendations with progressive filter relaxation."""
    db_params = {k: v for k, v in params.items()
                 if k in _FILTER_PRIORITY and v is not None}

    async with httpx.AsyncClient(timeout=10.0) as client:
        for drop in range(len(_FILTER_PRIORITY) + 1):
            keep = _FILTER_PRIORITY[: max(1, len(_FILTER_PRIORITY) - drop)]
            active = {k: db_params[k] for k in keep if k in db_params}
            if not active:
                break
            qp: dict = {"select": "*", "limit": "5"}
            for k, v in active.items():
                qp[k] = f"eq.{v}"
            r = await client.get(f"{SUPABASE_REST}/workout_recommendations",
                                 headers=DB_HEADERS, params=qp)
            if r.is_success and r.json():
                return r.json(), active

    return [], {}


async def fetch_exercises_dynamic(
    goal: str, level: str, muscle_groups: list[str], limit: int = 5
) -> list[dict]:
    """Dynamically look up exercises by goal, level, and target muscle groups."""
    level_map = {"Beginner": "beginner", "Intermediate": "intermediate", "Advanced": "advanced"}
    ex_goal = _EXERCISE_GOAL_MAP.get(goal, "muscle_gain")
    difficulty = level_map.get(level, "beginner")

    qp: dict = {
        "select": "id,name,gif_url,muscle_group,body_part",
        "difficulty": f"eq.{difficulty}",
        "limit": str(limit),
    }

    if goal == "Endurance/Stamina":
        qp["muscle_group"] = "eq.Cardio"
    elif muscle_groups:
        # filter by first mentioned muscle group
        qp["muscle_group"] = f"eq.{muscle_groups[0]}"
    else:
        qp["primary_goal"] = f"eq.{ex_goal}"

    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(f"{SUPABASE_REST}/exercises", headers=DB_HEADERS, params=qp)
        rows = r.json() if r.is_success else []

    # If fewer than 3 results, relax the difficulty filter
    if len(rows) < 3:
        fallback_qp = {k: v for k, v in qp.items() if k != "difficulty"}
        fallback_qp["limit"] = str(limit)
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{SUPABASE_REST}/exercises", headers=DB_HEADERS,
                                 params=fallback_qp)
            rows = r.json() if r.is_success else rows

    return rows[:limit]


async def fetch_videos_dynamic(goal: str, limit: int = 3) -> list[dict]:
    """Dynamically look up videos by category matching the fitness goal."""
    categories = _VIDEO_CAT_MAP.get(goal, ["muscle_gain"])
    cat_filter = "(" + ",".join(categories) + ")"

    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(
            f"{SUPABASE_REST}/youtube_videos",
            headers=DB_HEADERS,
            params={
                "select": "id,title,youtube_url,channel_name",
                "category": f"in.{cat_filter}",
                "limit": str(limit * 3),  # fetch more, then sample for variety
            },
        )
    rows = r.json() if r.is_success else []
    # Random sample so repeated calls return different videos
    random.shuffle(rows)
    return rows[:limit]


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", include_in_schema=False)
async def health():
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendationResponse)
@limiter.limit("10/minute")
async def recommend(request: Request, body: QueryRequest):
    # 1. Extract structured params + extra context from query
    params = await extract_params(body.query)
    if not params.get("fitness_goal"):
        raise HTTPException(
            status_code=422,
            detail=(
                "Could not determine a fitness goal. "
                "Try phrases like 'lose weight', 'build muscle', "
                "'get stronger', or 'improve stamina'."
            ),
        )

    # 2. Build condition note for unmapped conditions (e.g. Diabetes)
    raw_cond = params.get("raw_medical_conditions")
    db_cond  = params.get("medical_conditions", "None")
    condition_note: Optional[str] = None
    if raw_cond and db_cond == "None":
        condition_note = (
            f"Note: '{raw_cond}' was noted but isn't in our condition filter set. "
            "The AI summary has been personalized to account for it."
        )

    # 3. Find matching plan (progressive filter relaxation)
    plans, matched_filters = await search_plans(params)
    if not plans:
        raise HTTPException(
            status_code=404,
            detail="No matching workout plan found. Try specifying your goal and timeframe.",
        )
    plan = random.choice(plans[:3])

    # 4. Dynamic exercise + video lookup + AI summary — all in parallel
    goal   = plan["fitness_goal"]
    level  = plan["activity_level"]
    muscle = params.get("target_muscle_groups") or []

    exercises, videos, ai_summary = await asyncio.gather(
        fetch_exercises_dynamic(goal, level, muscle),
        fetch_videos_dynamic(goal),
        generate_summary(plan, body.query, params),
    )

    freq = params.get("workout_days_per_week")

    # Only surface profile fields the user actually provided — never assume from the plan
    return RecommendationResponse(
        plan_id=plan["plan_id"],
        fitness_goal=plan["fitness_goal"],
        target_timeframe=params.get("target_timeframe"),
        activity_level=params.get("activity_level"),
        gender=params.get("gender"),
        age_group=params.get("age_group"),
        bmi_category=params.get("bmi_category"),
        medical_conditions=params.get("medical_conditions") or (raw_cond and None),
        gym_workout_plan=plan["gym_workout_plan"],
        est_calories_burned=plan["est_calories_burned"],
        exercises=[Exercise(**e) for e in exercises],
        videos=[Video(**v) for v in videos],
        ai_summary=ai_summary,
        matched_filters=matched_filters,
        workout_frequency=f"{freq} days/week" if freq else None,
        target_muscle_groups=muscle or None,
        condition_note=condition_note,
    )
