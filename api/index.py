"""
Workout Recommendation API
POST /recommend  — natural-language query → structured per-day workout schedule
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
_SUPABASE_BASE  = os.environ["SUPABASE_URL"].rstrip("/")
SUPABASE_REST   = f"{_SUPABASE_BASE}/rest/v1"
SUPABASE_KEY    = os.environ["SUPABASE_SERVICE_KEY"]
GEMINI_API_KEY  = os.environ["GEMINI_API_KEY"]
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

_gemini       = genai.Client(api_key=GEMINI_API_KEY)
_GEMINI_MODEL = "gemini-2.5-flash"

DB_HEADERS = {
    "apikey":        SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type":  "application/json",
}

# ── Category constants ────────────────────────────────────────────────────────
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
VALID_MUSCLE_GROUPS = [
    "Back", "Cardio", "Chest", "Lower Arms", "Lower Legs",
    "Shoulders", "Upper Arms", "Upper Legs", "Waist",
]
_FILTER_PRIORITY = [
    "fitness_goal", "activity_level", "target_timeframe",
    "medical_conditions", "gender", "bmi_category", "age_group",
]
_EXERCISE_GOAL_MAP = {
    "Weight Loss":       "fat_loss",
    "Muscle Gain":       "muscle_gain",
    "Strength":          "muscle_gain",
    "Endurance/Stamina": "fat_loss",
}
_VIDEO_CAT_MAP: dict[str, list[str]] = {
    "Weight Loss":       ["fat_loss", "meal_prep"],
    "Muscle Gain":       ["muscle_gain", "bulking"],
    "Strength":          ["muscle_gain", "recomp"],
    "Endurance/Stamina": ["endurance", "pre_workout"],
}

# Focus keywords → muscle groups for per-day exercise lookup
_FOCUS_MUSCLE_MAP: dict[str, list[str]] = {
    "chest":     ["Chest"],
    "push":      ["Chest", "Shoulders", "Upper Arms"],
    "pull":      ["Back", "Upper Arms"],
    "back":      ["Back"],
    "shoulder":  ["Shoulders"],
    "delt":      ["Shoulders"],
    "tricep":    ["Upper Arms"],
    "bicep":     ["Upper Arms"],
    "arm":       ["Upper Arms", "Lower Arms"],
    "leg":       ["Upper Legs", "Lower Legs"],
    "squat":     ["Upper Legs"],
    "glute":     ["Upper Legs"],
    "hamstring": ["Upper Legs"],
    "calf":      ["Lower Legs"],
    "core":      ["Waist"],
    "ab":        ["Waist"],
    "cardio":    ["Cardio"],
    "full body": ["Chest", "Back", "Upper Legs"],
    "upper":     ["Chest", "Back", "Shoulders"],
    "lower":     ["Upper Legs", "Lower Legs"],
}

# Focus keywords → video categories
_FOCUS_VIDEO_MAP: dict[str, list[str]] = {
    "push":      ["muscle_gain", "bulking"],
    "chest":     ["muscle_gain", "bulking"],
    "pull":      ["muscle_gain", "bulking"],
    "back":      ["muscle_gain", "bulking"],
    "shoulder":  ["muscle_gain", "recomp"],
    "leg":       ["muscle_gain", "recomp"],
    "squat":     ["muscle_gain", "recomp"],
    "cardio":    ["endurance", "pre_workout"],
    "full body": ["muscle_gain", "meal_prep"],
    "upper":     ["muscle_gain", "pre_workout"],
    "lower":     ["muscle_gain", "recomp"],
    "core":      ["fat_loss", "meal_prep"],
}

# ── Rate limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Workout Recommendation API", version="3.0.0",
              docs_url="/docs", redoc_url=None)
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
    query: str = Field(..., min_length=10, max_length=1500)

    @field_validator("query")
    @classmethod
    def sanitize(cls, v: str) -> str:
        return re.sub(r"[<>{}\[\]\\]", "", v).strip()


class DayExercise(BaseModel):
    name: str
    gif_url: Optional[str] = None
    muscle_group: Optional[str] = None


class DayVideo(BaseModel):
    id: int
    title: str
    youtube_url: str
    channel_name: Optional[str] = None


class WorkoutDay(BaseModel):
    day: str
    focus: Optional[str] = None
    is_rest: bool
    exercises: list[DayExercise] = []
    videos: list[DayVideo] = []


class RecommendationResponse(BaseModel):
    plan_id: str
    fitness_goal: str
    target_timeframe: Optional[str] = None
    activity_level: Optional[str] = None
    gender: Optional[str] = None
    age_group: Optional[str] = None
    bmi_category: Optional[str] = None
    medical_conditions: Optional[str] = None
    est_calories_burned: int
    schedule: list[WorkoutDay]
    ai_summary: str
    matched_filters: dict[str, str]
    workout_frequency: Optional[str] = None
    target_muscle_groups: Optional[list[str]] = None
    condition_note: Optional[str] = None


# ── Plan text parser ──────────────────────────────────────────────────────────
_SETS_REPS_RE = re.compile(r"\s+\d+[xX×][\d\w\-]+\.?$", re.IGNORECASE)
_DAY_STARTERS = (
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday", "Weekend",
)

# Equipment words that are too common to use alone as ilike search terms
_EQUIPMENT_WORDS = {
    "barbell", "dumbbell", "cable", "machine", "weighted", "assisted",
    "smith", "band", "resistance", "lever", "kettlebell",
}


def _search_keywords(name: str) -> list[str]:
    """Return keywords ordered from most-specific to least-specific."""
    words = [w for w in re.sub(r"[^\w\s]", "", name.lower()).split() if len(w) > 3]
    specific = [w for w in words if w not in _EQUIPMENT_WORDS]
    generic  = [w for w in words if w in _EQUIPMENT_WORDS]
    # Put specific action words first, then equipment fallbacks; longest first within each group
    return (
        sorted(specific, key=len, reverse=True) +
        sorted(generic,  key=len, reverse=True)
    )


def _parse_plan(text: str) -> list[dict]:
    """Parse gym_workout_plan text into a list of day dicts."""
    days: list[dict] = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line or not any(line.startswith(d) for d in _DAY_STARTERS):
            continue
        if ":" not in line:
            continue

        colon = line.index(":")
        header  = line[:colon].strip()        # e.g. "Monday (Push – Chest/Shoulders/Triceps)"
        content = line[colon + 1:].strip()    # e.g. "Barbell Bench Press 4x10, ..."

        # Extract focus from parentheses in the header only
        focus_m = re.search(r"\(([^)]+)\)", header)
        focus   = focus_m.group(1).strip() if focus_m else None
        day_name = re.sub(r"\s*\([^)]*\)", "", header).strip()

        is_rest = bool(re.search(r"\brest\b", content, re.IGNORECASE)) or not content

        exercises: list[dict] = []
        if not is_rest:
            for part in content.split(","):
                display = part.strip().rstrip(".")          # e.g. "Barbell Bench Press 4x10"
                search  = _SETS_REPS_RE.sub("", display).strip()  # e.g. "Barbell Bench Press"
                if search and len(search) > 2:
                    exercises.append({"display": display, "search": search})

        days.append({
            "day":       day_name,
            "focus":     focus,
            "is_rest":   is_rest,
            "exercises": exercises,
        })
    return days


# ── Exercise & video helpers ──────────────────────────────────────────────────
def _focus_to_muscle_groups(focus: Optional[str]) -> list[str]:
    if not focus:
        return []
    fl = focus.lower()
    for keyword, groups in _FOCUS_MUSCLE_MAP.items():
        if keyword in fl:
            return groups
    return []


def _focus_to_video_cats(focus: Optional[str], fitness_goal: str) -> list[str]:
    if focus:
        fl = focus.lower()
        for keyword, cats in _FOCUS_VIDEO_MAP.items():
            if keyword in fl:
                return cats
    return _VIDEO_CAT_MAP.get(fitness_goal, ["muscle_gain"])


async def _find_exercise(
    client: httpx.AsyncClient,
    name: str,
    fallback_muscles: list[str],
    difficulty: str,
) -> Optional[dict]:
    """Look up an exercise by name, falling back to muscle group if needed."""
    try:
        keywords = _search_keywords(name)

        # Try 2-word phrase first (most precise), e.g. "bench press", "lat pulldown"
        if len(keywords) >= 2:
            phrase = f"{keywords[0]} {keywords[1]}"
            r = await client.get(
                f"{SUPABASE_REST}/exercises",
                headers=DB_HEADERS,
                params={"select": "id,name,gif_url,muscle_group",
                        "name": f"ilike.*{phrase}*", "limit": "1"},
            )
            if r.is_success:
                data = r.json()
                rows = data if isinstance(data, list) else []
                if rows:
                    return rows[0]

        # Try top-2 single keywords (specific words first, then equipment words)
        for kw in keywords[:2]:
            r = await client.get(
                f"{SUPABASE_REST}/exercises",
                headers=DB_HEADERS,
                params={"select": "id,name,gif_url,muscle_group",
                        "name": f"ilike.*{kw}*", "limit": "1"},
            )
            if r.is_success:
                data = r.json()
                rows = data if isinstance(data, list) else []
                if rows:
                    return rows[0]

        # Fallback: random exercise from matching muscle group
        for muscle in fallback_muscles:
            r = await client.get(
                f"{SUPABASE_REST}/exercises",
                headers=DB_HEADERS,
                params={"select": "id,name,gif_url,muscle_group",
                        "muscle_group": f"eq.{muscle}",
                        "difficulty": f"eq.{difficulty}",
                        "limit": "5"},
            )
            if r.is_success:
                data = r.json()
                rows = data if isinstance(data, list) else []
                if rows:
                    return random.choice(rows)
    except Exception:
        pass

    return None


async def _fetch_video_pool(
    client: httpx.AsyncClient,
    categories: list[str],
    n: int,
) -> list[dict]:
    if not categories:
        return []
    cat_filter = "(" + ",".join(categories) + ")"
    try:
        r = await client.get(
            f"{SUPABASE_REST}/youtube_videos",
            headers=DB_HEADERS,
            params={"select": "id,title,video_id,channel_name,category",
                    "category": f"in.{cat_filter}", "limit": str(n)},
        )
        if r.is_success:
            data = r.json()
            rows = data if isinstance(data, list) else []
        else:
            rows = []
    except Exception:
        rows = []
    random.shuffle(rows)
    return rows


async def _build_schedule(
    days: list[dict],
    fitness_goal: str,
    activity_level: str,
) -> list[WorkoutDay]:
    """Enrich each day with GIF-backed exercises and unique videos."""
    level_map  = {"Beginner": "beginner", "Intermediate": "intermediate", "Advanced": "advanced"}
    difficulty = level_map.get(activity_level, "beginner")

    active_days = [d for d in days if not d["is_rest"]]

    all_cats: list[str] = []
    for d in active_days:
        all_cats.extend(_focus_to_video_cats(d["focus"], fitness_goal))
    unique_cats = list(dict.fromkeys(all_cats))

    async with httpx.AsyncClient(timeout=12.0) as client:
        video_pool = await _fetch_video_pool(client, unique_cats, max(len(active_days) * 3, 20))

        # Build ALL exercise tasks for ALL active days at once, then run concurrently
        task_index: list[tuple[int, str]] = []   # (day_idx, display_name_with_sets_reps)
        all_ex_tasks = []
        for day_idx, day in enumerate(days):
            if day["is_rest"]:
                continue
            fallback = _focus_to_muscle_groups(day["focus"])
            for ex in day["exercises"]:
                task_index.append((day_idx, ex["display"]))          # keep sets/reps for response
                all_ex_tasks.append(_find_exercise(client, ex["search"], fallback, difficulty))

        raw_results = await asyncio.gather(*all_ex_tasks, return_exceptions=True)

        # Group results by day index
        day_results: dict[int, list[tuple[str, Optional[dict]]]] = {}
        for (day_idx, name), res in zip(task_index, raw_results):
            ex = res if isinstance(res, dict) else None
            day_results.setdefault(day_idx, []).append((name, ex))

        # Assemble final schedule
        schedule: list[WorkoutDay] = []
        for day_idx, day in enumerate(days):
            if day["is_rest"]:
                schedule.append(WorkoutDay(
                    day=day["day"], focus=day["focus"], is_rest=True,
                    exercises=[], videos=[],
                ))
                continue

            seen_ids: set = set()
            day_exercises: list[DayExercise] = []
            for name, result in day_results.get(day_idx, []):
                if result and result.get("id") not in seen_ids:
                    seen_ids.add(result["id"])
                    day_exercises.append(DayExercise(
                        name=name,                      # plan text — includes sets/reps
                        gif_url=result.get("gif_url"),
                        muscle_group=result.get("muscle_group"),
                    ))
                else:
                    day_exercises.append(DayExercise(name=name))

            day_videos: list[DayVideo] = []
            day_cats = set(_focus_to_video_cats(day["focus"], fitness_goal))
            preferred = [v for v in video_pool if v.get("category") in day_cats]
            other     = [v for v in video_pool if v.get("category") not in day_cats]
            for vid in preferred + other:
                if len(day_videos) >= 2:
                    break
                try:
                    video_pool.remove(vid)
                    day_videos.append(DayVideo(
                        id=vid["id"],
                        title=vid["title"],
                        youtube_url=f"https://www.youtube.com/watch?v={vid['video_id']}",
                        channel_name=vid.get("channel_name"),
                    ))
                except Exception:
                    pass

            schedule.append(WorkoutDay(
                day=day["day"], focus=day["focus"], is_rest=False,
                exercises=day_exercises, videos=day_videos,
            ))

    return schedule


# ── BMI helper ────────────────────────────────────────────────────────────────
def _classify_bmi(value: float) -> str:
    if value < 18.5:  return "Underweight"
    if value < 25.0:  return "Normal"
    if value < 30.0:  return "Overweight"
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
  "bmi_value":               numeric BMI if user gives a number (e.g. "BMI 23" → 23.0), else null,
  "bmi_category":            one of {VALID["bmi_category"]} or null — ONLY if user explicitly names it,
  "medical_conditions":      one of {VALID["medical_conditions"]} — map to closest; default "None",
  "raw_medical_conditions":  exact health condition text from query, or null if none mentioned,
  "target_muscle_groups":    list from {VALID_MUSCLE_GROUPS} for muscles user mentions, or [],
  "workout_days_per_week":   integer if user specifies frequency, else null
}}

Rules:
- Do NOT infer age_group unless user states their age.
- Do NOT infer gender unless user states it.
- Do NOT infer bmi_category unless user names a category — use bmi_value for numbers.
- Condition mapping: diabetes → "None" (capture raw); high blood pressure → "Hypertension";
  knee pain → "Knee Issues"; back pain → "Lower Back Pain"; breathing issues → "Asthma";
  heart issues → "Heart Disease". No condition mentioned → "None".
- Muscle groups: chest/pecs/bench → ["Chest"]; back/lats → ["Back"];
  shoulders/delts → ["Shoulders"]; legs/quads/glutes → ["Upper Legs","Lower Legs"];
  arms/biceps/triceps → ["Upper Arms","Lower Arms"]; cardio → ["Cardio"]; full body → [].

Goal mapping: lose weight/burn fat → "Weight Loss"; build muscle/bulk/hypertrophy → "Muscle Gain";
get stronger/powerlifting → "Strength"; endurance/stamina/cardio/run → "Endurance/Stamina".

Return ONLY valid JSON. No markdown, no explanation."""


async def extract_params(query: str) -> dict:
    try:
        resp = await _gemini.aio.models.generate_content(
            model=_GEMINI_MODEL,
            contents=_extract_prompt(query),
            config=genai_types.GenerateContentConfig(
                temperature=0.1, max_output_tokens=512,
                response_mime_type="application/json",
                thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
            ),
        )
        raw: dict = json.loads(resp.text)
    except Exception:
        return {}

    result: dict = {}
    for field in ("fitness_goal", "target_timeframe", "activity_level",
                  "gender", "age_group", "medical_conditions"):
        if raw.get(field) in VALID[field]:
            result[field] = raw[field]

    bmi_val = raw.get("bmi_value")
    if isinstance(bmi_val, (int, float)) and 10 < bmi_val < 60:
        result["bmi_category"] = _classify_bmi(float(bmi_val))
    elif raw.get("bmi_category") in VALID["bmi_category"]:
        result["bmi_category"] = raw["bmi_category"]

    result["raw_medical_conditions"] = raw.get("raw_medical_conditions")
    result["target_muscle_groups"]   = [
        m for m in (raw.get("target_muscle_groups") or []) if m in VALID_MUSCLE_GROUPS
    ]
    freq = raw.get("workout_days_per_week")
    result["workout_days_per_week"] = int(freq) if isinstance(freq, (int, float)) else None
    return result


# ── Gemini: AI summary ────────────────────────────────────────────────────────
async def generate_summary(plan: dict, query: str, params: dict) -> str:
    freq_note  = (f"{params['workout_days_per_week']} days/week as requested"
                  if params.get("workout_days_per_week") else "structured weekly sessions")
    cond_note  = (f"Your condition ({params['raw_medical_conditions']}) has been considered."
                  if params.get("raw_medical_conditions") else "")
    muscle_note = (f"Exercises focus on {', '.join(params['target_muscle_groups'])}."
                   if params.get("target_muscle_groups") else "")
    prompt = (
        f'User asked: "{query}"\n\n'
        f"Plan: {plan['target_timeframe']} {plan['fitness_goal']} for "
        f"{plan['activity_level']}, ~{plan['est_calories_burned']} kcal/week, {freq_note}. "
        f"{cond_note} {muscle_note}\n\n"
        "Write 2-3 friendly, encouraging sentences explaining why this plan suits the user. "
        "Be specific. Do NOT make medical claims or guarantee results."
    )
    try:
        resp = await _gemini.aio.models.generate_content(
            model=_GEMINI_MODEL, contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0.7, max_output_tokens=200,
                thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return resp.text.strip()
    except Exception:
        return (f"This {plan['target_timeframe']} {plan['fitness_goal']} plan is tailored "
                f"for {plan['activity_level']} level.")


# ── Supabase: plan search ─────────────────────────────────────────────────────
async def search_plans(params: dict) -> tuple[list[dict], dict[str, str]]:
    db_params = {k: v for k, v in params.items()
                 if k in _FILTER_PRIORITY and v is not None}
    async with httpx.AsyncClient(timeout=10.0) as client:
        for drop in range(len(_FILTER_PRIORITY) + 1):
            keep   = _FILTER_PRIORITY[: max(1, len(_FILTER_PRIORITY) - drop)]
            active = {k: db_params[k] for k in keep if k in db_params}
            if not active:
                break
            qp = {"select": "*", "limit": "5"}
            for k, v in active.items():
                qp[k] = f"eq.{v}"
            r = await client.get(f"{SUPABASE_REST}/workout_recommendations",
                                 headers=DB_HEADERS, params=qp)
            if r.is_success and r.json():
                return r.json(), active
    return [], {}


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", include_in_schema=False)
async def health():
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendationResponse)
@limiter.limit("10/minute")
async def recommend(request: Request, body: QueryRequest):
    # 1. Extract params
    params = await extract_params(body.query)
    if not params.get("fitness_goal"):
        raise HTTPException(status_code=422, detail=(
            "Could not determine a fitness goal. "
            "Try 'lose weight', 'build muscle', 'get stronger', or 'improve stamina'."
        ))

    # 2. Condition note for unmapped conditions
    raw_cond = params.get("raw_medical_conditions")
    db_cond  = params.get("medical_conditions")
    condition_note: Optional[str] = None
    if raw_cond and db_cond == "None":
        condition_note = (
            f"Note: '{raw_cond}' was noted but isn't in our filter set. "
            "The AI summary has been personalised to account for it."
        )

    # 3. Find matching plan
    plans, matched_filters = await search_plans(params)
    if not plans:
        raise HTTPException(status_code=404, detail=(
            "No matching workout plan found. Try specifying your goal and timeframe."
        ))
    plan = random.choice(plans[:3])

    # 4. Parse plan text → days, build schedule, generate summary — concurrently
    days = _parse_plan(plan["gym_workout_plan"])

    schedule, ai_summary = await asyncio.gather(
        _build_schedule(days, plan["fitness_goal"], plan["activity_level"]),
        generate_summary(plan, body.query, params),
    )

    freq = params.get("workout_days_per_week")
    return RecommendationResponse(
        plan_id=plan["plan_id"],
        fitness_goal=plan["fitness_goal"],
        target_timeframe=params.get("target_timeframe"),
        activity_level=params.get("activity_level"),
        gender=params.get("gender"),
        age_group=params.get("age_group"),
        bmi_category=params.get("bmi_category"),
        medical_conditions=db_cond if db_cond and db_cond != "None" else None,
        est_calories_burned=plan["est_calories_burned"],
        schedule=schedule,
        ai_summary=ai_summary,
        matched_filters=matched_filters,
        workout_frequency=f"{freq} days/week" if freq else None,
        target_muscle_groups=params.get("target_muscle_groups") or None,
        condition_note=condition_note,
    )
