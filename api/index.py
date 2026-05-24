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
_SUPABASE_BASE = os.environ["SUPABASE_URL"].rstrip("/")          # https://xxx.supabase.co
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

# ── Allowed category values (mirrors DB CHECK constraints) ────────────────────
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

# Most → least important for progressive filter relaxation
_FILTER_PRIORITY = [
    "fitness_goal", "activity_level", "target_timeframe",
    "medical_conditions", "gender", "bmi_category", "age_group",
]

# ── Rate limiter ──────────────────────────────────────────────────────────────
# NOTE: slowapi uses in-memory storage by default. On Vercel (stateless
# serverless), each warm instance has its own counter, so limits are
# per-instance, not global. For stricter enforcement use an Upstash Redis
# backend: Limiter(..., storage_uri="redis://...")
limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Workout Recommendation API",
    version="1.0.0",
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
    query: str = Field(..., min_length=10, max_length=500, description="Natural language fitness query")

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
    target_timeframe: str
    activity_level: str
    gender: str
    age_group: str
    bmi_category: str
    medical_conditions: str
    gym_workout_plan: str
    est_calories_burned: int
    exercises: list[Exercise]
    videos: list[Video]
    ai_summary: str
    matched_filters: dict[str, str]


# ── Gemini: extract structured params from free-text query ────────────────────
def _extract_prompt(query: str) -> str:
    return f"""You are a fitness profile extractor. Given a natural-language fitness query, \
extract structured parameters.

User query: "{query}"

Return ONLY a valid JSON object with these exact fields (use null if not determinable):
{{
  "fitness_goal":       one of {VALID["fitness_goal"]} or null,
  "target_timeframe":   one of {VALID["target_timeframe"]} or null,
  "activity_level":     one of {VALID["activity_level"]} or null,
  "gender":             one of {VALID["gender"]} or null,
  "age_group":          one of {VALID["age_group"]} or null,
  "bmi_category":       one of {VALID["bmi_category"]} or null,
  "medical_conditions": one of {VALID["medical_conditions"]} or null
}}

Mapping hints:
- lose weight / burn fat / slim / cut → "Weight Loss"
- build muscle / bulk / gain mass / hypertrophy → "Muscle Gain"
- get stronger / powerlifting / strength training → "Strength"
- endurance / stamina / cardio / marathon / run → "Endurance/Stamina"
- newbie / just starting / never worked out → "Beginner"
- some gym experience / worked out before → "Intermediate"
- years of training / advanced lifter → "Advanced"
- 1 month / 4 weeks → "4 Weeks"
- 2 months / 8 weeks → "8 Weeks"
- 3 months / 12 weeks / quarter → "12 Weeks"
- 6 months / half year → "6 Months"

Return ONLY valid JSON. No markdown fences, no explanation."""


async def extract_params(query: str) -> dict[str, str]:
    """Ask Gemini to parse the query; validate every value against the whitelist."""
    try:
        resp = await _gemini.aio.models.generate_content(
            model=_GEMINI_MODEL,
            contents=_extract_prompt(query),
            config=genai_types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=512,
                # Force clean JSON output — no markdown fences, no preamble
                response_mime_type="application/json",
                # Disable thinking for this deterministic extraction task
                thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
            ),
        )
        raw: dict = json.loads(resp.text)
    except Exception:
        return {}

    # Whitelist-validate every field — never trust LLM output directly
    return {
        field: raw[field]
        for field, allowed in VALID.items()
        if raw.get(field) in allowed
    }


# ── Gemini: personalized summary ──────────────────────────────────────────────
async def generate_summary(plan: dict, query: str) -> str:
    prompt = (
        f'User asked: "{query}"\n\n'
        f"Matched plan: {plan['target_timeframe']} {plan['fitness_goal']} program "
        f"for a {plan['activity_level']} ({plan['age_group']}), "
        f"medical notes: {plan['medical_conditions']}, "
        f"~{plan['est_calories_burned']} kcal/week.\n\n"
        "Write 2–3 friendly, encouraging sentences explaining why this plan suits the user. "
        "Be specific to their goal. Do NOT make medical claims or guarantee results."
    )
    try:
        resp = await _gemini.aio.models.generate_content(
            model=_GEMINI_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=200,
            ),
        )
        return resp.text.strip()
    except Exception:
        return (
            f"This {plan['target_timeframe']} {plan['fitness_goal']} plan is tailored "
            f"for {plan['activity_level']} level and accounts for your individual profile."
        )


# ── Supabase helpers ──────────────────────────────────────────────────────────
async def search_plans(
    params: dict[str, str],
) -> tuple[list[dict], dict[str, str]]:
    """Query workout_recommendations with progressive filter relaxation."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        for drop in range(len(_FILTER_PRIORITY) + 1):
            keep = _FILTER_PRIORITY[: max(1, len(_FILTER_PRIORITY) - drop)]
            active = {k: params[k] for k in keep if k in params}
            if not active:
                break

            qp: dict[str, str] = {"select": "*", "limit": "5"}
            for k, v in active.items():
                qp[k] = f"eq.{v}"

            r = await client.get(
                f"{SUPABASE_REST}/workout_recommendations",
                headers=DB_HEADERS,
                params=qp,
            )
            if r.is_success and r.json():
                return r.json(), active

    return [], {}


async def fetch_exercises(ids: list[str]) -> list[dict]:
    if not ids:
        return []
    id_filter = "(" + ",".join(f'"{i}"' for i in ids[:5]) + ")"
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(
            f"{SUPABASE_REST}/exercises",
            headers=DB_HEADERS,
            params={
                "select": "id,name,gif_url,muscle_group",
                "id": f"in.{id_filter}",
            },
        )
    return r.json() if r.is_success else []


async def fetch_videos(ids: list[int]) -> list[dict]:
    if not ids:
        return []
    id_filter = "(" + ",".join(str(i) for i in ids[:3]) + ")"
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(
            f"{SUPABASE_REST}/youtube_videos",
            headers=DB_HEADERS,
            params={
                "select": "id,title,youtube_url,channel_name",
                "id": f"in.{id_filter}",
            },
        )
    return r.json() if r.is_success else []


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", include_in_schema=False)
async def health():
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendationResponse)
@limiter.limit("10/minute")
async def recommend(request: Request, body: QueryRequest):
    # 1. Parse natural-language query into structured filters
    params = await extract_params(body.query)
    if not params.get("fitness_goal"):
        raise HTTPException(
            status_code=422,
            detail=(
                "Could not determine a fitness goal from your query. "
                "Try including phrases like 'lose weight', 'build muscle', "
                "'get stronger', or 'improve stamina'."
            ),
        )

    # 2. Find a matching plan (relaxes filters progressively until a row is found)
    plans, matched_filters = await search_plans(params)
    if not plans:
        raise HTTPException(
            status_code=404,
            detail="No matching workout plan found. Try specifying your goal and timeframe.",
        )

    # Pick randomly from up to the top 3 matches to add variety
    plan = random.choice(plans[:3])

    # 3. Fetch linked exercises (GIFs) + videos + AI summary — all in parallel
    ex_ids  = plan.get("related_exercise_ids") or []
    vid_ids = plan.get("related_video_ids") or []

    exercises, videos, ai_summary = await asyncio.gather(
        fetch_exercises(ex_ids),
        fetch_videos(vid_ids),
        generate_summary(plan, body.query),
    )

    return RecommendationResponse(
        plan_id=plan["plan_id"],
        fitness_goal=plan["fitness_goal"],
        target_timeframe=plan["target_timeframe"],
        activity_level=plan["activity_level"],
        gender=plan["gender"],
        age_group=plan["age_group"],
        bmi_category=plan["bmi_category"],
        medical_conditions=plan["medical_conditions"],
        gym_workout_plan=plan["gym_workout_plan"],
        est_calories_burned=plan["est_calories_burned"],
        exercises=[Exercise(**e) for e in exercises],
        videos=[Video(**v) for v in videos],
        ai_summary=ai_summary,
        matched_filters=matched_filters,
    )
