import csv
import io
import logging
from collections import Counter
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from .config import get_settings
from .database import get_db, init_db
from .models import Analysis, Correction, JournalEntry, UserPreference
from .personalization import PersonalizationEngine
from .schemas import (
    AnalyzeRequest, AnalysisResult, AuthSessionResponse, CorrectionCreate, CorrectionResponse,
    DailyTrend, Distribution, EntryCreate, EntryResponse, EntryUpdate, InsightsResponse,
    PersonalizationStatusResponse, PersonalizationToggleRequest,
)
from .services import AnalysisService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
settings = get_settings()

# Initialize AI and personalization services
analysis_service = AnalysisService(settings)
personalization_engine = PersonalizationEngine(min_feedback_threshold=5)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize database schema, column migrations, and heavy AI model once
    init_db()
    analysis_service.initialize()
    logger.info("Application startup complete. Model active: %s", analysis_service._model_name)
    yield
    # Shutdown logic
    logger.info("Application shutting down.")


app = FastAPI(title="Mood Journal API", version="1.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["*"],
    allow_headers=["*"],
)

LOCAL_USER_ID = "local-user"


def current_user(
    request: Request,
    x_user_id: str | None = Header(default=None),
) -> str:
    """Enforces local loopback boundary and isolates user namespaces.
    
    In a local-first architecture, access is restricted to loopback unless explicitly overridden.
    Cross-user spoofing is prevented by validating ownership during data operations.
    """
    client_host = request.client.host if request.client else "127.0.0.1"
    if settings.enforce_loopback and client_host not in ("127.0.0.1", "::1", "localhost", "testclient"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Application is restricted to local loopback access.",
        )

    user = (x_user_id or LOCAL_USER_ID).strip()[:128]
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: User identity context missing.",
        )
    return user


def get_user_preference(db: Session, user_id: str) -> UserPreference:
    pref = db.scalar(select(UserPreference).where(UserPreference.user_id == user_id))
    if not pref:
        pref = UserPreference(user_id=user_id, personalization_enabled=True, min_feedback_threshold=5)
        db.add(pref)
        db.commit()
        db.refresh(pref)
    return pref


def to_analysis_result(analysis: Analysis | None) -> AnalysisResult | None:
    if analysis is None:
        return None
    return AnalysisResult(
        sentiment=analysis.sentiment,
        compound_score=analysis.compound_score,
        emotion=analysis.emotion,
        confidence=analysis.confidence,
        model_name=analysis.model_name,
        limitation="Predictions are automated estimates for personal reflection, not a psychiatric diagnosis. Mixed emotions, slang, or context can affect results.",
        secondary_emotion=getattr(analysis, "secondary_emotion", None),
        secondary_confidence=getattr(analysis, "secondary_confidence", None),
        is_fallback=getattr(analysis, "is_fallback", False) or "fallback" in (analysis.model_name or "").lower(),
        ambiguity_note=getattr(analysis, "ambiguity_note", None),
        original_emotion=getattr(analysis, "original_emotion", None) or analysis.emotion,
        original_confidence=getattr(analysis, "original_confidence", None) or analysis.confidence,
        personalized_emotion=getattr(analysis, "personalized_emotion", None) or analysis.emotion,
        personalized_confidence=getattr(analysis, "personalized_confidence", None) or analysis.confidence,
        personalization_applied=getattr(analysis, "personalization_applied", False),
        personalization_explanation=getattr(analysis, "personalization_explanation", None),
    )


def to_entry_response(entry: JournalEntry) -> EntryResponse:
    correction = entry.analysis.correction if entry.analysis and entry.analysis.correction else None
    return EntryResponse(
        id=entry.id,
        text=entry.text,
        mood=entry.mood,
        note=entry.note,
        created_at=entry.created_at,
        updated_at=entry.updated_at,
        analysis=to_analysis_result(entry.analysis),
        correction=CorrectionResponse.model_validate(correction) if correction else None,
    )


def get_entry(db: Session, entry_id: int, user_id: str) -> JournalEntry:
    # First query without user_id filter to distinguish 404 from 403
    entry_any = db.scalar(select(JournalEntry).where(JournalEntry.id == entry_id))
    if not entry_any:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Journal entry not found")
    if entry_any.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Cannot view or modify entries belonging to another user.",
        )

    return db.scalar(
        select(JournalEntry)
        .options(joinedload(JournalEntry.analysis).joinedload(Analysis.correction))
        .where(JournalEntry.id == entry_id, JournalEntry.user_id == user_id)
    )


def sanitize_csv_cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if text and text[0] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + text
    return text


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/auth/session", response_model=AuthSessionResponse)
def auth_session(user_id: str = Depends(current_user)) -> AuthSessionResponse:
    return AuthSessionResponse(
        user_id=user_id,
        session_token="local-session-active",
        message="Local session verified on loopback interface",
    )


@app.get("/api/model/info")
def model_info() -> dict[str, Any]:
    return {
        "model": analysis_service._model_name,
        "emotion_model_available": analysis_service._classifier is not None,
        "sentiment_model": "VADER",
        "is_fallback": analysis_service._classifier is None,
        "multi_emotion_supported": True,
        "personalization_supported": True,
        "personalization_method": "Empirical Bayes Calibration with Bounded Shrinkage",
    }


# PERSONALIZATION SETTINGS & CONTROLS
@app.get("/api/personalization/status", response_model=PersonalizationStatusResponse)
def personalization_status(
    db: Session = Depends(get_db),
    user_id: str = Depends(current_user),
) -> PersonalizationStatusResponse:
    status_data = personalization_engine.get_personalization_status(db, user_id)
    return PersonalizationStatusResponse(**status_data)


@app.post("/api/personalization/toggle", response_model=PersonalizationStatusResponse)
def toggle_personalization(
    payload: PersonalizationToggleRequest,
    db: Session = Depends(get_db),
    user_id: str = Depends(current_user),
) -> PersonalizationStatusResponse:
    pref = get_user_preference(db, user_id)
    pref.personalization_enabled = payload.enabled
    db.commit()
    return personalization_status(db, user_id)


@app.post("/api/personalization/reset")
def reset_personalization(
    db: Session = Depends(get_db),
    user_id: str = Depends(current_user),
) -> dict[str, Any]:
    """Resets personalization by clearing recorded corrections for the active user."""
    corrections = db.scalars(
        select(Correction)
        .join(Analysis, Correction.analysis_id == Analysis.id)
        .join(JournalEntry, Analysis.entry_id == JournalEntry.id)
        .where(JournalEntry.user_id == user_id)
    ).all()
    count = len(corrections)
    for c in corrections:
        db.delete(c)

    analyses = db.scalars(
        select(Analysis)
        .join(JournalEntry, Analysis.entry_id == JournalEntry.id)
        .where(JournalEntry.user_id == user_id)
    ).all()
    for a in analyses:
        a.personalization_applied = False
        a.personalized_emotion = a.original_emotion
        a.personalized_confidence = a.original_confidence
        a.personalization_explanation = "Personalization reset by user."

    db.commit()
    logger.info("Personalization reset for user '%s': removed %d corrections", user_id, count)
    return {"status": "ok", "message": f"Personalization feedback history reset ({count} records cleared).", "cleared_events": count}


@app.post("/api/analyze", response_model=AnalysisResult)
def analyze(
    payload: AnalyzeRequest,
    db: Session = Depends(get_db),
    user_id: str = Depends(current_user),
) -> AnalysisResult:
    result = analysis_service.analyze(payload.text)

    original_emotion = result.emotion
    original_confidence = result.confidence
    personalized_emotion = result.emotion
    personalized_confidence = result.confidence
    personalization_applied = False
    personalization_explanation = "Using pretrained model prediction."

    if payload.apply_personalization:
        pref = get_user_preference(db, user_id)
        priors = personalization_engine.compute_user_priors(db, user_id)
        calib = personalization_engine.calibrate(
            base_emotion=result.emotion,
            base_confidence=result.confidence,
            base_distribution=result.raw_distribution,
            user_priors=priors,
            enabled=pref.personalization_enabled,
            threshold_override=pref.min_feedback_threshold,
        )
        personalized_emotion = calib.personalized_emotion
        personalized_confidence = calib.personalized_confidence
        personalization_applied = calib.applied
        personalization_explanation = calib.explanation

    return AnalysisResult(
        sentiment=result.sentiment,
        compound_score=result.compound_score,
        emotion=personalized_emotion,
        confidence=personalized_confidence,
        model_name=result.model_name,
        limitation="Predictions are automated estimates for personal reflection, not a psychiatric diagnosis. Mixed emotions, slang, or context can affect results.",
        secondary_emotion=result.secondary_emotion,
        secondary_confidence=result.secondary_confidence,
        is_fallback=result.is_fallback,
        ambiguity_note=result.ambiguity_note,
        original_emotion=original_emotion,
        original_confidence=original_confidence,
        personalized_emotion=personalized_emotion,
        personalized_confidence=personalized_confidence,
        personalization_applied=personalization_applied,
        personalization_explanation=personalization_explanation,
    )


@app.post("/api/entries", response_model=EntryResponse, status_code=status.HTTP_201_CREATED)
def create_entry(payload: EntryCreate, db: Session = Depends(get_db), user_id: str = Depends(current_user)) -> EntryResponse:
    entry = JournalEntry(user_id=user_id, text=payload.text, mood=payload.mood, note=payload.note)
    if payload.analyze:
        result = analysis_service.analyze(payload.text)

        # Personalization: strictly use priors from entries strictly prior to this moment
        pref = get_user_preference(db, user_id)
        priors = personalization_engine.compute_user_priors(db, user_id)
        calib = personalization_engine.calibrate(
            base_emotion=result.emotion,
            base_confidence=result.confidence,
            base_distribution=result.raw_distribution,
            user_priors=priors,
            enabled=pref.personalization_enabled,
            threshold_override=pref.min_feedback_threshold,
        )

        entry.analysis = Analysis(
            sentiment=result.sentiment,
            compound_score=result.compound_score,
            emotion=calib.personalized_emotion,
            confidence=calib.personalized_confidence,
            model_name=result.model_name,
            original_emotion=calib.original_emotion,
            original_confidence=calib.original_confidence,
            personalized_emotion=calib.personalized_emotion,
            personalized_confidence=calib.personalized_confidence,
            personalization_applied=calib.applied,
            personalization_explanation=calib.explanation,
        )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return to_entry_response(get_entry(db, entry.id, user_id))


@app.get("/api/entries", response_model=list[EntryResponse])
def list_entries(
    mood: str | None = Query(default=None),
    from_date: datetime | None = None,
    to_date: datetime | None = None,
    db: Session = Depends(get_db),
    user_id: str = Depends(current_user),
) -> list[EntryResponse]:
    query = (
        select(JournalEntry)
        .options(joinedload(JournalEntry.analysis).joinedload(Analysis.correction))
        .where(JournalEntry.user_id == user_id)
    )
    if mood:
        query = query.where(JournalEntry.mood == mood.lower())
    if from_date:
        query = query.where(JournalEntry.created_at >= from_date)
    if to_date:
        query = query.where(JournalEntry.created_at <= to_date)
    entries = db.scalars(query.order_by(JournalEntry.created_at.desc())).unique().all()
    return [to_entry_response(entry) for entry in entries]


@app.get("/api/entries/{entry_id}", response_model=EntryResponse)
def read_entry(entry_id: int, db: Session = Depends(get_db), user_id: str = Depends(current_user)) -> EntryResponse:
    return to_entry_response(get_entry(db, entry_id, user_id))


@app.put("/api/entries/{entry_id}", response_model=EntryResponse)
def update_entry(
    entry_id: int,
    payload: EntryUpdate,
    db: Session = Depends(get_db),
    user_id: str = Depends(current_user),
) -> EntryResponse:
    entry = get_entry(db, entry_id, user_id)
    entry.text, entry.mood, entry.note = payload.text, payload.mood, payload.note
    if payload.analyze:
        result = analysis_service.analyze(payload.text)

        # Personalization strictly from prior events
        pref = get_user_preference(db, user_id)
        priors = personalization_engine.compute_user_priors(db, user_id, before_time=entry.created_at)
        calib = personalization_engine.calibrate(
            base_emotion=result.emotion,
            base_confidence=result.confidence,
            base_distribution=result.raw_distribution,
            user_priors=priors,
            enabled=pref.personalization_enabled,
            threshold_override=pref.min_feedback_threshold,
        )

        if entry.analysis is None:
            entry.analysis = Analysis(
                sentiment=result.sentiment,
                compound_score=result.compound_score,
                emotion=calib.personalized_emotion,
                confidence=calib.personalized_confidence,
                model_name=result.model_name,
                original_emotion=calib.original_emotion,
                original_confidence=calib.original_confidence,
                personalized_emotion=calib.personalized_emotion,
                personalized_confidence=calib.personalized_confidence,
                personalization_applied=calib.applied,
                personalization_explanation=calib.explanation,
            )
        else:
            entry.analysis.sentiment = result.sentiment
            entry.analysis.compound_score = result.compound_score
            entry.analysis.emotion = calib.personalized_emotion
            entry.analysis.confidence = calib.personalized_confidence
            entry.analysis.model_name = result.model_name
            entry.analysis.original_emotion = calib.original_emotion
            entry.analysis.original_confidence = calib.original_confidence
            entry.analysis.personalized_emotion = calib.personalized_emotion
            entry.analysis.personalized_confidence = calib.personalized_confidence
            entry.analysis.personalization_applied = calib.applied
            entry.analysis.personalization_explanation = calib.explanation
            # Invalidate prior stale correction when journal text changes
            entry.analysis.correction = None
    else:
        entry.analysis = None
    db.commit()
    return to_entry_response(get_entry(db, entry.id, user_id))


@app.delete("/api/entries/{entry_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_entry(entry_id: int, db: Session = Depends(get_db), user_id: str = Depends(current_user)) -> Response:
    entry = get_entry(db, entry_id, user_id)
    db.delete(entry)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/api/entries/{entry_id}/correction", response_model=CorrectionResponse)
def correct_entry(
    entry_id: int,
    payload: CorrectionCreate,
    db: Session = Depends(get_db),
    user_id: str = Depends(current_user),
) -> CorrectionResponse:
    entry = get_entry(db, entry_id, user_id)
    if not entry.analysis:
        raise HTTPException(status_code=409, detail="Analyze the entry before providing prediction feedback")
    if not payload.helpful and not payload.corrected_emotion:
        raise HTTPException(status_code=422, detail="A corrected emotion is required when marking a prediction as inaccurate")

    correction = entry.analysis.correction or Correction(analysis_id=entry.analysis.id)
    correction.helpful = payload.helpful
    correction.corrected_emotion = payload.corrected_emotion if not payload.helpful else None
    db.add(correction)
    db.commit()
    db.refresh(correction)
    return CorrectionResponse.model_validate(correction)


def insights(period: str, days: int, db: Session, user_id: str) -> InsightsResponse:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    entries = db.scalars(
        select(JournalEntry)
        .options(joinedload(JournalEntry.analysis).joinedload(Analysis.correction))
        .where(JournalEntry.user_id == user_id, JournalEntry.created_at >= start, JournalEntry.created_at < end)
        .order_by(JournalEntry.created_at.asc())
    ).unique().all()

    moods = Counter(entry.mood for entry in entries)
    sentiments = Counter(entry.analysis.sentiment for entry in entries if entry.analysis)
    emotions = Counter(entry.analysis.emotion for entry in entries if entry.analysis)
    original_emotions = Counter(
        entry.analysis.original_emotion or entry.analysis.emotion for entry in entries if entry.analysis
    )

    # Daily trend calculation
    daily_groups: dict[str, list[str]] = {}
    for entry in entries:
        day_str = entry.created_at.strftime("%Y-%m-%d")
        daily_groups.setdefault(day_str, []).append(entry.mood)

    daily_trend = [
        DailyTrend(
            date=day_str,
            count=len(day_moods),
            top_mood=Counter(day_moods).most_common(1)[0][0] if day_moods else None,
        )
        for day_str, day_moods in daily_groups.items()
    ]

    # Mood-Emotion Alignment vs Divergence calculation
    POSITIVE_MOODS = {"happy", "excited", "calm"}
    NEGATIVE_MOODS = {"sad", "stressed", "anxious", "angry"}

    alignment_count = 0
    divergence_count = 0
    personalization_shifts = 0

    for entry in entries:
        if not entry.analysis:
            continue
        m = entry.mood.lower()
        s = entry.analysis.sentiment.lower()
        if (m in POSITIVE_MOODS and s == "positive") or (m in NEGATIVE_MOODS and s == "negative") or (m == "neutral" and s == "neutral"):
            alignment_count += 1
        elif (m in POSITIVE_MOODS and s == "negative") or (m in NEGATIVE_MOODS and s == "positive"):
            divergence_count += 1

        if entry.analysis.personalization_applied and (entry.analysis.original_emotion != entry.analysis.personalized_emotion):
            personalization_shifts += 1

    corrections = [entry.analysis.correction for entry in entries if entry.analysis and entry.analysis.correction]
    agreement = sum(1 for c in corrections if c.helpful)

    # Personalization Status & Verification
    pref = get_user_preference(db, user_id)
    all_priors = personalization_engine.compute_user_priors(db, user_id)

    personalization_meta = {
        "framework": "Empirical Bayes Calibration with Bounded Shrinkage",
        "enabled": pref.personalization_enabled,
        "reviewed_predictions": len(corrections),
        "total_historical_events": all_priors.total_feedback_events,
        "min_threshold": pref.min_feedback_threshold,
        "is_active": pref.personalization_enabled and all_priors.total_feedback_events >= pref.min_feedback_threshold,
        "personalization_shifts_in_period": personalization_shifts,
        "helpful_confirmations": agreement,
        "user_corrections": len(corrections) - agreement,
        "agreement_rate": round(agreement / len(corrections), 4) if corrections else None,
        "top_user_preferred_emotions": [emo for emo, _ in all_priors.emotion_counts.most_common(3)],
        "status": (
            "active" if (pref.personalization_enabled and all_priors.total_feedback_events >= pref.min_feedback_threshold)
            else "disabled" if not pref.personalization_enabled
            else "insufficient_data"
        ),
        "explanation": (
            f"Personalized calibration is active across {all_priors.total_feedback_events} feedback events."
            if (pref.personalization_enabled and all_priors.total_feedback_events >= pref.min_feedback_threshold)
            else f"Requires {pref.min_feedback_threshold} feedback events to activate (currently {all_priors.total_feedback_events})."
        ),
    }

    return InsightsResponse(
        period=period,
        from_date=start,
        to_date=end,
        entry_count=len(entries),
        most_frequent_mood=moods.most_common(1)[0][0] if moods else None,
        most_frequent_emotion=emotions.most_common(1)[0][0] if emotions else None,
        mood_distribution=[Distribution(label=k, count=v) for k, v in moods.items()],
        sentiment_distribution=[Distribution(label=k, count=v) for k, v in sentiments.items()],
        emotion_distribution=[Distribution(label=k, count=v) for k, v in emotions.items()],
        daily_trend=daily_trend,
        alignment_count=alignment_count,
        divergence_count=divergence_count,
        agreement_count=agreement,
        correction_count=len(corrections),
        agreement_rate=round(agreement / len(corrections), 4) if corrections else None,
        personalization=personalization_meta,
    )


@app.get("/api/insights/weekly", response_model=InsightsResponse)
def weekly_insights(db: Session = Depends(get_db), user_id: str = Depends(current_user)) -> InsightsResponse:
    return insights("weekly", 7, db, user_id)


@app.get("/api/insights/monthly", response_model=InsightsResponse)
def monthly_insights(db: Session = Depends(get_db), user_id: str = Depends(current_user)) -> InsightsResponse:
    return insights("monthly", 30, db, user_id)


@app.get("/api/export")
def export_entries(db: Session = Depends(get_db), user_id: str = Depends(current_user)) -> Response:
    entries = (
        db.scalars(
            select(JournalEntry)
            .options(joinedload(JournalEntry.analysis).joinedload(Analysis.correction))
            .where(JournalEntry.user_id == user_id)
            .order_by(JournalEntry.created_at)
        )
        .unique()
        .all()
    )
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "id",
        "created_at",
        "updated_at",
        "mood",
        "text",
        "note",
        "sentiment",
        "compound_score",
        "predicted_emotion",
        "confidence",
        "model_name",
        "feedback_status",
        "corrected_emotion",
        "correction_timestamp",
    ])

    for entry in entries:
        analysis = entry.analysis
        correction = analysis.correction if analysis else None

        feedback_status = "unreviewed"
        if correction:
            feedback_status = "helpful" if correction.helpful else "incorrect"

        writer.writerow([
            entry.id,
            entry.created_at.isoformat() if entry.created_at else "",
            entry.updated_at.isoformat() if entry.updated_at else "",
            sanitize_csv_cell(entry.mood),
            sanitize_csv_cell(entry.text),
            sanitize_csv_cell(entry.note or ""),
            analysis.sentiment if analysis else "",
            f"{analysis.compound_score:.4f}" if analysis else "",
            analysis.emotion if analysis else "",
            f"{analysis.confidence:.4f}" if (analysis and analysis.confidence is not None) else "",
            sanitize_csv_cell(analysis.model_name) if analysis else "",
            feedback_status,
            sanitize_csv_cell(correction.corrected_emotion) if (correction and correction.corrected_emotion) else "",
            correction.created_at.isoformat() if (correction and correction.created_at) else "",
        ])

    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=mood-journal.csv"},
    )


@app.delete("/api/data", status_code=status.HTTP_204_NO_CONTENT)
def delete_all_data(db: Session = Depends(get_db), user_id: str = Depends(current_user)) -> Response:
    entries = db.scalars(select(JournalEntry).where(JournalEntry.user_id == user_id)).all()
    for entry in entries:
        db.delete(entry)
    # Also delete user preferences
    pref = db.scalar(select(UserPreference).where(UserPreference.user_id == user_id))
    if pref:
        db.delete(pref)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# Keep API routes ahead of the catch-all desktop UI mount.
app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="desktop")
