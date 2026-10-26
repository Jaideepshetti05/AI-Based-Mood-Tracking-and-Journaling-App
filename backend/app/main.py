import csv
import io
import logging
from pathlib import Path
from collections import Counter
from datetime import datetime, timedelta, timezone

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from .config import get_settings
from .database import get_db, init_db
from .models import Analysis, Correction, JournalEntry
from .schemas import (
    AnalyzeRequest, AnalysisResult, CorrectionCreate, CorrectionResponse, Distribution,
    EntryCreate, EntryResponse, EntryUpdate, InsightsResponse,
)
from .services import AnalysisService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
settings = get_settings()
analysis_service = AnalysisService(settings)
app = FastAPI(title="Mood Journal API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origin_list, allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def startup() -> None:
    init_db()


def current_user(x_user_id: str | None = Header(default=None)) -> str:
    # Local-first deployments use an explicit stable device identifier. No cross-user data is exposed.
    return (x_user_id or "local-user").strip()[:128]


def to_analysis_result(analysis: Analysis | None) -> AnalysisResult | None:
    if analysis is None:
        return None
    return AnalysisResult(
        sentiment=analysis.sentiment, compound_score=analysis.compound_score, emotion=analysis.emotion,
        confidence=analysis.confidence, model_name=analysis.model_name,
        limitation="Predictions are estimates, not a diagnosis. Journal context and mixed emotions can reduce reliability.",
    )


def to_entry_response(entry: JournalEntry) -> EntryResponse:
    correction = entry.analysis.correction if entry.analysis and entry.analysis.correction else None
    return EntryResponse(
        id=entry.id, text=entry.text, mood=entry.mood, note=entry.note, created_at=entry.created_at,
        updated_at=entry.updated_at, analysis=to_analysis_result(entry.analysis),
        correction=CorrectionResponse.model_validate(correction) if correction else None,
    )


def get_entry(db: Session, entry_id: int, user_id: str) -> JournalEntry:
    entry = db.scalar(
        select(JournalEntry).options(joinedload(JournalEntry.analysis).joinedload(Analysis.correction))
        .where(JournalEntry.id == entry_id, JournalEntry.user_id == user_id)
    )
    if not entry:
        raise HTTPException(status_code=404, detail="Journal entry not found")
    return entry


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/model/info")
def model_info() -> dict[str, str | bool]:
    return {"model": analysis_service._model_name, "emotion_model_available": analysis_service._classifier is not None,
            "sentiment_model": "VADER"}


@app.post("/api/analyze", response_model=AnalysisResult)
def analyze(payload: AnalyzeRequest) -> AnalysisResult:
    return to_analysis_result(
        Analysis(
            id=0, entry_id=0, sentiment=(result := analysis_service.analyze(payload.text)).sentiment,
            compound_score=result.compound_score, emotion=result.emotion, confidence=result.confidence,
            model_name=result.model_name,
        )
    )


@app.post("/api/entries", response_model=EntryResponse, status_code=status.HTTP_201_CREATED)
def create_entry(payload: EntryCreate, db: Session = Depends(get_db), user_id: str = Depends(current_user)) -> EntryResponse:
    entry = JournalEntry(user_id=user_id, text=payload.text, mood=payload.mood, note=payload.note)
    if payload.analyze:
        result = analysis_service.analyze(payload.text)
        entry.analysis = Analysis(
            sentiment=result.sentiment, compound_score=result.compound_score, emotion=result.emotion,
            confidence=result.confidence, model_name=result.model_name,
        )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return to_entry_response(get_entry(db, entry.id, user_id))


@app.get("/api/entries", response_model=list[EntryResponse])
def list_entries(
    mood: str | None = Query(default=None), from_date: datetime | None = None, to_date: datetime | None = None,
    db: Session = Depends(get_db), user_id: str = Depends(current_user),
) -> list[EntryResponse]:
    query = select(JournalEntry).options(joinedload(JournalEntry.analysis).joinedload(Analysis.correction)).where(
        JournalEntry.user_id == user_id
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
def update_entry(entry_id: int, payload: EntryUpdate, db: Session = Depends(get_db), user_id: str = Depends(current_user)) -> EntryResponse:
    entry = get_entry(db, entry_id, user_id)
    entry.text, entry.mood, entry.note = payload.text, payload.mood, payload.note
    if payload.analyze:
        result = analysis_service.analyze(payload.text)
        if entry.analysis is None:
            entry.analysis = Analysis(
                sentiment=result.sentiment, compound_score=result.compound_score, emotion=result.emotion,
                confidence=result.confidence, model_name=result.model_name,
            )
        else:
            entry.analysis.sentiment = result.sentiment
            entry.analysis.compound_score = result.compound_score
            entry.analysis.emotion = result.emotion
            entry.analysis.confidence = result.confidence
            entry.analysis.model_name = result.model_name
            entry.analysis.correction = None
    else:
        entry.analysis = None
    db.commit()
    return to_entry_response(get_entry(db, entry_id, user_id))


@app.delete("/api/entries/{entry_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_entry(entry_id: int, db: Session = Depends(get_db), user_id: str = Depends(current_user)) -> Response:
    entry = get_entry(db, entry_id, user_id)
    db.delete(entry)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/api/entries/{entry_id}/correction", response_model=CorrectionResponse)
def correct_entry(
    entry_id: int, payload: CorrectionCreate, db: Session = Depends(get_db), user_id: str = Depends(current_user)
) -> CorrectionResponse:
    entry = get_entry(db, entry_id, user_id)
    if not entry.analysis:
        raise HTTPException(status_code=409, detail="Analyze the entry before correcting its prediction")
    if not payload.helpful and not payload.corrected_emotion:
        raise HTTPException(status_code=422, detail="A corrected emotion is required when prediction is not helpful")
    correction = entry.analysis.correction or Correction(analysis_id=entry.analysis.id)
    correction.helpful, correction.corrected_emotion = payload.helpful, payload.corrected_emotion
    db.add(correction)
    db.commit()
    db.refresh(correction)
    return CorrectionResponse.model_validate(correction)


def insights(period: str, days: int, db: Session, user_id: str) -> InsightsResponse:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    entries = db.scalars(
        select(JournalEntry).options(joinedload(JournalEntry.analysis).joinedload(Analysis.correction))
        .where(JournalEntry.user_id == user_id, JournalEntry.created_at >= start, JournalEntry.created_at < end)
    ).unique().all()
    moods = Counter(entry.mood for entry in entries)
    sentiments = Counter(entry.analysis.sentiment for entry in entries if entry.analysis)
    emotions = Counter(entry.analysis.emotion for entry in entries if entry.analysis)
    corrections = [entry.analysis.correction for entry in entries if entry.analysis and entry.analysis.correction]
    agreement = sum(c.helpful for c in corrections)
    personalization = None
    if corrections:
        personalization = {"labeled_examples": len(corrections), "helpful_rate": round(agreement / len(corrections), 4),
                           "status": "learning" if len(corrections) < 5 else "measured"}
    return InsightsResponse(
        period=period, from_date=start, to_date=end, entry_count=len(entries),
        most_frequent_mood=moods.most_common(1)[0][0] if moods else None,
        mood_distribution=[Distribution(label=k, count=v) for k, v in moods.items()],
        sentiment_distribution=[Distribution(label=k, count=v) for k, v in sentiments.items()],
        emotion_distribution=[Distribution(label=k, count=v) for k, v in emotions.items()],
        agreement_count=agreement, correction_count=len(corrections),
        agreement_rate=round(agreement / len(corrections), 4) if corrections else None,
        personalization=personalization,
    )


@app.get("/api/insights/weekly", response_model=InsightsResponse)
def weekly_insights(db: Session = Depends(get_db), user_id: str = Depends(current_user)) -> InsightsResponse:
    return insights("weekly", 7, db, user_id)


@app.get("/api/insights/monthly", response_model=InsightsResponse)
def monthly_insights(db: Session = Depends(get_db), user_id: str = Depends(current_user)) -> InsightsResponse:
    return insights("monthly", 30, db, user_id)


@app.get("/api/export")
def export_entries(db: Session = Depends(get_db), user_id: str = Depends(current_user)) -> Response:
    entries = db.scalars(select(JournalEntry).where(JournalEntry.user_id == user_id).order_by(JournalEntry.created_at)).all()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "created_at", "mood", "text", "note"])
    writer.writerows((entry.id, entry.created_at.isoformat(), entry.mood, entry.text, entry.note or "") for entry in entries)
    return Response(content=output.getvalue(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=mood-journal.csv"})


@app.delete("/api/data", status_code=status.HTTP_204_NO_CONTENT)
def delete_all_data(db: Session = Depends(get_db), user_id: str = Depends(current_user)) -> Response:
    entries = db.scalars(select(JournalEntry).where(JournalEntry.user_id == user_id)).all()
    for entry in entries:
        db.delete(entry)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# Keep API routes ahead of the catch-all desktop UI mount.
app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="desktop")
