from __future__ import annotations

from datetime import datetime, timezone
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class JournalEntry(Base):
    __tablename__ = "journal_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[str] = mapped_column(String(128), index=True)
    text: Mapped[str] = mapped_column(Text)
    mood: Mapped[str] = mapped_column(String(32))
    note: Mapped[str] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)
    analysis = relationship(
        "Analysis",
        back_populates="entry", cascade="all, delete-orphan", uselist=False
    )


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    entry_id: Mapped[int] = mapped_column(ForeignKey("journal_entries.id", ondelete="CASCADE"), unique=True)
    sentiment: Mapped[str] = mapped_column(String(16))
    compound_score: Mapped[float] = mapped_column(Float)
    emotion: Mapped[str] = mapped_column(String(64))
    confidence: Mapped[float] = mapped_column(Float, nullable=True)
    model_name: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    # Personalization tracking fields
    original_emotion: Mapped[str] = mapped_column(String(64), nullable=True)
    original_confidence: Mapped[float] = mapped_column(Float, nullable=True)
    personalized_emotion: Mapped[str] = mapped_column(String(64), nullable=True)
    personalized_confidence: Mapped[float] = mapped_column(Float, nullable=True)
    personalization_applied: Mapped[bool] = mapped_column(Boolean, default=False)
    personalization_explanation: Mapped[str] = mapped_column(Text, nullable=True)

    entry: Mapped[JournalEntry] = relationship(back_populates="analysis")
    correction = relationship(
        "Correction",
        back_populates="analysis", cascade="all, delete-orphan", uselist=False
    )


class Correction(Base):
    __tablename__ = "corrections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    analysis_id: Mapped[int] = mapped_column(ForeignKey("analyses.id", ondelete="CASCADE"), unique=True)
    helpful: Mapped[bool] = mapped_column(Boolean)
    corrected_emotion: Mapped[str] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    analysis: Mapped[Analysis] = relationship(back_populates="correction")


class UserPreference(Base):
    __tablename__ = "user_preferences"

    user_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    personalization_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    min_feedback_threshold: Mapped[int] = mapped_column(Integer, default=5)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)
