from datetime import datetime
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

MOODS = ("happy", "sad", "stressed", "anxious", "angry", "calm", "excited", "neutral")
EMOTIONS = (
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral",
)


class AnalysisResult(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    compound_score: float
    emotion: str
    confidence: Optional[float] = None
    model_name: str
    limitation: str
    secondary_emotion: Optional[str] = None
    secondary_confidence: Optional[float] = None
    is_fallback: bool = False
    ambiguity_note: Optional[str] = None

    # Transparent personalization tracking
    original_emotion: Optional[str] = None
    original_confidence: Optional[float] = None
    personalized_emotion: Optional[str] = None
    personalized_confidence: Optional[float] = None
    personalization_applied: bool = False
    personalization_explanation: Optional[str] = None


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)
    apply_personalization: bool = True

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text must not be blank")
        return value.strip()


class EntryCreate(BaseModel):
    text: str = Field(min_length=1, max_length=10000)
    mood: str
    note: Optional[str] = Field(default=None, max_length=500)
    analyze: bool = True

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Journal text must not be blank")
        return value.strip()

    @field_validator("mood")
    @classmethod
    def validate_mood(cls, value: str) -> str:
        value = value.lower().strip()
        if value not in MOODS:
            raise ValueError(f"Mood must be one of: {', '.join(MOODS)}")
        return value


class EntryUpdate(EntryCreate):
    pass


class CorrectionCreate(BaseModel):
    helpful: bool
    corrected_emotion: Optional[str] = None

    @field_validator("corrected_emotion")
    @classmethod
    def validate_emotion(cls, value: Optional[str]) -> Optional[str]:
        if value is not None:
            value = value.lower().strip()
            if value not in EMOTIONS:
                raise ValueError(f"Emotion must be one of: {', '.join(EMOTIONS)}")
        return value


class CorrectionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    helpful: bool
    corrected_emotion: Optional[str]
    created_at: datetime


class EntryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    text: str
    mood: str
    note: Optional[str]
    created_at: datetime
    updated_at: datetime
    analysis: Optional[AnalysisResult] = None
    correction: Optional[CorrectionResponse] = None


class Distribution(BaseModel):
    label: str
    count: int


class DailyTrend(BaseModel):
    date: str
    count: int
    top_mood: Optional[str] = None


class InsightsResponse(BaseModel):
    period: str
    from_date: datetime
    to_date: datetime
    entry_count: int
    most_frequent_mood: Optional[str]
    most_frequent_emotion: Optional[str] = None
    mood_distribution: list[Distribution]
    sentiment_distribution: list[Distribution]
    emotion_distribution: list[Distribution]
    daily_trend: list[DailyTrend] = []
    alignment_count: int = 0
    divergence_count: int = 0
    agreement_count: int
    correction_count: int
    agreement_rate: Optional[float]
    personalization: Optional[dict[str, Any]] = None


class AuthSessionResponse(BaseModel):
    user_id: str
    session_token: str
    message: str


class PersonalizationStatusResponse(BaseModel):
    user_id: str
    enabled: bool
    threshold: int
    feedback_count: int
    confirmations_count: int = 0
    corrections_count: int = 0
    active: bool
    total_entries_personalized: int = 0
    explanation: str
    top_corrected_emotions: list[str] = []


class PersonalizationToggleRequest(BaseModel):
    enabled: bool
