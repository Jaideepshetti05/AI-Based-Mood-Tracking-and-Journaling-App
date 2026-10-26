from datetime import datetime
from typing import Literal

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
    confidence: float | None = None
    model_name: str
    limitation: str


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text must not be blank")
        return value.strip()


class EntryCreate(BaseModel):
    text: str = Field(min_length=1, max_length=10000)
    mood: str
    note: str | None = Field(default=None, max_length=500)
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
    corrected_emotion: str | None = None

    @field_validator("corrected_emotion")
    @classmethod
    def validate_emotion(cls, value: str | None) -> str | None:
        if value is not None and value not in EMOTIONS:
            raise ValueError(f"Emotion must be one of: {', '.join(EMOTIONS)}")
        return value


class CorrectionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    helpful: bool
    corrected_emotion: str | None
    created_at: datetime


class EntryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    text: str
    mood: str
    note: str | None
    created_at: datetime
    updated_at: datetime
    analysis: AnalysisResult | None = None
    correction: CorrectionResponse | None = None


class Distribution(BaseModel):
    label: str
    count: int


class InsightsResponse(BaseModel):
    period: str
    from_date: datetime
    to_date: datetime
    entry_count: int
    most_frequent_mood: str | None
    mood_distribution: list[Distribution]
    sentiment_distribution: list[Distribution]
    emotion_distribution: list[Distribution]
    agreement_count: int
    correction_count: int
    agreement_rate: float | None
    personalization: dict[str, float | int | str] | None
