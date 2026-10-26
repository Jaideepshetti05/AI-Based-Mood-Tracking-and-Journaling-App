import logging
from dataclasses import dataclass

from .config import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Analysis:
    sentiment: str
    compound_score: float
    emotion: str
    confidence: float | None
    model_name: str


class AnalysisService:
    """Runs VADER and an optional transformer without logging user text."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._vader = None
        self._classifier = None
        self._model_name = "vader + emotion fallback"
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            self._vader = SentimentIntensityAnalyzer()
        except ImportError:
            logger.warning("vaderSentiment is unavailable; using a small sentiment fallback")
        if settings.enable_emotion_model:
            self._load_classifier()

    def _load_classifier(self) -> None:
        try:
            from transformers import pipeline

            self._classifier = pipeline(
                "text-classification",
                model=self.settings.emotion_model_name,
                top_k=1,
                device=-1,
            )
            self._model_name = self.settings.emotion_model_name
        except (ImportError, OSError, RuntimeError, ValueError) as exc:  # model downloads are optional and external
            logger.warning("Emotion model unavailable; using fallback (%s)", type(exc).__name__)
            self._classifier = None

    def analyze(self, text: str) -> Analysis:
        if self._vader:
            compound = float(self._vader.polarity_scores(text)["compound"])
        else:
            lowered = text.lower()
            positive = sum(word in lowered for word in ("good", "happy", "great", "love", "calm"))
            negative = sum(word in lowered for word in ("bad", "sad", "angry", "stress", "worry"))
            compound = max(-1.0, min(1.0, (positive - negative) / 5))
        sentiment = "positive" if compound >= 0.05 else "negative" if compound <= -0.05 else "neutral"

        emotion = "neutral"
        confidence = None
        if self._classifier:
            try:
                result = self._classifier(text[:10000])[0]
                if isinstance(result, list):
                    result = result[0]
                emotion = str(result["label"]).lower()
                confidence = round(float(result["score"]), 4)
            except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                logger.warning("Emotion inference failed (%s)", type(exc).__name__)
        elif sentiment == "positive":
            emotion = "joy"
        elif sentiment == "negative":
            emotion = "sadness"
        return Analysis(sentiment, round(compound, 4), emotion, confidence, self._model_name)
