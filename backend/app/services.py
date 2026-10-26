import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

from .config import Settings

logger = logging.getLogger(__name__)

EKMAN_LABELS = ("anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise")


@dataclass(frozen=True)
class Analysis:
    sentiment: str
    compound_score: float
    emotion: str
    confidence: Optional[float]
    model_name: str
    secondary_emotion: Optional[str] = None
    secondary_confidence: Optional[float] = None
    is_fallback: bool = False
    ambiguity_note: Optional[str] = None
    raw_distribution: Dict[str, float] = field(default_factory=dict)


class AnalysisService:
    """Runs VADER and an optional transformer without logging user text.
    
    Expensive model loading is deferred to the FastAPI lifespan handler
    so module import is non-blocking. Returns complete emotion score distribution
    to support transparent Bayesian personalization.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._vader = None
        self._classifier = None
        self._model_name = "vader + emotion fallback"
        self._initialized = False
        self._is_loading = False

        # Lightweight VADER initialization
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader = SentimentIntensityAnalyzer()
        except ImportError:
            logger.warning("vaderSentiment is unavailable; using a lexical fallback")

    def initialize(self) -> None:
        """Called during FastAPI lifespan startup to load heavy models once."""
        if self._initialized:
            return
        if self.settings.enable_emotion_model:
            self._load_classifier()
        self._initialized = True

    def _load_classifier(self) -> None:
        self._is_loading = True
        try:
            from transformers import pipeline

            # top_k=None returns complete distribution across all classes
            self._classifier = pipeline(
                "text-classification",
                model=self.settings.emotion_model_name,
                top_k=None,
                device=-1,
            )
            self._model_name = self.settings.emotion_model_name
            logger.info("Emotion classification model loaded: %s", self._model_name)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            logger.warning("Emotion model unavailable; using heuristic fallback (%s)", type(exc).__name__)
            self._classifier = None
            self._model_name = "vader + heuristic emotion fallback"
        finally:
            self._is_loading = False

    def analyze(self, text: str) -> Analysis:
        # 1. Sentiment polarity via VADER
        if self._vader:
            compound = float(self._vader.polarity_scores(text)["compound"])
        else:
            lowered = text.lower()
            positive = sum(word in lowered for word in ("good", "happy", "great", "love", "calm"))
            negative = sum(word in lowered for word in ("bad", "sad", "angry", "stress", "worry"))
            compound = max(-1.0, min(1.0, (positive - negative) / 5))
        sentiment = "positive" if compound >= 0.05 else "negative" if compound <= -0.05 else "neutral"

        # 2. Emotion classification
        emotion = "neutral"
        confidence = None
        secondary_emotion = None
        secondary_confidence = None
        is_fallback = False
        ambiguity_note = None
        raw_distribution: Dict[str, float] = {}

        if self._classifier:
            try:
                # Truncate to 10000 chars for safety
                results = self._classifier(text[:10000])
                if results and isinstance(results[0], list):
                    all_preds = results[0]
                elif results and isinstance(results, list):
                    all_preds = results
                else:
                    all_preds = []

                if all_preds:
                    # Sort predictions descending by score
                    sorted_preds = sorted(all_preds, key=lambda x: x["score"], reverse=True)
                    for item in sorted_preds:
                        lbl = str(item["label"]).lower()
                        raw_distribution[lbl] = round(float(item["score"]), 4)

                    primary = sorted_preds[0]
                    emotion = str(primary["label"]).lower()
                    confidence = round(float(primary["score"]), 4)

                    if len(sorted_preds) > 1:
                        sec = sorted_preds[1]
                        sec_label = str(sec["label"]).lower()
                        sec_score = round(float(sec["score"]), 4)
                        if sec_score >= 0.20 or (confidence - sec_score <= 0.25):
                            secondary_emotion = sec_label
                            secondary_confidence = sec_score
                            ambiguity_note = f"Mixed feelings detected; {emotion} ({round(confidence*100)}%) and {sec_label} ({round(sec_score*100)}%) both present."

                if confidence is not None and confidence < 0.60 and not ambiguity_note:
                    ambiguity_note = "Model confidence is moderate (<60%); complex or subtle phrasing may vary in interpretation."

            except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                logger.warning("Emotion inference failed (%s); using heuristic fallback", type(exc).__name__)
                is_fallback = True
        else:
            is_fallback = True

        if is_fallback:
            if sentiment == "positive":
                emotion = "joy"
                raw_distribution = {"joy": 0.65, "neutral": 0.20, "surprise": 0.05, "sadness": 0.03, "anger": 0.03, "fear": 0.02, "disgust": 0.02}
            elif sentiment == "negative":
                emotion = "sadness"
                raw_distribution = {"sadness": 0.65, "neutral": 0.15, "anger": 0.10, "fear": 0.05, "disgust": 0.03, "surprise": 0.01, "joy": 0.01}
            else:
                emotion = "neutral"
                raw_distribution = {"neutral": 0.70, "joy": 0.10, "sadness": 0.10, "surprise": 0.04, "anger": 0.02, "fear": 0.02, "disgust": 0.02}
            confidence = None
            ambiguity_note = "Neural model offline; estimated using basic sentiment heuristic."

        return Analysis(
            sentiment=sentiment,
            compound_score=round(compound, 4),
            emotion=emotion,
            confidence=confidence,
            model_name=self._model_name,
            secondary_emotion=secondary_emotion,
            secondary_confidence=secondary_confidence,
            is_fallback=is_fallback,
            ambiguity_note=ambiguity_note,
            raw_distribution=raw_distribution,
        )
