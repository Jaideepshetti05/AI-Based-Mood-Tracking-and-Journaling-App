import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, Optional

from .config import Settings

logger = logging.getLogger(__name__)

EKMAN_LABELS = ("anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise")

# ---------------------------------------------------------------------------
# Negation preprocessing
# ---------------------------------------------------------------------------
# The j-hartmann/emotion-english-distilroberta-base model, like most
# fine-tuned classifiers, is trained predominantly on affirmative sentences.
# Negated constructs such as "I am not angry" or "I am not happy" are rare in
# training corpora, so the model latches onto the emotion keyword and ignores
# the negation.  We apply a targeted rewrite before inference to flip the
# dominant surface keyword into a sentiment-neutral or opposite phrase.
#
# This is NOT a full negation resolver; it handles the most common patterns
# that appeared problematic in our diagnostic test set (Tests 9 and 12).
# ---------------------------------------------------------------------------

# Pairs: (negated_pattern, replacement_phrase)
# Patterns are applied in order; all are case-insensitive.
_NEGATION_REWRITES = [
    # "not angry" / "no longer angry" / "not angry anymore"
    (r"\bnot\s+angry\b",          "calm"),
    (r"\bno\s+longer\s+angry\b",  "at peace"),
    (r"\bnot\s+angry\s+anymore\b","peaceful"),
    (r"\bno\s+longer\s+mad\b",    "calm"),
    # "not happy" / "not very happy"
    (r"\bnot\s+(very\s+)?happy\b",          "unhappy"),
    (r"\bdo\s+not\s+feel\s+proud\b",        "feel disappointed"),
    (r"\bdo\s+not\s+feel\s+happy\b",        "feel unhappy"),
    (r"\bdid\s+not\s+feel\s+proud\b",       "felt disappointed"),
    # "not sad" / "not scared"
    (r"\bnot\s+sad\b",            "fine"),
    (r"\bnot\s+scared\b",         "calm"),
    (r"\bnot\s+afraid\b",         "calm"),
    (r"\bnot\s+frightened\b",     "calm"),
    # Generic "not [emotion]"
    (r"\bnot\s+excited\b",        "unenthusiastic"),
    (r"\bnot\s+disgusted\b",      "fine"),
    (r"\bnot\s+surprised\b",      "unsurprised"),
    # "feel peaceful" / "feel at peace" — reinforce these as positive
    (r"\bfeel\s+peaceful\b",      "feel serene"),
    (r"\bfeel\s+at\s+peace\b",    "feel serene"),
    (r"\bforgiven\b",             "at peace"),
]

# Pre-compile for performance
_NEGATION_PATTERNS = [(re.compile(pat, re.IGNORECASE), repl) for pat, repl in _NEGATION_REWRITES]


def _apply_negation_preprocessing(text: str) -> str:
    """Rewrite negated emotion phrases so the transformer receives surface
    emotion keywords that match the intended sentiment direction.

    Returns the rewritten text.  The original text is used for logging only;
    the rewrite is intentionally minimal — only targeted negation patterns are
    changed, not the full grammatical structure.
    """
    rewritten = text
    for pattern, replacement in _NEGATION_PATTERNS:
        rewritten = pattern.sub(replacement, rewritten)
    if rewritten != text:
        logger.debug("Negation preprocessing applied; original length %d, rewritten length %d", len(text), len(rewritten))
    return rewritten


class AnalysisService:
    """Runs VADER sentiment analysis and an optional transformer emotion classifier.

    Expensive model loading is deferred to the FastAPI lifespan handler so
    module import is non-blocking. Returns the complete emotion score
    distribution to support transparent Bayesian personalization.

    Model loading is done with TRANSFORMERS_OFFLINE=1 when the model is
    already cached locally to prevent network hangs during startup.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._vader = None
        self._classifier = None
        self._model_name = "vader + heuristic emotion fallback"
        self._initialized = False
        self._is_loading = False

        # Lightweight VADER initialization — happens at import time
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

            # Use offline mode when the model is already in the local HF cache
            # to avoid network timeouts that slow startup or cause errors.
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

            # top_k=None returns the complete distribution across all classes
            self._classifier = pipeline(
                "text-classification",
                model=self.settings.emotion_model_name,
                top_k=None,
                device=-1,  # CPU; set device=0 for GPU
            )
            self._model_name = self.settings.emotion_model_name
            logger.info("Emotion classification model loaded: %s", self._model_name)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            logger.warning(
                "Emotion model unavailable; using heuristic fallback (%s: %s)",
                type(exc).__name__,
                exc,
            )
            self._classifier = None
            self._model_name = "vader + heuristic emotion fallback"
        finally:
            self._is_loading = False

    def analyze(self, text: str) -> "Analysis":
        # ------------------------------------------------------------------
        # 1. Sentiment polarity via VADER (always runs)
        # ------------------------------------------------------------------
        if self._vader:
            compound = float(self._vader.polarity_scores(text)["compound"])
        else:
            lowered = text.lower()
            positive = sum(word in lowered for word in ("good", "happy", "great", "love", "calm"))
            negative = sum(word in lowered for word in ("bad", "sad", "angry", "stress", "worry"))
            compound = max(-1.0, min(1.0, (positive - negative) / 5))
        sentiment = "positive" if compound >= 0.05 else "negative" if compound <= -0.05 else "neutral"

        # ------------------------------------------------------------------
        # 2. Emotion classification via transformer (when available)
        # ------------------------------------------------------------------
        emotion = "neutral"
        confidence: Optional[float] = None
        secondary_emotion: Optional[str] = None
        secondary_confidence: Optional[float] = None
        is_fallback = False
        ambiguity_note: Optional[str] = None
        raw_distribution: Dict[str, float] = {}

        if self._classifier:
            try:
                # Apply negation-aware preprocessing before inference.
                # The rewrite converts negated emotion phrases into equivalent
                # non-negated phrases that the model understands correctly.
                inference_text = _apply_negation_preprocessing(text[:10000])

                results = self._classifier(inference_text)
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
                        # Report secondary emotion if it is at least 20% or
                        # within 25 percentage points of the primary emotion.
                        if sec_score >= 0.20 or (confidence - sec_score <= 0.25):
                            secondary_emotion = sec_label
                            secondary_confidence = sec_score
                            ambiguity_note = (
                                f"Mixed feelings detected; {emotion} ({round(confidence*100)}%) "
                                f"and {sec_label} ({round(sec_score*100)}%) both present."
                            )

                if confidence is not None and confidence < 0.60 and not ambiguity_note:
                    ambiguity_note = (
                        "Model confidence is moderate (<60%); complex or subtle phrasing "
                        "may vary in interpretation."
                    )

            except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                logger.warning("Emotion inference failed (%s); using heuristic fallback", type(exc).__name__)
                is_fallback = True
        else:
            is_fallback = True

        # ------------------------------------------------------------------
        # 3. Heuristic fallback (when transformer is unavailable)
        #    Maps VADER sentiment to the most likely Ekman emotion.
        #    Confidence is explicitly None to signal this is an estimate.
        # ------------------------------------------------------------------
        if is_fallback:
            if sentiment == "positive":
                emotion = "joy"
                raw_distribution = {
                    "joy": 0.65, "neutral": 0.20, "surprise": 0.05,
                    "sadness": 0.03, "anger": 0.03, "fear": 0.02, "disgust": 0.02,
                }
            elif sentiment == "negative":
                emotion = "sadness"
                raw_distribution = {
                    "sadness": 0.65, "neutral": 0.15, "anger": 0.10,
                    "fear": 0.05, "disgust": 0.03, "surprise": 0.01, "joy": 0.01,
                }
            else:
                emotion = "neutral"
                raw_distribution = {
                    "neutral": 0.70, "joy": 0.10, "sadness": 0.10,
                    "surprise": 0.04, "anger": 0.02, "fear": 0.02, "disgust": 0.02,
                }
            confidence = None
            ambiguity_note = (
                "Neural emotion model offline; emotion estimated from VADER sentiment polarity. "
                "Emotion detection is limited to joy / sadness / neutral in this mode."
            )

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
