import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from .models import Analysis, Correction, JournalEntry, UserPreference

logger = logging.getLogger(__name__)

EKMAN_LABELS = ("anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise")


@dataclass
class UserPriors:
    user_id: str
    total_feedback_events: int
    emotion_counts: Counter = field(default_factory=Counter)
    confusion_counts: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    last_updated: Optional[datetime] = None


@dataclass
class PersonalizationResult:
    original_emotion: str
    original_confidence: Optional[float]
    personalized_emotion: str
    personalized_confidence: Optional[float]
    applied: bool
    explanation: str
    all_personalized_scores: Dict[str, float] = field(default_factory=dict)


class PersonalizationEngine:
    """Lightweight, privacy-preserving Bayesian personalization engine.
    
    Adjusts model emotion scores strictly based on a specific user's past feedback history.
    Enforces a minimum evidence threshold (N_min) before activation and applies bounded
    shrinkage to prevent over-adjustment from small sample sizes.
    """

    def __init__(self, min_feedback_threshold: int = 5, max_beta: float = 0.65, half_life_k: float = 10.0):
        self.min_feedback_threshold = min_feedback_threshold
        self.max_beta = max_beta
        self.half_life_k = half_life_k

    def compute_user_priors(
        self,
        db: Session,
        user_id: str,
        before_time: Optional[datetime] = None,
    ) -> UserPriors:
        """Extracts feedback statistics strictly prior to before_time to prevent future data leakage."""
        query = (
            select(Correction)
            .join(Analysis, Correction.analysis_id == Analysis.id)
            .join(JournalEntry, Analysis.entry_id == JournalEntry.id)
            .where(JournalEntry.user_id == user_id)
        )
        if before_time:
            query = query.where(JournalEntry.created_at < before_time)

        corrections = db.scalars(query.options(joinedload(Correction.analysis))).all()

        total = len(corrections)
        emotion_counts = Counter()
        confusion = defaultdict(Counter)
        last_dt = None

        for c in corrections:
            pred_emo = c.analysis.emotion.lower().strip() if c.analysis else "neutral"
            if c.helpful:
                # User confirmed the model's prediction
                target_emo = pred_emo
            else:
                # User corrected the model's prediction
                target_emo = (c.corrected_emotion or "neutral").lower().strip()

            emotion_counts[target_emo] += 1
            confusion[pred_emo][target_emo] += 1
            if c.created_at and (last_dt is None or c.created_at > last_dt):
                last_dt = c.created_at

        return UserPriors(
            user_id=user_id,
            total_feedback_events=total,
            emotion_counts=emotion_counts,
            confusion_counts=confusion,
            last_updated=last_dt,
        )

    def calibrate(
        self,
        base_emotion: str,
        base_confidence: Optional[float],
        base_distribution: Dict[str, float],
        user_priors: UserPriors,
        enabled: bool = True,
        threshold_override: Optional[int] = None,
    ) -> PersonalizationResult:
        """Calibrates prediction scores using Bayesian prior shifting and empirical confusion weighting.
        
        Guarantees:
        - If disabled or feedback < threshold, returns base prediction with applied=False.
        - Bounded shrinkage: beta smoothly scales with sample size and never exceeds max_beta.
        - The original prediction and scores are strictly preserved.
        """
        N = user_priors.total_feedback_events
        threshold = threshold_override if threshold_override is not None else self.min_feedback_threshold

        if not enabled:
            return PersonalizationResult(
                original_emotion=base_emotion,
                original_confidence=base_confidence,
                personalized_emotion=base_emotion,
                personalized_confidence=base_confidence,
                applied=False,
                explanation="Personalization disabled in user settings.",
                all_personalized_scores=dict(base_distribution),
            )

        if N < threshold:
            return PersonalizationResult(
                original_emotion=base_emotion,
                original_confidence=base_confidence,
                personalized_emotion=base_emotion,
                personalized_confidence=base_confidence,
                applied=False,
                explanation=f"Personalization inactive: Insufficient evidence ({N}/{threshold} feedback events recorded).",
                all_personalized_scores=dict(base_distribution),
            )

        # 1. Compute empirical shrinkage factor beta(N)
        beta = self.max_beta * (N / (N + self.half_life_k))

        # 2. Build full label set (canonical Ekman + any user corrected labels present in base distribution)
        labels = list(base_distribution.keys()) if base_distribution else list(EKMAN_LABELS)
        K = len(labels)
        uniform_prior = 1.0 / K

        # 3. Calculate smoothed user prior P_user(e) and confusion transition T(e | e_base)
        user_counts = user_priors.emotion_counts
        total_user_events = sum(user_counts.values()) or 1
        alpha = 1.0  # Laplace smoothing for prior

        pred_confusion = user_priors.confusion_counts.get(base_emotion.lower(), Counter())
        total_pred_conf_events = sum(pred_confusion.values()) or 1
        gamma = 0.5  # Laplace smoothing for confusion transition

        personalized_logits = {}
        for emo in labels:
            p_base = max(1e-6, base_distribution.get(emo, 1e-4))
            base_logit = math.log(p_base)

            # Prior adjustment log-ratio
            p_user = (user_counts.get(emo, 0) + alpha) / (total_user_events + alpha * K)
            delta_prior = math.log(p_user / uniform_prior)

            # Confusion adjustment log-ratio
            t_user = (pred_confusion.get(emo, 0) + gamma) / (total_pred_conf_events + gamma * K)
            delta_conf = math.log(t_user / uniform_prior)

            # Composite adjusted logit
            personalized_logits[emo] = base_logit + beta * (0.6 * delta_prior + 0.4 * delta_conf)

        # 4. Softmax normalization over adjusted logits
        max_logit = max(personalized_logits.values())
        exp_logits = {emo: math.exp(val - max_logit) for emo, val in personalized_logits.items()}
        sum_exp = sum(exp_logits.values()) or 1.0
        personalized_scores = {emo: round(exp_logits[emo] / sum_exp, 4) for emo in labels}

        # 5. Extract top personalized emotion
        top_emo = max(personalized_scores.keys(), key=lambda k: personalized_scores[k])
        top_conf = personalized_scores[top_emo]

        # 6. Evaluate whether adjustment meaningfully shifted the decision
        is_shifted = top_emo.lower() != base_emotion.lower()
        conf_diff = abs((top_conf or 0.0) - (base_confidence or 0.0))
        applied = is_shifted or (conf_diff >= 0.05)

        if is_shifted:
            explanation = (
                f"Personalized: Adjusted from '{base_emotion}' to '{top_emo}' "
                f"based on your correction history ({N} feedback events, beta={beta:.2f})."
            )
        elif applied:
            explanation = (
                f"Personalized: Confirmed '{base_emotion}' with calibrated confidence "
                f"({round(top_conf * 100)}%) using {N} past feedback events."
            )
        else:
            explanation = f"Consistent with personal baseline ({N} feedback events); base prediction retained."

        return PersonalizationResult(
            original_emotion=base_emotion,
            original_confidence=base_confidence,
            personalized_emotion=top_emo,
            personalized_confidence=top_conf,
            applied=applied,
            explanation=explanation,
            all_personalized_scores=personalized_scores,
        )

    def get_personalization_status(self, db: Session, user_id: str) -> Dict[str, Any]:
        """Returns the current calibration status and statistics for a given user."""
        pref = db.scalar(select(UserPreference).where(UserPreference.user_id == user_id))
        enabled = pref.personalization_enabled if pref else True
        threshold = pref.min_feedback_threshold if pref else self.min_feedback_threshold

        priors = self.compute_user_priors(db, user_id)
        total_feedback = priors.total_feedback_events
        confirmations = sum(
            1 for c in db.scalars(
                select(Correction)
                .join(Analysis, Correction.analysis_id == Analysis.id)
                .join(JournalEntry, Analysis.entry_id == JournalEntry.id)
                .where(JournalEntry.user_id == user_id)
            ).all() if c.helpful
        )
        corrections_count = total_feedback - confirmations
        active = enabled and (total_feedback >= threshold)

        total_pers = len(
            db.scalars(
                select(Analysis)
                .join(JournalEntry, Analysis.entry_id == JournalEntry.id)
                .where(JournalEntry.user_id == user_id, Analysis.personalization_applied.is_(True))
            ).all()
        )

        top_emotions = [e for e, count in priors.emotion_counts.most_common(3)]

        if not enabled:
            explanation = "Personalization disabled in settings."
        elif not active:
            explanation = f"Building evidence: {total_feedback}/{threshold} reviews recorded."
        else:
            explanation = f"Calibrated using {total_feedback} feedback reviews ({corrections_count} corrections)."

        return {
            "user_id": user_id,
            "enabled": enabled,
            "threshold": threshold,
            "feedback_count": total_feedback,
            "confirmations_count": confirmations,
            "corrections_count": corrections_count,
            "active": active,
            "total_entries_personalized": total_pers,
            "explanation": explanation,
            "top_corrected_emotions": top_emotions,
        }
