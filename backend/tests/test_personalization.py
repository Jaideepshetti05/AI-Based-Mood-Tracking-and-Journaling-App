"""
test_personalization.py
-----------------------
Automated tests for the Empirical Bayes Personalization Engine.

Covers:
1. Personalization disabled.
2. Insufficient feedback threshold (N < 5).
3. Active personalization on valid correction history (N >= 5).
4. Prediction score adjustment on borderline predictions.
5. Prevention of extreme adjustments (high confidence predictions protected).
6. Cross-user isolation (User A feedback does not alter User B).
7. Resetting personalization via API.
8. Personalization toggle via API.
9. Persistence of preference and feedback records.
10. Strict prevention of future-data leakage (temporal causality).
"""

from datetime import datetime, timezone, timedelta
import pytest
from app.personalization import PersonalizationEngine, UserPriors


def test_personalization_disabled_leaves_predictions_unchanged():
    engine = PersonalizationEngine(min_feedback_threshold=5)
    
    # Simulate a user with 10 corrections towards 'anger'
    priors = UserPriors(
        user_id="user1",
        total_feedback_events=10,
        emotion_counts={"anger": 10},
    )
    
    base_dist = {"joy": 0.50, "anger": 0.30, "neutral": 0.20}
    
    res = engine.calibrate(
        base_emotion="joy",
        base_confidence=0.50,
        base_distribution=base_dist,
        user_priors=priors,
        enabled=False,
    )
    
    assert res.applied is False
    assert res.personalized_emotion == "joy"
    assert res.personalized_confidence == 0.50
    assert "disabled" in res.explanation.lower()


def test_insufficient_feedback_threshold_gates_personalization():
    engine = PersonalizationEngine(min_feedback_threshold=5)
    
    # Only 3 feedback items recorded (< 5)
    priors = UserPriors(
        user_id="user1",
        total_feedback_events=3,
        emotion_counts={"sadness": 3},
    )
    
    base_dist = {"joy": 0.55, "sadness": 0.35, "neutral": 0.10}
    
    res = engine.calibrate(
        base_emotion="joy",
        base_confidence=0.55,
        base_distribution=base_dist,
        user_priors=priors,
        enabled=True,
    )
    
    assert res.applied is False
    assert res.personalized_emotion == "joy"
    assert res.personalized_confidence == 0.55
    assert "insufficient evidence" in res.explanation.lower()
    assert "3/5" in res.explanation


def test_valid_correction_history_activates_calibration_and_adjusts_borderline():
    engine = PersonalizationEngine(min_feedback_threshold=5, max_beta=0.65, half_life_k=10.0)
    
    # User repeatedly corrects borderline 'neutral' to 'fear'
    from collections import Counter, defaultdict
    conf = defaultdict(Counter)
    conf["neutral"]["fear"] = 8
    
    priors = UserPriors(
        user_id="user_anxious",
        total_feedback_events=8,
        emotion_counts=Counter({"fear": 8}),
        confusion_counts=conf,
    )
    
    # Borderline ambiguous case where base model slightly favored neutral (0.42) over fear (0.38)
    base_dist = {
        "anger": 0.02,
        "disgust": 0.02,
        "fear": 0.38,
        "joy": 0.04,
        "neutral": 0.42,
        "sadness": 0.08,
        "surprise": 0.04,
    }
    
    res = engine.calibrate(
        base_emotion="neutral",
        base_confidence=0.42,
        base_distribution=base_dist,
        user_priors=priors,
        enabled=True,
    )
    
    # Should calibrate prediction to fear because user repeatedly corrected neutral -> fear
    assert res.applied is True
    assert res.original_emotion == "neutral"
    assert res.personalized_emotion == "fear"
    assert res.personalized_confidence > 0.40
    assert "personalized" in res.explanation.lower()


def test_prevention_of_extreme_adjustments_on_high_confidence():
    engine = PersonalizationEngine(min_feedback_threshold=5, max_beta=0.65, half_life_k=10.0)
    
    # User has 20 corrections for sadness
    priors = UserPriors(
        user_id="user_sad",
        total_feedback_events=20,
        emotion_counts={"sadness": 20},
    )
    
    # But model has overwhelming confidence in Joy (0.96) vs Sadness (0.01)
    base_dist = {
        "anger": 0.005,
        "disgust": 0.005,
        "fear": 0.005,
        "joy": 0.96,
        "neutral": 0.015,
        "sadness": 0.01,
        "surprise": 0.00,
    }
    
    res = engine.calibrate(
        base_emotion="joy",
        base_confidence=0.96,
        base_distribution=base_dist,
        user_priors=priors,
        enabled=True,
    )
    
    # Shrinkage bounding ensures 0.96 joy is NEVER flipped to sadness
    assert res.personalized_emotion == "joy"
    assert res.personalized_confidence > 0.85


def test_no_future_data_leakage(client):
    headers = {"X-User-ID": "temporal_user"}
    
    # Create 5 entries sequentially
    entry_ids = []
    for i in range(5):
        res = client.post(
            "/api/entries",
            headers=headers,
            json={"text": f"Entry number {i} with mixed feelings", "mood": "neutral"},
        )
        assert res.status_code == 201
        entry_ids.append(res.json()["id"])
        
        # Add feedback
        client.post(
            f"/api/entries/{entry_ids[-1]}/correction",
            headers=headers,
            json={"helpful": False, "corrected_emotion": "sadness"},
        )
        
    # Check status: 5 feedback items recorded -> Personalization now active
    status_res = client.get("/api/personalization/status", headers=headers).json()
    assert status_res["feedback_count"] == 5
    assert status_res["active"] is True
    
    # Inspect entry 1: When entry 1 was analyzed, feedback count was 0
    # Its original stored analysis must have personalization_applied = False
    get_e1 = client.get(f"/api/entries/{entry_ids[0]}", headers=headers).json()
    assert get_e1["analysis"]["personalization_applied"] is False


def test_cross_user_isolation(client):
    alice = {"X-User-ID": "alice_pers"}
    bob = {"X-User-ID": "bob_pers"}
    
    # Alice provides 6 corrections to 'fear'
    for i in range(6):
        e = client.post("/api/entries", headers=alice, json={"text": f"Alice event {i}", "mood": "anxious"}).json()
        client.post(f"/api/entries/{e['id']}/correction", headers=alice, json={"helpful": False, "corrected_emotion": "fear"})
        
    alice_status = client.get("/api/personalization/status", headers=alice).json()
    assert alice_status["feedback_count"] == 6
    assert alice_status["active"] is True
    
    # Bob has 0 feedback events
    bob_status = client.get("/api/personalization/status", headers=bob).json()
    assert bob_status["feedback_count"] == 0
    assert bob_status["active"] is False


def test_personalization_toggle_and_reset(client):
    headers = {"X-User-ID": "toggle_user"}
    
    # 1. Disable personalization via API
    toggle_off = client.post("/api/personalization/toggle", headers=headers, json={"enabled": False}).json()
    assert toggle_off["enabled"] is False
    
    status1 = client.get("/api/personalization/status", headers=headers).json()
    assert status1["enabled"] is False
    assert status1["active"] is False
    
    # 2. Re-enable personalization
    toggle_on = client.post("/api/personalization/toggle", headers=headers, json={"enabled": True}).json()
    assert toggle_on["enabled"] is True
    
    # 3. Add 5 feedback items
    for i in range(5):
        e = client.post("/api/entries", headers=headers, json={"text": f"Toggle entry {i}", "mood": "happy"}).json()
        client.post(f"/api/entries/{e['id']}/correction", headers=headers, json={"helpful": True})
        
    status2 = client.get("/api/personalization/status", headers=headers).json()
    assert status2["feedback_count"] == 5
    assert status2["active"] is True
    
    # 4. Reset personalization
    reset_res = client.post("/api/personalization/reset", headers=headers)
    assert reset_res.status_code == 200
    
    status3 = client.get("/api/personalization/status", headers=headers).json()
    assert status3["feedback_count"] == 0
    assert status3["active"] is False
