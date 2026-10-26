import csv
import io


def test_health_and_analysis(client):
    desktop = client.get("/")
    assert desktop.status_code == 200
    assert "My Mood Journal" in desktop.text

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}

    info = client.get("/api/model/info")
    assert info.status_code == 200
    assert "model" in info.json()

    response = client.post("/api/analyze", json={"text": "I am very happy and joyful today!"})
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "positive"
    assert -1.0 <= data["compound_score"] <= 1.0
    assert data["emotion"] in ("joy", "neutral", "admiration", "happy")
    assert "limitation" in data
    assert "Predictions are automated estimates" in data["limitation"]


def test_entry_creation_and_positive_feedback(client):
    headers = {"X-User-ID": "alice"}
    created = client.post(
        "/api/entries",
        headers=headers,
        json={"text": "Everything went great today at work!", "mood": "happy", "note": "promotion"},
    ).json()
    entry_id = created["id"]
    assert created["mood"] == "happy"
    assert created["analysis"] is not None

    # Positive feedback: user confirms the prediction was accurate
    feedback_res = client.post(
        f"/api/entries/{entry_id}/correction",
        headers=headers,
        json={"helpful": True, "corrected_emotion": None},
    )
    assert feedback_res.status_code == 200
    feedback_data = feedback_res.json()
    assert feedback_data["helpful"] is True
    assert feedback_data["corrected_emotion"] is None

    # Check updated entry response
    entry_res = client.get(f"/api/entries/{entry_id}", headers=headers)
    assert entry_res.status_code == 200
    assert entry_res.json()["correction"]["helpful"] is True

    # Check insights reflect agreement
    insights = client.get("/api/insights/weekly", headers=headers).json()
    assert insights["entry_count"] == 1
    assert insights["correction_count"] == 1
    assert insights["agreement_count"] == 1
    assert insights["agreement_rate"] == 1.0


def test_entry_negative_feedback_and_correction(client):
    headers = {"X-User-ID": "alice"}
    created = client.post(
        "/api/entries",
        headers=headers,
        json={"text": "A difficult and sad day", "mood": "sad"},
    ).json()
    entry_id = created["id"]

    # Negative feedback: user corrects emotion to grief
    correction = client.post(
        f"/api/entries/{entry_id}/correction",
        headers=headers,
        json={"helpful": False, "corrected_emotion": "grief"},
    )
    assert correction.status_code == 200
    data = correction.json()
    assert data["helpful"] is False
    assert data["corrected_emotion"] == "grief"

    # Validation: unhelpful feedback without corrected emotion is rejected
    invalid_correction = client.post(
        f"/api/entries/{entry_id}/correction",
        headers=headers,
        json={"helpful": False, "corrected_emotion": None},
    )
    assert invalid_correction.status_code == 422


def test_updating_analyzed_entry_clears_stale_correction(client):
    headers = {"X-User-ID": "alice"}
    created = client.post(
        "/api/entries",
        headers=headers,
        json={"text": "A good day celebrating", "mood": "happy"},
    ).json()
    entry_id = created["id"]

    # Submit a correction
    client.post(
        f"/api/entries/{entry_id}/correction",
        headers=headers,
        json={"helpful": True},
    )

    # Edit the entry text
    updated = client.put(
        f"/api/entries/{entry_id}",
        headers=headers,
        json={"text": "Actually it turned into a hard day", "mood": "sad", "analyze": True},
    )
    assert updated.status_code == 200
    updated_data = updated.json()
    assert updated_data["mood"] == "sad"
    # Prior correction must be cleared because underlying text was modified
    assert updated_data["correction"] is None


def test_multiline_editing_preserves_content(client):
    headers = {"X-User-ID": "alice"}
    multiline_text = "Paragraph 1: Morning thoughts.\n\nParagraph 2: Afternoon reflection.\n\nParagraph 3: Evening gratitude."
    created = client.post(
        "/api/entries",
        headers=headers,
        json={"text": multiline_text, "mood": "calm", "note": "Daily reflection"},
    ).json()
    assert created["text"] == multiline_text

    read_back = client.get(f"/api/entries/{created['id']}", headers=headers).json()
    assert read_back["text"] == multiline_text
    assert "\n\n" in read_back["text"]


def test_cross_user_isolation_and_forbidden_access(client):
    alice_headers = {"X-User-ID": "alice"}
    bob_headers = {"X-User-ID": "bob"}

    alice_entry = client.post(
        "/api/entries",
        headers=alice_headers,
        json={"text": "Alice's secret thoughts", "mood": "calm"},
    ).json()
    entry_id = alice_entry["id"]

    # Bob should not see Alice's entry in their list
    bob_list = client.get("/api/entries", headers=bob_headers).json()
    assert len(bob_list) == 0

    # Bob attempting to read Alice's specific entry directly must return 403 Forbidden
    cross_read = client.get(f"/api/entries/{entry_id}", headers=bob_headers)
    assert cross_read.status_code == 403
    assert "Forbidden" in cross_read.json()["detail"]

    # Bob attempting to edit Alice's entry must return 403 Forbidden
    cross_edit = client.put(
        f"/api/entries/{entry_id}",
        headers=bob_headers,
        json={"text": "Hacked", "mood": "angry"},
    )
    assert cross_edit.status_code == 403

    # Bob attempting to delete Alice's entry must return 403 Forbidden
    cross_del = client.delete(f"/api/entries/{entry_id}", headers=bob_headers)
    assert cross_del.status_code == 403


def test_csv_export_comprehensive_schema_and_sanitization(client):
    headers = {"X-User-ID": "alice"}

    # 1. Normal entry with AI
    entry1 = client.post(
        "/api/entries",
        headers=headers,
        json={"text": "Joyful celebration today", "mood": "happy", "note": "Party"},
    ).json()

    # Add correction
    client.post(
        f"/api/entries/{entry1['id']}/correction",
        headers=headers,
        json={"helpful": True},
    )

    # 2. Entry with potential CSV formula injection text
    client.post(
        "/api/entries",
        headers=headers,
        json={"text": "=SUM(1+1)*cmd|' /C calc'!A0", "mood": "neutral", "note": "@formula_test"},
    )

    export_res = client.get("/api/export", headers=headers)
    assert export_res.status_code == 200
    assert export_res.headers["content-type"].startswith("text/csv")

    csv_reader = list(csv.reader(io.StringIO(export_res.text)))
    assert len(csv_reader) >= 3  # Header + 2 entries

    expected_headers = [
        "id", "created_at", "updated_at", "mood", "text", "note",
        "sentiment", "compound_score", "predicted_emotion", "confidence",
        "model_name", "feedback_status", "corrected_emotion", "correction_timestamp"
    ]
    assert csv_reader[0] == expected_headers

    # Check formula injection was sanitized with leading quote
    formula_row = next(row for row in csv_reader[1:] if "calc" in row[4])
    assert formula_row[4].startswith("'=")
    assert formula_row[5].startswith("'@")


def test_validation_and_deletion(client):
    # Blank text
    invalid = client.post("/api/entries", json={"text": "   ", "mood": "sad"})
    assert invalid.status_code == 422

    # Invalid mood
    invalid_mood = client.post("/api/entries", json={"text": "hello", "mood": "unknown_mood"})
    assert invalid_mood.status_code == 422

    # Valid creation and deletion
    created = client.post("/api/entries", json={"text": "hello world", "mood": "calm"}).json()
    del_res = client.delete(f"/api/entries/{created['id']}")
    assert del_res.status_code == 204

    # Subsequent read is 404
    assert client.get(f"/api/entries/{created['id']}").status_code == 404


def test_delete_all_data_is_scoped(client):
    client.post("/api/entries", headers={"X-User-ID": "alice"}, json={"text": "one", "mood": "calm"})
    client.post("/api/entries", headers={"X-User-ID": "bob"}, json={"text": "two", "mood": "calm"})

    del_res = client.delete("/api/data", headers={"X-User-ID": "alice"})
    assert del_res.status_code == 204

    assert client.get("/api/entries", headers={"X-User-ID": "alice"}).json() == []
    assert len(client.get("/api/entries", headers={"X-User-ID": "bob"}).json()) == 1


def test_insights_analytics_and_daily_trend(client):
    headers = {"X-User-ID": "alice"}
    client.post("/api/entries", headers=headers, json={"text": "Happy day", "mood": "happy"})
    client.post("/api/entries", headers=headers, json={"text": "Exciting event", "mood": "excited"})

    insights = client.get("/api/insights/weekly", headers=headers).json()
    assert insights["entry_count"] == 2
    assert "daily_trend" in insights
    assert isinstance(insights["daily_trend"], list)
    assert len(insights["daily_trend"]) >= 1
    assert "alignment_count" in insights
    assert "divergence_count" in insights
