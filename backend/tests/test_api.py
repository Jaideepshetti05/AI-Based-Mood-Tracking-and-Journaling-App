def test_health_and_analysis(client):
    desktop = client.get("/")
    assert desktop.status_code == 200
    assert "My Mood Journal" in desktop.text
    assert client.get("/api/health").json() == {"status": "ok"}
    response = client.post("/api/analyze", json={"text": "I am very happy today!"})
    assert response.status_code == 200
    assert response.json()["sentiment"] == "positive"
    assert -1 <= response.json()["compound_score"] <= 1


def test_entry_correction_insights_and_isolation(client):
    headers = {"X-User-ID": "alice"}
    created = client.post("/api/entries", headers=headers, json={"text": "A difficult day", "mood": "sad"}).json()
    entry_id = created["id"]
    assert created["analysis"]["emotion"] == "sadness"
    correction = client.post(
        f"/api/entries/{entry_id}/correction", headers=headers,
        json={"helpful": False, "corrected_emotion": "grief"},
    )
    assert correction.status_code == 200
    assert client.get("/api/entries", headers={"X-User-ID": "bob"}).json() == []
    insights = client.get("/api/insights/weekly", headers=headers).json()
    assert insights["entry_count"] == 1
    assert insights["correction_count"] == 1
    assert insights["personalization"]["status"] == "learning"


def test_updating_analyzed_entry_reuses_analysis_row(client):
    headers = {"X-User-ID": "alice"}
    created = client.post("/api/entries", headers=headers, json={"text": "A good day", "mood": "happy"}).json()
    updated = client.put(
        f"/api/entries/{created['id']}", headers=headers,
        json={"text": "A hard day", "mood": "sad", "analyze": True},
    )
    assert updated.status_code == 200
    assert updated.json()["mood"] == "sad"
    assert updated.json()["analysis"]["sentiment"] == "negative"


def test_validation_and_deletion(client):
    invalid = client.post("/api/entries", json={"text": " ", "mood": "sad"})
    assert invalid.status_code == 422
    invalid_mood = client.post("/api/entries", json={"text": "hello", "mood": "unknown"})
    assert invalid_mood.status_code == 422
    created = client.post("/api/entries", json={"text": "hello", "mood": "calm"}).json()
    assert client.delete(f"/api/entries/{created['id']}").status_code == 204
    assert client.get(f"/api/entries/{created['id']}").status_code == 404


def test_delete_all_data_is_scoped(client):
    client.post("/api/entries", headers={"X-User-ID": "alice"}, json={"text": "one", "mood": "calm"})
    client.post("/api/entries", headers={"X-User-ID": "bob"}, json={"text": "two", "mood": "calm"})
    assert client.delete("/api/data", headers={"X-User-ID": "alice"}).status_code == 204
    assert client.get("/api/entries", headers={"X-User-ID": "alice"}).json() == []
    assert len(client.get("/api/entries", headers={"X-User-ID": "bob"}).json()) == 1
