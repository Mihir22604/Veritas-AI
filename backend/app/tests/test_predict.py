from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# -----------------------------
# Successful prediction test
# -----------------------------
def test_prediction_success():
    response = client.post(
        "/predict/",
        json={"text": "The sun rises in the east every day"}
    )

    assert response.status_code == 200

    data = response.json()
    assert "label" in data
    assert "confidence" in data

    assert data["label"] in ["Fake", "Genuine"]
    assert isinstance(data["confidence"], float)
    assert 0.0 <= data["confidence"] <= 100.0


# -----------------------------
# Empty text validation test
# -----------------------------
def test_prediction_empty_text():
    response = client.post(
        "/predict/",
        json={"text": "   "}
    )

    assert response.status_code == 422


# -----------------------------
# Missing field validation test
# -----------------------------
def test_prediction_missing_text():
    response = client.post(
        "/predict/",
        json={}
    )

    assert response.status_code == 422
