"""Tests for Flask web app endpoints."""

import os
import sys
import json

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Skip if model files don't exist (CI environment)
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "demand_model.pkl"
)
pytestmark = pytest.mark.skipif(
    not os.path.exists(MODEL_PATH),
    reason="Model files not found (CI environment)"
)


@pytest.fixture
def client():
    from web.app import app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_index_returns_200(client):
    resp = client.get("/")
    assert resp.status_code == 200


def test_api_products_returns_list(client):
    resp = client.get("/api/products?bakery=test")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert isinstance(data, list)


def test_predict_missing_data_returns_error(client):
    resp = client.post("/predict", json={
        "bakery": "NONEXISTENT",
        "product": "NONEXISTENT",
        "date": "2099-01-01",
    })
    data = json.loads(resp.data)
    assert data["success"] is False
