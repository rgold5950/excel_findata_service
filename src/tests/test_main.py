from fastapi.testclient import TestClient
import json
from src.main import app
import pandas as pd

client = TestClient(app)


def test_welcome():
    response = client.get("/")
    print(response, response.text)
    assert response.status_code == 200
    assert response.text == '"Welcome to the investment data service."'


def test_get_tickers():
    response = client.get("/tickers")
    assert response.status_code == 200
    assert response.json() == ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]


def test_get_returns():
    response = client.get("/returns/AAPL/2022-01-01/2022-12-31")
    assert response.status_code == 200
    assert "ticker" in response.json()
    assert "returns" in response.json()


def test_get_correlation():
    response = client.get("/correlation/AAPL/GOOGL/2022-01-01/2022-12-31")
    assert response.status_code == 200
    assert "ticker1" in response.json()
    assert "ticker2" in response.json()
    assert "correlation" in response.json()


def test_post_correlation_matrix():
    payload = {
        "tickers": [{"ticker": "AAPL"}, {"ticker": "GOOGL"}, {"ticker": "MSFT"}],
        "start_date": "2022-01-01",
        "end_date": "2022-12-31",
    }
    response = client.post("/correlation_matrix/", json=payload)
    sample = {
        "AAPL": {
            "AAPL": 1.0,
            "GOOGL": 0.7982688200067611,
            "MSFT": 0.8249012032887364,
        },
        "GOOGL": {
            "AAPL": 0.7982688200067611,
            "GOOGL": 1.0,
            "MSFT": 0.8503358558555891,
        },
        "MSFT": {
            "AAPL": 0.8249012032887364,
            "GOOGL": 0.8503358558555891,
            "MSFT": 1.0,
        },
    }
    sample_df = pd.DataFrame(sample).round(5)
    test_df = pd.DataFrame(response.json()).round(5)
    assert response.status_code == 200
    assert test_df.equals(sample_df) == True
