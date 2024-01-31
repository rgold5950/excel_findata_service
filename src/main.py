from fastapi import FastAPI, HTTPException, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
from typing import List
import logging
from pydantic import BaseModel, field_validator
from functools import cache

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPPORTED_TICKERS = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]


class TickerParam(BaseModel):
    ticker: str

    @field_validator("ticker")
    def validate(cls, t):
        if not isinstance(t, str):
            raise HTTPException(status_code=422, detail="Ticker must be a valid string")
        if t not in SUPPORTED_TICKERS:
            raise HTTPException(status_code=422, detail="Ticker not supported")
        return t


class TickerParamDependency:
    def __call__(self, ticker_param: TickerParam = Depends()):
        return ticker_param.ticker


class CorrelationMatrixParams(BaseModel):
    tickers: List[TickerParam] = []
    start_date: str
    end_date: str


@cache
def price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Wrapper for yfinance api to handle errors and malformed inputs"""
    logger.info(f"Fetching Data for {ticker} between {start_date} and {end_date}")
    try:
        return yf.download(ticker, start_date, end_date)
    except Exception as e:
        logger.error(
            f"Encountered an error while trying to fetch data from yfinance: \n{e}"
        )
        raise HTTPException(status_code=503, detail="Gateway Timeout")


@cache
def returns(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    return (
        price_data(ticker, start_date=start_date, end_date=end_date)["Adj Close"]
        .pct_change()
        .dropna()
    )


def correlations(
    tickers: List[TickerParam], start_date: str, end_date: str
) -> pd.DataFrame:
    return pd.concat(
        [returns(t, start_date, end_date).rename(t) for t in tickers], axis=1
    ).corr()


@app.get("/")
async def welcome():
    return "Welcome to the investment data service."


@app.get("/tickers", response_model=List[str])
async def get_tickers():
    return SUPPORTED_TICKERS


@app.get("/returns/{ticker}/{start_date}/{end_date}")
async def get_returns(
    *,
    ticker: TickerParam = Depends(TickerParamDependency()),
    start_date: str,
    end_date: str,
):
    return {
        "ticker": ticker,
        "returns": returns(ticker, start_date, end_date).to_dict(),
    }


@app.get("/correlation/{ticker1}/{ticker2}/{start_date}/{end_date}")
async def get_correlation(
    *,
    ticker1: str,
    ticker2: str,
    start_date: str,
    end_date: str,
):
    return {
        "ticker1": ticker1,
        "ticker2": ticker2,
        "correlation": correlations([ticker1, ticker2], start_date, end_date).iloc[
            1, 0
        ],
    }


@app.post("/correlation_matrix/")
async def post_correlation_matrix(params: CorrelationMatrixParams):
    tickers = [t.ticker for t in params.tickers]
    start_date = params.start_date
    end_date = params.end_date
    return correlations(tickers, start_date, end_date).to_dict()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
