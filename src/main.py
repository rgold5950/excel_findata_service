from fastapi import FastAPI, HTTPException, Depends, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
from typing import List
import logging
from pydantic import BaseModel, field_validator
from functools import cache
import json
from starlette.middleware.base import BaseHTTPMiddleware
from IPython import embed
from enum import Enum
from datetime import datetime
from fastapi.responses import JSONResponse
from enum import Enum


class ExcelParamType(Enum):
    BOOLEAN = "boolean"
    NUMBER = "number"
    STRING = "string"
    ANY = "any"


from enum import Enum


class ExcelParamType(Enum):
    BOOLEAN = "boolean"
    NUMBER = "number"
    STRING = "string"
    ANY = "any"


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
BATCHABLE = ["/agg_return/{ticker}/{return_window}"]
BATCH_ENDPOINT_MAPPING = {"/agg_return/{ticker}/{return_window}": "/batch_agg_return/"}


class TickerParam(BaseModel):
    ticker: str

    @field_validator("ticker")
    def validate(cls, t):
        if not isinstance(t, str):
            raise HTTPException(status_code=422, detail="Ticker must be a valid string")
        if t not in SUPPORTED_TICKERS:
            raise HTTPException(status_code=422, detail="Ticker not supported")
        return t


class ReturnWindow(Enum):
    YTD = "YTD"
    MTD = "MTD"


class ReturnWindowParam(BaseModel):
    return_window: ReturnWindow


class AggReturnParam(BaseModel):
    ticker: str
    return_window: ReturnWindow


class AggReturnParams(BaseModel):
    params: List[AggReturnParam]


class TickerParamDependency:
    def __call__(self, ticker_param: TickerParam = Depends()):
        return ticker_param.ticker


class ReturnParamDependency:
    def __call__(self, return_param: ReturnWindowParam = Depends()):
        return return_param.return_window


class CorrelationMatrixParams(BaseModel):
    tickers: List[TickerParam]
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


@app.get("/tickers")
async def get_tickers() -> list:
    return [[t] for t in SUPPORTED_TICKERS]


@app.get("/returns/{ticker}/{start_date}/{end_date}")
def get_returns(
    *,
    ticker: TickerParam = Depends(TickerParamDependency()),
    start_date: str,
    end_date: str,
):
    rtns_df = returns(ticker, start_date, end_date).reset_index(drop=False)
    rtns_df["Date"] = rtns_df["Date"].dt.strftime("%Y-%m-%d")
    return json.dumps(
        {
            "ticker": ticker,
            "returns": rtns_df.values.tolist(),
        }
    )


def calc_agg_return(ticker: str, return_window: ReturnWindow) -> float:
    today = datetime.today()
    last_bday = today - pd.offsets.BDay(1)
    last_bday = last_bday.normalize()
    if return_window == ReturnWindow.MTD:
        start_date = datetime(last_bday.year, last_bday.month, 1)
    elif return_window == ReturnWindow.YTD:
        start_date = datetime(last_bday.year, 1, 2)
    else:
        return f"Return Window {return_window} Not Supported"
    if last_bday <= start_date:
        return json.dumps(0.0)
    rtns_df = price_data(
        ticker,
        start_date=start_date.date().isoformat(),
        end_date=today.date().isoformat(),
    ).sort_index(ascending=True)["Adj Close"]
    start = rtns_df.iloc[0]
    end = rtns_df.iloc[-1]
    return (start - end) / start


@app.get("/agg_return/{ticker}/{return_window}")
def get_returns_agg(
    *,
    ticker: TickerParam = Depends(TickerParamDependency()),
    return_window: ReturnWindow,
) -> float:
    return json.dumps(calc_agg_return(ticker, return_window))


@app.post("/batch_agg_return/")
def batch_get_returns_agg(agg_return_params: AggReturnParams) -> list[dict[str, float]]:
    results = []
    for params in agg_return_params.params:
        try:
            result = {"result": calc_agg_return(params.ticker, params.return_window)}
        except Exception as e:
            result = {"error": e}
        results.append(result)
    return results


@app.get("/spilled_returns/{ticker}/{start_date}/{end_date}")
def spilled_returns(
    *,
    ticker: TickerParam = Depends(TickerParamDependency()),
    start_date: str,
    end_date: str,
) -> list:
    rtns_df = returns(ticker, start_date, end_date).reset_index(drop=False)
    rtns_df["Date"] = rtns_df["Date"].dt.strftime("%Y-%m-%d")
    return rtns_df.values.tolist()


@app.get("/correlation/{ticker1}/{ticker2}/{start_date}/{end_date}")
def get_correlation(
    *,
    ticker1: str,
    ticker2: str,
    start_date: str,
    end_date: str,
):
    return json.dumps(
        {
            "ticker1": ticker1,
            "ticker2": ticker2,
            "correlation": correlations([ticker1, ticker2], start_date, end_date).iloc[
                1, 0
            ],
        }
    )


@app.post("/correlation_matrix/")
def post_correlation_matrix(params: CorrelationMatrixParams):
    tickers = [t.ticker for t in params.tickers]
    start_date = params.start_date
    end_date = params.end_date
    return json.dumps(correlations(tickers, start_date, end_date).to_dict())


@app.post("/correlation_matrix_spilled/")
def post_correlation_matrix_spilled(params: CorrelationMatrixParams) -> list:
    tickers = [t.ticker for t in params.tickers]
    start_date = params.start_date
    end_date = params.end_date
    df = correlations(tickers, start_date, end_date)
    return [[None] + df.columns.tolist()] + df.reset_index(drop=False).values.tolist()


@app.get("/goldman/")
async def goldman():
    return "hi goldman"


@app.get("/changesalot")
async def thisonechangesalot(
    some_param: str = Query(...), another_param: str = Query(...)
):
    return json.dumps({"some_param": some_param, "another_param": another_param})


def openapi_to_custom_functions(openapi_schema: dict) -> dict:
    custom_functions = {"allowCustomDataForDataTypeAny": True, "functions": []}
    endpoint_map = {}
    now = ""

    valid_types = {
        ExcelParamType.BOOLEAN.value,
        ExcelParamType.NUMBER.value,
        ExcelParamType.STRING.value,
    }

    for path, path_item in openapi_schema.get("paths", {}).items():
        for method, operation in path_item.items():
            if operation.get("summary") == "Get Functions":
                continue  # skip the functions endpoint
            name = operation.get("summary").upper().replace(" ", "_")
            if name.upper().startswith("BATCH"):
                continue  # these functions should only be called indirectly via the non-batch equivalents
            function = {
                "description": operation.get("summary", ""),
                "id": name + now,  # Use the modified path as the ID
                "name": name + now,  # Use the modified path as the name
                "parameters": [],
                "result": {
                    "type": ExcelParamType.ANY.value,
                    "dimensionality": "scalar",
                },
            }
            batchable = False
            batch_endpoint = ""
            if path in BATCHABLE:
                batchable = True
                batch_endpoint = BATCH_ENDPOINT_MAPPING[path]
            endpoint_info_item = {
                "endpoint": path,
                "method": method,
                "path_parameters": [
                    {"name": p["name"], "required": p.get("required", False)}
                    for p in operation.get("parameters", [])
                    if p["in"] == "path"
                ],
                "query_parameters": [
                    {"name": p["name"], "required": p.get("required", False)}
                    for p in operation.get("parameters", [])
                    if p["in"] == "query"
                ],
                "batchable": batchable,
                "batch_endpoint": batch_endpoint,
            }
            # Parse body parameters
            body_parameters = []
            if "requestBody" in operation:
                request_body = operation["requestBody"]
                if "content" in request_body:
                    for content_type, content_info in request_body["content"].items():
                        if "schema" in content_info:
                            schema = content_info["schema"]
                            if "$ref" in schema:
                                ref = schema["$ref"]
                                # Resolve the reference if needed (this example assumes direct schema)
                                ref_name = ref.split("/")[-1]
                                body_parameters = [
                                    {
                                        "name": k,
                                        "required": k
                                        in openapi_schema["components"]["schemas"][
                                            ref_name
                                        ].get("required", []),
                                        "type": v.get("type", ExcelParamType.ANY.value),
                                        "dimensionality": "matrix"
                                        if v.get("type") == "array"
                                        else "scalar",
                                    }
                                    for k, v in openapi_schema["components"]["schemas"][
                                        ref_name
                                    ]["properties"].items()
                                ]
                            elif "properties" in schema:
                                body_parameters = [
                                    {
                                        "name": k,
                                        "required": k in schema.get("required", []),
                                        "type": v.get("type", ExcelParamType.ANY.value),
                                        "dimensionality": "matrix"
                                        if v.get("type") == "array"
                                        else "scalar",
                                    }
                                    for k, v in schema["properties"].items()
                                ]

            endpoint_info_item["body_parameters"] = body_parameters
            endpoint_map[name + now] = endpoint_info_item

            # Extract parameters
            for param in operation.get("parameters", []):
                param_type = param.get("schema", {}).get(
                    "type", ExcelParamType.ANY.value
                )
                if param_type not in valid_types:
                    param_type = ExcelParamType.ANY.value
                function["parameters"].append(
                    {
                        "description": param.get("description", ""),
                        "name": param.get("name", ""),
                        "type": param_type,
                        "dimensionality": "scalar",
                    }
                )

            # Extract request body parameters if any
            if method.lower() == "post" and body_parameters:
                for param in body_parameters:
                    param_type = param.get("type", ExcelParamType.ANY.value)
                    if param_type not in valid_types:
                        param_type = ExcelParamType.ANY.value
                    function["parameters"].append(
                        {
                            "description": "",  # Description can be added if available in schema
                            "name": param["name"],
                            "type": param_type,
                            "dimensionality": param["dimensionality"],
                        }
                    )

            # Extract response type and dimensionality
            responses = operation.get("responses", {})
            for status, response in responses.items():
                if status.startswith("2"):  # Look for successful responses
                    content = response.get("content", {})
                    for content_type, media_type in content.items():
                        schema = media_type.get("schema", {})
                        result_type = schema.get("type", ExcelParamType.ANY.value)
                        if result_type not in valid_types:
                            result_type = ExcelParamType.ANY.value
                        function["result"]["type"] = result_type
                        function["result"]["dimensionality"] = (
                            "matrix"
                            if schema.get("type") == "array" or "items" in schema
                            else "scalar"
                        )
                        break

            custom_functions["functions"].append(function)
    return custom_functions, endpoint_map


@app.get("/functions.json", response_class=JSONResponse)
def get_functions():
    openapi_schema = app.openapi()
    data, endpoint_map = openapi_to_custom_functions(openapi_schema)
    res = {"functions_metadata": data, "endpoint_map": endpoint_map}
    return JSONResponse(content=res)


if __name__ == "__main__":
    import uvicorn

    import os

    base_dir = os.path.expanduser("~/.office-addin-dev-certs")

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        ssl_keyfile="/Users/richardgoldman/.office-addin-dev-certs/localhost.key",
        ssl_certfile="/Users/richardgoldman/.office-addin-dev-certs/localhost.crt",
    )
